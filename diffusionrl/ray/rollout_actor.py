"""diffusionrl Rollout actor implementation (generation side)."""

import logging
import os
from typing import Any, Dict, List, Optional

import ray
import torch
from omegaconf import DictConfig

from diffusionrl.ray.actor_config import ConfigActor
from diffusionrl.ray.distributed import DistributedMixin
from diffusionrl.ray.mixins import RolloutPipelineMixin, RolloutWeightSyncMixin
from diffusionrl.ray.utils.gpu import log_gpu_state, log_resource_ids
from diffusionrl.ray.utils.net import get_free_port, get_node_ip
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.samplers.engine import chunked_engine_generate
from diffusionrl.transfer.buffer import Buffer
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse

logger = logging.getLogger(__name__)


@ray.remote
class RolloutActor(ConfigActor, RolloutWeightSyncMixin, DistributedMixin, RolloutPipelineMixin, Buffer):
    """
    Rollout Actor - Manages sampling and generation via Engine interface.

    This actor hosts dedicated rollout-side services only:
    - SGLang: Distributed rollout inference service

    Direct-sampling engines (for example the default FSDP sampler path) run on
    ``TrainActor`` and should never be instantiated here.

    GPU Allocation:
        GPU count is configured at actor creation via .options(num_gpus=N).
        - FSDP: num_gpus=1 (single GPU per actor, default)
        - FSDP multi-GPU: num_gpus>1 (uses FSDP wrapper for model parallelism)
    Example:
        actor = RolloutActor.options(num_gpus=1).remote(
            rank=0, world_size=1, num_gpus_allocated=1
        )
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        num_gpus_allocated: int = 1,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        cfg: Optional[DictConfig] = None,
    ):
        """Initialize rollout actor and its underlying engine.

        Reads ``cfg.algorithm`` / ``cfg.rollout.engine`` / ``cfg.model`` and
        ``cfg.reward`` from the cfg ``ConfigActor`` installs into
        ``actor_config._current``. Algorithm + engine sections carry
        ``_target_`` and are materialized via ``build()``; engine takes
        ``rank`` and ``model_config`` as runtime deps. ``cfg.model`` is
        materialized via ``materialize``. ``cfg.reward`` is kept as a
        ``DictConfig`` and forwarded into ``RewardPipeline.from_configs``,
        which dispatches each component through ``build()``.

        Args:
            rank: This actor's rank in the rollout group
            world_size: Total number of rollout actors
            num_gpus_allocated: Number of GPUs allocated to this actor
                               (must match Ray's num_gpus option)
            master_addr: Master node address for distributed (multi-GPU)
            master_port: Master node port for distributed (multi-GPU)
            gpu_ids: Physical GPU ids this actor owns (Slime NOSET pattern).
                     When set (len > 1), the actor manually sets
                     ``CUDA_VISIBLE_DEVICES`` to the listed GPUs.
            cfg: Full composed Hydra cfg. ``ConfigActor`` installs it into
                 ``actor_config._current`` for ambient access by helpers.
        """
        from diffusionrl.config.instantiate import build, materialize
        from diffusionrl.utils import set_seed

        super().__init__(
            cfg=cfg,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        self.num_gpus_allocated = num_gpus_allocated
        self.gpu_ids = list(gpu_ids) if gpu_ids else []
        self._reward_pipeline: Optional[RewardPipeline] = None

        set_seed(int(self._cfg.run.seed))
        logger.info(
            "Rank %s: Initializing rollout actor (num_gpus=%d)...",
            self.rank,
            self.num_gpus_allocated,
        )
        # env-first: _setup_distributed_env writes CUDA_VISIBLE_DEVICES under
        # NOSET; must land before torch caches the visible-device list.
        self._setup_distributed_env()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.algorithm = build(self._cfg.algorithm)
        self._reward_config = self._cfg.reward
        self.engine = build(
            self._cfg.rollout.engine,
            rank=self.rank,
            model_config=materialize(self._cfg.model),
        )
        self.engine.strategy = build(self._cfg.sampling.sde_strategy)

        self._rollout_plan = materialize(self._cfg.rollout.plan)
        self.engine.initialize(self._device)

        logger.info(
            "Rank %s: Rollout actor initialized (forward_batch_size=%s)",
            self.rank,
            self._rollout_plan.forward_batch_size,
        )
        self._log_resource_ids("rollout_init")
        self._log_gpu_state("rollout_init")

    def _log_resource_ids(self, tag: str) -> None:
        log_resource_ids(tag, self.rank)

    def _log_gpu_state(self, tag: str) -> None:
        offloaded = None
        if self.engine is not None:
            try:
                offloaded = self.engine.is_offloaded
            except Exception:
                offloaded = None
        log_gpu_state(tag, self.rank, device=self._device, offloaded=offloaded)

    def _setup_distributed_env(self) -> None:
        """
        Setup environment variables for multi-GPU distributed rollout.

        This is called before engine initialization when num_gpus_allocated > 1.
        When using the Slime NOSET pattern (``gpu_ids`` supplied), also sets
        CUDA_VISIBLE_DEVICES manually since Ray won't do it.
        """
        if self.num_gpus_allocated <= 1:
            return

        if self.gpu_ids:
            cvd = ",".join(str(g) for g in self.gpu_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            logger.info(f"Rank {self.rank}: Set CUDA_VISIBLE_DEVICES={cvd}")

        master_addr = self.master_addr or get_node_ip()
        master_port = int(self.master_port or get_free_port())

        self._write_distributed_env(
            master_addr=master_addr,
            master_port=master_port,
            world_size=self.num_gpus_allocated,
            rank=0,
            local_rank=0,
        )

        logger.info(
            f"Rank {self.rank}: Distributed env setup - "
            f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, "
            f"WORLD_SIZE={self.num_gpus_allocated}"
        )

    def _ensure_engine_ready_for_generate(self) -> None:
        """Ensure generation path always starts from an active engine state."""
        if not self.engine.is_initialized:
            raise RuntimeError("Engine initialization incomplete.")
        self.engine.wake_up()

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        if self._reward_pipeline is None:
            self._reward_pipeline = RewardPipeline.from_configs(self._reward_config)
        return self._reward_pipeline

    def generate(self, request: RolloutRequest) -> RolloutResponse:
        if not request.prompts or not request.prompts.prompts:
            raise ValueError("RolloutActor.generate requires non-empty prompts.")

        self._ensure_engine_ready_for_generate()
        self._log_gpu_state("inference_generate_start")
        return RolloutResponse(
            request=request,
            samples=chunked_engine_generate(
                self.engine,
                request,
                chunk_size=self._rollout_plan.forward_batch_size,
            ),
        )

    def get_num_gpus_allocated(self) -> int:
        """Return physical GPU count allocated to this rollout actor."""
        return int(self.num_gpus_allocated)

    def sleep(self) -> None:
        """Put engine into sleep mode to release runtime resources."""
        if self.engine is not None:
            self.engine.sleep()
        if self._reward_pipeline is not None:
            self._reward_pipeline.offload()
        logger.info(f"Rank {self.rank}: Engine entered sleep mode")
        self._log_gpu_state("inference_sleep")

    def wake_up(self) -> None:
        """Wake engine up for generation or weight update."""
        if self.engine is not None:
            self.engine.wake_up()
        if self._reward_pipeline is not None:
            self._reward_pipeline.onload()
        logger.info(f"Rank {self.rank}: Engine wake_up complete")
        self._log_gpu_state("inference_wake_up")

    def health_check(self) -> bool:
        """Check if actor is healthy."""
        if self.engine is None:
            return False
        return self.engine.health_check()

    def is_offloaded(self) -> bool:
        """Check if actor is currently offloaded to CPU."""
        if self.engine is None:
            return False
        return self.engine.is_offloaded

    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self.engine is not None:
            return self.engine.get_memory_info()
        return {}


__all__ = ["RolloutActor"]
