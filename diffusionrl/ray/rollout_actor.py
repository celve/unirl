"""diffusionrl Rollout actor implementation (generation side)."""

import logging
import os
from typing import Any, Dict, Optional

import ray
import torch

from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import RolloutActorConfig
from diffusionrl.ray.mixins import RolloutPipelineMixin, RolloutWeightSyncMixin
from diffusionrl.ray.utils.gpu import log_gpu_state, log_resource_ids
from diffusionrl.ray.utils.net import get_free_port, get_node_ip
from diffusionrl.reward.config import RewardSpec
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.samplers.construction import create_rollout_engine_from_init_payload
from diffusionrl.samplers.engine import BaseRolloutEngine
from diffusionrl.samplers.registry import derive_rollout_engine_class
from diffusionrl.transfer.buffer import Buffer
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.sample import RolloutSamples

logger = logging.getLogger(__name__)


@ray.remote
class RolloutActor(RolloutWeightSyncMixin, RolloutPipelineMixin, Buffer):
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
        config: Optional[dict] = None,
        num_gpus_allocated: int = 1,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        base_gpu_id: int = 0,
        force_set_cuda_visible_devices: bool = False,
    ):
        """
        Initialize rollout actor.

        Args:
            rank: This actor's rank in the rollout group
            world_size: Total number of rollout actors
            config: Optional initial configuration
            num_gpus_allocated: Number of GPUs allocated to this actor
                               (must match Ray's num_gpus option)
            master_addr: Master node address for distributed (multi-GPU)
            master_port: Master node port for distributed (multi-GPU)
            base_gpu_id: Starting physical GPU ID (for Slime NOSET pattern).
                        When > 0, CUDA_VISIBLE_DEVICES is set manually to
                        [base_gpu_id, base_gpu_id+1, ..., base_gpu_id+num_gpus-1].
            force_set_cuda_visible_devices: Force manual CUDA_VISIBLE_DEVICES setup
                        even when base_gpu_id is 0 (needed for NOSET mode).
        """
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.config = config or {}
        self._rollout_batch_size: Optional[int] = None
        self.num_gpus_allocated = num_gpus_allocated
        self.master_addr = master_addr
        self.master_port = master_port
        self.base_gpu_id = base_gpu_id
        self.force_set_cuda_visible_devices = bool(force_set_cuda_visible_devices)
        self.engine: Optional[BaseRolloutEngine] = None
        self.algorithm: Optional[Any] = None
        self._device = None
        self._reward_spec: Optional[RewardSpec] = None
        self._reward_pipeline: Optional[RewardPipeline] = None

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
        When using the Slime NOSET pattern (base_gpu_id > 0), also sets
        CUDA_VISIBLE_DEVICES manually since Ray won't do it.
        """
        if self.num_gpus_allocated <= 1:
            return

        # When using NOSET mode, manually set CUDA_VISIBLE_DEVICES.
        # base_gpu_id can be 0 when actor is assigned the first physical GPU group.
        if self.force_set_cuda_visible_devices or self.base_gpu_id > 0:
            gpu_range = ",".join(str(self.base_gpu_id + i) for i in range(self.num_gpus_allocated))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_range
            logger.info(f"Rank {self.rank}: Set CUDA_VISIBLE_DEVICES={gpu_range}")

        master_addr = self.master_addr or get_node_ip()
        master_port = self.master_port or get_free_port()

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(self.num_gpus_allocated)
        os.environ["RANK"] = "0"  # Single actor manages all GPUs
        os.environ["LOCAL_RANK"] = "0"

        logger.info(
            f"Rank {self.rank}: Distributed env setup - "
            f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, "
            f"WORLD_SIZE={self.num_gpus_allocated}"
        )

    def _ensure_engine_ready_for_generate(self) -> None:
        """Ensure generation path always starts from an active engine state."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not self.engine.is_initialized:
            raise RuntimeError("Engine initialization incomplete.")
        self.engine.wake_up()

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        if self._reward_spec is None:
            raise RuntimeError("Reward pipeline requested before reward spec initialization.")
        if self._reward_pipeline is None:
            self._reward_pipeline = RewardPipeline.from_spec(self._reward_spec)
        return self._reward_pipeline

    def init(self, config: RolloutActorConfig) -> None:
        """
        Initialize the rollout actor and underlying engine.

        Args:
            config: Typed RolloutActorConfig including engine_init_payload and
                reward_config.

        Raises:
            ValueError: If required sections or fields are not provided
        """
        if not isinstance(config, RolloutActorConfig):
            raise ValueError(f"rollout actor init config must be a RolloutActorConfig, got: {type(config).__name__}")

        # Per-actor determinism setup: must run BEFORE any CUDA op so that
        # cuDNN / cuBLAS / deterministic-algorithm flags are in effect for
        # the subsequent engine initialization and sampling.
        from diffusionrl.utils import set_seed

        set_seed(int(config.seed))

        logger.info(f"Rank {self.rank}: Initializing rollout actor (num_gpus={self.num_gpus_allocated})...")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed environment for multi-GPU
        self._setup_distributed_env()
        engine_init_payload = config.engine_init_payload
        if not isinstance(engine_init_payload, ComponentInitPayload):
            raise ValueError(
                "rollout actor init config must provide engine_init_payload as ComponentInitPayload. "
                f"Got: {type(engine_init_payload).__name__}"
            )
        base_engine_config = engine_init_payload.component_config
        if not isinstance(base_engine_config, EngineConfig):
            raise ValueError(
                "rollout actor init config must provide EngineConfig inside engine_init_payload. "
                f"Got: {type(base_engine_config).__name__}"
            )
        algorithm_init_payload = config.algorithm_init_payload
        if not isinstance(algorithm_init_payload, ComponentInitPayload):
            raise ValueError(
                "rollout actor init config must provide algorithm_init_payload as ComponentInitPayload. "
                f"Got: {type(algorithm_init_payload).__name__}"
            )
        from diffusionrl.algorithms import create_algorithm_from_init_payload

        self.algorithm = create_algorithm_from_init_payload(algorithm_init_payload)
        engine_cls = derive_rollout_engine_class(engine_init_payload.component_dotpath)
        sampler_engine_type = (
            str(getattr(engine_cls, "_component_name", "") or getattr(engine_cls, "__name__", "")).strip().lower()
        )
        if not sampler_engine_type:
            raise ValueError("Failed to resolve rollout engine type from engine_init_payload.")

        self._reward_spec = config.reward_config

        resolved_engine_config = base_engine_config
        if sampler_engine_type == "sglang":
            resolved_engine_config = resolved_engine_config.with_sglang_ports(self.rank)

        self._rollout_batch_size = int(config.rollout_batch_size) if config.rollout_batch_size is not None else None
        self.engine = create_rollout_engine_from_init_payload(
            ComponentInitPayload(
                component_dotpath=engine_init_payload.component_dotpath,
                component_config=resolved_engine_config,
            )
        )

        # Initialize engine
        self.engine.initialize(self._device)

        logger.info(
            "Rank %s: Rollout actor initialized with %s engine%s",
            self.rank,
            sampler_engine_type,
            f" (rollout_batch_size={self._rollout_batch_size})" if self._rollout_batch_size else "",
        )
        self._log_resource_ids("rollout_init")
        self._log_gpu_state("rollout_init")

    def generate(self, request: RolloutRequest) -> RolloutResponse:
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not request.prompts or not request.prompts.prompts:
            raise ValueError("RolloutActor.generate requires non-empty prompts.")

        self._ensure_engine_ready_for_generate()
        self._log_gpu_state("inference_generate_start")
        batch_size = self._rollout_batch_size
        n_prompts = len(request.prompts.prompts)
        if batch_size and n_prompts > batch_size:
            outputs = []
            for i in range(0, n_prompts, batch_size):
                sub_request = request.slice(i, min(i + batch_size, n_prompts))
                outputs.append(self.engine.generate(sub_request))
            output = RolloutSamples.concat(outputs)
        else:
            output = self.engine.generate(request)
        return RolloutResponse(request=request, samples=output)

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
