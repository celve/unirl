"""diffusionrl rollout actor.

Wraps a :class:`diffusionrl.rollout.engine.base.BaseRolloutEngine`
(currently ``VLLMOmniRolloutEngine`` for HunyuanImage-3,
``SGLangRolloutEngine``, ``TrainsideRolloutEngine``) and speaks
``RolloutReq``/``RolloutResp`` end-to-end.

Engine construction is one-shot:
``build(cfg.rollout.engine, device=..., strategy=..., rank=..., model_config=...)``
returns a fully-usable engine; no separate ``engine.initialize(device)``
step and no post-ctor ``engine.strategy = ...`` mutation. The strategy
travels via the build kwarg and (for per-call control) via
``RolloutReq.stage_params``.

``generate(req)`` returns a ``RolloutResp`` directly — chunking is
handled by :func:`diffusionrl.rollout.engine.chunked_engine_generate_req`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import ray
import torch
from omegaconf import DictConfig

from diffusionrl.ray.actor_config import ConfigActor
from diffusionrl.ray.distributed import DistributedMixin
from diffusionrl.ray.mixins import RolloutWeightSyncMixin
from diffusionrl.ray.mixins.rollout_pipeline import RolloutPipelineMixin
from diffusionrl.ray.utils.gpu import log_gpu_state, log_resource_ids
from diffusionrl.ray.utils.net import get_free_port, get_node_ip
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.rollout.engine import chunked_engine_generate_req
from diffusionrl.transfer.buffer import Buffer
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

logger = logging.getLogger(__name__)


@ray.remote
class RolloutActor(ConfigActor, RolloutWeightSyncMixin, DistributedMixin, RolloutPipelineMixin, Buffer):
    """Rollout actor that drives a new-ABC engine end-to-end on RolloutReq/Resp.

    Hosts dedicated rollout-side services only — direct-sampling engines stay
    on ``TrainActor``. GPU allocation is configured at actor creation via
    ``.options(num_gpus=N)``.

    Example::

        actor = RolloutActor.options(num_gpus=8).remote(
            rank=0, world_size=1, num_gpus_allocated=8, gpu_ids=[0,1,2,3,4,5,6,7], cfg=cfg,
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
        """Initialize rollout actor and its underlying new-ABC engine.

        Reads ``cfg.algorithm`` / ``cfg.rollout.engine`` / ``cfg.rollout.plan``
        / ``cfg.model`` / ``cfg.sampling.sde_strategy`` / ``cfg.reward`` from
        the cfg ``ConfigActor`` installs into ``actor_config._current``.
        Algorithm + engine sections carry ``_target_`` and are materialized via
        ``build()``; ``cfg.model`` is materialized via ``materialize``;
        ``cfg.reward`` is kept as a ``DictConfig`` and forwarded into
        ``RewardPipeline.from_configs``.

        Engine construction is one-shot: ``device``, ``strategy``, ``rank``,
        ``model_config`` flow as ctor kwargs, and the engine is fully usable
        on return — there is no separate ``initialize(device)`` step (cf.
        :class:`diffusionrl.rollout.engine.base.BaseRolloutEngine`).

        Args:
            rank: This actor's rank in the rollout group.
            world_size: Total number of rollout actors.
            num_gpus_allocated: Number of GPUs allocated to this actor (must
                match Ray's ``num_gpus`` option).
            master_addr / master_port: Master node coordinates for
                distributed (multi-GPU) setup.
            gpu_ids: Physical GPU ids this actor owns (Slime NOSET pattern).
                When set (len > 1) the actor manually sets
                ``CUDA_VISIBLE_DEVICES`` to the listed GPUs.
            cfg: Full composed Hydra cfg.
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
            "Rank %s: Initializing new rollout actor (num_gpus=%d)...",
            self.rank,
            self.num_gpus_allocated,
        )
        # env-first: _setup_distributed_env writes CUDA_VISIBLE_DEVICES under
        # NOSET; must land before torch caches the visible-device list.
        self._setup_distributed_env()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.algorithm = build(self._cfg.algorithm)
        self._reward_config = self._cfg.reward
        # One-shot engine construction. ``strategy`` rides as a ctor kwarg;
        # vllm-omni currently stores it as an attribute and does not consume
        # it (the SDE math lives in the worker subprocess), but the contract
        # is: per-request control flows through ``RolloutReq.stage_params``,
        # ctor-time defaults flow through here.
        self.engine = build(
            self._cfg.rollout.engine,
            device=self._device,
            strategy=build(self._cfg.sampling.sde_strategy),
            rank=self.rank,
            model_config=materialize(self._cfg.model),
        )

        self._rollout_plan = materialize(self._cfg.rollout.plan)

        logger.info(
            "Rank %s: New rollout actor initialized (forward_batch_size=%s)",
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
        """Env setup so multi-GPU rollout actors can be NOSET-pattern
        Slime-allocated regardless of which engine the actor wraps.
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
        """Ensure generation path always starts from an active engine state.

        New-ABC engines are fully constructed in ``__init__`` (no
        ``is_initialized`` flag). ``wake_up`` is a default no-op on the new
        ABC; vllm-omni overrides it as needed. We still call it so engines
        that *do* implement runtime offload (future) are restored before
        generate.
        """
        if self.engine is None:
            raise RuntimeError("Engine not constructed.")
        self.engine.wake_up()

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        if self._reward_pipeline is None:
            self._reward_pipeline = RewardPipeline.from_configs(self._reward_config)
        return self._reward_pipeline

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Generate one rollout. Chunks ``req`` at ``forward_batch_size`` and
        concatenates the per-chunk ``RolloutResp`` results.
        """
        if int(req.batch_size) == 0:
            raise ValueError("RolloutActor.generate requires non-empty req (batch_size>0).")

        self._ensure_engine_ready_for_generate()
        self._log_gpu_state("inference_generate_start")
        return chunked_engine_generate_req(
            self.engine,
            req,
            chunk_size=self._rollout_plan.forward_batch_size,
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
        return bool(self.engine.is_offloaded)

    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self.engine is not None:
            return self.engine.get_memory_info()
        return {}

    # ------------------------------------------------------------------
    # Weight-sync verification bridges (vllm-omni only).
    #
    # The engine's ``loaded_param_checksums`` / per-stage ``collective_rpc``
    # surface is not part of the new-ABC ``BaseRolloutEngine`` contract —
    # only the vllm-omni engine + its ``BucketedIPCReceiveMixin``-composed
    # worker extensions implement them. These thin bridges expose the
    # engine surface as Ray methods so the e2e weight-sync smokes can
    # collect rollout-side checksums without holding the engine handle
    # in-process.
    #
    # Both methods raise ``NotImplementedError`` if the underlying engine
    # doesn't carry the checksum surface (e.g. a future SGLang-backed
    # ``BaseRolloutEngine`` would land here cleanly without a crash).
    # ------------------------------------------------------------------

    def num_stages(self) -> int:
        """Return the underlying engine's stage count.

        Used by the IPC weight-sync handler to drive its ZMQ socket list
        without hardcoding the HI3 ``[AR, DiT]`` pair. SD3 has 1 stage,
        HI3 t2i/it2i has 2.
        """
        if self.engine is None:
            raise RuntimeError("RolloutActor.num_stages: engine not initialized")
        omni = getattr(self.engine, "_omni", None)
        if omni is None:
            raise NotImplementedError(
                f"engine {type(self.engine).__name__} does not expose ``_omni`` (vllm-omni only today)."
            )
        return int(omni.engine.num_stages)

    def tp_per_stage(self) -> dict:
        """Return ``{stage_id: tensor_parallel_size}`` for each stage.

        Used by the IPC weight-sync handler to determine how many TP
        receivers each stage has, so train ranks beyond that TP size
        drain the iterator without opening a ZMQ sender.
        """
        if self.engine is None:
            raise RuntimeError("RolloutActor.tp_per_stage: engine not initialized")
        fn = getattr(self.engine, "tp_per_stage", None)
        if callable(fn):
            return fn()
        raise NotImplementedError(f"engine {type(self.engine).__name__} does not expose ``tp_per_stage``.")

    def loaded_param_checksums(
        self,
        *,
        names: List[str],
        stage_ids: Optional[List[int]] = None,
    ) -> Dict[int, Any]:
        """Return per-stage, per-rank loaded-weight checksums on this actor.

        Forwards to ``VLLMOmniRolloutEngine.loaded_param_checksums`` which
        fans ``_diffrl_loaded_param_checksums`` across the requested stages.
        """
        if self.engine is None:
            raise RuntimeError("RolloutActor.loaded_param_checksums: engine not initialized")
        fn = getattr(self.engine, "loaded_param_checksums", None)
        if not callable(fn):
            raise NotImplementedError(
                f"engine {type(self.engine).__name__} does not expose loaded_param_checksums (vllm-omni only today)."
            )
        return fn(names=list(names), stage_ids=list(stage_ids) if stage_ids else None)

    def list_param_names(
        self,
        *,
        stage_ids: Optional[List[int]] = None,
    ) -> Dict[int, List[str]]:
        """Return real worker parameter names per stage.

        Issues a ``collective_rpc("_diffrl_param_checksums", args=(None,))``
        which returns ``{name: short_sha256_hex}`` for every parameter the
        worker actually holds (the smoke filters this down to TP-flat names
        it can verify byte-for-byte). Returns one list per stage_id.
        """
        if self.engine is None:
            raise RuntimeError("RolloutActor.list_param_names: engine not initialized")
        omni = getattr(self.engine, "_omni", None)
        if omni is None:
            raise NotImplementedError(
                f"engine {type(self.engine).__name__} does not expose ``_omni`` (vllm-omni only today)."
            )
        if stage_ids is None:
            stage_ids = list(range(int(omni.engine.num_stages)))
        out: Dict[int, List[str]] = {}
        for sid in stage_ids:
            results = omni.engine.collective_rpc(
                method="_diffrl_param_checksums",
                args=(None,),
                stage_ids=[int(sid)],
            )
            # ``collective_rpc`` returns ``[stage_results]`` where
            # ``stage_results`` is ``[rank0_dict, rank1_dict, ...]``. Take
            # the rank-0 dict's keys: TP-replicated params are identical
            # across ranks for our purposes (the smoke filters TP-sharded
            # names out anyway).
            stage_results = results[0] if isinstance(results, list) and results else results
            rank0 = stage_results[0] if isinstance(stage_results, list) and stage_results else stage_results
            out[int(sid)] = sorted(rank0.keys()) if isinstance(rank0, dict) else []
        return out


__all__ = ["RolloutActor"]
