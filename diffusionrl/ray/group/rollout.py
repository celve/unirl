"""Rollout actor group: spawn + dispatch + control plane for the
``RolloutReq`` / ``RolloutResp`` path.

Spawns :class:`diffusionrl.ray.rollout_actor.RolloutActor` handles and
exposes the control-plane surface (``sleep`` / ``wake_up`` / weight-sync
group setup). The data plane drives sharding via
:meth:`RolloutPlan.shard_grouped_req`.

Construction is single-step: ``RolloutActorGroup(cfg=cfg,
placement=placement)`` validates the engine class, spawns Ray handles, syncs
on ``health_check``, and wires up the rollout plan in one call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.group.base import ActorGroup
from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.rollout.plan import RolloutPlan
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

if TYPE_CHECKING:
    from diffusionrl.ray.group.train import TrainActorGroup
    from diffusionrl.ray.placement import Placement

logger = logging.getLogger(__name__)


def _spawn_actors(
    *,
    placement: "Placement",
    allow_noset_multi_gpu_inference: bool,
    colocate_gpu_fraction: float,
    cfg: Optional[DictConfig] = None,
) -> Tuple[List[ActorHandle], int]:
    """Spawn ``RolloutActor`` Ray handles per ``placement.rollout_actors``.

    Does NOT force construction — the caller blocks via ``health_check.remote()``
    round trip. Returns ``(handles, stride)`` where ``stride`` matches
    ``placement.config.num_rollout_gpus_per_actor``.
    """
    from diffusionrl.ray.rollout_actor import RolloutActor
    from diffusionrl.ray.utils.node import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

    actors = placement.rollout_actors
    if not actors:
        raise ValueError("placement has no rollout actors")

    stride = placement.config.num_rollout_gpus_per_actor
    colocate = placement.config.colocate
    noset = stride > 1

    if noset and not bool(allow_noset_multi_gpu_inference):
        raise ValueError("Multi-GPU rollout actor layout requires allow_noset_multi_gpu_inference=True.")

    # cuBLAS workspace must land in os.environ before torch import.
    _DETERMINISM_ENV_VARS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

    if noset:
        ray_num_gpus = 0.5  # fractional claim per Slime pattern
        env_vars = {
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **_DETERMINISM_ENV_VARS,
        }
    elif colocate:
        ray_num_gpus = float(colocate_gpu_fraction)
        env_vars = dict(_DETERMINISM_ENV_VARS)
    else:
        ray_num_gpus = float(stride)
        env_vars = dict(_DETERMINISM_ENV_VARS)

    logger.info(
        "Creating %d rollout actors (noset=%s, colocate=%s, stride=%d, ray_num_gpus=%s)",
        len(actors),
        noset,
        colocate,
        stride,
        ray_num_gpus,
    )

    rollout_actors: List[ActorHandle] = []
    for actor in actors:
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement.pg,
            placement_group_bundle_index=actor.bundle_idx,
            placement_group_capture_child_tasks=noset,
        )
        handle = RolloutActor.options(
            num_gpus=ray_num_gpus,
            scheduling_strategy=strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            rank=actor.rank,
            world_size=len(actors),
            num_gpus_allocated=len(actor.gpu_ids) if actor.gpu_ids else stride,
            gpu_ids=list(actor.gpu_ids) if noset else None,
            cfg=cfg,
        )
        rollout_actors.append(handle)

    return rollout_actors, stride


class RolloutActorGroup(ActorGroup):
    """Rollout actor group for the new ``RolloutReq``/``RolloutResp`` path.

    Inherits handle storage and ``scatter_gather(_async)`` from
    :class:`ActorGroup`. Adds the typed control-plane surface (``sleep``,
    ``wake_up``, weight-sync group setup). The driver-side pipeline drives
    data-plane sharding through :meth:`rollout_plan.shard_grouped_req` and
    ``scatter_gather("run_rollout_pipeline", shards)``.

    Single-step construction: ``RolloutActorGroup(cfg=cfg, placement=placement)``
    validates that ``cfg.rollout.engine._target_`` resolves to a subclass of
    :class:`BaseRolloutEngine`, spawns ``RolloutActor`` handles, syncs on
    ``health_check``, and materializes the rollout plan — in one call.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        placement: "Placement",
    ) -> None:
        """Spawn ``RolloutActor`` handles directly from a Hydra ``DictConfig``.

        Reads ``cfg.rollout.engine`` / ``cfg.rollout.plan`` — registered sections
        whose ``_target_`` (set by ``@register_config``) the actor and the plan
        resolve via ``build()`` / ``materialize()``. Spawn scalars live on their
        owner configs: ``forward_batch_size`` on :class:`RolloutPlan`,
        ``allow_noset_multi_gpu_inference`` on the placement config. The
        colocated GPU fraction is a fixed runtime constant, not a cfg knob.

        Sanity-checks ``cfg.rollout.engine._target_`` resolves to a subclass of
        :class:`BaseRolloutEngine`.
        """
        import hydra

        from diffusionrl.config.instantiate import materialize

        engine_target = cfg.rollout.engine.get("_target_")
        if engine_target is None:
            raise ValueError(
                "cfg.rollout.engine must carry _target_; use a registered preset "
                "with target set (e.g. `rollout/engine: vllm_omni`)."
            )
        engine_cls = hydra.utils.get_method(engine_target)
        if not (isinstance(engine_cls, type) and issubclass(engine_cls, BaseRolloutEngine)):
            raise ValueError(
                f"RolloutActorGroup requires the engine at {engine_target!r} to subclass "
                f"diffusionrl.rollout.engine.base.BaseRolloutEngine."
            )
        sampler_engine_type = (
            str(getattr(engine_cls, "_component_name", "") or getattr(engine_cls, "__name__", "")).strip().lower()
        )
        if not sampler_engine_type:
            raise ValueError(f"Failed to resolve rollout engine type from {engine_target!r}.")

        actors, stride = _spawn_actors(
            placement=placement,
            allow_noset_multi_gpu_inference=bool(placement.config.allow_noset_multi_gpu_inference),
            colocate_gpu_fraction=float(placement.config.colocate_gpu_fraction),
            cfg=cfg,
        )

        # Force synchronous construction so any actor __init__ errors surface
        # here rather than at the first real method call downstream.
        ray.get([a.health_check.remote() for a in actors])
        logger.info("Rollout actors initialized: %d handles", len(actors))

        super().__init__(actors)
        self.rollout_plan: RolloutPlan = materialize(cfg.rollout.plan)
        self._num_gpus_allocated = int(stride or 1)
        self._sampler_engine_type = sampler_engine_type
        logger.info("RolloutActorGroup ready (engine=%s)", sampler_engine_type)

    # ------------------------------------------------------------------
    # Alt constructor: adopt train actor handles (direct-sampling mode)
    # ------------------------------------------------------------------

    @classmethod
    def from_train_group(
        cls,
        train_group: "TrainActorGroup",
        *,
        cfg: DictConfig,
    ) -> "RolloutActorGroup":
        """Adopt train actor handles as rollout actors (direct-sampling mode).

        No new actors are spawned. The returned group reuses
        ``train_group``'s actor handles and materializes its own
        :class:`RolloutPlan` from ``cfg.rollout.plan``. Valid only when
        :func:`is_direct_sampling` returns True; each train actor must
        already host a ``TrainsideRolloutEngine`` and the
        :class:`RolloutPipelineMixin` surface (see
        :class:`TrainActor` for the host-contract install).

        Health checks are skipped — the train actors already passed their
        own during ``TrainActorGroup`` construction; doing them again
        only buys latency.
        """
        from diffusionrl.config.instantiate import materialize
        from diffusionrl.config.validation import is_direct_sampling

        if not is_direct_sampling(cfg):
            raise ValueError(
                "RolloutActorGroup.from_train_group requires "
                "cfg.rollout.engine to target a direct-sampling engine "
                "(e.g. `rollout/engine: trainside`); got "
                f"_target_={cfg.rollout.engine.get('_target_')!r}."
            )

        actors = train_group.get_actors()
        if not actors:
            raise ValueError("RolloutActorGroup.from_train_group: train_group has no actor handles to adopt.")

        instance = cls.__new__(cls)
        ActorGroup.__init__(instance, actors)
        instance.rollout_plan = materialize(cfg.rollout.plan)
        instance._num_gpus_allocated = 1
        instance._sampler_engine_type = "trainside"
        logger.info(
            "RolloutActorGroup.from_train_group: adopted %d train actor handle(s) as rollout actors (engine=trainside)",
            len(actors),
        )
        return instance

    # ------------------------------------------------------------------
    # Data plane: generate (production path uses
    # the driver-side RolloutPipeline for shard_grouped_req routing)
    # ------------------------------------------------------------------

    def _build_shards(
        self,
        req: RolloutReq,
        *,
        samples_per_prompt: int,
    ) -> List[Optional[RolloutReq]]:
        return self.rollout_plan.shard_grouped_req(
            req,
            num_actors=self.num_actors,
            samples_per_prompt=int(samples_per_prompt),
        )

    def generate_async(
        self,
        req: RolloutReq,
        *,
        samples_per_prompt: int,
    ) -> List[ray.ObjectRef]:
        return self.scatter_gather_async("generate", self._build_shards(req, samples_per_prompt=samples_per_prompt))

    def generate(
        self,
        req: RolloutReq,
        *,
        samples_per_prompt: int,
    ) -> List[RolloutResp]:
        return self.scatter_gather("generate", self._build_shards(req, samples_per_prompt=samples_per_prompt))

    # ------------------------------------------------------------------
    # Control plane (parity with legacy)
    # ------------------------------------------------------------------

    def get_rollout_actors(self) -> List[ActorHandle]:
        """Return concrete rollout actor handles for direct handler injection."""
        return list(self._actors)

    def async_init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> list:
        return [
            actor.init_weights_update_group.remote(
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=1 + idx * self._num_gpus_allocated,
                world_size=int(world_size),
                group_name=str(group_name),
                backend=str(backend),
            )
            for idx, actor in enumerate(self._actors)
        ]

    def destroy_weights_update_group(self, group_name: str) -> None:
        ray.get([a.destroy_weights_update_group.remote(group_name=str(group_name)) for a in self._actors])

    def sleep(self) -> None:
        ray.get([a.sleep.remote() for a in self._actors])

    def wake_up(self) -> None:
        ray.get([a.wake_up.remote() for a in self._actors])


__all__ = ["RolloutActorGroup"]
