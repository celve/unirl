"""Rollout actor group: spawn + dispatch + control plane in one class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.group.base import ActorGroup
from diffusionrl.rollout.plan import RolloutPlan
from diffusionrl.transfer.buffer import BufferHandle
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse

if TYPE_CHECKING:
    from diffusionrl.ray.placement import Placement

logger = logging.getLogger(__name__)


class RolloutActorGroup(ActorGroup):
    """Rollout actor group: spawn + dispatch + control plane in one class.

    Inherits handle storage and ``scatter_gather(_async)`` from
    :class:`ActorGroup`. Adds rollout-specific spawn (``bootstrap``)
    plus the typed control-plane surface (``sleep``, ``wake_up``,
    weight-sync group setup).
    """

    def __init__(
        self,
        actors: Sequence[ActorHandle],
        *,
        rollout_plan: RolloutPlan,
        num_gpus_allocated: int = 1,
        sampler_engine_type: str = "unknown",
    ) -> None:
        super().__init__(actors)
        self.rollout_plan = rollout_plan
        self._num_gpus_allocated = int(num_gpus_allocated or 1)
        self._sampler_engine_type = str(sampler_engine_type or "unknown")

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    @classmethod
    def _spawn_actors(
        cls,
        *,
        placement: "Placement",
        allow_noset_multi_gpu_inference: bool,
        colocate_gpu_fraction: float,
        cfg: Optional[DictConfig] = None,
    ) -> Tuple[List[ActorHandle], int]:
        """Spawn ``RolloutActor`` Ray handles per ``placement.rollout_actors``.

        Shared spawn logic for both ``bootstrap`` (argparse path) and
        ``bootstrap_from_cfg`` (Hydra path). Does NOT force construction —
        the caller issues a cheap round-trip (e.g. ``health_check``) to
        block until each actor finishes ``__init__``.

        Returns ``(handles, stride)`` where ``stride`` is
        ``placement.config.num_rollout_gpus_per_actor``. Use
        ``group.get_actors()`` / ``group.dispose()`` on the resulting
        ``RolloutActorGroup`` for raw handles and teardown.
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

        # Determinism env vars propagated to every rollout actor process so
        # they land in os.environ *before* Python imports torch. Required for
        # cuBLAS deterministic workspace; cudnn.deterministic and
        # torch.use_deterministic_algorithms are set by set_seed() in the
        # actor's ``__init__``.
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

    @classmethod
    def bootstrap(
        cls,
        *,
        cfg: DictConfig,
        placement: "Placement",
    ) -> "RolloutActorGroup":
        """Spawn ``RolloutActor`` handles directly from a Hydra ``DictConfig``.

        Reads ``cfg.rollout.engine`` / ``cfg.algorithm`` / ``cfg.reward`` —
        registered sections whose ``_target_`` (set by ``@register_config``)
        the actor resolves via ``build()`` during construction. Spawn scalars
        live on their owner configs: ``forward_batch_size`` on ``RolloutPlan``,
        ``allow_noset_multi_gpu_inference`` on ``PlacementConfig``. The
        colocated GPU fraction is a fixed runtime constant, not a cfg knob.
        """
        import hydra

        engine_target = cfg.rollout.engine.get("_target_")
        if engine_target is None:
            raise ValueError(
                "cfg.rollout.engine must carry _target_; use a registered preset "
                "with target set (e.g. `rollout/engine: sglang`)."
            )
        engine_cls = hydra.utils.get_method(engine_target)
        sampler_engine_type = (
            str(getattr(engine_cls, "_component_name", "") or getattr(engine_cls, "__name__", "")).strip().lower()
        )
        if not sampler_engine_type:
            raise ValueError(f"Failed to resolve rollout engine type from {engine_target!r}.")

        allow_noset = bool(placement.config.allow_noset_multi_gpu_inference)

        rollout_actors, stride = cls._spawn_actors(
            placement=placement,
            allow_noset_multi_gpu_inference=allow_noset,
            colocate_gpu_fraction=float(placement.config.colocate_gpu_fraction),
            cfg=cfg,
        )

        # Force synchronous construction so any __init__ errors surface here
        # rather than at the first real method call.
        ray.get([a.health_check.remote() for a in rollout_actors])
        logger.info("Rollout actors initialized: %d handles", len(rollout_actors))

        from diffusionrl.config.instantiate import materialize

        group = cls(
            rollout_actors,
            rollout_plan=materialize(cfg.rollout.plan),
            num_gpus_allocated=stride,
            sampler_engine_type=sampler_engine_type,
        )
        logger.info("RolloutActorGroup ready (engine=%s)", sampler_engine_type)
        return group

    # ------------------------------------------------------------------
    # Data plane: generate
    # ------------------------------------------------------------------
    #
    # Mirrors the legacy ``RolloutActorGroup.generate*`` shape. Note:
    # train.py drives generation directly through actor handles via
    # ``RolloutPipeline.run_once`` and does not call these methods today.
    # They're here so future direct-dispatch consumers can use the new path.

    def _build_shards(self, request: RolloutRequest) -> List[Optional[RolloutRequest]]:
        return self.rollout_plan.shard(
            request,
            num_actors=self.num_actors,
            pad_to_actor_count=False,
        )

    def generate_async(self, request: RolloutRequest) -> List[ray.ObjectRef]:
        return self.scatter_gather_async("generate", self._build_shards(request))

    def generate(self, request: RolloutRequest) -> List[RolloutResponse]:
        return self.scatter_gather("generate", self._build_shards(request))

    def generate_buffered(self, request: RolloutRequest) -> List[BufferHandle]:
        plan = self._build_generate_plan(request)
        nested = self.scatter_gather("generate_buffered", plan.shards)
        return [handle for actor_handles in nested for handle in actor_handles]

    # ------------------------------------------------------------------
    # Control plane
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
