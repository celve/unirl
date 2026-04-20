"""Rollout actor group: spawn + dispatch + control plane in one class."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.generate_sharding import build_generate_shard_plan
from diffusionrl.ray.group_base import ActorHandleGroup
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.transfer.buffer import BufferHandle

logger = logging.getLogger(__name__)


class RolloutActorGroup:
    """Rollout actor group: spawn + dispatch + control plane in one class.

    Used by ``diffusionrl.train``. Replaces a previous two-layer split
    where a control-plane ``GroupRuntime`` wrapped a data-plane actor
    group via ``from_group``.
    """

    def __init__(
        self,
        *,
        handle: ActorHandleGroup,
        num_gpus_allocated: int = 1,
        sampler_engine_type: str = "unknown",
    ):
        self._handle = handle.snapshot()
        self._num_gpus_allocated = int(num_gpus_allocated or 1)
        self._sampler_engine_type = str(sampler_engine_type or "unknown")
        self.num_actors = int(self._handle.num_actors)

    @property
    def handle(self) -> ActorHandleGroup:
        """The underlying actor handle group (for dispatch primitives)."""
        return self._handle

    def get_actors(self) -> List[ActorHandle]:
        """Return the raw actor handles."""
        return self._handle.get_actors()

    def dispose(self) -> None:
        """Kill all managed actors. Safe to call multiple times."""
        for actor in self._handle.get_actors():
            try:
                ray.kill(actor)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(
        cls,
        *,
        launch_config: Any,
        rollout_pgs: Tuple[Any, List[int], Any],
    ) -> "RolloutActorGroup":
        """Spawn dedicated ``RolloutActor`` handles per launch_config.

        Validates that a dedicated rollout launch is present, resolves Ray
        options (NOSET multi-GPU / colocate fractional / standard single-GPU),
        spawns the ``RolloutActor`` handles on the given placement-group
        bundles, and round-trips a blocking ``init`` call.

        Use ``group.get_actors()`` for raw handles, ``group.handle`` for the
        underlying ``ActorHandleGroup``, ``group.dispose()`` for teardown.
        """
        from diffusionrl.config.validation import validate_rollout_actor_init_config
        from diffusionrl.ray.rollout_actor import RolloutActor
        from diffusionrl.ray.utils.node import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        rollout_launch = launch_config.rollout
        if rollout_launch is None:
            raise ValueError(
                "launch_config.rollout has no dedicated rollout launch — "
                "RolloutActorGroup.bootstrap requires a dedicated rollout actor group."
            )

        pg, bundle_indices, gpu_ids = rollout_pgs
        rollout_engine = rollout_launch.rollout_engine

        actor_init_config = deepcopy(rollout_launch.actor_init_config)
        validate_rollout_actor_init_config(actor_init_config)

        num_gpus_per_actor_int = int(rollout_launch.actor_gpu_requirement)
        colocate = bool(launch_config.placement.colocate_rollout)

        available = len(bundle_indices)
        if available < 1:
            raise ValueError("Rollout placement group has no GPU bundles allocated.")

        rollout_actors: List[ActorHandle] = []

        # Determinism env vars propagated to every rollout actor process so
        # they land in os.environ *before* Python imports torch. Required for
        # cuBLAS deterministic workspace; cudnn.deterministic and
        # torch.use_deterministic_algorithms are set by set_seed() in
        # RolloutActor.init().
        _DETERMINISM_ENV_VARS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

        if num_gpus_per_actor_int > 1:
            # Multi-GPU NOSET / Slime pattern (sglang TP)
            if not bool(rollout_launch.allow_noset_multi_gpu_inference):
                raise ValueError(
                    "Multi-GPU rollout actor layout requires "
                    "--allow-noset-multi-gpu-inference=true."
                )
            if available % num_gpus_per_actor_int != 0:
                raise ValueError(
                    f"Rollout bundle count ({available}) must be divisible by "
                    f"num_gpus_per_actor ({num_gpus_per_actor_int})."
                )
            num_actors = available // num_gpus_per_actor_int
            ray_num_gpus = 0.5  # fractional claim per Slime pattern
            runtime_env = {
                "env_vars": {
                    **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
                    **_DETERMINISM_ENV_VARS,
                }
            }
            logger.info(
                "Creating %d rollout actors (Slime NOSET pattern, colocate=%s), "
                "%d GPU(s) per engine, ray_num_gpus=%s",
                num_actors,
                colocate,
                num_gpus_per_actor_int,
                ray_num_gpus,
            )
            for rank in range(num_actors):
                bundle_idx = bundle_indices[rank * num_gpus_per_actor_int]
                base_gpu_id = (
                    gpu_ids[rank * num_gpus_per_actor_int] if gpu_ids else 0
                )
                strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    placement_group_capture_child_tasks=True,
                )
                handle = RolloutActor.options(
                    num_gpus=ray_num_gpus,
                    scheduling_strategy=strategy,
                    runtime_env=runtime_env,
                ).remote(
                    rank=rank,
                    world_size=num_actors,
                    num_gpus_allocated=num_gpus_per_actor_int,
                    base_gpu_id=base_gpu_id,
                    force_set_cuda_visible_devices=True,
                )
                rollout_actors.append(handle)
        else:
            # Single-GPU or colocate-fractional path
            if colocate:
                ray_num_gpus = float(rollout_launch.colocate_gpu_fraction)
                num_actors = available
            else:
                ray_num_gpus = float(num_gpus_per_actor_int)
                num_actors = int(available / num_gpus_per_actor_int)

            if num_actors < 1:
                raise ValueError("Not enough GPUs allocated for rollout actors.")

            logger.info(
                "Creating %d rollout actors, %s GPU(s) per actor",
                num_actors,
                ray_num_gpus,
            )
            for rank in range(num_actors):
                bundle_idx = bundle_indices[rank]
                strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    placement_group_capture_child_tasks=False,
                )
                handle = RolloutActor.options(
                    num_gpus=ray_num_gpus,
                    scheduling_strategy=strategy,
                    runtime_env={"env_vars": dict(_DETERMINISM_ENV_VARS)},
                ).remote(
                    rank=rank,
                    world_size=num_actors,
                    num_gpus_allocated=max(1, num_gpus_per_actor_int),
                )
                rollout_actors.append(handle)

        ray.get([a.init.remote(actor_init_config) for a in rollout_actors])
        logger.info("Rollout actors initialized: %d handles", len(rollout_actors))

        handle_group = ActorHandleGroup(rollout_actors, num_actors=len(rollout_actors))
        group = cls(
            handle=handle_group,
            num_gpus_allocated=int(num_gpus_per_actor_int),
            sampler_engine_type=rollout_engine,
        )
        logger.info("RolloutActorGroup ready (engine=%s)", rollout_engine)
        return group

    # ------------------------------------------------------------------
    # Data plane: generate
    # ------------------------------------------------------------------
    #
    # Mirrors the legacy ``RolloutActorGroup.generate*`` shape. Note:
    # train.py drives generation directly through actor handles via
    # ``RolloutPipeline.run_once`` and does not call these methods today.
    # They're here so future direct-dispatch consumers can use the new path.

    def _build_generate_plan(self, request: RolloutRequest):
        return build_generate_shard_plan(
            request,
            num_actors=self.num_actors,
            pad_to_actor_count=False,
        )

    def generate_async(self, request: RolloutRequest) -> List[ray.ObjectRef]:
        plan = self._build_generate_plan(request)
        return self._handle.scatter_gather_async("generate", plan.shards)

    def generate(self, request: RolloutRequest) -> List[RolloutResponse]:
        plan = self._build_generate_plan(request)
        return ray.get(self._handle.scatter_gather_async("generate", plan.shards))

    def generate_buffered(self, request: RolloutRequest) -> List[BufferHandle]:
        plan = self._build_generate_plan(request)
        nested = ray.get(
            self._handle.scatter_gather_async("generate_buffered", plan.shards)
        )
        return [handle for actor_handles in nested for handle in actor_handles]

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    def get_rollout_actors(self) -> List[ActorHandle]:
        """Return concrete rollout actor handles for direct handler injection."""
        return self._handle.get_actors()

    def async_init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> list:
        num_actors = self._handle.num_actors
        per_actor_args: List[Optional[tuple[Any, ...]]] = [None] * num_actors
        per_actor_kwargs: List[Optional[Dict[str, Any]]] = [None] * num_actors
        for idx in range(num_actors):
            rank_offset = 1 + idx * self._num_gpus_allocated
            per_actor_args[idx] = ()
            per_actor_kwargs[idx] = {
                "master_address": master_address,
                "master_port": int(master_port),
                "rank_offset": int(rank_offset),
                "world_size": int(world_size),
                "group_name": str(group_name),
                "backend": str(backend),
            }
        return self._handle.call_per_actor_async(
            "init_weights_update_group",
            per_actor_args=per_actor_args,
            per_actor_kwargs=per_actor_kwargs,
        )

    def destroy_weights_update_group(self, group_name: str) -> None:
        self._handle.call_all(
            "destroy_weights_update_group",
            group_name=str(group_name),
        )

    def get_weight_sync_topology(self) -> Dict[str, int]:
        return {
            "num_actors": int(self._handle.num_actors),
            "num_gpus_per_actor": int(self._num_gpus_allocated),
            "total_gpus": int(self._handle.num_actors * self._num_gpus_allocated),
        }

    def sleep(self) -> None:
        self._handle.call_all("sleep")

    def wake_up(self) -> None:
        self._handle.call_all("wake_up")


__all__ = ["RolloutActorGroup"]
