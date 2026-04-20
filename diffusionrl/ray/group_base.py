"""Actor-group launch and dispatch primitives."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Type

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


class ActorHandleGroup:
    """Serializable handle set with shared Ray dispatch primitives.

    num_actors is the size of this remote actor set for dispatch/scatter logic.
    It is an orchestration concept, not a training batch-geometry concept.
    """

    def __init__(
        self,
        actor_handles: Sequence[ray.actor.ActorHandle],
        *,
        num_actors: Optional[int] = None,
    ) -> None:
        self._actor_handles: List[ray.actor.ActorHandle] = list(actor_handles)
        self.num_actors = int(num_actors if num_actors is not None else len(self._actor_handles))

    def snapshot(self) -> "ActorHandleGroup":
        """Return a lightweight copy that keeps only actor handles."""
        return ActorHandleGroup(self._actor_handles, num_actors=self.num_actors)

    def get_actors(self) -> List[ray.actor.ActorHandle]:
        return list(self._actor_handles)

    def get_actor(self, index: int) -> ray.actor.ActorHandle:
        return self._actor_handles[index]

    def call_all_async(self, method: str, *args: Any, **kwargs: Any) -> List[ray.ObjectRef]:
        return [getattr(actor, method).remote(*args, **kwargs) for actor in self._actor_handles]

    def call_all(self, method: str, *args: Any, **kwargs: Any) -> List[Any]:
        return ray.get(self.call_all_async(method, *args, **kwargs))

    def call_rank0_async(self, method: str, *args: Any, **kwargs: Any) -> ray.ObjectRef:
        if not self._actor_handles:
            raise RuntimeError("Actor group is empty.")
        return getattr(self._actor_handles[0], method).remote(*args, **kwargs)

    def call_rank0(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return ray.get(self.call_rank0_async(method, *args, **kwargs))

    def call_subset_async(
        self,
        indices: Sequence[int],
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[ray.ObjectRef]:
        return [getattr(self._actor_handles[int(index)], method).remote(*args, **kwargs) for index in indices]

    def call_subset(
        self,
        indices: Sequence[int],
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Any]:
        return ray.get(self.call_subset_async(indices, method, *args, **kwargs))

    def call_per_actor_async(
        self,
        method: str,
        *,
        per_actor_args: Sequence[Optional[Sequence[Any]]],
        per_actor_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[ray.ObjectRef]:
        if len(per_actor_args) != len(self._actor_handles):
            raise ValueError(
                f"per_actor_args length {len(per_actor_args)} does not match actor count {len(self._actor_handles)}"
            )
        if per_actor_kwargs is not None and len(per_actor_kwargs) != len(self._actor_handles):
            raise ValueError(
                f"per_actor_kwargs length {len(per_actor_kwargs)} does not match actor count {len(self._actor_handles)}"
            )

        refs: List[ray.ObjectRef] = []
        for index, actor in enumerate(self._actor_handles):
            args_i = per_actor_args[index]
            kwargs_i = per_actor_kwargs[index] if per_actor_kwargs is not None else None
            if args_i is None and kwargs_i is None:
                continue
            refs.append(
                getattr(actor, method).remote(
                    *(tuple(args_i) if args_i is not None else ()),
                    **dict(kwargs_i or {}),
                )
            )
        return refs

    def call_per_actor(
        self,
        method: str,
        *,
        per_actor_args: Sequence[Optional[Sequence[Any]]],
        per_actor_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[Any]:
        return ray.get(
            self.call_per_actor_async(
                method,
                per_actor_args=per_actor_args,
                per_actor_kwargs=per_actor_kwargs,
            )
        )

    def scatter_gather_async(
        self,
        method: str,
        shards: Sequence[Optional[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> List[ray.ObjectRef]:
        if len(shards) != len(self._actor_handles):
            raise ValueError(f"shards length {len(shards)} does not match actor count {len(self._actor_handles)}")
        refs: List[ray.ObjectRef] = []
        for actor, shard in zip(self._actor_handles, shards):
            if shard is None:
                continue
            refs.append(getattr(actor, method).remote(shard, *args, **kwargs))
        return refs

    def scatter_gather(
        self,
        method: str,
        shards: Sequence[Optional[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> List[Any]:
        return ray.get(self.scatter_gather_async(method, shards, *args, **kwargs))


class PlacementGroupActorPool(ActorHandleGroup):
    """Generic actor pool launched on placement-group bundles."""

    def __init__(
        self,
        actor_class: Type,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        gpu_ids: Optional[List[int]] = None,
        num_gpus_per_actor: float = 1.0,
        num_cpus_per_actor: float = 1.0,
        num_gpus_per_engine: int = 1,
        capture_child_tasks: bool = False,
        runtime_env: Optional[dict] = None,
        inject_rank_world_size: bool = False,
        inject_base_gpu_id: bool = False,
        per_actor_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **actor_kwargs: Any,
    ) -> None:
        self.actor_class = actor_class
        self.pg = pg
        self.bundle_indices = list(bundle_indices)
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_gpus_per_actor = float(num_gpus_per_actor)
        self.num_cpus_per_actor = float(num_cpus_per_actor)
        self.num_gpus_per_engine = int(num_gpus_per_engine)

        if per_actor_kwargs is not None and len(per_actor_kwargs) != int(num_actors):
            raise ValueError(f"per_actor_kwargs length {len(per_actor_kwargs)} does not match actor count {num_actors}")

        actor_handles: List[ray.actor.ActorHandle] = []
        for index in range(num_actors):
            actor_kwargs_i = dict(actor_kwargs)
            if per_actor_kwargs is not None and per_actor_kwargs[index] is not None:
                actor_kwargs_i.update(dict(per_actor_kwargs[index] or {}))

            if num_gpus_per_engine > 1 and gpu_ids is not None:
                bundle_index = bundle_indices[index * num_gpus_per_engine]
                if inject_base_gpu_id:
                    actor_kwargs_i.setdefault(
                        "base_gpu_id",
                        gpu_ids[index * num_gpus_per_engine],
                    )
            else:
                bundle_index = bundle_indices[index] if index < len(bundle_indices) else index

            if inject_rank_world_size:
                actor_kwargs_i.setdefault("rank", index)
                actor_kwargs_i.setdefault("world_size", num_actors)

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=capture_child_tasks,
            )
            options = {
                "num_gpus": num_gpus_per_actor,
                "num_cpus": num_cpus_per_actor,
                "scheduling_strategy": scheduling_strategy,
            }
            if runtime_env:
                options["runtime_env"] = runtime_env

            actor_handles.append(actor_class.options(**options).remote(**actor_kwargs_i))

        super().__init__(actor_handles, num_actors=num_actors)
        logger.info("Created %d actors of type %s", num_actors, actor_class)

    def dispose(self) -> None:
        for actor in self._actor_handles:
            try:
                ray.kill(actor)
            except Exception as exc:
                logger.warning("Error killing actor: %s", exc)
        self._actor_handles.clear()
        logger.info("Actor pool disposed")


class ActorGroup(PlacementGroupActorPool):
    """Launch Ray actors on specific placement-group bundles and dispatch RPCs."""

    def __init__(
        self,
        actor_class: Type,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        gpu_ids: Optional[List[int]] = None,
        num_gpus_per_actor: float = 1.0,
        num_cpus_per_actor: float = 1.0,
        num_gpus_per_engine: int = 1,
        capture_child_tasks: bool = False,
        runtime_env: Optional[dict] = None,
        **actor_kwargs: Any,
    ) -> None:
        super().__init__(
            actor_class=actor_class,
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
            num_gpus_per_actor=num_gpus_per_actor,
            num_cpus_per_actor=num_cpus_per_actor,
            num_gpus_per_engine=num_gpus_per_engine,
            capture_child_tasks=capture_child_tasks,
            runtime_env=runtime_env,
            inject_rank_world_size=True,
            inject_base_gpu_id=True,
            **actor_kwargs,
        )

    def async_init(self, config: dict) -> List[ray.ObjectRef]:
        return self.call_all_async("init", config)

    def init(self, config: dict) -> List[Any]:
        return ray.get(self.async_init(config))

    def health_check(self) -> List[bool]:
        return self.call_all("health_check")


__all__ = ["ActorHandleGroup", "PlacementGroupActorPool", "ActorGroup"]
