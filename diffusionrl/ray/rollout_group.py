"""Rollout worker-group typed facade."""

from __future__ import annotations

from typing import Any, List, Optional

from ray.util.placement_group import PlacementGroup

from diffusionrl.types import RolloutRequest

from .generate_sharding import build_generate_shard_plan
from .group_base import ActorGroup


class RolloutActorGroup(ActorGroup):
    """Thin typed wrapper around rollout sampling actors."""

    def __init__(
        self,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        gpu_ids: Optional[List[int]] = None,
        num_gpus_per_actor: float = 1.0,
        num_gpus_per_engine: int = 1,
        capture_child_tasks: bool = False,
        runtime_env: Optional[dict] = None,
        sampler_engine_type: str = "sglang",
        **kwargs,
    ):
        from .rollout_actor import RolloutActor

        if "num_gpus_allocated" not in kwargs:
            if num_gpus_per_engine > 1:
                kwargs["num_gpus_allocated"] = num_gpus_per_engine
            elif num_gpus_per_actor < 1:
                kwargs["num_gpus_allocated"] = 1
            else:
                kwargs["num_gpus_allocated"] = int(num_gpus_per_actor)

        super().__init__(
            actor_class=RolloutActor,
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
            num_gpus_per_actor=num_gpus_per_actor,
            num_gpus_per_engine=num_gpus_per_engine,
            capture_child_tasks=capture_child_tasks,
            runtime_env=runtime_env,
            **kwargs,
        )
        self.sampler_engine_type = str(sampler_engine_type or "sglang").lower()
        self.num_gpus_allocated = int(kwargs.get("num_gpus_allocated", 1) or 1)

    def _build_generate_plan(self, request: RolloutRequest):
        return build_generate_shard_plan(
            request,
            num_actors=self.num_actors,
            pad_to_actor_count=False,
        )

    def async_generate(self, request: RolloutRequest) -> List[Any]:
        plan = self._build_generate_plan(request)
        return self.scatter_gather_async("generate", plan.shards)

    def generate(self, request: RolloutRequest) -> List[Any]:
        import ray

        plan = self._build_generate_plan(request)
        return ray.get(self.scatter_gather_async("generate", plan.shards))


__all__ = ["RolloutActorGroup"]
