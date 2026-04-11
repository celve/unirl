"""Training worker-group typed facade."""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Dict, List, Optional

import ray
from ray.util.placement_group import PlacementGroup

from diffusionrl.types import RolloutRequest
from diffusionrl.utils import load_function

from .generate_sharding import build_generate_shard_plan, trim_generate_outputs
from .group_base import ActorGroup

logger = logging.getLogger(__name__)


class TrainingActorGroup(ActorGroup):
    """Thin typed wrapper around the shared actor-group dispatch kernel.

    num_actors here means the number of launched training actors in this Ray
    group. It should not be read as dp_size by itself.
    """

    def __init__(
        self,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        actor_class_path: Optional[str] = None,
        actor_init_kwargs: Optional[Dict[str, Any]] = None,
        runtime_env: Optional[dict] = None,
        **kwargs: Any,
    ):
        if actor_class_path:
            actor_class = load_function(actor_class_path)
            if not hasattr(actor_class, "remote"):
                raise TypeError(
                    f"actor_class_path must resolve to a Ray actor class, got: {actor_class}"
                )
        else:
            from .training_actor import TrainingActor

            actor_class = TrainingActor

        merged_actor_kwargs: Dict[str, Any] = {}
        if isinstance(actor_init_kwargs, dict):
            merged_actor_kwargs.update(actor_init_kwargs)
        merged_actor_kwargs.update(kwargs)

        super().__init__(
            actor_class=actor_class,
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            runtime_env=runtime_env,
            **merged_actor_kwargs,
        )

    def async_train(self, rollout_id: int, training_data_handle: Any) -> List[ray.ObjectRef]:
        if isinstance(training_data_handle, list):
            if len(training_data_handle) != self.num_actors:
                raise ValueError(
                    "training_data_handle list length "
                    f"{len(training_data_handle)} does not match num_actors {self.num_actors}"
                )
            per_actor_args = [
                (rollout_id, ref)
                for ref in training_data_handle
            ]
            return self.call_per_actor_async("train", per_actor_args=per_actor_args)
        return self.call_all_async("train", rollout_id, training_data_handle)

    def train(self, rollout_id: int, training_data_handle: Any) -> List[Dict[str, Any]]:
        return ray.get(self.async_train(rollout_id, training_data_handle))

    def _build_generate_plan(self, request: RolloutRequest):
        return build_generate_shard_plan(
            request,
            num_actors=self.num_actors,
            pad_to_actor_count=True,
        )

    def async_generate(self, request: RolloutRequest) -> List[ray.ObjectRef]:
        plan = self._build_generate_plan(request)
        return self.scatter_gather_async("generate", plan.shards)

    def generate(self, request: RolloutRequest) -> List[Any]:
        plan = self._build_generate_plan(request)

        t0 = _time.perf_counter()
        refs = self.scatter_gather_async("generate", plan.shards)
        t1 = _time.perf_counter()
        outputs = ray.get(refs)
        t2 = _time.perf_counter()
        logger.debug(
            "[TIMING] training_group.generate: dispatch=%.2fs ray_get=%.2fs total=%.2fs prompts=%s",
            t1 - t0,
            t2 - t1,
            t2 - t0,
            plan.original_batch_size,
        )
        return trim_generate_outputs(outputs, plan=plan)


__all__ = ["TrainingActorGroup"]
