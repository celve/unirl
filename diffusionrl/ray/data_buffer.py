"""Rollout data buffering helpers for rollout->train handoff."""

from __future__ import annotations

from typing import Any, Optional

from diffusionrl.runtime.pipeline.partition_stage import maybe_partition_training_batch


def _ray_put(value: Any) -> Any:
    import ray

    return ray.put(value)


def _is_ray_object_ref(value: Any) -> bool:
    try:
        import ray
    except ModuleNotFoundError:
        return False
    object_ref_type = getattr(ray, "ObjectRef", None)
    if object_ref_type is None:
        return False
    return isinstance(value, object_ref_type)


class RolloutDataBuffer:
    """
    Encapsulate rollout data serialization and optional partitioning.

    This keeps RolloutManager focused on pipeline orchestration while the buffer
    owns object-store handoff details.
    """

    def __init__(self, *, partition_train_data: bool = True) -> None:
        self.partition_train_data = bool(partition_train_data)

    def put(self, *, train_data: Any, world_size: Optional[int]) -> Any:
        """Put a typed training batch into Ray object store."""
        partitioned_batches = maybe_partition_training_batch(
            train_data=train_data,
            world_size=world_size,
            partition_train_data=self.partition_train_data,
        )
        if partitioned_batches is not None:
            return [_ray_put(part) for part in partitioned_batches]
        return _ray_put(train_data)


def normalize_rollout_result(rollout_result: Any) -> Any:
    """
    Normalize rollout output to training-compatible references.

    Handles Ray nested-ref behavior differences by ensuring non-ref payloads are
    wrapped once in object store.
    """
    if isinstance(rollout_result, list):
        return rollout_result
    if _is_ray_object_ref(rollout_result):
        return rollout_result
    return _ray_put(rollout_result)


__all__ = ["RolloutDataBuffer", "normalize_rollout_result"]
