"""Ray actor shell for the buffer subsystem."""

from __future__ import annotations

from typing import Any, Dict, Optional

import ray

from diffusionrl.buffer import (
    BufferPlugin,
    BufferPluginContext,
    BufferRuntime,
    FiniteTensorFilterPlugin,
    MinSamplesGuardPlugin,
    RayBatchStore,
    RewardRangeFilterPlugin,
    build_buffer_plugins,
)
from diffusionrl.types.buffer_contracts import (
    BufferedTrainingPayload,
    RolloutPayload,
)
from diffusionrl.types.training_batch import TrainingBatch


@ray.remote(num_cpus=1, num_gpus=0)
class BufferActor:
    """Remote shell that delegates buffer semantics to BufferRuntime."""

    def __init__(self, args: Any):
        self.runtime = BufferRuntime.from_args(
            args,
            batch_store=RayBatchStore(),
        )

    def size(self) -> int:
        return self.runtime.size()

    def clear(self) -> None:
        self.runtime.clear()

    def push(
        self,
        *,
        rollout_id: int,
        train_data: TrainingBatch,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.runtime.push(
            rollout_id=rollout_id,
            train_data=train_data,
            metadata=metadata,
        )

    def push_payload_ref(
        self,
        *,
        payload_ref: Any,
    ) -> Dict[str, Any]:
        """Admit a rollout payload ref into the buffer."""
        payload = ray.get(payload_ref) if isinstance(payload_ref, ray.ObjectRef) else payload_ref
        if not isinstance(payload, RolloutPayload):
            raise TypeError(
                "BufferActor.push_payload_ref expects RolloutPayload instances, "
                f"got {type(payload).__name__}."
            )

        push_result = self.runtime.push(
            rollout_id=int(payload.rollout_id),
            train_data=payload.training_batch,
            metadata=payload.metadata,
        )
        receipt = dict(push_result)
        receipt["payload_rollout_id"] = int(payload.rollout_id)
        return receipt

    def pop(
        self,
    ) -> Optional[BufferedTrainingPayload]:
        return self.runtime.pop()

    def pop_training_data(
        self,
        *,
        expected_rollout_id: Optional[int] = None,
    ) -> BufferedTrainingPayload:
        return self.runtime.pop_training_data(
            expected_rollout_id=expected_rollout_id,
        )

    def get_stats(self) -> Dict[str, Any]:
        return self.runtime.get_stats()

    def dispose(self) -> None:
        self.runtime.dispose()


def create_buffer_actor(args: Any):
    """Factory for the Ray-backed buffer actor."""
    return BufferActor.options(num_cpus=1, num_gpus=0).remote(args)


__all__ = [
    "BufferActor",
    "BufferPlugin",
    "BufferPluginContext",
    "FiniteTensorFilterPlugin",
    "MinSamplesGuardPlugin",
    "RewardRangeFilterPlugin",
    "build_buffer_plugins",
    "create_buffer_actor",
]
