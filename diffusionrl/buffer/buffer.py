"""Stateful rollout buffer independent of Ray actor plumbing."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from diffusionrl.buffer.buffer_plugins import BufferPlugin, BufferPluginContext, build_buffer_plugins
from diffusionrl.buffer.buffer_store import BatchStore
from diffusionrl.types.buffer_contracts import BufferedTrainingPayload
from diffusionrl.types.training_batch import TrainingBatch

logger = logging.getLogger(__name__)


@dataclass
class BufferItem:
    """Single buffered training-batch entry in dispatch queue."""

    item_id: str
    rollout_id: int
    batch_handle: Any
    sample_count: int
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BufferRuntime:
    """Queueing and filtering for rollout training batches."""

    def __init__(
        self,
        *,
        batch_store: BatchStore,
        max_queue_size: int,
        plugins: Sequence[BufferPlugin],
    ) -> None:
        self.batch_store = batch_store
        self.max_queue_size = int(max_queue_size)
        self.plugins = list(plugins)

        self._dispatch_queue: Deque[BufferItem] = deque()

        self._counter = 0
        self._dropped_queue_items = 0
        self._dropped_batches = 0
        self._dropped_samples = 0
        self._pushed_batches = 0
        self._popped_batches = 0
        self._pushed_samples = 0
        self._popped_samples = 0

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        batch_store: BatchStore,
    ) -> "BufferRuntime":
        rollout_buffer = args.rollout
        return cls(
            batch_store=batch_store,
            max_queue_size=int(rollout_buffer.max_queue_size),
            plugins=build_buffer_plugins(args),
        )

    def _new_item_id(self) -> str:
        self._counter += 1
        return f"buffer_item_{self._counter}"

    def _release_dispatch_item(self, item: BufferItem) -> None:
        self.batch_store.release(item.batch_handle)

    def _maybe_drop_dispatch_head(self) -> None:
        if self.max_queue_size > 0 and len(self._dispatch_queue) >= self.max_queue_size:
            dropped = self._dispatch_queue.popleft()
            self._release_dispatch_item(dropped)
            self._dropped_queue_items += 1
            self._dropped_batches += 1
            self._dropped_samples += int(dropped.sample_count)

    def materialize_training_data(self, handle: Any) -> Any:
        if isinstance(handle, list):
            return [self.batch_store.get(item) for item in handle]
        return self.batch_store.get(handle)

    def release_training_data(self, handle: Any) -> None:
        if isinstance(handle, list):
            for item in handle:
                self.batch_store.release(item)
            return
        self.batch_store.release(handle)

    def size(self) -> int:
        return len(self._dispatch_queue)

    def clear(self) -> None:
        while self._dispatch_queue:
            self._release_dispatch_item(self._dispatch_queue.popleft())

    def push(
        self,
        *,
        rollout_id: int,
        train_data: TrainingBatch,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = BufferPluginContext(
            rollout_id=int(rollout_id),
            metadata=dict(metadata or {}),
        )

        try:
            current = train_data
            for plugin in self.plugins:
                current = plugin.process(current, context=context)

            current.validate()

            sample_count = int(current.batch_size)
            if sample_count <= 0:
                raise ValueError("Processed batch is empty.")

            self._maybe_drop_dispatch_head()
            item = BufferItem(
                item_id=self._new_item_id(),
                rollout_id=int(rollout_id),
                batch_handle=self.batch_store.put(current),
                sample_count=sample_count,
                created_at=time.time(),
                metadata=context.metadata,
            )
            self._dispatch_queue.append(item)
            self._pushed_batches += 1
            self._pushed_samples += sample_count

            return {
                "accepted": True,
                "item_id": item.item_id,
                "rollout_id": item.rollout_id,
                "sample_count": sample_count,
                "queue_size": len(self._dispatch_queue),
                "admission_state": "admitted_ready",
                "ready_for_consume": True,
            }
        except Exception as exc:
            logger.warning("BufferRuntime drop rollout_id=%s due to: %s", rollout_id, exc)
            self._dropped_batches += 1
            return {
                "accepted": False,
                "rollout_id": int(rollout_id),
                "error": str(exc),
                "queue_size": len(self._dispatch_queue),
                "admission_state": "rejected",
                "ready_for_consume": False,
            }

    def pop(self) -> Optional[BufferedTrainingPayload]:
        if not self._dispatch_queue:
            return None

        item = self._dispatch_queue.popleft()
        self._popped_batches += 1
        self._popped_samples += int(item.sample_count)

        return BufferedTrainingPayload(
            rollout_id=int(item.rollout_id),
            sample_count=int(item.sample_count),
            metadata=dict(item.metadata),
            training_data=item.batch_handle,
        )

    def pop_training_data(
        self,
        *,
        expected_rollout_id: Optional[int] = None,
    ) -> BufferedTrainingPayload:
        payload = self.pop()
        if payload is None:
            raise RuntimeError("Rollout buffer is empty; no training data available.")
        if expected_rollout_id is not None:
            got = int(payload.rollout_id)
            if got != int(expected_rollout_id):
                raise RuntimeError(
                    "Rollout/training payload mismatch: "
                    f"expected rollout_id={expected_rollout_id}, got {got}."
                )
        return payload

    def get_stats(self) -> Dict[str, Any]:
        plugin_stats = {plugin.name: plugin.stats() for plugin in self.plugins}

        return {
            "queue_size": len(self._dispatch_queue),
            "pushed_batches": self._pushed_batches,
            "popped_batches": self._popped_batches,
            "pushed_samples": self._pushed_samples,
            "popped_samples": self._popped_samples,
            "dropped_queue_items": self._dropped_queue_items,
            "dropped_batches": self._dropped_batches,
            "dropped_samples": self._dropped_samples,
            "max_queue_size": self.max_queue_size,
            "plugins": plugin_stats,
        }

    def dispose(self) -> None:
        self.clear()


__all__ = [
    "BufferRuntime",
]
