"""Stateful rollout buffer independent of Ray actor plumbing."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from diffusionrl.buffer.buffer_batch_ops import concat_training_batches, index_training_batch
from diffusionrl.buffer.buffer_plugins import BufferPlugin, BufferPluginContext, build_buffer_plugins
from diffusionrl.buffer.buffer_store import BatchStore
from diffusionrl.types.buffer_contracts import BufferedTrainingPayload
from diffusionrl.types.training_batch import BackwardTrainingBatch, ForwardTrainingBatch, TrainingBatch

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


@dataclass
class GroupSampleLocator:
    """Pointer to one sample inside a stored rollout TrainingBatch."""

    batch_handle: Any
    sample_idx: int
    rollout_id: int
    created_at: float
    group_id: Optional[str]
    reward: Optional[float]
    modality: str

class BufferRuntime:
    """Own queueing, filtering, and group reassembly for rollout training batches."""

    def __init__(
        self,
        *,
        batch_store: BatchStore,
        max_queue_size: int,
        reassemble_by_group: bool,
        group_size: Optional[int],
        expected_global_batch_size: Optional[int],
        group_ttl_seconds: float,
        max_pending_samples: int,
        plugins: Sequence[BufferPlugin],
    ) -> None:
        self.batch_store = batch_store
        self.max_queue_size = int(max_queue_size)
        self.reassemble_by_group = bool(reassemble_by_group)
        self.group_size = None if group_size is None else int(group_size)
        self.expected_global_batch_size = (
            None
            if expected_global_batch_size is None
            else int(expected_global_batch_size)
        )
        self.group_ttl_seconds = float(group_ttl_seconds)
        self.max_pending_samples = int(max_pending_samples)
        self.plugins = list(plugins)

        self._dispatch_queue: Deque[BufferItem] = deque()
        self._groups: Dict[str, Deque[GroupSampleLocator]] = {}
        self._group_batch_ref_counts: Dict[Any, int] = {}

        self._counter = 0
        self._dropped_queue_items = 0
        self._dropped_batches = 0
        self._dropped_samples = 0
        self._expired_samples = 0
        self._pending_overflow_drops = 0
        self._pushed_batches = 0
        self._popped_batches = 0
        self._pushed_samples = 0
        self._popped_samples = 0
        self._assembled_batches = 0

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        batch_store: BatchStore,
    ) -> "BufferRuntime":
        rollout_buffer = args.rollout.buffer
        reassemble_by_group = bool(rollout_buffer.reassemble_by_group)
        raw_group_size = rollout_buffer.group_size
        if reassemble_by_group and raw_group_size is None:
            raise ValueError(
                "rollout.buffer.reassemble_by_group requires rollout.buffer.group_size to be "
                "validated before BufferRuntime initialization."
            )
        return cls(
            batch_store=batch_store,
            max_queue_size=int(rollout_buffer.max_queue_size),
            reassemble_by_group=reassemble_by_group,
            group_size=raw_group_size,
            expected_global_batch_size=None,
            group_ttl_seconds=float(rollout_buffer.group_ttl_seconds),
            max_pending_samples=int(rollout_buffer.max_pending_samples),
            plugins=build_buffer_plugins(args),
        )

    def configure_expected_global_batch_size(
        self,
        *,
        expected_global_batch_size: int,
    ) -> int:
        self.expected_global_batch_size = int(expected_global_batch_size)
        return int(self.expected_global_batch_size)

    def _require_expected_global_batch_size(self) -> int:
        expected_global_batch_size = self.expected_global_batch_size
        if expected_global_batch_size is None:
            raise RuntimeError(
                "BufferRuntime must be configured with expected_global_batch_size before use."
            )
        return int(expected_global_batch_size)

    def _new_item_id(self) -> str:
        self._counter += 1
        return f"buffer_item_{self._counter}"

    def _pending_samples(self) -> int:
        return sum(len(items) for items in self._groups.values())

    def _ready_groups(self) -> int:
        ready = 0
        for key, items in self._groups.items():
            if len(items) >= self._required_group_size(key):
                ready += 1
        return ready

    def _track_group_batch_handle(self, batch_handle: Any, sample_count: int) -> None:
        self._group_batch_ref_counts[batch_handle] = self._group_batch_ref_counts.get(batch_handle, 0) + int(sample_count)

    def _release_group_batch_sample(self, batch_handle: Any) -> None:
        remaining = int(self._group_batch_ref_counts.get(batch_handle, 0)) - 1
        if remaining <= 0:
            self._group_batch_ref_counts.pop(batch_handle, None)
            self.batch_store.release(batch_handle)
            return
        self._group_batch_ref_counts[batch_handle] = remaining

    def _detect_modality(self, batch: TrainingBatch) -> str:
        if isinstance(batch, BackwardTrainingBatch):
            return "video" if int(batch.trajectories.ndim) >= 6 else "image"
        if isinstance(batch, ForwardTrainingBatch):
            return "video" if int(batch.clean_latents.ndim) >= 5 else "image"
        return "unknown"

    def _normalize_group_id(self, group_id: Any) -> Optional[str]:
        if group_id is None:
            return None
        text = str(group_id).strip()
        return text if text else None

    def _group_key_for_sample(self, *, group_id: Optional[str], rollout_id: int, sample_idx: int) -> str:
        if group_id is not None:
            return f"rollout:{int(rollout_id)}:group:{group_id}"
        raise ValueError(
            "rollout.buffer.reassemble_by_group requires explicit group_ids; "
            "fallback grouping is removed. "
            f"Got rollout_id={rollout_id}, sample_idx={sample_idx}."
        )

    def _release_dispatch_item(self, item: BufferItem) -> None:
        self.batch_store.release(item.batch_handle)

    def _maybe_drop_dispatch_head(self) -> None:
        if self.max_queue_size > 0 and len(self._dispatch_queue) >= self.max_queue_size:
            dropped = self._dispatch_queue.popleft()
            self._release_dispatch_item(dropped)
            self._dropped_queue_items += 1
            self._dropped_batches += 1
            self._dropped_samples += int(dropped.sample_count)

    def _queue_has_capacity(self) -> bool:
        return self.max_queue_size <= 0 or len(self._dispatch_queue) < self.max_queue_size

    def _cleanup_empty_group(self, group_key: str) -> None:
        group = self._groups.get(group_key)
        if group is not None and len(group) == 0:
            del self._groups[group_key]

    def _required_group_size(self, group_key: str) -> int:
        del group_key
        if self.group_size is None:
            return 1
        return int(self.group_size)

    def _cleanup_expired_groups(self) -> None:
        if self.group_ttl_seconds <= 0:
            return
        now = time.time()
        cutoff = now - self.group_ttl_seconds
        for key in list(self._groups.keys()):
            group = self._groups.get(key)
            if group is None:
                continue
            while group and group[0].created_at < cutoff:
                locator = group.popleft()
                self._release_group_batch_sample(locator.batch_handle)
                self._expired_samples += 1
                self._dropped_samples += 1
            self._cleanup_empty_group(key)

    def _drop_oldest_pending_sample(self) -> bool:
        oldest_key: Optional[str] = None
        oldest_ts: Optional[float] = None

        for key, group in self._groups.items():
            if not group:
                continue
            ts = float(group[0].created_at)
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
                oldest_key = key

        if oldest_key is None:
            return False

        group = self._groups[oldest_key]
        locator = group.popleft()
        self._release_group_batch_sample(locator.batch_handle)
        self._pending_overflow_drops += 1
        self._dropped_samples += 1
        self._cleanup_empty_group(oldest_key)
        return True

    def _enforce_pending_capacity(self) -> None:
        if self.max_pending_samples <= 0:
            return
        while self._pending_samples() > self.max_pending_samples:
            if not self._drop_oldest_pending_sample():
                break

    def _pop_group_samples(self, group_key: str, count: int) -> List[GroupSampleLocator]:
        group = self._groups.get(group_key)
        if group is None:
            return []

        items: List[GroupSampleLocator] = []
        for _ in range(min(int(count), len(group))):
            items.append(group.popleft())
        self._cleanup_empty_group(group_key)
        return items

    def _peek_group_samples(self, group_key: str, count: int) -> List[GroupSampleLocator]:
        group = self._groups.get(group_key)
        if group is None:
            return []
        return list(group)[: int(count)]

    def _select_dispatch_plan(self) -> List[Tuple[str, int]]:
        target_samples = self._require_expected_global_batch_size()
        if target_samples <= 0:
            raise ValueError(
                f"expected_global_batch_size must be positive, got {self.expected_global_batch_size}."
            )
        plan: List[Tuple[str, int]] = []
        selected_samples = 0
        for key in list(self._groups.keys()):
            group = self._groups.get(key)
            required_size = self._required_group_size(key)
            if group is None or len(group) < required_size:
                continue
            if selected_samples + required_size > target_samples:
                continue
            plan.append((key, required_size))
            selected_samples += required_size
            if selected_samples == target_samples:
                return plan

        return plan if selected_samples == target_samples else []

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

    def _materialize_batch_from_locators(self, locators: Sequence[GroupSampleLocator]) -> TrainingBatch:
        if not locators:
            raise ValueError("Cannot materialize batch from empty locator list.")

        batch_cache: Dict[Any, TrainingBatch] = {}
        sample_batches: List[TrainingBatch] = []
        for locator in locators:
            base_batch = batch_cache.get(locator.batch_handle)
            if base_batch is None:
                base_batch = self.materialize_training_data(locator.batch_handle)
                batch_cache[locator.batch_handle] = base_batch

            sample_idx = int(locator.sample_idx)
            if sample_idx < 0 or sample_idx >= int(base_batch.batch_size):
                raise IndexError(
                    "Sample index out of range while materializing grouped batch: "
                    f"sample_idx={sample_idx}, batch_size={base_batch.batch_size}"
                )
            sample_batches.append(index_training_batch(base_batch, [sample_idx]))

        out = concat_training_batches(sample_batches)
        out.validate()
        return out

    def _enqueue_dispatch_item(
        self,
        *,
        rollout_id: int,
        batch: TrainingBatch,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._maybe_drop_dispatch_head()
        item = BufferItem(
            item_id=self._new_item_id(),
            rollout_id=int(rollout_id),
            batch_handle=self.batch_store.put(batch),
            sample_count=int(batch.batch_size),
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self._dispatch_queue.append(item)
        self._assembled_batches += 1

    def _promote_ready_groups(self) -> int:
        if not self.reassemble_by_group:
            return 0

        self._cleanup_expired_groups()
        self._enforce_pending_capacity()
        promoted = 0

        while self._queue_has_capacity():
            plan = self._select_dispatch_plan()
            if not plan:
                break

            selected: List[GroupSampleLocator] = []
            selected_group_keys: List[str] = []
            for key, take in plan:
                selected_group_keys.append(key)
                selected.extend(self._peek_group_samples(key, take))

            if not selected:
                break

            batch = self._materialize_batch_from_locators(selected)
            committed: List[GroupSampleLocator] = []
            for key, take in plan:
                committed.extend(self._pop_group_samples(key, take))
            for locator in committed:
                self._release_group_batch_sample(locator.batch_handle)
            self._enqueue_dispatch_item(
                rollout_id=max(int(s.rollout_id) for s in committed),
                batch=batch,
                metadata={
                    "group_keys": selected_group_keys,
                    "group_size": (
                        self._required_group_size(selected_group_keys[0])
                        if selected_group_keys
                        else None
                    ),
                },
            )
            promoted += 1

        return promoted

    def _push_grouped(
        self,
        *,
        rollout_id: int,
        current: TrainingBatch,
        context: BufferPluginContext,
    ) -> Dict[str, Any]:
        sample_count = int(current.batch_size)
        if sample_count <= 0:
            raise ValueError("Processed batch is empty.")

        created_at = time.time()
        modality = self._detect_modality(current)

        group_ids = current.group_ids
        if group_ids is None or len(group_ids) != sample_count:
            raise ValueError(
                "rollout.buffer.reassemble_by_group requires explicit sample-aligned group_ids. "
                f"Got batch_size={sample_count}, group_ids_len={len(group_ids) if group_ids is not None else None}."
            )

        normalized_group_ids: List[str] = []
        group_counts: Dict[str, int] = {}
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                raise ValueError(
                    "rollout.buffer.reassemble_by_group requires non-empty group_ids for every sample. "
                    f"Found invalid group_id at sample_idx={sample_idx}."
                )
            group_key = self._group_key_for_sample(
                group_id=group_id,
                rollout_id=int(rollout_id),
                sample_idx=sample_idx,
            )
            normalized_group_ids.append(group_id)
            group_counts[group_key] = group_counts.get(group_key, 0) + 1

        for group_key, count in group_counts.items():
            pending_count = len(self._groups.get(group_key, ()))
            required_group_size = self._required_group_size(group_key)
            if pending_count > 0:
                raise ValueError(
                    "rollout.buffer.reassemble_by_group encountered a duplicate pending group key. "
                    f"Got group_key={group_key}, pending={pending_count}. "
                    "Each rollout must provide one complete logical group per group_id."
                )
            if int(count) != required_group_size:
                raise ValueError(
                    "rollout.buffer.reassemble_by_group requires each incoming group "
                    "to remain complete after buffer plugins. "
                    f"Got group_key={group_key}, incoming={int(count)}, "
                    f"required_group_size={required_group_size}. "
                    "Disable sample-dropping filters or keep this mode off."
                )
            if pending_count + int(count) > required_group_size:
                raise ValueError(
                    "rollout.buffer.reassemble_by_group received more samples than the configured group size "
                    f"for {group_key}: pending={pending_count}, incoming={int(count)}, "
                    f"group_size={required_group_size}. "
                    "Set rollout.buffer.group_size explicitly to match the producer contract."
                )

        rewards_tensor = current.rewards
        batch_handle = self.batch_store.put(current)
        self._track_group_batch_handle(batch_handle, sample_count)
        pending_locators: List[Tuple[str, GroupSampleLocator]] = []
        try:
            for sample_idx in range(sample_count):
                group_id = normalized_group_ids[sample_idx]
                reward = (
                    float(rewards_tensor[sample_idx].item())
                    if rewards_tensor is not None
                    else None
                )
                group_key = self._group_key_for_sample(
                    group_id=group_id,
                    rollout_id=int(rollout_id),
                    sample_idx=sample_idx,
                )
                locator = GroupSampleLocator(
                    batch_handle=batch_handle,
                    sample_idx=sample_idx,
                    rollout_id=int(rollout_id),
                    created_at=created_at,
                    group_id=group_id,
                    reward=reward,
                    modality=modality,
                )
                if group_key not in self._groups:
                    self._groups[group_key] = deque()
                self._groups[group_key].append(locator)
                pending_locators.append((group_key, locator))

            promoted = self._promote_ready_groups()
        except Exception:
            for group_key, locator in reversed(pending_locators):
                group = self._groups.get(group_key)
                if group is None:
                    continue
                removed = False
                if group and group[-1] is locator:
                    group.pop()
                    removed = True
                if not removed:
                    for index in range(len(group) - 1, -1, -1):
                        if group[index] is locator:
                            del group[index]
                            break
                self._cleanup_empty_group(group_key)
            self._group_batch_ref_counts.pop(batch_handle, None)
            self.batch_store.release(batch_handle)
            raise

        self._pushed_batches += 1
        self._pushed_samples += sample_count

        return {
            "accepted": True,
            "rollout_id": int(rollout_id),
            "sample_count": sample_count,
            "ready_queue_size": len(self._dispatch_queue),
            "pending_groups": len(self._groups),
            "pending_samples": self._pending_samples(),
            "promoted_batches": promoted,
            "admission_state": "admitted_ready" if promoted > 0 else "admitted_pending",
            "ready_for_consume": bool(promoted > 0),
            "reassemble_by_group_mode": True,
            "metadata": context.metadata,
        }

    def size(self) -> int:
        return len(self._dispatch_queue)

    def clear(self) -> None:
        while self._dispatch_queue:
            self._release_dispatch_item(self._dispatch_queue.popleft())
        for handle in list(self._group_batch_ref_counts.keys()):
            self.batch_store.release(handle)
        self._groups.clear()
        self._group_batch_ref_counts.clear()

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
            self._require_expected_global_batch_size()
            current = train_data
            for plugin in self.plugins:
                current = plugin.process(current, context=context)

            current.validate()

            if self.reassemble_by_group:
                return self._push_grouped(
                    rollout_id=rollout_id,
                    current=current,
                    context=context,
                )

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
                "reassemble_by_group_mode": False,
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
        if self.reassemble_by_group and not self._dispatch_queue:
            self._promote_ready_groups()

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
                    f"expected rollout_id={expected_rollout_id}, got {got}. "
                    "Disable strict alignment when using rollout.buffer.reassemble_by_group."
                )
        return payload

    def get_stats(self) -> Dict[str, Any]:
        plugin_stats = {plugin.name: plugin.stats() for plugin in self.plugins}

        pending_modalities: Dict[str, int] = {}
        reward_sum = 0.0
        reward_count = 0
        for group in self._groups.values():
            for locator in group:
                pending_modalities[locator.modality] = pending_modalities.get(locator.modality, 0) + 1
                if locator.reward is not None:
                    reward_sum += float(locator.reward)
                    reward_count += 1

        avg_pending_reward = reward_sum / reward_count if reward_count > 0 else None

        return {
            "reassemble_by_group_mode": self.reassemble_by_group,
            "queue_size": len(self._dispatch_queue),
            "pushed_batches": self._pushed_batches,
            "popped_batches": self._popped_batches,
            "pushed_samples": self._pushed_samples,
            "popped_samples": self._popped_samples,
            "assembled_batches": self._assembled_batches,
            "dropped_queue_items": self._dropped_queue_items,
            "dropped_batches": self._dropped_batches,
            "dropped_samples": self._dropped_samples,
            "expired_samples": self._expired_samples,
            "pending_overflow_drops": self._pending_overflow_drops,
            "max_queue_size": self.max_queue_size,
            "plugins": plugin_stats,
            "group_size": self.group_size,
            "expected_global_batch_size": self.expected_global_batch_size,
            "group_ttl_seconds": self.group_ttl_seconds,
            "max_pending_samples": self.max_pending_samples,
            "pending_group_count": len(self._groups),
            "ready_group_count": self._ready_groups(),
            "pending_sample_count": self._pending_samples(),
            "pending_modality_counts": pending_modalities,
            "avg_pending_reward": avg_pending_reward,
        }

    def dispose(self) -> None:
        self.clear()


__all__ = [
    "BufferRuntime",
]
