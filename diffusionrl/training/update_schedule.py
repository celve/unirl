"""Training update scheduling policies driven by an explicit execution plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from diffusionrl.types.training_batch import TrainingBatch


@dataclass(frozen=True)
class TrainingExecutionPlan:
    """Runtime plan for one local training consumer batch."""

    local_batch_size: int
    local_mini_batch_size: int
    micro_batch_size: int
    num_updates_per_batch: int
    update_slices: Tuple[Tuple[int, int], ...]
    mini_batch_slices_per_update: Tuple[Tuple[Tuple[int, int], ...], ...]


def _positive_int(*, name: str, value: Any) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1. Got {resolved}.")
    return resolved


def _coerce_slice_pair(*, name: str, value: Any) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence, got: {value!r}")
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end <= start:
        raise ValueError(f"{name} must satisfy 0 <= start < end, got: {value!r}")
    return (start, end)


def _build_micro_batch_slices(
    *,
    total_size: int,
    micro_batch_size: int,
) -> Tuple[Tuple[int, int], ...]:
    resolved_total_size = _positive_int(name="total_size", value=total_size)
    resolved_micro_batch_size = _positive_int(name="micro_batch_size", value=micro_batch_size)
    slices: List[Tuple[int, int]] = []
    start = 0
    while start < resolved_total_size:
        end = min(start + resolved_micro_batch_size, resolved_total_size)
        slices.append((start, end))
        start = end
    return tuple(slices)


def _coerce_update_slices(raw: Any, *, local_batch_size: int) -> Tuple[Tuple[int, int], ...]:
    if raw is None:
        raise ValueError("training_plan.update_slices is required.")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("training_plan.update_slices must be a non-empty sequence.")

    slices = tuple(
        _coerce_slice_pair(name=f"training_plan.update_slices[{index}]", value=item) for index, item in enumerate(raw)
    )
    for index, (start, end) in enumerate(slices):
        if end > int(local_batch_size):
            raise ValueError(
                "training_plan.update_slices exceeds local_batch_size. "
                f"Slice {index}={start, end}, local_batch_size={local_batch_size}."
            )
    return slices


def _coerce_per_update_mini_batch_slices(
    raw: Any,
    *,
    update_slices: Sequence[Tuple[int, int]],
    micro_batch_size: int,
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    if raw is None:
        return tuple(
            _build_micro_batch_slices(
                total_size=int(end) - int(start),
                micro_batch_size=int(micro_batch_size),
            )
            for start, end in update_slices
        )
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("training_plan.mini_batch_slices_per_update must be a non-empty sequence.")

    if len(raw) != len(update_slices):
        raise ValueError(
            "training_plan.mini_batch_slices_per_update must align with training_plan.update_slices. "
            f"Got {len(raw)} mini-batch collections for {len(update_slices)} update slices."
        )

    resolved: List[Tuple[Tuple[int, int], ...]] = []
    for update_index, (per_update, update_slice) in enumerate(zip(raw, update_slices)):
        if not isinstance(per_update, (list, tuple)) or not per_update:
            raise ValueError(
                "training_plan.mini_batch_slices_per_update entries must be non-empty sequences. "
                f"Got index={update_index}, value={per_update!r}."
            )
        update_size = int(update_slice[1]) - int(update_slice[0])
        mini_slices = tuple(
            _coerce_slice_pair(
                name=(f"training_plan.mini_batch_slices_per_update[{update_index}][{mini_index}]"),
                value=item,
            )
            for mini_index, item in enumerate(per_update)
        )
        for mini_index, (_, end) in enumerate(mini_slices):
            if end > update_size:
                raise ValueError(
                    "training_plan.mini_batch_slices_per_update exceeds its update chunk. "
                    f"update_index={update_index}, mini_index={mini_index}, "
                    f"mini_slice={mini_slices[mini_index]}, update_size={update_size}."
                )
        resolved.append(mini_slices)
    return tuple(resolved)


def coerce_training_execution_plan(raw: Any) -> TrainingExecutionPlan:
    """Build a runtime execution plan from a serialized config payload."""

    if isinstance(raw, TrainingExecutionPlan):
        return raw
    # Lazy import to avoid the diffusionrl.config <-> diffusionrl.training import
    # cycle. TrainingPlan is the resolution-time view that carries
    # global_batch_size; we drop that field here and reuse the dict path.
    from diffusionrl.config.spec import TrainingPlan

    if isinstance(raw, TrainingPlan):
        raw = raw.as_dict()
    if not isinstance(raw, dict):
        raise ValueError(
            f"training_plan must be a dict, TrainingPlan, or TrainingExecutionPlan. Got: {type(raw).__name__}"
        )

    local_batch_size = _positive_int(
        name="training_plan.local_batch_size",
        value=raw["local_batch_size"],
    )
    local_mini_batch_size = _positive_int(
        name="training_plan.local_mini_batch_size",
        value=raw["local_mini_batch_size"],
    )
    micro_batch_size = _positive_int(
        name="training_plan.micro_batch_size",
        value=raw["micro_batch_size"],
    )
    num_updates_per_batch = _positive_int(
        name="training_plan.num_updates_per_batch",
        value=raw["num_updates_per_batch"],
    )
    if local_batch_size != local_mini_batch_size * num_updates_per_batch:
        raise ValueError(
            "training_plan.local_batch_size must equal "
            "training_plan.local_mini_batch_size * training_plan.num_updates_per_batch. "
            f"Got local_batch_size={local_batch_size}, "
            f"local_mini_batch_size={local_mini_batch_size}, "
            f"num_updates_per_batch={num_updates_per_batch}."
        )
    if local_mini_batch_size % micro_batch_size != 0:
        raise ValueError(
            "training_plan.micro_batch_size must evenly divide "
            "training_plan.local_mini_batch_size. "
            f"Got micro_batch_size={micro_batch_size}, "
            f"local_mini_batch_size={local_mini_batch_size}."
        )
    update_slices = _coerce_update_slices(raw.get("update_slices"), local_batch_size=local_batch_size)
    mini_batch_slices_per_update = _coerce_per_update_mini_batch_slices(
        raw.get("mini_batch_slices_per_update"),
        update_slices=update_slices,
        micro_batch_size=micro_batch_size,
    )

    return TrainingExecutionPlan(
        local_batch_size=local_batch_size,
        local_mini_batch_size=local_mini_batch_size,
        micro_batch_size=micro_batch_size,
        num_updates_per_batch=num_updates_per_batch,
        update_slices=update_slices,
        mini_batch_slices_per_update=mini_batch_slices_per_update,
    )


def validate_batch_against_plan(
    *,
    batch_size: int,
    plan: TrainingExecutionPlan,
) -> None:
    """Ensure runtime payload still matches the validated training plan."""

    resolved_batch_size = _positive_int(name="batch_size", value=batch_size)
    if resolved_batch_size != int(plan.local_batch_size):
        raise ValueError(
            "Runtime training batch violates the resolved training plan. "
            f"Got local_batch_size={resolved_batch_size}, "
            f"expected={plan.local_batch_size}."
        )


@dataclass(frozen=True)
class TrainingUpdateChunk:
    """One optimizer-update chunk produced by a training schedule."""

    batch: TrainingBatch
    update_batch_size: int
    update_index: int
    mini_batch_slices: Tuple[Tuple[int, int], ...]


class TrainingUpdateSchedule:
    """Schedule interface for splitting one rollout batch into optimizer updates."""

    def __init__(self, plan: TrainingExecutionPlan):
        self.plan = coerce_training_execution_plan(plan)
        self.name = "single_update" if int(self.plan.num_updates_per_batch) <= 1 else "multi_update"

    def iter_update_chunks(
        self,
        *,
        batch: TrainingBatch,
    ):
        raise NotImplementedError


class SingleUpdateSchedule(TrainingUpdateSchedule):
    """One optimizer step per rollout pass, with explicit mini-batch slices."""

    def iter_update_chunks(
        self,
        *,
        batch: TrainingBatch,
    ):
        validate_batch_against_plan(batch_size=int(batch.batch_size), plan=self.plan)
        if len(self.plan.update_slices) != 1:
            raise ValueError(
                f"single_update plan must contain exactly one update slice. Got {len(self.plan.update_slices)}."
            )
        update_start, update_end = self.plan.update_slices[0]
        yield TrainingUpdateChunk(
            batch=batch.slice(int(update_start), int(update_end)),
            update_batch_size=int(self.plan.local_mini_batch_size),
            update_index=0,
            mini_batch_slices=tuple((int(start), int(end)) for start, end in self.plan.mini_batch_slices_per_update[0]),
        )


class MultiUpdateSchedule(TrainingUpdateSchedule):
    """One optimizer step per explicit update chunk inside a rollout pass."""

    def iter_update_chunks(
        self,
        *,
        batch: TrainingBatch,
    ):
        validate_batch_against_plan(batch_size=int(batch.batch_size), plan=self.plan)
        for update_index, ((start, end), mini_batch_slices) in enumerate(
            zip(self.plan.update_slices, self.plan.mini_batch_slices_per_update)
        ):
            yield TrainingUpdateChunk(
                batch=batch.slice(int(start), int(end)),
                update_batch_size=int(end) - int(start),
                update_index=int(update_index),
                mini_batch_slices=tuple((int(mini_start), int(mini_end)) for mini_start, mini_end in mini_batch_slices),
            )


def create_training_update_schedule(plan: Any) -> TrainingUpdateSchedule:
    """Create a training update schedule from an explicit execution plan."""
    resolved_plan = coerce_training_execution_plan(plan)
    if int(resolved_plan.num_updates_per_batch) <= 1:
        return SingleUpdateSchedule(resolved_plan)
    return MultiUpdateSchedule(resolved_plan)


__all__ = [
    "TrainingExecutionPlan",
    "TrainingUpdateChunk",
    "TrainingUpdateSchedule",
    "coerce_training_execution_plan",
    "create_training_update_schedule",
    "validate_batch_against_plan",
    "SingleUpdateSchedule",
    "MultiUpdateSchedule",
]
