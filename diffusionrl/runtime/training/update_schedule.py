"""Training update scheduling policies for typed training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from diffusionrl.types.training_batch import TrainingBatch


def _require_runtime_positive_int(*, name: str, value: Optional[int]) -> int:
    if value is None:
        raise ValueError(
            f"{name} must be normalized to an explicit int before entering training runtime."
        )

    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1. Got {resolved}.")
    return resolved


def resolve_gradient_accumulation_plan(
    *,
    batch_size: int,
    gradient_accumulation_batch_size: int,
) -> Tuple[List[Tuple[int, int]], int]:
    """Resolve sample-dimension micro-batches for one optimizer update."""
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"local update batch size must be >= 1. Got {batch_size}.")

    resolved_gradient_accumulation_batch_size = _require_runtime_positive_int(
        name="gradient_accumulation_batch_size",
        value=gradient_accumulation_batch_size,
    )
    if resolved_gradient_accumulation_batch_size > batch_size:
        raise ValueError(
            "gradient_accumulation_batch_size must be <= local update batch size before "
            "entering training runtime. "
            f"Got local_update_batch_size={batch_size}, "
            f"gradient_accumulation_batch_size={resolved_gradient_accumulation_batch_size}."
        )
    if batch_size % resolved_gradient_accumulation_batch_size != 0:
        raise ValueError(
            "local update batch size must be divisible by gradient_accumulation_batch_size. "
            f"Got local_update_batch_size={batch_size}, "
            f"gradient_accumulation_batch_size={resolved_gradient_accumulation_batch_size}."
        )

    micro_batches: List[Tuple[int, int]] = []
    for start in range(0, batch_size, resolved_gradient_accumulation_batch_size):
        end = min(start + resolved_gradient_accumulation_batch_size, batch_size)
        micro_batches.append((start, end))

    actual_micro_batches = max(1, len(micro_batches))
    return micro_batches, actual_micro_batches


@dataclass(frozen=True)
class TrainingUpdateChunk:
    """One optimizer-update chunk produced by a training schedule."""

    batch: TrainingBatch
    gradient_accumulation_batch_size: int
    update_batch_size: int
    update_index: int


class TrainingUpdateSchedule:
    """Policy interface for splitting one rollout batch into optimizer updates."""

    name = "single_update"

    def iter_update_chunks(
        self,
        *,
        batch: TrainingBatch,
        gradient_accumulation_batch_size: int,
        multi_update_batch_size: Optional[int] = None,
    ):
        raise NotImplementedError


class SingleUpdateSchedule(TrainingUpdateSchedule):
    """One optimizer step per rollout pass, with optional mini-batch accumulation."""

    name = "single_update"

    def iter_update_chunks(
        self,
        *,
        batch: TrainingBatch,
        gradient_accumulation_batch_size: int,
        multi_update_batch_size: Optional[int] = None,
    ):
        del multi_update_batch_size
        batch_size = int(batch.batch_size)
        resolved_gradient_accumulation_batch_size = _require_runtime_positive_int(
            name="gradient_accumulation_batch_size",
            value=gradient_accumulation_batch_size,
        )
        if resolved_gradient_accumulation_batch_size > batch_size:
            raise ValueError(
                "single_update requires gradient_accumulation_batch_size to be <= local rollout "
                "batch size before entering training runtime. "
                f"Got local_batch_size={batch_size}, "
                f"gradient_accumulation_batch_size={resolved_gradient_accumulation_batch_size}."
            )
        if batch_size % resolved_gradient_accumulation_batch_size != 0:
            raise ValueError(
                "single_update requires local rollout batch size to be divisible by "
                "gradient_accumulation_batch_size. "
                f"Got local_batch_size={batch_size}, "
                f"gradient_accumulation_batch_size={resolved_gradient_accumulation_batch_size}."
            )

        yield TrainingUpdateChunk(
            batch=batch,
            gradient_accumulation_batch_size=resolved_gradient_accumulation_batch_size,
            update_batch_size=batch_size,
            update_index=0,
        )


class MultiUpdateSchedule(TrainingUpdateSchedule):
    """One optimizer step per update chunk inside a rollout pass."""

    name = "multi_update"

    def iter_update_chunks(
        self,
        *,
        batch: TrainingBatch,
        gradient_accumulation_batch_size: int,
        multi_update_batch_size: Optional[int] = None,
    ):
        batch_size = int(batch.batch_size)
        resolved_update_batch_size = _require_runtime_positive_int(
            name="multi_update_batch_size",
            value=multi_update_batch_size,
        )
        if resolved_update_batch_size > batch_size:
            raise ValueError(
                "multi_update_batch_size must be <= local rollout batch size before entering "
                "training runtime. "
                f"Got local_batch_size={batch_size}, "
                f"multi_update_batch_size={resolved_update_batch_size}."
            )
        if batch_size % resolved_update_batch_size != 0:
            raise ValueError(
                "multi_update requires local rollout batch size to be divisible by "
                "multi_update_batch_size. "
                f"Got local_batch_size={batch_size}, "
                f"multi_update_batch_size={resolved_update_batch_size}."
            )

        resolved_gradient_accumulation_batch_size = _require_runtime_positive_int(
            name="gradient_accumulation_batch_size",
            value=gradient_accumulation_batch_size,
        )
        if resolved_gradient_accumulation_batch_size > resolved_update_batch_size:
            raise ValueError(
                "multi_update requires gradient_accumulation_batch_size to be <= "
                "multi_update_batch_size before entering training runtime. "
                f"Got multi_update_batch_size={resolved_update_batch_size}, "
                f"gradient_accumulation_batch_size={resolved_gradient_accumulation_batch_size}."
            )
        if resolved_update_batch_size % resolved_gradient_accumulation_batch_size != 0:
            raise ValueError(
                "multi_update requires multi_update_batch_size to be divisible by "
                "gradient_accumulation_batch_size. "
                f"Got multi_update_batch_size={resolved_update_batch_size}, "
                f"gradient_accumulation_batch_size={resolved_gradient_accumulation_batch_size}."
            )

        update_index = 0
        for start in range(0, batch_size, resolved_update_batch_size):
            end = start + resolved_update_batch_size
            yield TrainingUpdateChunk(
                batch=batch.slice(start, end),
                gradient_accumulation_batch_size=resolved_gradient_accumulation_batch_size,
                update_batch_size=resolved_update_batch_size,
                update_index=update_index,
            )
            update_index += 1


def create_training_update_schedule(name: str) -> TrainingUpdateSchedule:
    """Create a training update schedule by name."""
    normalized = str(name).strip().lower()
    if normalized == "single_update":
        return SingleUpdateSchedule()
    if normalized == "multi_update":
        return MultiUpdateSchedule()
    raise ValueError(
        f"Unsupported training update schedule: {name!r}. "
        "Expected one of: single_update, multi_update."
    )


__all__ = [
    "resolve_gradient_accumulation_plan",
    "TrainingUpdateChunk",
    "TrainingUpdateSchedule",
    "create_training_update_schedule",
    "SingleUpdateSchedule",
    "MultiUpdateSchedule",
]
