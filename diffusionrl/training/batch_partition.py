"""Shared training-batch partition helpers."""

from __future__ import annotations

from typing import List, Optional

from diffusionrl.types.training_batch import TrainingBatch


def _resolve_shard_sizes(
    batch: TrainingBatch,
    *,
    dp_size: int,
    per_rank_batch_size: Optional[int] = None,
) -> List[int]:
    batch_size = getattr(batch, "batch_size", None)
    if batch_size is None:
        raise ValueError(f"Training batch {type(batch).__name__} does not expose batch_size for DP partitioning.")

    resolved_batch_size = int(batch_size)
    resolved_dp_size = int(dp_size)
    if resolved_dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}.")

    if per_rank_batch_size is None:
        raise ValueError("Data-parallel batch partition requires an explicit per_rank_batch_size plan.")

    resolved_per_rank = int(per_rank_batch_size)
    if resolved_per_rank <= 0:
        raise ValueError(f"per_rank_batch_size must be positive, got {per_rank_batch_size}.")

    expected_total = resolved_per_rank * resolved_dp_size
    if resolved_batch_size != expected_total:
        raise ValueError(
            "Training batch violates the resolved per-rank partition plan. "
            f"Got batch_size={resolved_batch_size}, expected={expected_total}, "
            f"dp_size={resolved_dp_size}, per_rank_batch_size={resolved_per_rank}."
        )
    return [resolved_per_rank] * resolved_dp_size


def shard_training_batch_for_rank(
    batch: TrainingBatch,
    *,
    dp_size: int,
    rank: int,
    per_rank_batch_size: Optional[int] = None,
) -> TrainingBatch:
    """Return the shard owned by one data-parallel rank."""
    resolved_dp_size = int(dp_size)
    if resolved_dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}.")
    if resolved_dp_size <= 1:
        return batch

    if rank < 0 or rank >= resolved_dp_size:
        raise ValueError(f"rank must satisfy 0 <= rank < dp_size, got rank={rank}, dp_size={resolved_dp_size}.")

    shard_sizes = _resolve_shard_sizes(
        batch,
        dp_size=resolved_dp_size,
        per_rank_batch_size=per_rank_batch_size,
    )
    start = sum(int(size) for size in shard_sizes[: int(rank)])
    end = start + int(shard_sizes[int(rank)])
    return batch.slice(start, end)
