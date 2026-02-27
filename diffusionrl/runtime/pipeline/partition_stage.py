"""Partition stage helpers for rollout->train handoff."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def maybe_partition_training_batch(
    *,
    train_data: Any,
    dp_size: Optional[int],
    partition_train_data: bool = True,
) -> Optional[List[Any]]:
    """
    Optionally partition a typed training batch across training ranks.

    Returns:
        List of per-rank typed batch partitions when partition is applied; otherwise None.
    """
    if not partition_train_data or not dp_size:
        return None

    batch_size = getattr(train_data, "batch_size", None)
    if batch_size is None:
        logger.warning("Training batch does not expose batch_size; skipping partition.")
        return None

    per_rank = batch_size // dp_size
    remainder = batch_size % dp_size

    if per_rank == 0:
        logger.warning(
            "Batch size %d too small for dp_size %d; skipping partition.",
            batch_size,
            dp_size,
        )
        return None

    if remainder != 0:
        logger.warning(
            "Batch size %d not divisible by dp_size %d; dropping %d samples for even partition.",
            batch_size,
            dp_size,
            remainder,
        )

    partitions: List[Any] = []
    for rank in range(dp_size):
        start = rank * per_rank
        end = start + per_rank
        part = train_data.slice(start, end)
        partitions.append(part)
    return partitions
