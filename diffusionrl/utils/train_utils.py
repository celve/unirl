"""Shared helpers for sync/async training entrypoints."""

import os
import re
from typing import Optional

from omegaconf import DictConfig


def should_save(rollout_id: int, cfg: DictConfig) -> bool:
    """Check if we should save a checkpoint at this rollout."""
    interval = int(cfg.resume.save_steps)
    return interval > 0 and (rollout_id + 1) % interval == 0


def should_eval(rollout_id: int, cfg: DictConfig) -> bool:
    """Check if we should run evaluation at this rollout."""
    interval = int(cfg.evaluation.eval_steps)
    return interval > 0 and (rollout_id + 1) % interval == 0


def should_log(rollout_id: int, cfg: DictConfig) -> bool:
    """Check if we should emit periodic rollout logs at this rollout."""
    interval = int(cfg.logging.logging_steps)
    return interval > 0 and (rollout_id + 1) % interval == 0


def maybe_restore_start_rollout_id_from_checkpoint(cfg: DictConfig, checkpoint_path: Optional[str]) -> Optional[int]:
    """Infer and persist the next rollout id when resuming from a checkpoint path.

    Writes the resolved ``start_rollout_id`` back onto ``cfg.resume`` so
    the outer loop picks it up. Returns the value for logging, or ``None``
    when no restore is needed.
    """
    if not checkpoint_path:
        return None

    if int(cfg.resume.start_rollout_id) != 0:
        return None

    match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(checkpoint_path)))
    if not match:
        return None

    next_rollout_id = int(match.group(1)) + 1
    cfg.resume.start_rollout_id = next_rollout_id
    return next_rollout_id


__all__ = [
    "should_eval",
    "should_log",
    "should_save",
    "maybe_restore_start_rollout_id_from_checkpoint",
]
