"""Shared helpers for sync/async training entrypoints."""

import logging
import os
import re
from typing import Any, Optional

from diffusionrl.algorithms.construction import (
    instantiate_algorithm_from_config,
)

logger = logging.getLogger(__name__)


def should_save(rollout_id: int, args) -> bool:
    """Check if we should save a checkpoint at this rollout."""
    interval = int(args.rollout.artifacts.save_steps)
    return interval > 0 and (rollout_id + 1) % interval == 0


def should_eval(rollout_id: int, args) -> bool:
    """Check if we should run evaluation at this rollout."""
    interval = int(args.rollout.evaluation.eval_steps)
    return interval > 0 and rollout_id % interval == 0


def should_log(rollout_id: int, args) -> bool:
    """Check if we should emit periodic rollout logs at this rollout."""
    interval = int(args.rollout.logging.logging_steps)
    return interval > 0 and rollout_id % interval == 0


def maybe_restore_start_rollout_id_from_checkpoint(args, checkpoint_path: Optional[str]) -> Optional[int]:
    """Infer and persist the next rollout id when resuming from a checkpoint path."""
    if not checkpoint_path:
        return None

    rollout_control = args.rollout.control
    if int(rollout_control.start_rollout_id) != 0:
        return None

    match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(checkpoint_path)))
    if not match:
        return None

    next_rollout_id = int(match.group(1)) + 1
    args.rollout.set_start_rollout_id(next_rollout_id)
    return next_rollout_id


def collect_rollout_batch_metrics(*, ray_module, batch_ref, compute_rollout_batch_metrics_fn) -> dict:
    """Best-effort materialization for rollout-level observability only."""
    if batch_ref is None:
        return {}
    try:
        training_data = ray_module.get(batch_ref)
    except Exception as exc:
        logger.warning("Failed to materialize training batch for rollout metrics: %s", exc)
        return {}
    return compute_rollout_batch_metrics_fn(training_data=training_data)


def build_control_algorithm(algorithm_config: dict) -> tuple[dict, Any]:
    """Instantiate the driver-local control-plane algorithm."""
    if not isinstance(algorithm_config, dict):
        raise ValueError(
            "build_control_algorithm requires a canonical algorithm_config dict from the driver."
        )
    resolved_algorithm_config = dict(algorithm_config)
    return resolved_algorithm_config, instantiate_algorithm_from_config(resolved_algorithm_config)


__all__ = [
    "build_control_algorithm",
    "collect_rollout_batch_metrics",
    "should_eval",
    "should_log",
    "should_save",
]
