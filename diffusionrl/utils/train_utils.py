"""Shared helpers for sync/async training entrypoints."""

import logging
import os
import re
from typing import Any, Optional, Set

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


def create_rollout_timestep_scheduler(args, *, algorithm: Any):
    """Create the control-plane timestep scheduler for rollout SDE selection."""
    from diffusionrl.config.resolution import collect_sampling_requirements
    from diffusionrl.samplers.schedulers import get_scheduler
    from diffusionrl.samplers.schedulers.timestep_window import _normalize_timestep_fraction

    requirements = collect_sampling_requirements(algorithm=algorithm)
    scheduler_type = args.algorithm.window.timestep_strategy
    num_timesteps = int(args.sampling.num_inference_steps)
    timestep_fraction = args.sampling.timestep_fraction

    if scheduler_type == "all" and requirements.sde_ratio < 1.0:
        group_size = max(1, int(num_timesteps * requirements.sde_ratio))
        scheduler_type = "window"
        logger.info("Auto-configured window scheduler from sde_ratio=%s", requirements.sde_ratio)
    else:
        group_size = None

    if scheduler_type == "window":
        explicit_group_size = args.algorithm.window.window_group_size
        if explicit_group_size is None and requirements.sde_ratio < 1.0:
            group_size = max(1, int(num_timesteps * requirements.sde_ratio))
        else:
            group_size = explicit_group_size or 4
        scheduler = get_scheduler(
            scheduler_type="window",
            num_timesteps=num_timesteps,
            timestep_fraction=timestep_fraction,
            strategy=args.algorithm.window.window_strategy,
            group_size=group_size,
            iters_per_group=args.algorithm.window.window_iters_per_group,
            max_iters_per_group=args.algorithm.window.window_max_iters_per_group,
            min_iters_per_group=args.algorithm.window.window_min_iters_per_group,
            overlap=args.algorithm.window.window_overlap,
            overlap_step=args.algorithm.window.window_overlap_step,
            roll_back=args.algorithm.window.window_roll_back,
        )
        logger.info("Control-plane window scheduler initialized: group_size=%s", group_size)
        return scheduler

    scheduler = get_scheduler(
        scheduler_type="all",
        num_timesteps=num_timesteps,
        timestep_fraction=timestep_fraction,
        num_sde_steps=args.sampling.num_sde_steps,
    )
    frac_start, frac_end = _normalize_timestep_fraction(timestep_fraction)
    if frac_start > 0.0 or frac_end < 1.0 or args.sampling.num_sde_steps is not None:
        eff_start = int(num_timesteps * frac_start)
        eff_end = int(num_timesteps * frac_end)
        logger.info(
            "Control-plane all-SDE scheduler initialized; timestep_fraction=%s num_sde_steps=%s (SDE pool [%s, %s)/%s)",
            timestep_fraction,
            args.sampling.num_sde_steps,
            eff_start,
            eff_end,
            num_timesteps,
        )
    else:
        logger.info("Control-plane all-SDE scheduler initialized")
    return scheduler


class RolloutSDEController:
    """Driver-side control-plane owner of rollout SDE scheduling."""

    def __init__(
        self,
        *,
        algorithm: Any,
        timestep_scheduler: Optional[Any],
    ) -> None:
        self.algorithm = algorithm
        self.timestep_scheduler = timestep_scheduler
        self._current_step = 0

    def next_sde_indices(self) -> Optional[Set[int]]:
        """Return the current rollout's SDE indices and advance local scheduler state."""
        sde_indices = self.algorithm.resolve_rollout_sde_indices(
            timestep_scheduler=self.timestep_scheduler,
            current_step=self._current_step,
        )
        normalized = set(int(i) for i in sde_indices) if sde_indices is not None else None
        self._current_step += 1
        if self.timestep_scheduler is not None:
            self.timestep_scheduler.update(self._current_step)
        return normalized


__all__ = [
    "RolloutSDEController",
    "build_control_algorithm",
    "collect_rollout_batch_metrics",
    "create_rollout_timestep_scheduler",
    "should_eval",
    "should_log",
    "should_save",
]
