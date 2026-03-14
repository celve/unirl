"""Forward (NFT) training step implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusionrl.runtime.training.update_schedule import resolve_gradient_accumulation_plan
from diffusionrl.types.training_batch import ForwardTrainingBatch
from diffusionrl.samplers.schedulers.timestep_window import _normalize_timestep_fraction

logger = logging.getLogger(__name__)


def _compute_forward_batch_loss(
    *,
    loss_fn: Any,
    model: torch.nn.Module,
    batch: ForwardTrainingBatch,
    timestep_values: Optional[torch.Tensor] = None,
    apply_shift: Optional[bool] = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    compute_batch = getattr(loss_fn, "compute_batch", None)
    if not callable(compute_batch):
        raise TypeError(
            f"Loss {type(loss_fn).__name__} must implement compute_batch(). "
            "Legacy compute() is no longer supported."
        )
    kwargs: Dict[str, Any] = {
        "model": model,
        "batch": batch,
    }
    if timestep_values is not None:
        kwargs["timestep_values"] = timestep_values
    if apply_shift is not None:
        kwargs["apply_shift"] = apply_shift
    return compute_batch(**kwargs)


def train_forward_batch(
    *,
    batch: ForwardTrainingBatch,
    loss_fn: Any,
    model: torch.nn.Module,
    gradient_accumulation_batch_size: int,
    timestep_mode: str,
    shuffle_timesteps: bool,
    apply_shift: bool,
    timestep_fraction: Union[float, Tuple[float, float]] = 1.0,
) -> tuple[float, Dict[str, Any], int, int, bool]:
    """Run one forward-path update (random timestep or all-timestep mode).

    When timestep_mode is \"all\", timestep_fraction restricts which timesteps
    are trained: only indices in [frac_start*N, frac_end*N) are used (same
    semantics as DanceGRPO/rollout timestep_fraction).
    """
    mini_batches, actual_mini_batches = resolve_gradient_accumulation_plan(
        batch_size=batch.batch_size,
        gradient_accumulation_batch_size=gradient_accumulation_batch_size,
    )
    num_mini_batches = len(mini_batches)

    if timestep_mode == "all" and batch.timesteps is not None:
        timesteps = batch.timesteps
    else:
        timesteps = torch.rand(batch.batch_size, device=batch.advantages.device)

    if isinstance(timesteps, torch.Tensor):
        timesteps = timesteps.detach()
    else:
        timesteps = torch.tensor(timesteps, device=batch.advantages.device)

    timesteps = timesteps.flatten()
    if timesteps.numel() > 1 and torch.isclose(
        timesteps[-1],
        torch.zeros((), device=timesteps.device, dtype=timesteps.dtype),
        atol=1e-8,
    ).item():
        timesteps = timesteps[:-1]

    # Apply timestep_fraction: train only on [frac_start*N, frac_end*N)
    if timesteps.numel() > 0 and timestep_fraction is not None and timestep_fraction != 1.0:
        frac_start, frac_end = _normalize_timestep_fraction(timestep_fraction)
        n = timesteps.numel()
        effective_start = int(n * frac_start)
        effective_end = int(n * frac_end)
        effective_end = min(effective_end, n)
        if effective_start < effective_end:
            timesteps = timesteps[effective_start:effective_end]
        else:
            timesteps = timesteps[:0]

    if timesteps.numel() == 0:
        logger.warning("NFT all-timestep mode: empty timesteps (or empty after timestep_fraction), falling back to random mode")
    else:
        if shuffle_timesteps:
            perm = torch.randperm(timesteps.numel(), device=timesteps.device)
            timesteps = timesteps[perm]

        effective_mini_batches = actual_mini_batches * timesteps.numel()
        total_loss_accum = 0.0
        mini_batch_metrics_list: List[Dict[str, Any]] = []
        has_backward = False

        for start, end in mini_batches:
            mini_batch = batch.slice(start, end)
            mini_loss_sum = 0.0
            mini_metrics: Dict[str, Any] = {}
            metric_sums: Dict[str, float] = {}
            metric_counts: Dict[str, int] = {}

            for t in timesteps:
                loss, metrics = _compute_forward_batch_loss(
                    loss_fn=loss_fn,
                    model=model,
                    batch=mini_batch,
                    timestep_values=t,
                    apply_shift=apply_shift,
                )

                (loss / effective_mini_batches).backward()
                has_backward = True

                mini_loss_sum += loss.detach().item()

                for key, value in metrics.items():
                    metric_val = value.item() if isinstance(value, torch.Tensor) else float(value)
                    metric_sums[key] = metric_sums.get(key, 0.0) + metric_val
                    metric_counts[key] = metric_counts.get(key, 0) + 1
                    if key not in mini_metrics:
                        mini_metrics[key] = metric_val

            for key, total in metric_sums.items():
                count = metric_counts.get(key, 0)
                if count > 0:
                    mini_metrics[key] = total / count

            mini_batch_metrics_list.append(mini_metrics)
            total_loss_accum += mini_loss_sum / timesteps.numel()

        all_metrics: Dict[str, Any] = {}
        if mini_batch_metrics_list:
            keys = mini_batch_metrics_list[0].keys()
            for key in keys:
                values = [m.get(key) for m in mini_batch_metrics_list if m.get(key) is not None]
                if values and isinstance(values[0], (int, float)):
                    all_metrics[key] = sum(values) / len(values)
                else:
                    all_metrics[key] = mini_batch_metrics_list[-1].get(key)

        return (
            total_loss_accum / max(1, num_mini_batches),
            all_metrics,
            timesteps.numel(),
            effective_mini_batches,
            has_backward,
        )