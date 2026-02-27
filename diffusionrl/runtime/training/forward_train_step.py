"""Forward (NFT) training step implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from diffusionrl.types.training_batch import ForwardTrainingBatch

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
    gradient_accumulation_steps: int,
    timestep_mode: str,
    shuffle_timesteps: bool,
    apply_shift: bool,
) -> tuple[torch.Tensor, Dict[str, Any], int, int, bool]:
    """Run one forward-path update (random timestep or all-timestep mode)."""
    if timestep_mode == "all" and batch.timesteps is not None:
        timesteps = batch.timesteps
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

        if timesteps.numel() == 0:
            logger.warning("NFT all-timestep mode: empty timesteps, falling back to random mode")
        else:
            if shuffle_timesteps:
                perm = torch.randperm(timesteps.numel(), device=timesteps.device)
                timesteps = timesteps[perm]

            effective_grad_accum = max(1, int(gradient_accumulation_steps)) * timesteps.numel()
            total_loss = torch.zeros((), device=batch.advantages.device)
            metrics_sum: Dict[str, float] = {}

            for t in timesteps:
                loss, metrics = _compute_forward_batch_loss(
                    loss_fn=loss_fn,
                    model=model,
                    batch=batch,
                    timestep_values=t,
                    apply_shift=apply_shift,
                )

                (loss / effective_grad_accum).backward()
                total_loss = total_loss + loss.detach()

                for key, value in metrics.items():
                    metric_val = value.item() if isinstance(value, torch.Tensor) else float(value)
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + metric_val

            all_metrics: Dict[str, Any] = {}
            for key, value in metrics_sum.items():
                all_metrics[key] = value / timesteps.numel()

            return (
                total_loss / timesteps.numel(),
                all_metrics,
                timesteps.numel(),
                effective_grad_accum,
                True,
            )

    loss, metrics = _compute_forward_batch_loss(
        loss_fn=loss_fn,
        model=model,
        batch=batch,
    )

    all_metrics: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            all_metrics[key] = value.item()
        else:
            all_metrics[key] = value

    return loss, all_metrics, 1, 1, False
