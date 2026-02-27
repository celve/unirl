"""Backward (GRPO/MixGRPO) training step implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch

from diffusionrl.types.training_batch import BackwardTrainingBatch

logger = logging.getLogger(__name__)


def compute_backward_timestep_loss(
    *,
    loss_fn: Any,
    model: torch.nn.Module,
    batch: BackwardTrainingBatch,
    timestep_idx: int,
    guidance_scale: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute backward-path loss/metrics for one logical diffusion step."""
    timestep_data = batch.get_timestep_data_by_step(timestep_idx)
    compute_timestep = getattr(loss_fn, "compute_timestep", None)
    if not callable(compute_timestep):
        raise TypeError(
            f"Loss {type(loss_fn).__name__} must implement compute_timestep(). "
            "Legacy compute() is no longer supported."
        )
    return compute_timestep(
        model=model,
        timestep_data=timestep_data,
        advantages=batch.advantages,
        embeddings=batch.embeddings,
        guidance_scale=guidance_scale,
    )


def train_backward_with_accumulation(
    *,
    batch: BackwardTrainingBatch,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    model: torch.nn.Module,
    guidance_scale: float,
    gradient_accumulation_steps: int,
) -> tuple[float, Dict[str, Any], int, int, bool]:
    """Run backward-path micro-batch gradient accumulation over selected SDE steps."""
    batch_size = batch.batch_size
    grad_accum = max(1, int(gradient_accumulation_steps))

    if batch_size % grad_accum != 0:
        logger.warning(
            "batch_size %s not divisible by gradient_accumulation_steps %s; fallback to grad_accum=1",
            batch_size,
            grad_accum,
        )
        grad_accum = 1

    micro_batch_size = batch_size // grad_accum
    num_micro_batches = grad_accum

    optimizer.zero_grad()
    total_loss_accum = 0.0
    has_backward = False

    available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
    valid_step_indices = sorted(int(i) for i in batch.sde_indices if int(i) in available_steps)
    num_timesteps_per_sample = len(valid_step_indices)
    if num_timesteps_per_sample == 0:
        logger.warning("No valid SDE timesteps in batch, skipping GRPO backward")
        return 0.0, {}, 0, grad_accum, False

    micro_batch_metrics_list: list[Dict[str, Any]] = []

    for mbs_id in range(num_micro_batches):
        start = mbs_id * micro_batch_size
        end = start + micro_batch_size
        micro_batch = batch.slice(start, end)

        micro_loss_sum = 0.0
        micro_metrics: Dict[str, Any] = {}
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        for t_idx in valid_step_indices:
            loss_t, metrics_t = compute_backward_timestep_loss(
                loss_fn=loss_fn,
                model=model,
                batch=micro_batch,
                timestep_idx=t_idx,
                guidance_scale=guidance_scale,
            )
            scaled_loss = loss_t / (num_micro_batches * num_timesteps_per_sample)
            scaled_loss.backward()
            has_backward = True
            micro_loss_sum += loss_t.detach().item()

            for key, value in metrics_t.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                metric_key = f"t{t_idx}_{key}"
                if metric_key not in micro_metrics:
                    micro_metrics[metric_key] = val
                if isinstance(val, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(val)
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        for key, total in metric_sums.items():
            count = metric_counts.get(key, 0)
            if count > 0:
                micro_metrics[key] = total / count

        micro_loss = micro_loss_sum / num_timesteps_per_sample
        micro_batch_metrics_list.append(micro_metrics)
        total_loss_accum += micro_loss

    all_metrics: Dict[str, Any] = {}
    if micro_batch_metrics_list:
        keys = micro_batch_metrics_list[0].keys()
        for key in keys:
            values = [m.get(key) for m in micro_batch_metrics_list if m.get(key) is not None]
            if values and isinstance(values[0], (int, float)):
                all_metrics[key] = sum(values) / len(values)
            else:
                all_metrics[key] = micro_batch_metrics_list[-1].get(key)

    avg_loss = total_loss_accum / num_micro_batches if num_micro_batches > 0 else 0.0
    return avg_loss, all_metrics, num_timesteps_per_sample, grad_accum, has_backward
