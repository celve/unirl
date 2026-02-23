"""Forward (NFT) training step implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from diffusionrl.types.training_batch import ForwardTrainingBatch

logger = logging.getLogger(__name__)


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
                if hasattr(loss_fn, "compute_batch"):
                    loss, metrics = loss_fn.compute_batch(
                        model=model,
                        batch=batch,
                        timestep_values=t,
                        apply_shift=apply_shift,
                    )
                else:
                    loss, metrics = loss_fn.compute(
                        model=model,
                        samples=batch.to_loss_dict(),
                        timestep_idx=0,
                        advantages=batch.advantages,
                        prompt_embeds=batch.embeddings.prompt_embeds,
                        pooled_prompt_embeds=batch.embeddings.pooled_prompt_embeds,
                        text_ids=batch.embeddings.text_ids,
                        image_ids=batch.embeddings.image_ids,
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

    if hasattr(loss_fn, "compute_batch"):
        loss, metrics = loss_fn.compute_batch(
            model=model,
            batch=batch,
        )
    else:
        loss, metrics = loss_fn.compute(
            model=model,
            samples=batch.to_loss_dict(),
            timestep_idx=0,
            advantages=batch.advantages,
            prompt_embeds=batch.embeddings.prompt_embeds,
            pooled_prompt_embeds=batch.embeddings.pooled_prompt_embeds,
            text_ids=batch.embeddings.text_ids,
            image_ids=batch.embeddings.image_ids,
        )

    all_metrics: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            all_metrics[key] = value.item()
        else:
            all_metrics[key] = value

    return loss, all_metrics, 1, 1, False
