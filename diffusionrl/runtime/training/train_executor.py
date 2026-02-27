"""Training executor that owns core optimization flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.types.training_batch import (
    BackwardTrainingBatch,
    ForwardTrainingBatch,
    TrainingBatch,
)
from diffusionrl.runtime.training.backward_train_step import (
    train_backward_with_accumulation,
)
from diffusionrl.runtime.training.forward_train_step import train_forward_batch

logger = logging.getLogger(__name__)


@dataclass
class TrainExecutorConfig:
    """Runtime knobs for training execution."""

    rank: int
    dp_size: int
    device: torch.device
    use_fsdp: bool
    loss_type: str
    guidance_scale: float
    max_grad_norm: float
    gradient_accumulation_steps: int
    num_inner_epochs: int
    use_ema: bool
    ema_updater: Optional[Any]
    nft_timestep_mode: str
    nft_shuffle_timesteps: bool
    nft_apply_shift: bool


def resolve_grad_accum(training_config: dict) -> int:
    """Compute gradient accumulation steps with optional auto mode."""
    raw = training_config.get("gradient_accumulation_steps", 1)
    if isinstance(raw, int):
        return max(1, raw)
    if isinstance(raw, str) and raw.lower() == "auto":
        prompts_per_batch = training_config.get("prompts_per_batch", 1)
        k = training_config.get("num_samples_per_prompt", 1)
        dp_size = training_config.get("dp_size", 1)
        batch_size = training_config.get("batch_size", 1)
        grad_steps_per_epoch = training_config.get("gradient_steps_per_epoch", 1)
        try:
            total_gen = prompts_per_batch * k
            target_per_update = total_gen / max(1, grad_steps_per_epoch)
            denom = batch_size * max(1, dp_size)
            accum = int((target_per_update + denom - 1) // denom)
            accum = max(1, accum)
            logger.info(
                "Auto gradient_accumulation_steps=%d (prompts_per_batch=%d, k=%d, dp_size=%d, batch_size=%d, grad_steps_per_epoch=%d)",
                accum,
                prompts_per_batch,
                k,
                dp_size,
                batch_size,
                grad_steps_per_epoch,
            )
            return accum
        except Exception as e:
            logger.warning("Auto gradient accumulation failed (%s), fallback to 1", e)
            return 1
    try:
        return max(1, int(raw))
    except Exception:
        logger.warning("Invalid gradient_accumulation_steps=%s, fallback to 1", raw)
        return 1


def aggregate_numeric_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate numeric metrics from repeated inner-epoch updates."""
    aggregated: Dict[str, float] = {}
    if not metrics_list:
        return aggregated

    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    for key in all_keys:
        values: List[float] = []
        for metrics in metrics_list:
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            aggregated[key] = sum(values) / len(values)

    return aggregated


class TrainExecutor:
    """Execution-layer trainer used by RPC actors."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: Any,
        config: TrainExecutorConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.config = config

    def _shard_batch_by_rank(self, batch: TrainingBatch) -> Optional[TrainingBatch]:
        dp_size = max(1, int(self.config.dp_size))
        if dp_size <= 1 or getattr(batch, "is_partitioned", False):
            return batch

        batch_size = batch.batch_size
        per_rank = batch_size // dp_size
        remainder = batch_size % dp_size

        if per_rank == 0:
            logger.error(
                "Rank %s: batch_size=%s too small for dp_size=%s; skipping train step",
                self.config.rank,
                batch_size,
                dp_size,
            )
            return None

        if remainder != 0 and self.config.rank == 0:
            logger.warning(
                "Batch size %d not divisible by dp_size %d; dropping %d samples for even sharding",
                batch_size,
                dp_size,
                remainder,
            )

        start = self.config.rank * per_rank
        end = start + per_rank
        return batch.slice(start, end)

    def prepare_batch(self, batch: TrainingBatch) -> Optional[TrainingBatch]:
        """Validate, shard, and move training batch to compute device."""
        if not isinstance(batch, (BackwardTrainingBatch, ForwardTrainingBatch)):
            raise TypeError(
                f"Unsupported batch type: {type(batch).__name__}. "
                "Expected BackwardTrainingBatch or ForwardTrainingBatch. "
                "Legacy dict format is not supported - use typed batches."
            )

        sharded = self._shard_batch_by_rank(batch)
        if sharded is None:
            return None

        prepared = sharded.to_device(self.config.device)
        prepared.validate()

        if (
            isinstance(prepared, ForwardTrainingBatch)
            and self.config.gradient_accumulation_steps > 1
            and self.config.nft_timestep_mode != "all"
        ):
            logger.warning(
                "gradient_accumulation_steps=%s is ignored for NFT loss (single forward pass)",
                self.config.gradient_accumulation_steps,
            )

        return prepared

    def skipped_metrics(self, rollout_id: int) -> Dict[str, Any]:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "lr": self.optimizer.param_groups[0]["lr"],
            "rollout_id": rollout_id,
            "skipped": True,
        }

    def execute_prepared_batch(self, *, rollout_id: int, batch: TrainingBatch) -> Dict[str, Any]:
        """Run optimization on a pre-validated, on-device typed batch."""
        inner_metrics: List[Dict[str, Any]] = []
        total_timesteps = 0
        last_grad_accum = 1

        for _inner_epoch_id in range(max(1, int(self.config.num_inner_epochs))):
            has_backward = False

            if isinstance(batch, ForwardTrainingBatch):
                self.optimizer.zero_grad()
                total_loss, all_metrics, num_timesteps, actual_grad_accum, has_backward = train_forward_batch(
                    batch=batch,
                    loss_fn=self.loss_fn,
                    model=self.model,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    timestep_mode=self.config.nft_timestep_mode,
                    shuffle_timesteps=self.config.nft_shuffle_timesteps,
                    apply_shift=self.config.nft_apply_shift,
                )
                if not has_backward:
                    total_loss.backward()
                    has_backward = True
            else:
                total_loss, all_metrics, num_timesteps, actual_grad_accum, has_backward = train_backward_with_accumulation(
                    batch=batch,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_fn,
                    model=self.model,
                    guidance_scale=self.config.guidance_scale,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                )

            if has_backward:
                if self.config.use_fsdp:
                    grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm,
                    )

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            else:
                grad_norm = 0.0
                logger.warning("No valid timesteps to train, skipping optimizer step")

            if self.config.use_ema and self.config.ema_updater is not None:
                ema_success = self.config.ema_updater.update(self.model)
                all_metrics["ema_updated"] = ema_success

            last_grad_accum = actual_grad_accum
            total_timesteps += num_timesteps

            step_metrics = {
                "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": self.optimizer.param_groups[0]["lr"],
                "num_timesteps_trained": num_timesteps,
                "gradient_accumulation_steps": actual_grad_accum,
                **all_metrics,
            }
            inner_metrics.append(step_metrics)

        metrics = aggregate_numeric_metrics(inner_metrics)
        metrics.update(
            {
                "rollout_id": rollout_id,
                "loss_type": self.config.loss_type,
                "num_inner_epochs": self.config.num_inner_epochs,
                "num_timesteps_trained": total_timesteps,
                "gradient_accumulation_steps": last_grad_accum,
                "effective_gradient_accumulation_steps": last_grad_accum * self.config.num_inner_epochs,
            }
        )
        return metrics
