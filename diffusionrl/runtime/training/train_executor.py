"""Training executor that owns core optimization flow."""

from __future__ import annotations

import os
import re
from functools import partial
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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
from diffusionrl.runtime.training.update_schedule import (
    TrainingUpdateSchedule,
    create_training_update_schedule,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


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
    gradient_accumulation_batch_size: int
    multi_update_batch_size: Optional[int]
    update_mode: str
    use_ema: bool
    ema_updater: Optional[Any]
    nft_timestep_mode: str
    nft_shuffle_timesteps: bool
    nft_apply_shift: bool
    clip_grad_norm_fn: Optional[Callable[..., Any]] = None


def aggregate_numeric_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate numeric metrics from repeated update chunks."""
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
        self.update_schedule: TrainingUpdateSchedule = create_training_update_schedule(
            config.update_mode
        )

    def _shard_batch_by_rank(self, batch: TrainingBatch) -> Optional[TrainingBatch]:
        dp_size = max(1, int(self.config.dp_size))
        if dp_size <= 1 or getattr(batch, "is_partitioned", False):
            return batch

        batch_size = batch.batch_size
        per_rank = batch_size // dp_size
        remainder = batch_size % dp_size

        if per_rank == 0:
            raise ValueError(
                "Training batch size is smaller than dp_size; each rank needs at least one sample. "
                f"Got batch_size={batch_size}, dp_size={dp_size}."
            )

        if remainder != 0 and self.config.rank == 0:
            raise ValueError(
                "Training batch size must be divisible by dp_size. "
                "DiffusionRL no longer drops remainder samples implicitly. "
                f"Got batch_size={batch_size}, dp_size={dp_size}, remainder={remainder}."
            )
        if remainder != 0:
            raise ValueError(
                "Training batch size must be divisible by dp_size. "
                f"Got batch_size={batch_size}, dp_size={dp_size}, remainder={remainder}."
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

        return prepared

    def _current_lr(self) -> float:
        """Best-effort LR lookup compatible with custom optimizer wrappers."""
        param_groups = getattr(self.optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            try:
                return float(param_groups[0]["lr"])
            except Exception:
                pass
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "get_last_lr"):
            try:
                last = self.lr_scheduler.get_last_lr()
                if isinstance(last, list) and last:
                    return float(last[0])
            except Exception:
                pass
        return 0.0

    def _train_update_chunk(
        self,
        *,
        batch: TrainingBatch,
        gradient_accumulation_batch_size: int,
    ) -> tuple[float, Dict[str, Any], int, int, bool]:
        if isinstance(batch, ForwardTrainingBatch):
            return train_forward_batch(
                batch=batch,
                loss_fn=self.loss_fn,
                model=self.model,
                gradient_accumulation_batch_size=gradient_accumulation_batch_size,
                timestep_mode=self.config.nft_timestep_mode,
                shuffle_timesteps=self.config.nft_shuffle_timesteps,
                apply_shift=self.config.nft_apply_shift,
            )
        return train_backward_with_accumulation(
            batch=batch,
            loss_fn=self.loss_fn,
            model=self.model,
            guidance_scale=self.config.guidance_scale,
            gradient_accumulation_batch_size=gradient_accumulation_batch_size,
        )

    def _clip_grad_norm(self) -> float:
        """Clip gradients using backend-aware semantics and return the norm."""
        if callable(self.config.clip_grad_norm_fn):
            grad_norm = self.config.clip_grad_norm_fn(
                model=self.model,
                max_grad_norm=self.config.max_grad_norm,
            )
        elif self.config.use_fsdp:
            grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm,
            )
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    def _apply_optimizer_step(self) -> None:
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def skipped_metrics(self, rollout_id: int) -> Dict[str, Any]:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "lr": self._current_lr(),
            "rollout_id": rollout_id,
            "skipped": True,
        }

    def execute_prepared_batch(self, *, rollout_id: int, batch: TrainingBatch) -> Dict[str, Any]:
        """Run optimization on a pre-validated, on-device typed batch."""
        inner_metrics: List[Dict[str, Any]] = []
        total_timesteps = 0
        last_mini_batches_per_update = 1
        total_mini_batches_consumed = 0
        optimizer_steps = 0
        last_effective_update_batch_size = int(batch.batch_size)
        last_effective_gradient_accumulation_batch_size = int(
            self.config.gradient_accumulation_batch_size
        )
        update_mode = self.update_schedule.name

        rank = self.config.rank
        for update_chunk in tqdm(
            self.update_schedule.iter_update_chunks(
                batch=batch,
                gradient_accumulation_batch_size=self.config.gradient_accumulation_batch_size,
                multi_update_batch_size=self.config.multi_update_batch_size,
            ),
            desc=f"Training {rollout_id}:",
            unit="update",
            disable=(rank != 0),
        ):
            self.optimizer.zero_grad()
            total_loss, all_metrics, num_timesteps, actual_mini_batches, has_backward = self._train_update_chunk(
                batch=update_chunk.batch,
                gradient_accumulation_batch_size=update_chunk.gradient_accumulation_batch_size,
            )

            if has_backward:
                grad_norm = self._clip_grad_norm()
                self._apply_optimizer_step()
                optimizer_steps += 1
            else:
                grad_norm = 0.0
                logger.warning(
                    "No valid timesteps to train in %s update, skipping optimizer step",
                    update_mode,
                )

            if self.config.use_ema and self.config.ema_updater is not None:
                ema_success = self.config.ema_updater.update(self.model)
                all_metrics["ema_updated"] = ema_success

            last_mini_batches_per_update = actual_mini_batches
            last_effective_update_batch_size = int(update_chunk.update_batch_size)
            last_effective_gradient_accumulation_batch_size = int(
                update_chunk.gradient_accumulation_batch_size
            )
            total_mini_batches_consumed += actual_mini_batches
            total_timesteps += num_timesteps

            step_metrics = {
                "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": self._current_lr(),
                "num_timesteps_trained": num_timesteps,
                "mini_batches_per_update": actual_mini_batches,
                "has_backward": has_backward,
                **all_metrics,
            }
            inner_metrics.append(step_metrics)

        metrics = aggregate_numeric_metrics(inner_metrics)
        # Filter out per-timestep metrics (t{N}_...) from aggregated output;
        # these are exposed only via _per_optimizer_step_metrics for train/ namespace.
        metrics = {k: v for k, v in metrics.items() if not re.match(r"^t\d+_", k)}
        metrics.update(
            {
                "rollout_id": rollout_id,
                "loss_type": self.config.loss_type,
                "training_update_mode": update_mode,
                "configured_gradient_accumulation_batch_size": self.config.gradient_accumulation_batch_size,
                "configured_multi_update_batch_size": self.config.multi_update_batch_size,
                "effective_gradient_accumulation_batch_size": last_effective_gradient_accumulation_batch_size,
                "effective_update_batch_size": last_effective_update_batch_size,
                "num_timesteps_trained": total_timesteps,
                "mini_batches_per_update": last_mini_batches_per_update,
                "mini_batches_consumed": total_mini_batches_consumed,
                "optimizer_steps": optimizer_steps,
            }
        )
        metrics["_per_optimizer_step_metrics"] = inner_metrics
        return metrics
