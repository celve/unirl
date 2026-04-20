from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from diffusionrl.algorithms import BaseAlgorithm
from diffusionrl.training.backends.base import TrainBackend
from diffusionrl.training.update_schedule import _build_micro_batch_slices
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils.ema import EMAManager
from diffusionrl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicroBatchResult:
    """Result of a single forward + scaled backward on one micro-batch."""

    loss: float
    num_timesteps: int
    has_backward: bool
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class MiniBatchResult:
    """Result of one optimizer step over a full mini-batch."""

    loss: float
    grad_norm: float
    lr: float
    num_timesteps_trained: int
    has_backward: bool
    micro_batch_results: List[MicroBatchResult]
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class BatchResult:
    """Result of training over a full batch (one or more optimizer steps)."""

    rollout_step: int
    loss: float
    grad_norm: float
    lr: float
    num_timesteps_trained: int
    optimizer_steps: int
    num_mini_batches: int
    has_backward: bool
    mini_batch_results: List[MiniBatchResult]
    metrics: Dict[str, Any]

    def to_legacy_metric_dict(self) -> Dict[str, Any]:
        """Convert to the dict shape legacy train.py callers (and wandb) expect."""
        per_step = [
            {
                **dict(mini.metrics or {}),
                "loss": float(mini.loss),
                "grad_norm": float(mini.grad_norm),
                "lr": float(mini.lr),
                "has_backward": bool(mini.has_backward),
            }
            for mini in self.mini_batch_results
        ]
        return {
            **dict(self.metrics or {}),
            "loss": float(self.loss),
            "grad_norm": float(self.grad_norm),
            "lr": float(self.lr),
            "has_backward": bool(self.has_backward),
            "optimizer_steps": int(self.optimizer_steps),
            "rollout_step": int(self.rollout_step),
            "_per_optimizer_step_metrics": per_step,
        }


@dataclass
class TrainStack:
    backend: TrainBackend
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    algorithm: BaseAlgorithm
    ema_manager: EMAManager
    max_grad_norm: float = 1.0

    def train_batch(
        self,
        batch: TrainingBatch,
        *,
        mini_batch_size: int,
        micro_batch_size: int,
        rollout_step: int,
    ) -> BatchResult:
        mini_slices = _build_micro_batch_slices(
            total_size=batch.batch_size,
            micro_batch_size=mini_batch_size,
        )
        if not mini_slices:
            raise ValueError("train_batch requires a non-empty batch.")

        mini_results: List[MiniBatchResult] = []
        for start, end in mini_slices:
            mini_results.append(
                self.train_minibatch(
                    batch.slice(start, end),
                    micro_batch_size=micro_batch_size,
                    rollout_step=rollout_step,
                )
            )

        n = len(mini_results)
        mean_loss = sum(r.loss for r in mini_results) / n
        mean_grad_norm = sum(r.grad_norm for r in mini_results) / n
        last_lr = mini_results[-1].lr
        optimizer_steps = sum(1 for r in mini_results if r.has_backward)
        total_timesteps = sum(r.num_timesteps_trained for r in mini_results)
        has_backward = any(r.has_backward for r in mini_results)

        aggregated_metrics = aggregate_numeric_metrics(
            [r.metrics for r in mini_results if r.metrics]
        )

        if optimizer_steps > 0:
            self.ema_manager.post_rollout_end(
                self.backend.model, aggregated_metrics
            )

        return BatchResult(
            rollout_step=rollout_step,
            loss=mean_loss,
            grad_norm=mean_grad_norm,
            lr=last_lr,
            num_timesteps_trained=total_timesteps,
            optimizer_steps=optimizer_steps,
            num_mini_batches=n,
            has_backward=has_backward,
            mini_batch_results=mini_results,
            metrics=aggregated_metrics,
        )

    def train_minibatch(
        self,
        batch: TrainingBatch,
        *,
        micro_batch_size: int,
        rollout_step: int,
    ) -> MiniBatchResult:
        self.optimizer.zero_grad()

        timesteps = self.algorithm.resolve_training_timesteps(
            batch=batch,
            current_step=rollout_step,
        )

        micro_slices = _build_micro_batch_slices(
            total_size=batch.batch_size,
            micro_batch_size=micro_batch_size,
        )
        if not micro_slices:
            raise ValueError("train_minibatch requires a non-empty batch.")

        loss_scale = 1.0 / len(micro_slices)
        micro_results: List[MicroBatchResult] = []
        total_loss = 0.0
        update_num_timesteps = 0
        has_backward = False

        for start, end in micro_slices:
            result = self._run_micro_step(
                micro_batch=batch.slice(start, end),
                timesteps=timesteps,
                loss_scale=loss_scale,
            )
            micro_results.append(result)
            total_loss += result.loss
            update_num_timesteps = max(update_num_timesteps, result.num_timesteps)
            has_backward = has_backward or result.has_backward

        aggregated_metrics = aggregate_numeric_metrics(
            [r.metrics for r in micro_results if r.metrics]
        )

        if has_backward:
            clipped = self.backend.clip_grad_norm(self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.ema_manager.post_optimizer_step(
                self.backend.model, aggregated_metrics
            )
        else:
            clipped = 0.0
            logger.warning(
                "train_minibatch: no valid timesteps to train, skipping optimizer step",
            )

        if clipped is None:
            grad_norm_value = 0.0
        elif isinstance(clipped, torch.Tensor):
            grad_norm_value = float(clipped.item())
        else:
            grad_norm_value = float(clipped)

        return MiniBatchResult(
            loss=total_loss,
            grad_norm=grad_norm_value,
            lr=self._current_lr(),
            num_timesteps_trained=update_num_timesteps,
            has_backward=has_backward,
            micro_batch_results=micro_results,
            metrics=aggregated_metrics,
        )

    def train_microbatch(
        self,
        batch: TrainingBatch,
        *,
        rollout_step: int,
    ) -> MicroBatchResult:
        timesteps = self.algorithm.resolve_training_timesteps(
            batch=batch,
            current_step=rollout_step,
        )
        return self._run_micro_step(
            micro_batch=batch,
            timesteps=timesteps,
            loss_scale=1.0,
        )

    def _run_micro_step(
        self,
        *,
        micro_batch: TrainingBatch,
        timesteps: Any,
        loss_scale: float,
    ) -> MicroBatchResult:
        """Run one forward + scaled backward; no optimizer mutation."""
        (
            micro_loss,
            micro_metrics,
            num_timesteps,
            has_backward,
        ) = self.algorithm.compute_loss_and_backward(
            model=self.backend.model,
            batch=micro_batch,
            timesteps=timesteps,
            loss_scale=loss_scale,
        )
        if isinstance(micro_loss, torch.Tensor):
            loss_value = float(micro_loss.item())
        else:
            loss_value = float(micro_loss)
        return MicroBatchResult(
            loss=loss_value,
            num_timesteps=int(num_timesteps),
            has_backward=bool(has_backward),
            metrics=dict(micro_metrics or {}),
        )

    def _current_lr(self) -> float:
        """Best-effort LR lookup compatible with custom optimizer wrappers."""
        param_groups = getattr(self.optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            try:
                return float(param_groups[0]["lr"])
            except Exception:
                pass
        if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
            try:
                last = self.scheduler.get_last_lr()
                if isinstance(last, list) and last:
                    return float(last[0])
            except Exception:
                pass
        return 0.0
