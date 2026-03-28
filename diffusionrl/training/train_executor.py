"""Training executor that owns core optimization flow."""

from __future__ import annotations

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
from diffusionrl.training.batch_partition import shard_training_batch_for_rank
from diffusionrl.training.update_schedule import (
    TrainingUpdateSchedule,
    create_training_update_schedule,
)
from diffusionrl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainExecutorConfig:
    """Runtime knobs for training execution."""

    rank: int
    dp_size: int
    device: torch.device
    use_fsdp: bool
    algorithm_type: str
    guidance_scale: float
    max_grad_norm: float
    local_micro_batch_size: int
    local_update_batch_size: int
    num_updates_per_local_batch: int
    training_plan: Dict[str, Any]
    ema_manager: Optional[Any]
    shuffle_samples: bool = True
    shuffle_seed: Optional[int] = None
    clip_grad_norm_fn: Optional[Callable[..., Any]] = None

class TrainExecutor:
    """Execution-layer trainer used by RPC actors."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        algorithm: Any,
        config: TrainExecutorConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.algorithm = algorithm
        self.config = config
        self.update_schedule: TrainingUpdateSchedule = create_training_update_schedule(
            config.training_plan
        )

    def prepare_batch(self, batch: TrainingBatch) -> Optional[TrainingBatch]:
        """Validate, shard, and move training batch to compute device."""
        if not isinstance(batch, (BackwardTrainingBatch, ForwardTrainingBatch)):
            raise TypeError(
                f"Unsupported batch type: {type(batch).__name__}. "
                "Expected BackwardTrainingBatch or ForwardTrainingBatch. "
                "Legacy dict format is not supported - use typed batches."
            )

        sharded = shard_training_batch_for_rank(
            batch,
            dp_size=int(self.config.dp_size),
            rank=int(self.config.rank),
            per_rank_batch_size=int(self.config.training_plan.get("local_batch_size", 0) or 0),
        )
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

    def _update_ema_trackers(self, metrics: dict) -> None:
        """Step all active EMA trackers after an optimizer step."""
        if self.config.ema_manager is not None:
            self.config.ema_manager.post_optimizer_step(self.model, metrics)

    def skipped_metrics(self, rollout_id: int) -> Dict[str, Any]:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "lr": self._current_lr(),
            "rollout_id": rollout_id,
            "skipped": True,
        }

    def _shuffle_batch(self, batch: TrainingBatch, rollout_id: int) -> TrainingBatch:
        """Shuffle samples in the training batch before optimization.

        Analogous to Flow-Factory's per-inner-epoch shuffle: applies a
        deterministic permutation (seeded by shuffle_seed + rollout_id) to
        all sample-level tensors so that mini-batch composition varies
        across rollouts.  All ranks derive the identical permutation from
        the same seed, preserving distributed consistency.

        Args:
            batch: Typed training batch (Backward or Forward).
            rollout_id: Current rollout iteration, mixed into the RNG seed
                        for reproducible yet varying permutations.

        Returns:
            A new TrainingBatch with shuffled sample order.
        """
        batch_size = batch.batch_size
        if batch_size <= 1:
            return batch

        # Build a deterministic generator from (shuffle_seed, rollout_id)
        # so that every rank produces the same permutation.
        g = torch.Generator()
        seed = self.config.shuffle_seed if self.config.shuffle_seed is not None else 42
        g.manual_seed(seed + rollout_id)
        perm = torch.randperm(batch_size, generator=g)

        logger.info(
            "Shuffling %d samples before training (rollout_id=%d, seed=%d)",
            batch_size, rollout_id, seed + rollout_id,
        )
        return batch.shuffle(perm)

    def execute_prepared_batch(self, *, rollout_id: int, batch: TrainingBatch) -> Dict[str, Any]:
        """Run optimization on a pre-validated, on-device typed batch."""
        # Shuffle samples before training to break ordering bias,
        # analogous to Flow-Factory's inner-epoch shuffle.
        if self.config.shuffle_samples:
            batch = self._shuffle_batch(batch, rollout_id)

        inner_metrics: List[Dict[str, Any]] = []
        total_timesteps = 0
        last_mini_batches_per_update = 1
        total_mini_batches_consumed = 0
        optimizer_steps = 0
        last_effective_update_batch_size = int(self.config.local_update_batch_size)
        last_effective_local_micro_batch_size = int(
            self.config.local_micro_batch_size
        )
        training_schedule = self.update_schedule.name

        rank = self.config.rank
        for update_chunk in tqdm(
            self.update_schedule.iter_update_chunks(
                batch=batch,
            ),
            desc=f"Training {rollout_id}:",
            unit="update",
            disable=(rank != 0),
        ):
            self.optimizer.zero_grad()

            mini_batches = tuple(
                (int(start), int(end)) for start, end in update_chunk.mini_batch_slices
            )
            if not mini_batches:
                raise ValueError(
                    "TrainExecutor requires non-empty mini_batch_slices per update."
                )

            loss_scale = 1.0 / len(mini_batches)
            training_timesteps = self.algorithm.resolve_training_timesteps(
                batch=update_chunk.batch,
                current_step=rollout_id,
            )
            total_loss = 0.0
            update_num_timesteps = 0
            total_mini_batches_this_update = len(mini_batches)
            has_backward = False
            micro_batch_metrics: List[Dict[str, Any]] = []

            # Mixed precision for model forwards is handled inside forward plugins
            # (see forward_plugin.autocast_dtype / plugin.forward), not here.
            for start, end in mini_batches:
                micro_batch = update_chunk.batch.slice(start, end)
                (
                    micro_loss,
                    micro_metrics,
                    num_timesteps,
                    micro_has_backward,
                ) = self.algorithm.compute_loss_and_backward(
                    model=self.model,
                    batch=micro_batch,
                    guidance_scale=self.config.guidance_scale,
                    timesteps=training_timesteps,
                    loss_scale=loss_scale,
                )
                total_loss += (
                    micro_loss.item()
                    if isinstance(micro_loss, torch.Tensor)
                    else micro_loss
                )
                if num_timesteps > 0:
                    update_num_timesteps = max(update_num_timesteps, num_timesteps)
                has_backward = has_backward or micro_has_backward
                if micro_metrics:
                    micro_batch_metrics.append(micro_metrics)

            all_metrics = aggregate_numeric_metrics(micro_batch_metrics)

            if has_backward:
                grad_norm = self._clip_grad_norm()
                self._apply_optimizer_step()
                optimizer_steps += 1
                self._update_ema_trackers(all_metrics)
            else:
                grad_norm = 0.0
                logger.warning(
                    "No valid timesteps to train in %s update, skipping optimizer step",
                    training_schedule,
                )

            last_mini_batches_per_update = total_mini_batches_this_update
            last_effective_update_batch_size = int(update_chunk.update_batch_size)
            total_mini_batches_consumed += total_mini_batches_this_update
            total_timesteps += update_num_timesteps

            step_metrics = {
                "loss": total_loss,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": self._current_lr(),
                "num_timesteps_trained": update_num_timesteps,
                "mini_batches_per_update": total_mini_batches_this_update,
                "has_backward": has_backward,
                **all_metrics,
            }
            inner_metrics.append(step_metrics)

        metrics = aggregate_numeric_metrics(inner_metrics)

        # Filter out per-timestep metrics (t{N}_...) from aggregated output;
        # these are exposed only via _per_optimizer_step_metrics for train/ namespace.
        metrics = {k: v for k, v in metrics.items() if not re.match(r"^t\d+_", k)}
        if self.config.ema_manager is not None and optimizer_steps > 0:
            self.config.ema_manager.post_rollout_end(self.model, metrics)
        metrics.update(
            {
                "rollout_id": rollout_id,
                "algorithm_type": self.config.algorithm_type,
                "training_schedule": training_schedule,
                "configured_local_micro_batch_size": self.config.local_micro_batch_size,
                "configured_local_update_batch_size": self.config.local_update_batch_size,
                "configured_num_updates_per_local_batch": self.config.num_updates_per_local_batch,
                "effective_local_micro_batch_size": last_effective_local_micro_batch_size,
                "effective_local_update_batch_size": last_effective_update_batch_size,
                "num_timesteps_trained": total_timesteps,
                "mini_batches_per_update": last_mini_batches_per_update,
                "mini_batches_consumed": total_mini_batches_consumed,
                "optimizer_steps": optimizer_steps,
            }
        )
        metrics["_per_optimizer_step_metrics"] = inner_metrics
        return metrics
