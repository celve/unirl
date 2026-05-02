"""Authoritative training batch/update plan.

Colocated with the training package — consumed by ``train_executor``,
``update_schedule``, and ``TrainStack`` (which reads ``cfg.training.plan``
directly). Authored in cfg via the ``training/plan`` preset; slice tuples
used downstream are computed from these scalars by ``update_schedule``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require


@register_config(group="training/plan", name="default")
@dataclass
class TrainingPlan:
    """Batch geometry scalars that drive training updates."""

    global_batch_size: int = 1
    local_batch_size: int = 1
    local_mini_batch_size: int = 1
    micro_batch_size: int = 1
    num_updates_per_batch: int = 1

    def __post_init__(self) -> None:
        require(
            self.global_batch_size >= 1, f"TrainingPlan.global_batch_size must be >= 1; got {self.global_batch_size!r}"
        )
        require(
            self.local_batch_size >= 1, f"TrainingPlan.local_batch_size must be >= 1; got {self.local_batch_size!r}"
        )
        require(
            self.local_mini_batch_size >= 1,
            f"TrainingPlan.local_mini_batch_size must be >= 1; got {self.local_mini_batch_size!r}",
        )
        require(
            self.micro_batch_size >= 1, f"TrainingPlan.micro_batch_size must be >= 1; got {self.micro_batch_size!r}"
        )
        require(
            self.num_updates_per_batch >= 1,
            f"TrainingPlan.num_updates_per_batch must be >= 1; got {self.num_updates_per_batch!r}",
        )
        expected = self.local_mini_batch_size * self.num_updates_per_batch
        require(
            self.local_batch_size == expected,
            f"TrainingPlan.local_batch_size ({self.local_batch_size}) must equal local_mini_batch_size * num_updates_per_batch ({expected})",
        )
        require(
            self.local_mini_batch_size % self.micro_batch_size == 0,
            f"TrainingPlan.micro_batch_size ({self.micro_batch_size}) must evenly divide local_mini_batch_size ({self.local_mini_batch_size})",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_batch_size": self.global_batch_size,
            "local_batch_size": self.local_batch_size,
            "local_mini_batch_size": self.local_mini_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "num_updates_per_batch": self.num_updates_per_batch,
        }


__all__ = ["TrainingPlan"]
