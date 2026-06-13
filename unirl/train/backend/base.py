"""Schema dataclasses for the training stack.

Config-layer schemas (torch-free — shared by every train backend):

* :class:`OptimizerConfig` — AdamW hyperparameters
* :class:`LrSchedulerConfig` — LR schedule hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """AdamW-style optimizer hyperparameters consumed by the training actor."""

    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float


@dataclass
class LrSchedulerConfig:
    """Learning-rate scheduler hyperparameters."""

    type: str
    warmup_steps: int
    total_steps: int


__all__ = [
    "LrSchedulerConfig",
    "OptimizerConfig",
]
