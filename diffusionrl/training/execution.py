"""Per-step training-execution knobs.

Consumed by ``train.py`` (offload gating) and
:class:`diffusionrl.training.stack.StageTrainStack` (max_grad_norm).
Registered under ``training/execution``.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffusionrl.config.registration import register_config
from diffusionrl.config.validation import validate_precision_type


@register_config(group="training/execution", name="default")
@dataclass
class TrainingExecutionConfig:
    """Per-step training-execution knobs.

    Read sites:
      - ``training/stack.py::StageTrainStack`` reads ``max_grad_norm``
      - ``train.py`` reads ``offload_train`` / ``offload_rollout`` to
        gate per-rollout sleep/wake of the training + rollout groups.
    """

    max_grad_norm: float
    training_autocast_precision: str = "bf16"
    offload_train: bool = False
    offload_rollout: bool = False

    def __post_init__(self) -> None:
        if self.max_grad_norm <= 0:
            raise ValueError(f"TrainingExecutionConfig.max_grad_norm must be > 0; got {self.max_grad_norm!r}")
        validate_precision_type(
            self.training_autocast_precision, field="TrainingExecutionConfig.training_autocast_precision"
        )


__all__ = ["TrainingExecutionConfig"]
