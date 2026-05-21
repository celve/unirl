"""diffusionrl stage-driven training stack.

Public surface for the ``models`` training contract. The trainable-module
facade is the :class:`Policy` Protocol from :mod:`diffusionrl.training.policy`
(FSDP / LoRA / EMA composed via :func:`compose_policy`).
"""

from __future__ import annotations

from .stack import StageTrainStack, TrainOptimizerStepResult

__all__ = [
    "StageTrainStack",
    "TrainOptimizerStepResult",
]
