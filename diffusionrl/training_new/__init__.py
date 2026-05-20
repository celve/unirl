"""diffusionrl stage-driven training stack.

Public surface for the ``models_new`` training contract. Coexists with the
legacy :mod:`diffusionrl.training` package; the trainable-module facade is
the :class:`Policy` Protocol from :mod:`diffusionrl.training_new.policy` (FSDP /
LoRA / EMA composed via :func:`compose_policy`).
"""

from __future__ import annotations

from .stack import StageMiniBatchResult, StageTrainStack

__all__ = [
    "StageMiniBatchResult",
    "StageTrainStack",
]
