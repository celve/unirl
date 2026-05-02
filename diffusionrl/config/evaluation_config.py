"""Typed evaluation config registered under ``cfg.evaluation``."""

from __future__ import annotations

from dataclasses import dataclass

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require


@register_config(group="evaluation", name="default")
@dataclass
class EvaluationConfig:
    """Eval cadence read by the rollout driver."""

    eval_steps: int = 0
    eval_batch_size: int = 1

    def __post_init__(self) -> None:
        require(self.eval_steps >= 0, f"EvaluationConfig.eval_steps must be >= 0; got {self.eval_steps!r}")
        require(
            self.eval_batch_size >= 1, f"EvaluationConfig.eval_batch_size must be >= 1; got {self.eval_batch_size!r}"
        )


__all__ = ["EvaluationConfig"]
