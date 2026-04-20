"""Typed sub-config dataclasses for ``TrainingActorConfig``.

These dataclasses replace the ad-hoc ``Dict[str, Any]`` payloads previously
used for the optimizer / LR-scheduler / training-execution sections of the
training-actor init config. Each dataclass mirrors the keys that
``diffusionrl.config.build_domain_args`` builds and that the training actor
consumes in ``__init__``.

Backend plugin compatibility: ``BaseTrainBackend.build_optimizer`` and
``BaseTrainBackend.build_scheduler`` keep their ``Mapping[str, Any]``
signature. Call sites convert via ``dataclasses.asdict(typed_config)`` so
third-party backends do not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW-style optimizer hyperparameters consumed by the training actor."""

    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float


@dataclass(frozen=True)
class LrSchedulerConfig:
    """Learning-rate scheduler hyperparameters.

    The name distinguishes this LR scheduler from
    ``diffusionrl.config.arguments.SchedulerConfig`` which is the
    timestep-index scheduler used by ``diffusionrl.algorithms.grpo`` etc.
    """

    type: str
    warmup_steps: int
    total_steps: int


@dataclass(frozen=True)
class TrainingExecutionConfig:
    """Per-step training-execution knobs read by the training actor.

    The optional fields' defaults match the ``.get(..., default)`` defaults
    that the previous dict-based consumer used, so behavior is preserved.
    """

    max_grad_norm: float
    replay_enabled: bool
    algorithm_type: str
    guidance_scale: float
    shuffle_samples: bool = True
    shuffle_seed: Optional[int] = None
    training_autocast_precision: str = "bf16"
    debug_output_dir: Optional[str] = None


__all__ = [
    "LrSchedulerConfig",
    "OptimizerConfig",
    "TrainingExecutionConfig",
]
