"""Compatibility exports for grouped config classes.

Canonical definitions now live in ``diffusionrl.config.arguments`` so newcomers
can understand argument structure from a single file.
"""

from __future__ import annotations

from .arguments import (
    AlgorithmConfig,
    EMAConfig,
    ModelConfig,
    NFTConfig,
    RayConfig,
    RewardConfig,
    RolloutLoggingConfig,
    SamplingConfig,
    TrainingConfig,
    WindowSchedulerConfig,
)

__all__ = [
    "AlgorithmConfig",
    "NFTConfig",
    "EMAConfig",
    "WindowSchedulerConfig",
    "ModelConfig",
    "TrainingConfig",
    "RolloutLoggingConfig",
    "SamplingConfig",
    "RewardConfig",
    "RayConfig",
]
