"""Sampling mode plugin selection."""

from __future__ import annotations

from .base import SamplingModePlugin
from .inference_mode import InferenceSamplingMode
from .training_mode import TrainingSamplingMode


def create_sampling_mode_plugin(args) -> SamplingModePlugin:
    backend = getattr(args, "sampling_backend", "inference")
    if backend == "training":
        return TrainingSamplingMode(args)
    return InferenceSamplingMode(args)


__all__ = [
    "SamplingModePlugin",
    "InferenceSamplingMode",
    "TrainingSamplingMode",
    "create_sampling_mode_plugin",
]
