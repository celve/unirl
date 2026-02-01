"""Sampling mode adapter selection."""

from __future__ import annotations

from .base import SamplingModeAdapter
from .inference_mode import InferenceSamplingMode
from .training_mode import TrainingSamplingMode


def create_sampling_mode_adapter(args) -> SamplingModeAdapter:
    backend = getattr(args, "sampling_backend", "inference")
    if backend == "training":
        return TrainingSamplingMode(args)
    return InferenceSamplingMode(args)


__all__ = [
    "SamplingModeAdapter",
    "InferenceSamplingMode",
    "TrainingSamplingMode",
    "create_sampling_mode_adapter",
]
