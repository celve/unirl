"""Sampling mode plugin selection."""

from __future__ import annotations

from .base import SamplingModePlugin


def create_sampling_mode_plugin(args) -> SamplingModePlugin:
    backend = getattr(args, "sampling_backend", "inference")
    if backend == "training":
        from .training_mode import TrainingSamplingMode

        return TrainingSamplingMode(args)
    from .inference_mode import InferenceSamplingMode

    return InferenceSamplingMode(args)


def __getattr__(name: str):
    if name == "InferenceSamplingMode":
        from .inference_mode import InferenceSamplingMode

        return InferenceSamplingMode
    if name == "TrainingSamplingMode":
        from .training_mode import TrainingSamplingMode

        return TrainingSamplingMode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SamplingModePlugin",
    "InferenceSamplingMode",
    "TrainingSamplingMode",
    "create_sampling_mode_plugin",
]
