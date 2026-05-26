"""diffusionrl stage-driven training stack.

Wrap functions mutate the nn.Module tree at build time; handles (Shadow,
EMA) give per-step access to installed state.  TrainTrack groups
per-track objects; StageTrainStack drives the training loop.
"""

from __future__ import annotations

from .ema import EMA
from .shadow import Shadow
from .stack import StageTrainStack, TrackMiniBatchResult
from .train_track import TrainTrack

__all__ = [
    "EMA",
    "Shadow",
    "StageTrainStack",
    "TrackMiniBatchResult",
    "TrainTrack",
]
