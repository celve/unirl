"""Typed contracts for rollout-buffer-training handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from diffusionrl.types.training_batch import TrainingBatch


@dataclass(frozen=True)
class RolloutPayload:
    """One rollout-produced training batch handed to the buffer."""

    rollout_id: int
    training_batch: TrainingBatch
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BufferedTrainingPayload:
    """One buffer-produced training payload handed to training."""

    rollout_id: int
    training_data: Any
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
