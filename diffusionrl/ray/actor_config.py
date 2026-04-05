"""Typed configs for fixed Ray actor runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from diffusionrl.construction import ComponentInitPayload


@dataclass(frozen=True)
class TrainingActorConfig:
    model_init_payload: ComponentInitPayload
    reward_config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    scheduler_config: Dict[str, Any]
    algorithm_init_payload: ComponentInitPayload
    training_config: Dict[str, Any]
    topology_config: Dict[str, Any]
    training_plan_config: Dict[str, Any]
    sampling_config: Dict[str, Any]
    train_backend_init_payload: ComponentInitPayload


@dataclass(frozen=True)
class RolloutActorConfig:
    engine_init_payload: ComponentInitPayload
    reward_config: Dict[str, Any]
    rollout_batch_size: int | None = None


__all__ = ["RolloutActorConfig", "TrainingActorConfig"]
