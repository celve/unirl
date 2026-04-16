"""Typed configs for fixed Ray actor runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from diffusionrl.config.spec import TrainingPlan
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.reward.config import RewardSpec
from diffusionrl.training.backends.base import TrainTopology


@dataclass(frozen=True)
class TrainingOptimizerConfig:
    type: str
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingLrSchedulerConfig:
    type: str = "constant"
    warmup_steps: int = 0
    total_steps: int = 1


@dataclass(frozen=True)
class TrainingExecutionConfig:
    max_grad_norm: float
    replay_enabled: bool
    shuffle_samples: bool = True
    shuffle_seed: int | None = None
    training_autocast_precision: str = "bf16"
    debug_output_dir: str | None = None
    guidance_scale: float = 1.0
    algorithm_type: str = ""


@dataclass(frozen=True)
class TrainingActorConfig:
    model_init_payload: ComponentInitPayload
    algorithm_init_payload: ComponentInitPayload
    train_backend_init_payload: ComponentInitPayload

    sampling_config: Dict[str, Any]
    reward_config: RewardSpec

    optimizer_config: TrainingOptimizerConfig
    scheduler_config: TrainingLrSchedulerConfig
    training_config: TrainingExecutionConfig
    training_plan_config: TrainingPlan

    topology_config: TrainTopology


@dataclass(frozen=True)
class RolloutActorConfig:
    engine_init_payload: ComponentInitPayload
    reward_config: RewardSpec
    rollout_batch_size: int | None = None


__all__ = [
    "TrainingLrSchedulerConfig",
    "TrainingOptimizerConfig",
    "RolloutActorConfig",
    "TrainingActorConfig",
    "TrainingExecutionConfig",
]
