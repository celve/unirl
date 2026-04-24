"""Typed configs for fixed Ray actor runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from diffusionrl.config.spec import TrainingPlan
from diffusionrl.config.training_sections import (
    LrSchedulerConfig,
    OptimizerConfig,
    TrainingExecutionConfig,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.reward.config import RewardSpec
from diffusionrl.training.backends import (
    FSDPBackendConfig,
    VeOmniBackendConfig,
)
from diffusionrl.training.types import TrainTopology
from diffusionrl.types.sampling import SamplingParams


@dataclass(frozen=True)
class TrainingActorConfig:
    model_init_payload: ComponentInitPayload
    reward_config: RewardSpec
    optimizer_config: OptimizerConfig
    scheduler_config: LrSchedulerConfig
    algorithm_init_payload: ComponentInitPayload
    training_config: TrainingExecutionConfig
    topology_config: TrainTopology
    training_plan_config: TrainingPlan
    # None means the training actor will not perform sampling itself
    # (train-from-buffer mode) and will not run the replay-logprob patch.
    sampling_config: Optional[SamplingParams]
    train_backend_init_payload: ComponentInitPayload
    seed: int = 42


@dataclass(frozen=True)
class AdvantageParams:
    """Advantage normalization params, sourced from algorithm config."""

    epsilon: float = 1e-8
    clip_max: Optional[float] = 5.0
    trim_outliers_ratio: float = 0.0


@dataclass(frozen=True)
class RolloutActorConfig:
    engine_init_payload: ComponentInitPayload
    reward_config: RewardSpec
    algorithm_init_payload: ComponentInitPayload
    rollout_batch_size: int | None = None
    advantage_params: AdvantageParams = AdvantageParams()
    seed: int = 42


def build_train_actor_init_kwargs(
    *,
    training_launch: Any,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    sampling_config: Any = None,
) -> Dict[str, Any]:
    """Map a resolved ``TrainingLaunch`` into eager-init ``TrainActor`` kwargs.

    Used by the new-actor path (``TrainActorGroup.bootstrap``). The
    ``training_launch`` argument is duck-typed to avoid a hard import cycle
    with ``diffusionrl.config.assembly``; in practice it is always a
    ``LaunchConfig.training`` (``TrainingLaunch``) instance.

    When *sampling_config* (a ``SamplingParams``) is provided, the
    ``TrainActor`` will initialise an ``FSDPSamplingEngine`` for direct
    sampling mode.
    """
    actor_cfg = training_launch.actor_init_config
    backend_cfg = actor_cfg.train_backend_init_payload.component_config
    if not isinstance(backend_cfg, (FSDPBackendConfig, VeOmniBackendConfig)):
        raise NotImplementedError(
            "The new-actor training path supports FSDPBackendConfig and "
            f"VeOmniBackendConfig. Got backend config type: {type(backend_cfg).__name__}."
        )
    training_plan = actor_cfg.training_plan_config
    training_exec = actor_cfg.training_config
    kwargs = dict(
        world_size=int(world_size),
        rank=int(rank),
        master_addr=str(master_addr),
        master_port=int(master_port),
        mini_batch_size=int(training_plan.local_mini_batch_size),
        micro_batch_size=int(training_plan.micro_batch_size),
        max_grad_norm=float(training_exec.max_grad_norm),
        backend_config=backend_cfg,
        optimizer_config=actor_cfg.optimizer_config,
        scheduler_config=actor_cfg.scheduler_config,
        reward_spec=actor_cfg.reward_config,
        algorithm_init_payload=actor_cfg.algorithm_init_payload,
        model_init_payload=actor_cfg.model_init_payload,
        training_autocast_precision=str(training_exec.training_autocast_precision),
        debug_output_dir=training_exec.debug_output_dir,
        seed=int(getattr(actor_cfg, "seed", 42)),
    )
    if sampling_config is not None:
        kwargs["sampling_config"] = sampling_config
    return kwargs


__all__ = [
    "AdvantageParams",
    "RolloutActorConfig",
    "TrainingActorConfig",
    "build_train_actor_init_kwargs",
]
