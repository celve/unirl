"""Cmdline adaptation helpers for framework runtime actor configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
from diffusionrl.cmdline.models import build_model_bundle_init_payload_from_args
from diffusionrl.cmdline.rollout_engine import (
    build_rollout_engine_init_payload_from_args,
)
from diffusionrl.config.spec import SamplingSpec, TrainingPlan
from diffusionrl.config.training_sections import (
    LrSchedulerConfig as TrainingLrSchedulerConfig,
)
from diffusionrl.config.training_sections import (
    OptimizerConfig as TrainingOptimizerConfig,
)
from diffusionrl.config.training_sections import (
    TrainingExecutionConfig,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import (
    RolloutActorConfig,
    TrainingActorConfig,
)
from diffusionrl.reward.config import RewardSpec

if TYPE_CHECKING:
    from diffusionrl.config.assembly import DerivedConfig


def build_training_sampling_config_from_args(
    *,
    args: Any,
    sampling_spec: SamplingSpec,
    sampler_engine_type: str,
) -> Dict[str, Any]:
    """Build training-actor sampling config from the canonical derived sampling spec."""
    return sampling_spec.as_training_sampling_payload(
        sampler_engine_type=sampler_engine_type,
        precision_settings=args.precision,
    )


def build_rollout_actor_init_config_from_args(
    args: Any,
    *,
    derived_config: DerivedConfig,
    model_init_payload: ComponentInitPayload | None = None,
    reward_config: RewardSpec | None = None,
    engine_init_payload: ComponentInitPayload | None = None,
    algorithm_init_payload: ComponentInitPayload | None = None,
) -> RolloutActorConfig:
    """Build RolloutActorConfig from args and derived config."""
    if engine_init_payload is None:
        if model_init_payload is None:
            model_init_payload = build_model_bundle_init_payload_from_args(
                args,
                model_spec=derived_config.model_spec,
            )
        engine_init_payload = build_rollout_engine_init_payload_from_args(
            args,
            model_init_payload=model_init_payload,
            sampling_spec=derived_config.sampling_spec.to_params(args.precision),
            rollout_info=derived_config.rollout_info,
            sampler_dotpath=derived_config.model_spec.sampler_dotpath,
        )
    if algorithm_init_payload is None:
        algorithm_init_payload = build_algorithm_init_payload_from_args(
            args,
            sampling_spec=derived_config.sampling_spec.to_params(args.precision),
        )
    return RolloutActorConfig(
        engine_init_payload=engine_init_payload,
        reward_config=(reward_config if reward_config is not None else RewardSpec.from_args(args)),
        algorithm_init_payload=algorithm_init_payload,
        forward_batch_size=(
            int(args.sampling.forward_batch_size) if args.sampling.forward_batch_size is not None else None
        ),
    )


def _build_optimizer_config_from_args(*, args: Any) -> TrainingOptimizerConfig:
    return TrainingOptimizerConfig(
        learning_rate=float(args.training.learning_rate),
        adam_beta1=float(args.training.adam_beta1),
        adam_beta2=float(args.training.adam_beta2),
        adam_epsilon=float(args.training.adam_epsilon),
        weight_decay=float(args.training.weight_decay),
    )


def _build_scheduler_config_from_args(*, args: Any) -> TrainingLrSchedulerConfig:
    return TrainingLrSchedulerConfig(
        type=str(args.training.lr_scheduler_type),
        warmup_steps=int(args.training.warmup_steps),
        total_steps=int(args.rollout.num_rollout),
    )


def _build_training_execution_config_from_args(
    *,
    args: Any,
    guidance_scale: float,
    algorithm_type: str,
    replay_enabled: bool,
) -> TrainingExecutionConfig:
    return TrainingExecutionConfig(
        max_grad_norm=float(args.training.max_grad_norm),
        replay_enabled=bool(replay_enabled),
        training_autocast_precision=str(args.precision.training_autocast_precision),
        debug_output_dir=args.debug.output_dir,
        guidance_scale=float(guidance_scale),
        algorithm_type=str(algorithm_type),
    )


def build_training_actor_init_config_from_args(
    args: Any,
    *,
    derived_config: DerivedConfig,
    train_backend_init_payload: ComponentInitPayload,
    algorithm_init_payload: ComponentInitPayload | None = None,
    model_init_payload: ComponentInitPayload | None = None,
    reward_config: RewardSpec | None = None,
    sampling_config: Dict[str, Any] | None = None,
    training_plan: TrainingPlan | None = None,
) -> TrainingActorConfig:
    """Build TrainingActorConfig from args and derived config."""
    sampling_spec = derived_config.sampling_spec
    rollout_info = derived_config.rollout_info

    if algorithm_init_payload is None:
        algorithm_init_payload = build_algorithm_init_payload_from_args(
            args,
            sampling_spec=sampling_spec.to_params(args.precision),
        )
    if model_init_payload is None:
        model_init_payload = build_model_bundle_init_payload_from_args(
            args,
            model_spec=derived_config.model_spec,
        )
    if reward_config is None:
        reward_config = RewardSpec.from_args(args)
    if sampling_config is None:
        sampling_config = build_training_sampling_config_from_args(
            args=args,
            sampling_spec=sampling_spec,
            sampler_engine_type=rollout_info.sampling_engine,
        )
    if training_plan is None:
        training_plan = derived_config.require_training_plan()

    sampling_params = sampling_spec.to_params(args.precision)

    return TrainingActorConfig(
        algorithm_init_payload=algorithm_init_payload,
        model_init_payload=model_init_payload,
        reward_config=reward_config,
        optimizer_config=_build_optimizer_config_from_args(args=args),
        scheduler_config=_build_scheduler_config_from_args(args=args),
        training_config=_build_training_execution_config_from_args(
            args=args,
            guidance_scale=float(sampling_config["guidance_scale"]),
            algorithm_type=str(getattr(args.algorithm, "algorithm_type", "") or ""),
            replay_enabled=bool(rollout_info.replay_enabled),
        ),
        topology_config=derived_config.training_topology,
        training_plan_config=training_plan,
        sampling_config=sampling_params,
        train_backend_init_payload=train_backend_init_payload,
    )


__all__ = [
    "build_training_sampling_config_from_args",
    "build_rollout_actor_init_config_from_args",
    "build_training_actor_init_config_from_args",
]
