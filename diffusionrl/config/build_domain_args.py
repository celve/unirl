"""Projection helpers from resolved config slices to actor/service payload dicts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
from diffusionrl.cmdline.models import build_model_bundle_init_payload_from_args
from diffusionrl.cmdline.rollout_engine import (
    build_rollout_engine_init_payload_from_args,
)
from diffusionrl.cmdline.train_backend import build_train_backend_init_payload_from_args
from diffusionrl.config.resolution import (
    TrainingPlan,
    TrainTopology,
    derive_sampling_host_engine_type,
    normalize_rollout_engine,
    resolve_config,
)
from diffusionrl.config.spec import ModelSpec
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import RolloutActorConfig, TrainingActorConfig
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.types.sampling import SamplingSpec

# ---------------------------------------------------------------------------
# Reward domain
# ---------------------------------------------------------------------------


def build_reward_config(
    *,
    reward_schema: RewardSchema,
) -> Dict[str, Any]:
    """Build reward config consumed by actors and services."""
    return asdict(reward_schema)


# ---------------------------------------------------------------------------
# Sampling / rollout domain
# ---------------------------------------------------------------------------

def _build_shared_sampling_payload(sampling_spec: SamplingSpec) -> Dict[str, Any]:
    return {
        "sampler_dotpath": sampling_spec.sampler_dotpath,
        "num_inference_steps": int(sampling_spec.num_inference_steps),
        "sde_config": sampling_spec.sde_config.to_dict(),
        "guidance_scale": float(sampling_spec.guidance_scale),
        "height": int(sampling_spec.height),
        "width": int(sampling_spec.width),
        "num_frames": int(sampling_spec.num_frames),
    }


def build_training_sampling_config(
    *,
    precision_settings,
    sampling_spec: SamplingSpec,
    sampler_engine_type: str,
) -> Dict[str, Any]:
    """Build training-actor sampling config from the canonical resolved sampling spec."""
    payload = _build_shared_sampling_payload(sampling_spec)
    payload.update({
        "sampler_engine_type": normalize_rollout_engine(sampler_engine_type)
        or str(sampler_engine_type).strip().lower(),
        "replay_sampler_dotpath": sampling_spec.replay_sampler_dotpath,
        "seed": int(sampling_spec.seed),
        "sampling_adapter": sampling_spec.sampling_adapter,
        "init_same_noise": bool(sampling_spec.init_same_noise),
        "sampler_kwargs": dict(sampling_spec.sampler_kwargs),
        # Rollout precision lives at sampling_config top-level, not inside
        # sampler_kwargs, because it is a framework contract — not a sampler
        # constructor parameter.
        "autocast_precision": precision_settings.rollout_autocast_precision,
        "trajectory_precision": precision_settings.trajectory_precision,
        "logprob_precision": precision_settings.logprob_precision,
    })
    return payload

def build_rollout_actor_init_config_from_args(
    args: Any,
    *,
    config_bundle: Any = None,
    model_init_payload: ComponentInitPayload | None = None,
    reward_config: Dict[str, Any] | None = None,
    engine_init_payload: ComponentInitPayload | None = None,
) -> RolloutActorConfig:
    """Build RolloutActorConfig from framework args with optional prebuilt slices."""

    bundle = config_bundle
    needs_bundle = any(
        value is None
        for value in (
            model_init_payload,
            reward_config,
            engine_init_payload,
        )
    )
    if bundle is None and needs_bundle:
        bundle = resolve_config(args, include_training_plan=False)

    actor_model_init_payload = (
        model_init_payload
        if model_init_payload is not None
        else build_model_bundle_init_payload_from_args(
            args, model_spec=bundle.model_spec
        )
    )
    actor_reward_config = (
        dict(reward_config)
        if reward_config is not None
        else build_reward_config(reward_schema=RewardSchema.from_args(args))
    )
    actor_engine_init_payload = (
        engine_init_payload
        if engine_init_payload is not None
        else build_rollout_engine_init_payload_from_args(
            args,
            model_init_payload=actor_model_init_payload,
            sampling_spec=bundle.sampling_spec,
            rollout_mode_info=bundle.rollout_mode_info,
        )
    )

    return RolloutActorConfig(
        engine_init_payload=actor_engine_init_payload,
        reward_config=dict(actor_reward_config),
        rollout_batch_size=(
            int(args.rollout.rollout_batch_size)
            if getattr(args.rollout, "rollout_batch_size", None) is not None
            else None
        ),
    )

# ---------------------------------------------------------------------------
# Training domain
# ---------------------------------------------------------------------------

def _build_optimizer_config(training_settings) -> Dict[str, Any]:
    return {
        "learning_rate": training_settings.learning_rate,
        "adam_beta1": training_settings.adam_beta1,
        "adam_beta2": training_settings.adam_beta2,
        "adam_epsilon": training_settings.adam_epsilon,
        "weight_decay": training_settings.weight_decay,
    }


def _build_scheduler_config(training_settings, *, total_steps: int) -> Dict[str, Any]:
    return {
        "type": training_settings.lr_scheduler_type,
        "warmup_steps": training_settings.warmup_steps,
        "total_steps": total_steps,
    }


def _build_training_execution_config(
    *,
    training_settings,
    algorithm_settings,
    precision_settings,
    debug_settings,
    guidance_scale: float,
    algorithm_type: str,
    replay_enabled: bool,
) -> Dict[str, Any]:
    return {
        "max_grad_norm": training_settings.max_grad_norm,
        "replay_enabled": bool(replay_enabled),
        "shuffle_samples": bool(algorithm_settings.shuffle_samples),
        "shuffle_seed": algorithm_settings.shuffle_seed,
        "training_autocast_precision": precision_settings.training_autocast_precision,
        "debug_output_dir": debug_settings.debug_output_dir,
        "guidance_scale": float(guidance_scale),
        "algorithm_type": str(algorithm_type),
    }


def _build_training_topology_config(topology: TrainTopology) -> Dict[str, Any]:
    return topology.as_dict()


def _build_training_plan_config(training_plan: TrainingPlan) -> Dict[str, Any]:
    return training_plan.as_dict()


def build_training_actor_init_config_from_args(
    args: Any,
    *,
    config_bundle: Any = None,
    replay_enabled: bool | None = None,
    topology: TrainTopology | None = None,
    training_plan: TrainingPlan | None = None,
    algorithm_init_payload=None,
    model_init_payload: ComponentInitPayload | None = None,
    reward_config: Dict[str, Any] | None = None,
    sampling_config: Dict[str, Any] | None = None,
    train_backend_init_payload: ComponentInitPayload | None = None,
) -> TrainingActorConfig:
    """Build TrainingActorConfig from framework args with optional prebuilt slices."""
    from diffusionrl.construction import ComponentInitPayload

    bundle = config_bundle
    needs_bundle = any(
        value is None
        for value in (
            replay_enabled,
            topology,
            training_plan,
            algorithm_init_payload,
            model_init_payload,
            reward_config,
            sampling_config,
            train_backend_init_payload,
        )
    )
    if bundle is None and needs_bundle:
        bundle = resolve_config(args, include_training_plan=True)

    actor_topology = topology if topology is not None else bundle.training_topology
    actor_training_plan = (
        training_plan if training_plan is not None else bundle.training_plan
    )
    if actor_training_plan is None:
        raise ValueError(
            "build_training_actor_init_config_from_args requires training_plan, "
            "either explicitly or via config_bundle/resolve_config(args)."
        )

    actor_algorithm_init_payload = (
        algorithm_init_payload
        if algorithm_init_payload is not None
        else build_algorithm_init_payload_from_args(
            args,
            sampling_spec=bundle.sampling_spec,
        )
    )
    actor_model_init_payload = (
        model_init_payload
        if model_init_payload is not None
        else build_model_bundle_init_payload_from_args(
            args, model_spec=bundle.model_spec
        )
    )
    actor_reward_config = (
        dict(reward_config)
        if reward_config is not None
        else build_reward_config(reward_schema=RewardSchema.from_args(args))
    )
    actor_sampling_config = (
        dict(sampling_config)
        if sampling_config is not None
        else build_training_sampling_config(
            precision_settings=args.precision,
            sampling_spec=bundle.sampling_spec,
            sampler_engine_type=derive_sampling_host_engine_type(
                args,
                rollout_mode_info=bundle.rollout_mode_info,
            ),
        )
    )
    actor_train_backend_init_payload = (
        train_backend_init_payload
        if train_backend_init_payload is not None
        else build_train_backend_init_payload_from_args(args)
    )
    actor_replay_enabled = (
        bool(replay_enabled)
        if replay_enabled is not None
        else bool(bundle.rollout_mode_info.replay_enabled)
    )
    optimizer_config = _build_optimizer_config(args.training)
    scheduler_config = _build_scheduler_config(
        args.training,
        total_steps=args.rollout.num_rollout,
    )
    training_config = _build_training_execution_config(
        training_settings=args.training,
        algorithm_settings=args.algorithm,
        precision_settings=args.precision,
        debug_settings=args.debug,
        guidance_scale=float(actor_sampling_config["guidance_scale"]),
        algorithm_type=str(getattr(args.algorithm, "algorithm_type", "") or ""),
        replay_enabled=actor_replay_enabled,
    )
    topology_config = _build_training_topology_config(actor_topology)
    training_plan_config = _build_training_plan_config(actor_training_plan)
    if not isinstance(actor_algorithm_init_payload, ComponentInitPayload):
        raise ValueError(
            "build_training_actor_init_config_from_args requires algorithm_init_payload "
            "to be a ComponentInitPayload."
        )
    if not isinstance(actor_model_init_payload, ComponentInitPayload):
        raise ValueError(
            "build_training_actor_init_config_from_args requires model_init_payload "
            "to be a ComponentInitPayload."
        )

    return TrainingActorConfig(
        algorithm_init_payload=actor_algorithm_init_payload,
        model_init_payload=actor_model_init_payload,
        reward_config=dict(actor_reward_config),
        optimizer_config=dict(optimizer_config),
        scheduler_config=dict(scheduler_config),
        training_config=dict(training_config),
        topology_config=dict(topology_config),
        training_plan_config=dict(training_plan_config),
        sampling_config=dict(actor_sampling_config),
        train_backend_init_payload=actor_train_backend_init_payload,
    )

__all__ = [
    "build_reward_config",
    "build_training_sampling_config",
    "build_rollout_actor_init_config_from_args",
    "build_training_actor_init_config_from_args",
]
