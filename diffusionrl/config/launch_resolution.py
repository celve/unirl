"""Launch-side config assembly built once on the driver.

This module consumes ``ConfigBundle`` from ``resolution.py`` and builds the
driver-facing launch payloads used by placement, group creation, and rollout /
training actor initialization. It does not own config semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
from diffusionrl.cmdline.models import build_model_bundle_init_payload_from_args
from diffusionrl.cmdline.rollout_engine import (
    build_rollout_engine_init_payload_from_args,
)
from diffusionrl.cmdline.train_backend import build_train_backend_init_payload_from_args
from diffusionrl.config.build_domain_args import (
    build_reward_config,
    build_rollout_actor_init_config_from_args,
    build_training_actor_init_config_from_args,
    build_training_sampling_config,
)
from diffusionrl.config.resolution import (
    ConfigBundle,
    RolloutModeInfo,
    TrainingPlan,
    TrainTopology,
    derive_sampling_host_engine_type,
    derive_training_plan,
    require_rollout_num_gpus_per_actor,
    resolve_config,
    rollout_mode_is_colocated,
)
from diffusionrl.config.spec import ModelSpec
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import RolloutActorConfig, TrainingActorConfig
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.training.backends import resolve_train_backend_launch_spec
from diffusionrl.training.backends.base import TrainBackendLaunchSpec
from diffusionrl.types.sampling import SamplingSpec


@dataclass(frozen=True)
class PlacementSpec:
    rollout_num_nodes: int
    rollout_num_gpus_per_node: int
    training_num_nodes: int
    training_num_gpus_per_node: int
    reward_dedicated_num_gpus: int
    reward_dedicated_num_nodes: int
    reward_dedicated_num_gpus_per_node: int
    reward_dedicated_gpus_per_actor: int
    colocate_rollout: bool
    strategy: str


@dataclass(frozen=True)
class RolloutLaunch:
    mode: str
    rollout_engine: str
    actor_gpu_requirement: int
    colocate: bool
    colocate_gpu_fraction: float
    allow_noset_multi_gpu_inference: bool
    engine_init_payload: ComponentInitPayload
    actor_init_config: RolloutActorConfig


@dataclass(frozen=True)
class TrainingLaunch:
    topology: TrainTopology
    training_plan: TrainingPlan
    backend_name: str
    backend_capabilities: Dict[str, Any]
    launch_spec: TrainBackendLaunchSpec
    colocate: bool
    colocate_gpu_fraction: float
    backend_init_payload: ComponentInitPayload
    actor_init_config: TrainingActorConfig


@dataclass(frozen=True)
class LaunchConfig:
    algorithm_init_payload: Any
    model_spec: ModelSpec
    sampling_spec: SamplingSpec
    training_sampling_config: Dict[str, Any]
    rollout_mode_info: RolloutModeInfo
    placement: PlacementSpec
    rollout: Optional[RolloutLaunch]
    training: TrainingLaunch


def resolve_launch_placement_spec(
    args: Any,
    *,
    rollout_mode_info: Optional[RolloutModeInfo] = None,
) -> PlacementSpec:
    resolved_mode_info = (
        rollout_mode_info
        if rollout_mode_info is not None
        else resolve_config(args).rollout_mode_info
    )
    debug_mode = str(args.debug.debug_mode or "none").strip().lower()
    rollout_num_nodes = int(args.ray.rollout_num_nodes)
    rollout_num_gpus_per_node = int(args.ray.rollout_num_gpus_per_node)
    training_num_nodes = int(args.ray.training_num_nodes)
    training_num_gpus_per_node = int(args.ray.training_num_gpus_per_node)
    reward_dedicated_num_gpus = int(args.reward.reward_dedicated_num_gpus)
    reward_dedicated_num_nodes = int(args.reward.reward_dedicated_num_nodes)
    reward_dedicated_num_gpus_per_node = int(args.reward.reward_dedicated_num_gpus_per_node)

    if debug_mode == "train_only":
        rollout_num_nodes = 0
        rollout_num_gpus_per_node = 0
        reward_dedicated_num_gpus = 0
        reward_dedicated_num_nodes = 0
        reward_dedicated_num_gpus_per_node = 0
    elif resolved_mode_info.training_actor_sampling_mode:
        rollout_num_nodes = 0
        rollout_num_gpus_per_node = 0

    return PlacementSpec(
        rollout_num_nodes=rollout_num_nodes,
        rollout_num_gpus_per_node=rollout_num_gpus_per_node,
        training_num_nodes=training_num_nodes,
        training_num_gpus_per_node=training_num_gpus_per_node,
        reward_dedicated_num_gpus=reward_dedicated_num_gpus,
        reward_dedicated_num_nodes=reward_dedicated_num_nodes,
        reward_dedicated_num_gpus_per_node=reward_dedicated_num_gpus_per_node,
        reward_dedicated_gpus_per_actor=int(args.reward.reward_dedicated_gpus_per_actor),
        colocate_rollout=rollout_mode_is_colocated(
            resolved_mode_info.rollout_topology.mode
        ),
        strategy=str(args.ray.placement_strategy),
    )


def resolve_launch_config(
    args: Any,
    *,
    resolved: Optional[ConfigBundle] = None,
) -> LaunchConfig:
    resolved_config = (
        resolved
        if resolved is not None
        else resolve_config(args, include_training_plan=True)
    )
    model_spec = resolved_config.model_spec
    rollout_mode_info = resolved_config.rollout_mode_info
    sampling_spec = resolved_config.sampling_spec
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args,
        sampling_spec=sampling_spec,
    )
    train_backend_config = resolved_config.train_backend_config
    training_topology = resolved_config.training_topology
    train_backend_capabilities = resolved_config.train_backend_capabilities
    train_backend_launch_spec = resolve_train_backend_launch_spec(
        train_backend_config,
        args=args,
        topology=training_topology,
    )
    training_plan = resolved_config.training_plan
    if training_plan is None:
        training_plan = derive_training_plan(
            args,
            training_topology=training_topology,
        )
    placement = resolve_launch_placement_spec(
        args,
        rollout_mode_info=rollout_mode_info,
    )
    reward_schema = RewardSchema.from_args(args)

    model_init_payload = build_model_bundle_init_payload_from_args(
        args, model_spec=model_spec
    )
    reward_config = build_reward_config(
        reward_schema=reward_schema,
    )
    training_sampling_config = build_training_sampling_config(
        precision_settings=args.precision,
        sampling_spec=sampling_spec,
        sampler_engine_type=derive_sampling_host_engine_type(
            args,
            rollout_mode_info=rollout_mode_info,
        ),
    )
    train_backend_init_payload = build_train_backend_init_payload_from_args(args)
    training_actor_init_config = build_training_actor_init_config_from_args(
        args,
        config_bundle=resolved_config,
        replay_enabled=bool(rollout_mode_info.replay_enabled),
        topology=training_topology,
        training_plan=training_plan,
        algorithm_init_payload=algorithm_init_payload,
        model_init_payload=model_init_payload,
        reward_config=reward_config,
        sampling_config=training_sampling_config,
        train_backend_init_payload=train_backend_init_payload,
    )
    training = TrainingLaunch(
        topology=training_topology,
        training_plan=training_plan,
        backend_name=train_backend_config.name,
        backend_capabilities=train_backend_capabilities.as_dict(),
        launch_spec=train_backend_launch_spec,
        colocate=placement.colocate_rollout,
        colocate_gpu_fraction=float(args.ray.colocate_training_gpu_fraction),
        backend_init_payload=train_backend_init_payload,
        actor_init_config=training_actor_init_config,
    )

    rollout: Optional[RolloutLaunch] = None
    if placement.rollout_num_nodes > 0 and placement.rollout_num_gpus_per_node > 0:
        rollout_engine = rollout_mode_info.rollout_topology.rollout_engine
        if not rollout_engine:
            raise ValueError(
                "Resolved rollout launch requires rollout.rollout_engine to be set."
            )
        rollout_engine_init_payload = build_rollout_engine_init_payload_from_args(
            args,
            model_init_payload=model_init_payload,
            sampling_spec=sampling_spec,
            rollout_mode_info=rollout_mode_info,
        )
        rollout_actor_init_config = build_rollout_actor_init_config_from_args(
            args,
            config_bundle=resolved_config,
            model_init_payload=model_init_payload,
            reward_config=reward_config,
            engine_init_payload=rollout_engine_init_payload,
        )
        rollout = RolloutLaunch(
            mode=rollout_mode_info.rollout_topology.mode,
            rollout_engine=rollout_engine,
            actor_gpu_requirement=require_rollout_num_gpus_per_actor(args),
            colocate=placement.colocate_rollout,
            colocate_gpu_fraction=float(args.ray.colocate_rollout_gpu_fraction),
            allow_noset_multi_gpu_inference=bool(args.ray.allow_noset_multi_gpu_inference),
            engine_init_payload=rollout_engine_init_payload,
            actor_init_config=rollout_actor_init_config,
        )

    return LaunchConfig(
        algorithm_init_payload=algorithm_init_payload,
        model_spec=model_spec,
        sampling_spec=sampling_spec,
        training_sampling_config=training_sampling_config,
        rollout_mode_info=rollout_mode_info,
        placement=placement,
        rollout=rollout,
        training=training,
    )


__all__ = [
    "PlacementSpec",
    "RolloutLaunch",
    "LaunchConfig",
    "TrainingLaunch",
    "resolve_launch_config",
    "resolve_launch_placement_spec",
]
