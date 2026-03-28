"""Launch-side config assembly built once on the driver.

This module consumes ``ConfigBundle`` from ``resolution.py`` and builds the
driver-facing launch payloads used by placement, group creation, and rollout /
training actor initialization. It does not own config semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from diffusionrl.algorithms.construction import build_algorithm_config
from diffusionrl.config.build_domain_args import (
    build_model_config,
    build_reward_config,
    build_training_sampling_config,
    build_train_backend_config,
    build_rollout_actor_init_config,
    build_rollout_engine_config,
    build_training_actor_init_config,
)
from diffusionrl.config.resolution import (
    ConfigBundle,
    ModelSpec,
    RolloutModeInfo,
    TrainingPlan,
    TrainTopology,
    derive_sampling_host_engine_type,
    derive_training_plan,
    require_rollout_service_num_gpus,
    resolve_config,
    rollout_mode_is_colocated,
)
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.training.backends import (
    resolve_train_backend_launch_spec,
)
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
    service_engine: str
    actor_gpu_requirement: int
    colocate: bool
    colocate_gpu_fraction: float
    allow_noset_multi_gpu_inference: bool
    engine_runtime_config: Dict[str, Any]
    actor_init_config: Dict[str, Any]


@dataclass(frozen=True)
class TrainingLaunch:
    topology: TrainTopology
    training_plan: TrainingPlan
    backend_name: str
    backend_capabilities: Dict[str, Any]
    launch_spec: TrainBackendLaunchSpec
    colocate: bool
    colocate_gpu_fraction: float
    backend_config: Dict[str, Any]
    actor_init_config: Dict[str, Any]


@dataclass(frozen=True)
class LaunchConfig:
    algorithm_config: Dict[str, Any]
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
    algorithm_config = build_algorithm_config(args, sampling_spec=sampling_spec)
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

    model_config = build_model_config(
        model_spec=model_spec,
        model_settings=args.model,
        training_settings=args.training,
        precision_settings=args.precision,
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
    train_backend_payload = build_train_backend_config(
        resolved_backend_config=train_backend_config,
    )
    training_actor_init_config = build_training_actor_init_config(
        training_settings=args.training,
        rollout_control_settings=args.rollout.control,
        replay_log_probs=bool(args.sampling.replay_log_probs),
        topology=training_topology,
        training_plan=training_plan,
        algorithm_config=algorithm_config,
        model_config=model_config,
        reward_config=reward_config,
        sampling_config=training_sampling_config,
        train_backend_config=train_backend_payload,
    )
    training = TrainingLaunch(
        topology=training_topology,
        training_plan=training_plan,
        backend_name=train_backend_config.name,
        backend_capabilities=train_backend_capabilities.as_dict(),
        launch_spec=train_backend_launch_spec,
        colocate=placement.colocate_rollout,
        colocate_gpu_fraction=float(args.ray.colocate_training_gpu_fraction),
        backend_config=train_backend_payload,
        actor_init_config=training_actor_init_config,
    )

    rollout: Optional[RolloutLaunch] = None
    if placement.rollout_num_nodes > 0 and placement.rollout_num_gpus_per_node > 0:
        service_engine = rollout_mode_info.rollout_topology.service_engine
        if not service_engine:
            raise ValueError(
                "Resolved rollout launch requires rollout.topology.service_engine to be set."
            )
        rollout_engine_runtime_config = build_rollout_engine_config(
            rollout_topology_settings=args.rollout.topology,
            precision_settings=args.precision,
            sync_settings=args.sync,
            fps=int(args.fps),
            logprob_source=rollout_mode_info.logprob_source,
            sampler_engine_type=service_engine,
            model_config=model_config,
            sampling_spec=sampling_spec,
            offload_rollout=bool(args.ray.offload_rollout),
        )
        rollout_actor_init_config = build_rollout_actor_init_config(
            engine_runtime_config=rollout_engine_runtime_config,
            reward_config=reward_config,
        )
        rollout = RolloutLaunch(
            mode=rollout_mode_info.rollout_topology.mode,
            service_engine=service_engine,
            actor_gpu_requirement=require_rollout_service_num_gpus(args),
            colocate=placement.colocate_rollout,
            colocate_gpu_fraction=float(args.ray.colocate_rollout_gpu_fraction),
            allow_noset_multi_gpu_inference=bool(args.ray.allow_noset_multi_gpu_inference),
            engine_runtime_config=rollout_engine_runtime_config,
            actor_init_config=rollout_actor_init_config,
        )

    return LaunchConfig(
        algorithm_config=algorithm_config,
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
