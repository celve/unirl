"""Worker-group construction factories."""

from __future__ import annotations

import copy
import logging

from diffusionrl.config.launch_resolution import LaunchConfig
from diffusionrl.config.validation import (
    validate_rollout_actor_init_config,
    validate_training_actor_init_config,
)

from .rollout_group import RolloutActorGroup
from .training_group import TrainingActorGroup

logger = logging.getLogger(__name__)

def create_rollout_actor_group(
    launch_config: LaunchConfig,
    pgs,
) -> RolloutActorGroup:
    """
    Factory function to create RolloutActorGroup for dedicated rollout engines.

    Args:
        launch_config: Resolved launch config built on the driver
        pgs: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized RolloutActorGroup
    """
    rollout = launch_config.rollout
    if rollout is None:
        raise ValueError(
            "Resolved runtime config does not include a dedicated rollout bootstrap payload."
        )
    pg, bundle_indices, gpu_ids = pgs
    rollout_service_engine = rollout.service_engine

    actor_init_config = copy.deepcopy(rollout.actor_init_config)
    engine_runtime_config = dict(rollout.engine_runtime_config)
    engine_kwargs = dict(engine_runtime_config.get("engine_kwargs") or {})
    num_gpus_per_actor = int(rollout.actor_gpu_requirement)

    if rollout_service_engine == "sglang":
        logger.info(
            "SGLang engine: num_gpus_per_actor=%s, tp_size=%s, sp_degree=%s",
            num_gpus_per_actor,
            engine_kwargs.get("tp_size", "auto"),
            engine_kwargs.get("sp_degree", engine_kwargs.get("sp_size", 1)),
        )
    else:
        raise ValueError(
            f"Unsupported dedicated rollout engine: {rollout_service_engine!r}. "
            "Expected: sglang."
        )

    # Calculate number of actors
    # Multi-GPU engines allocate more than one GPU per actor.
    available_rollout_bundles = len(bundle_indices)
    if available_rollout_bundles < 1:
        raise ValueError(
            "Rollout placement group did not allocate any GPU bundles for dedicated rollout actors."
        )

    # In colocate mode with single-GPU setup, use fractional GPU allocation
    # This allows both rollout and training actors to share the same GPU bundle
    colocate = bool(rollout.colocate)
    if colocate and num_gpus_per_actor == 1:
        num_gpus_per_actor = float(rollout.colocate_gpu_fraction)
        logger.info(
            f"Colocate mode: RolloutActorGroup actors using {num_gpus_per_actor} GPU each"
        )

    # Multi-GPU engine: use Slime/NOSET pattern.
    # This path supports both separate and colocate rollout modes.
    if num_gpus_per_actor > 1:
        if not bool(rollout.allow_noset_multi_gpu_inference):
            raise ValueError(
                "Multi-GPU rollout actor layout requires --allow-noset-multi-gpu-inference=true. "
                "Default layout only supports integer single-GPU actors."
            )
        from diffusionrl.ray.ray_utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        actual_gpus_per_engine = int(num_gpus_per_actor)
        ray_num_gpus = 0.5  # Fractional claim to satisfy Ray scheduler
        if available_rollout_bundles % actual_gpus_per_engine != 0:
            raise ValueError(
                "Placement-group rollout bundle count must be divisible by rollout.topology.service_num_gpus "
                "for multi-GPU rollout actors. "
                f"Available rollout bundles: {available_rollout_bundles}, "
                f"rollout.topology.service_num_gpus: {actual_gpus_per_engine}."
            )
        num_actors = available_rollout_bundles // actual_gpus_per_engine

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for rollout. "
                f"Available rollout bundles: {available_rollout_bundles}, GPUs per engine: {actual_gpus_per_engine}"
            )

        noset_env = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

        logger.info(
            f"Creating {num_actors} rollout actors (Slime pattern, colocate={colocate}), "
            f"{actual_gpus_per_engine} GPU(s) per engine, "
            f"ray_num_gpus={ray_num_gpus}"
        )

        group = RolloutActorGroup(
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
            num_gpus_per_actor=ray_num_gpus,
            num_gpus_per_engine=actual_gpus_per_engine,
            capture_child_tasks=True,
            runtime_env={"env_vars": noset_env},
            sampler_engine_type=rollout_service_engine,
            num_gpus_allocated=actual_gpus_per_engine,
            force_set_cuda_visible_devices=True,
        )
    else:
        # Single GPU or colocate mode: standard scheduling
        if colocate and num_gpus_per_actor < 1:
            num_actors = available_rollout_bundles
        else:
            num_actors = int(available_rollout_bundles / num_gpus_per_actor)

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for rollout. "
                f"Available rollout bundles: {available_rollout_bundles}, GPUs per actor: {num_gpus_per_actor}"
            )

        logger.info(
            f"Creating {num_actors} rollout actors, "
            f"{num_gpus_per_actor} GPU(s) per actor"
        )

        group = RolloutActorGroup(
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            num_gpus_per_actor=num_gpus_per_actor,
            sampler_engine_type=rollout_service_engine,
        )

    # Initialize actors with domain-split runtime/model schema.
    validate_rollout_actor_init_config(actor_init_config)
    group.init(actor_init_config)
    return group


def create_training_actor_group(
    launch_config: LaunchConfig,
    pgs,
) -> TrainingActorGroup:
    """
    Factory function to create TrainingActorGroup from args.

    Args:
        launch_config: Resolved launch config built on the driver
        pgs: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized TrainingActorGroup
    """
    pg, bundle_indices, gpu_ids = pgs
    training = launch_config.training
    training_topology = training.topology
    # Actor-group size is currently driven by the resolved training actor_count.
    # This is a launch/orchestration quantity; it should not be conflated with
    # world_size or dp_size even when they happen to match in the mainline path.
    num_actors = int(training_topology.actor_count)

    config = copy.deepcopy(training.actor_init_config)
    validate_training_actor_init_config(config)
    launch_spec = training.launch_spec
    caps = training.backend_capabilities
    if bool(caps.get("requires_custom_actor_class", False)) and not launch_spec.actor_class_path:
        raise ValueError(
            f"train_backend={training.backend_name!r} requires a backend-specific actor class. "
            "Provide train_backend_kwargs with `actor_class_path`, "
            "or override via --train-backend-path to a backend that does not require a custom actor."
        )

    logger.info(
        "Training backend=%s launch spec: %s; resolved topology=%s",
        training.backend_name,
        launch_spec.as_dict(),
        training_topology.as_dict(),
    )

    # In colocate mode, use fractional GPU allocation to allow sharing
    # Both rollout and training actors will claim fractional GPU each
    colocate = bool(training.colocate)
    if launch_spec.num_gpus_per_actor is not None:
        num_gpus_per_actor = float(launch_spec.num_gpus_per_actor)
    else:
        num_gpus_per_actor = float(training.colocate_gpu_fraction) if colocate else 1.0

    if colocate:
        logger.info(
            f"Colocate mode: TrainingActors using {num_gpus_per_actor} GPU each"
        )

    group = TrainingActorGroup(
        num_actors=num_actors,
        pg=pg,
        bundle_indices=bundle_indices,
        num_gpus_per_actor=num_gpus_per_actor,
        actor_class_path=launch_spec.actor_class_path,
        actor_init_kwargs=dict(launch_spec.actor_kwargs or {}),
        runtime_env=dict(launch_spec.runtime_env or {}) or None,
    )

    master_info = group.call_rank0("get_master_info")
    if not isinstance(master_info, dict):
        raise RuntimeError(f"Invalid rank0 master payload: {master_info!r}")
    master_addr = str(master_info["master_addr"])
    master_port = int(master_info["master_port"])
    group.call_all("set_master_info", master_addr, master_port)

    group.init(config)

    return group

__all__ = [
    "create_rollout_actor_group",
    "create_training_actor_group",
]
