"""Worker-group construction factories."""

from __future__ import annotations

import logging

from diffusionrl.config.build_domain_args import (
    build_rollout_engine_config,
    build_training_actor_init_config,
    validate_rollout_engine_config,
    validate_training_actor_init_config,
)
from diffusionrl.runtime.training import create_train_backend

from .rollout_group import RolloutActorGroup
from .training_group import TrainingActorGroup

logger = logging.getLogger(__name__)

def create_rollout_actor_group(
    args,
    pg_result,
) -> RolloutActorGroup:
    """
    Factory function to create RolloutActorGroup from args.

    GPU Allocation Strategy:
    - FSDP engine: 1 GPU per rollout actor (default)

    Args:
        args: TrainingArguments instance
        pg_result: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized RolloutActorGroup
    """
    pg, bundle_indices, gpu_ids = pg_result

    # Get sampler_engine_type (should be set by validate_args in arguments.py)
    sampler_engine_type = args.sampling.sampler_engine_type
    if sampler_engine_type is None:
        raise ValueError(
            "sampling.sampler_engine_type is unset after normalize/validate. "
            "Expected non-empty engine type."
        )

    # Determine GPUs per actor based on engine type
    engine_kwargs = args.sampling.engine_kwargs
    if not isinstance(engine_kwargs, dict):
        raise ValueError(
            "sampling.engine_kwargs must be a dict after normalize/validate, "
            f"got: {type(engine_kwargs).__name__}"
        )

    # [FastVideo-suspended] BEGIN — FastVideo GPU allocation branch
    # if sampler_engine_type == "fastvideo":
    #     ...
    # [FastVideo-suspended] END
    if sampler_engine_type == "fsdp":
        # FSDP GPU allocation:
        # - fsdp_num_gpus: GPUs per FSDP rollout actor (default: 1)
        # - fsdp_sharding_strategy: Sharding strategy (NO_SHARD for inference)
        #
        # Scenarios:
        # 1. fsdp_num_gpus=1: Single GPU, data parallel across actors (default)
        # 2. fsdp_num_gpus=4: 4 GPUs with FSDP model parallelism
        #
        # For multi-node: each node gets one actor, actor uses local GPUs only
        # (torch.distributed doesn't support cross-node within single actor)

        fsdp_num_gpus = args.sampling.fsdp_num_gpus
        fsdp_sharding_strategy = args.sampling.fsdp_inference_sharding_strategy

        num_gpus_per_actor = fsdp_num_gpus

        # Update engine_kwargs
        engine_kwargs["num_gpus"] = num_gpus_per_actor
        engine_kwargs["fsdp_sharding_strategy"] = fsdp_sharding_strategy
        engine_kwargs.setdefault("cpu_offload", args.training.fsdp_cpu_offload)

        if num_gpus_per_actor > 1:
            logger.info(
                f"FSDP engine: num_gpus_per_actor={num_gpus_per_actor}, "
                f"sharding_strategy={fsdp_sharding_strategy}"
            )
        else:
            logger.info("FSDP engine: single GPU per actor (default)")
    elif sampler_engine_type == "sglang":
        # sglang-diffusion GPU allocation:
        # - num_gpus: worker processes launched by DiffGenerator (authoritative)
        # - tp_size/sp_degree: optional parallel hints forwarded to ServerArgs
        raw_num_gpus = engine_kwargs.get("num_gpus")
        if raw_num_gpus is None:
            raw_num_gpus = engine_kwargs.get("tp_size", args.sampling.tp_size)
        num_gpus_per_actor = int(raw_num_gpus)
        if num_gpus_per_actor < 1:
            raise ValueError(f"sglang engine num_gpus must be >= 1, got: {num_gpus_per_actor}")

        engine_kwargs["num_gpus"] = num_gpus_per_actor
        if "tp_size" not in engine_kwargs:
            engine_kwargs["tp_size"] = int(args.sampling.tp_size)
        if "sp_degree" not in engine_kwargs:
            if engine_kwargs.get("sp_size") is not None:
                engine_kwargs["sp_degree"] = int(engine_kwargs["sp_size"])
            elif args.sampling.sp_size > 1:
                engine_kwargs["sp_degree"] = int(args.sampling.sp_size)
        if bool(args.ray.offload_rollout):
            # Pre-release policy: when rollout offload is requested, require
            # concrete SGLang memory handlers instead of best-effort fallbacks.
            engine_kwargs["require_memory_api"] = True

        logger.info(
            "SGLang engine: num_gpus_per_actor=%s, tp_size=%s, sp_degree=%s",
            num_gpus_per_actor,
            engine_kwargs.get("tp_size", "auto"),
            engine_kwargs.get("sp_degree", 1),
        )
    else:
        # Other engines: single GPU per actor.
        num_gpus_per_actor = 1

    # Calculate number of actors
    # Multi-GPU engines allocate more than one GPU per actor.
    total_gpus = args.ray.rollout_num_nodes * args.ray.rollout_num_gpus_per_node

    # In colocate mode with single-GPU setup, use fractional GPU allocation
    # This allows both rollout and training actors to share the same GPU bundle
    colocate = bool(args.ray.colocate_rollout_training)
    if colocate and num_gpus_per_actor == 1:
        num_gpus_per_actor = float(args.ray.colocate_rollout_gpu_fraction)
        logger.info(
            f"Colocate mode: RolloutActorGroup actors using {num_gpus_per_actor} GPU each"
        )

    # Multi-GPU engine: use Slime/NOSET pattern.
    # This path supports both separate and colocate runtime modes.
    if num_gpus_per_actor > 1:
        if not bool(args.ray.allow_noset_multi_gpu_inference):
            raise ValueError(
                "Multi-GPU rollout actor layout requires --allow-noset-multi-gpu-inference=true. "
                "Default layout only supports integer single-GPU actors."
            )
        from diffusionrl.ray.ray_utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        actual_gpus_per_engine = int(num_gpus_per_actor)
        ray_num_gpus = 0.5  # Fractional claim to satisfy Ray scheduler
        num_actors = total_gpus // actual_gpus_per_engine

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for rollout. "
                f"Total GPUs: {total_gpus}, GPUs per engine: {actual_gpus_per_engine}"
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
            sampler_engine_type=sampler_engine_type,
            num_gpus_allocated=actual_gpus_per_engine,
            force_set_cuda_visible_devices=True,
        )
    else:
        # Single GPU or colocate mode: standard scheduling
        if colocate and num_gpus_per_actor < 1:
            num_actors = total_gpus
        else:
            num_actors = int(total_gpus / num_gpus_per_actor)

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for rollout. "
                f"Total GPUs: {total_gpus}, GPUs per actor: {num_gpus_per_actor}"
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
            sampler_engine_type=sampler_engine_type,
        )

    # Initialize actors with domain-split runtime/model schema.
    engine_config = build_rollout_engine_config(
        args=args,
        sampler_engine_type=sampler_engine_type,
        engine_kwargs=engine_kwargs,
    )
    validate_rollout_engine_config(engine_config)
    group.init(engine_config)
    if hasattr(group, "refresh_weight_update_targets"):
        try:
            dedupe_payload = group.refresh_weight_update_targets()
            logger.info(
                "Rollout weight-update target topology: %s",
                dedupe_payload,
            )
        except Exception as exc:
            logger.warning(
                "Failed to refresh rollout weight-update targets; fallback to per-actor updates. %s",
                exc,
            )

    return group


def create_training_actor_group(
    args,
    pg_result,
) -> TrainingActorGroup:
    """
    Factory function to create TrainingActorGroup from args.

    Args:
        args: TrainingArguments instance
        pg_result: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized TrainingActorGroup
    """
    pg, bundle_indices, gpu_ids = pg_result

    default_num_actors = args.ray.training_num_nodes * args.ray.training_num_gpus_per_node

    config = build_training_actor_init_config(args=args, dp_size=default_num_actors)
    validate_training_actor_init_config(config)
    backend_config = config["train_backend_config"]
    backend_name = str(backend_config["name"]).strip().lower()
    backend_kwargs = dict(backend_config.get("kwargs") or {})
    backend = create_train_backend(
        backend_name,
        backend_path=backend_config.get("backend_path"),
        backend_kwargs=backend_kwargs,
    )
    launch_spec = backend.launch_spec(args=args, default_num_actors=default_num_actors)
    caps = backend.capabilities
    if bool(caps.requires_custom_actor_class) and not launch_spec.actor_class_path:
        raise ValueError(
            f"train_backend={backend.name!r} requires a backend-specific actor class. "
            "Provide train_backend_kwargs with `actor_class_path`, "
            "or override via --train-backend-path to a backend that does not require a custom actor."
        )

    num_actors = int(launch_spec.num_actors) if launch_spec.num_actors else int(default_num_actors)
    if num_actors != default_num_actors:
        logger.info(
            "Training backend=%s overrides num_actors: %s -> %s",
            backend.name,
            default_num_actors,
            num_actors,
        )
        config = build_training_actor_init_config(args=args, dp_size=num_actors)
        validate_training_actor_init_config(config)
    else:
        logger.info("Training backend=%s launch spec: %s", backend.name, launch_spec.as_dict())

    # In colocate mode, use fractional GPU allocation to allow sharing
    # Both rollout and training actors will claim fractional GPU each
    colocate = bool(args.ray.colocate_rollout_training)
    if launch_spec.num_gpus_per_actor is not None:
        num_gpus_per_actor = float(launch_spec.num_gpus_per_actor)
    else:
        num_gpus_per_actor = float(args.ray.colocate_training_gpu_fraction) if colocate else 1.0

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

    group.init(config)

    return group

__all__ = [
    "create_rollout_actor_group",
    "create_training_actor_group",
]
