"""Worker-group construction factories."""

from __future__ import annotations

import logging

from diffusionrl.config.build_domain_args import (
    build_rollout_engine_config,
    build_training_actor_init_config,
)
from diffusionrl.runtime.training import create_train_backend

from .rollout import RolloutActorGroup
from .training import TrainingActorGroup

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
    sampler_engine_type = getattr(args, "sampler_engine_type", None)

    # Fallback to fsdp if not set (should not happen if validate_args was called)
    if sampler_engine_type is None:
        logger.warning(
            "sampler_engine_type not set. This should have been auto-selected in validate_args(). "
            "Falling back to 'fsdp'."
        )
        sampler_engine_type = "fsdp"

    # Determine GPUs per actor based on engine type
    engine_kwargs = getattr(args, "engine_kwargs", {})
    if not isinstance(engine_kwargs, dict):
        logger.warning("engine_kwargs is not a dict in create_rollout_actor_group; resetting to empty dict.")
        engine_kwargs = {}

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

        fsdp_num_gpus = getattr(args, "fsdp_num_gpus", 1)
        fsdp_sharding_strategy = getattr(args, "fsdp_inference_sharding_strategy", "NO_SHARD")

        num_gpus_per_actor = fsdp_num_gpus

        # Update engine_kwargs
        engine_kwargs["num_gpus"] = num_gpus_per_actor
        engine_kwargs["fsdp_sharding_strategy"] = fsdp_sharding_strategy
        engine_kwargs.setdefault("cpu_offload", getattr(args, "fsdp_cpu_offload", False))

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
            raw_num_gpus = engine_kwargs.get("tp_size", getattr(args, "tp_size", 1))
        num_gpus_per_actor = int(raw_num_gpus)
        if num_gpus_per_actor < 1:
            raise ValueError(f"sglang engine num_gpus must be >= 1, got: {num_gpus_per_actor}")

        engine_kwargs["num_gpus"] = num_gpus_per_actor
        if "tp_size" not in engine_kwargs and getattr(args, "tp_size", None) is not None:
            engine_kwargs["tp_size"] = int(getattr(args, "tp_size"))
        if "sp_degree" not in engine_kwargs:
            if engine_kwargs.get("sp_size") is not None:
                engine_kwargs["sp_degree"] = int(engine_kwargs["sp_size"])
            elif getattr(args, "sp_size", 1) > 1:
                engine_kwargs["sp_degree"] = int(getattr(args, "sp_size"))
        if bool(getattr(args, "offload_rollout", False)):
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
    total_gpus = args.rollout_num_nodes * args.rollout_num_gpus_per_node

    # In colocate mode with single-GPU setup, use fractional GPU allocation
    # This allows both rollout and training actors to share the same GPU bundle
    colocate = getattr(args, "colocate_rollout_training", False)
    if colocate and num_gpus_per_actor == 1:
        num_gpus_per_actor = float(getattr(args, "colocate_rollout_gpu_fraction", 0.4))
        logger.info(
            f"Colocate mode: RolloutActorGroup actors using {num_gpus_per_actor} GPU each"
        )

    # Multi-GPU engine: use Slime/NOSET pattern.
    # This path supports both separate and colocate runtime modes.
    if num_gpus_per_actor > 1:
        if not bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
            raise ValueError(
                "Multi-GPU rollout actor layout requires --allow-noset-multi-gpu-inference=true. "
                "Default layout only supports integer single-GPU actors."
            )
        from diffusionrl.ray.utils.distributed import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

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
    group.init(engine_config)

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

    default_num_actors = args.training_num_nodes * args.training_num_gpus_per_node

    config = build_training_actor_init_config(args=args, dp_size=default_num_actors)
    backend = create_train_backend(
        config["train_backend"],
        backend_path=config.get("train_backend_path"),
        backend_kwargs=config.get("train_backend_kwargs"),
    )
    launch_spec = backend.launch_spec(args=args, default_num_actors=default_num_actors)
    caps = backend.capabilities
    if bool(caps.requires_custom_actor_class) and not launch_spec.actor_class_path:
        raise ValueError(
            f"train_backend={backend.name!r} requires a backend-specific actor class. "
            "Provide train_backend_kwargs_json with `actor_class_path`, "
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
    else:
        logger.info("Training backend=%s launch spec: %s", backend.name, launch_spec.as_dict())

    # In colocate mode, use fractional GPU allocation to allow sharing
    # Both rollout and training actors will claim fractional GPU each
    colocate = getattr(args, "colocate_rollout_training", False)
    if launch_spec.num_gpus_per_actor is not None:
        num_gpus_per_actor = float(launch_spec.num_gpus_per_actor)
    else:
        num_gpus_per_actor = float(getattr(args, "colocate_training_gpu_fraction", 0.4)) if colocate else 1.0

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
