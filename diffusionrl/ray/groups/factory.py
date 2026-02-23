"""Worker-group construction factories."""

from __future__ import annotations

import logging

from diffusionrl.config.schema import (
    build_inference_engine_config,
    build_training_actor_init_config,
)

from .inference import InferenceActorGroup
from .training import TrainingActorGroup

logger = logging.getLogger(__name__)

def create_inference_actor_group(
    args,
    pg_result,
) -> InferenceActorGroup:
    """
    Factory function to create InferenceActorGroup from args.

    GPU Allocation Strategy:
    - FSDP engine: 1 GPU per actor (default)
    - FastVideo engine: sp_size GPUs per actor (for sequence parallelism)

    Args:
        args: TrainingArguments instance
        pg_result: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized InferenceActorGroup
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
        logger.warning("engine_kwargs is not a dict in create_inference_actor_group; resetting to empty dict.")
        engine_kwargs = {}

    if sampler_engine_type == "fastvideo":
        # FastVideo GPU allocation:
        # - num_gpus: GPUs per FastVideo instance (per Ray actor)
        # - sp_size: Sequence parallelism size (must <= num_gpus)
        #
        # Scenarios:
        # 1. sp_size=1, num_gpus=1: Single GPU, no parallelism
        # 2. sp_size=4, num_gpus=4: 4 GPUs with SP
        # 3. sp_size=1, num_gpus=4: 4 GPUs, no SP (data parallel within executor)
        #
        # For multi-node: each node gets one actor, actor uses local GPUs only
        # (MultiprocExecutor uses multiprocessing, doesn't support cross-node)

        sp_size = engine_kwargs.get("sp_size", getattr(args, "sp_size", 1))
        tp_size = engine_kwargs.get("tp_size", getattr(args, "tp_size", 1))

        # Determine num_gpus per actor
        # If fastvideo_num_gpus is set, use it; otherwise use sp_size
        fastvideo_num_gpus = getattr(args, "fastvideo_num_gpus", None)
        if fastvideo_num_gpus is not None:
            num_gpus_per_actor = fastvideo_num_gpus
        else:
            # Default: each actor gets sp_size GPUs
            # This ensures SP works correctly
            num_gpus_per_actor = sp_size

        # Validate
        if sp_size > num_gpus_per_actor:
            raise ValueError(
                f"sp_size ({sp_size}) must be <= num_gpus_per_actor ({num_gpus_per_actor})"
            )

        # Update engine_kwargs
        engine_kwargs["sp_size"] = sp_size
        engine_kwargs["num_gpus"] = num_gpus_per_actor
        engine_kwargs["tp_size"] = tp_size

        logger.info(
            f"FastVideo engine: num_gpus_per_actor={num_gpus_per_actor}, "
            f"sp_size={sp_size}, tp_size={tp_size}"
        )
    elif sampler_engine_type == "fsdp":
        # FSDP GPU allocation:
        # - fsdp_num_gpus: GPUs per FSDP inference actor (default: 1)
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
    # For FastVideo with sp_size>1, each actor uses sp_size GPUs
    total_gpus = args.inference_num_nodes * args.inference_num_gpus_per_node

    # In colocate mode with single-GPU setup, use fractional GPU allocation
    # This allows both inference and training actors to share the same GPU bundle
    colocate = getattr(args, "colocate_inference_training", False)
    if colocate and num_gpus_per_actor == 1:
        num_gpus_per_actor = float(getattr(args, "colocate_inference_gpu_fraction", 0.4))
        logger.info(
            f"Colocate mode: InferenceActors using {num_gpus_per_actor} GPU each"
        )

    # Multi-GPU engine (non-colocate): use Slime pattern
    if num_gpus_per_actor > 1 and not colocate:
        if not bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
            raise ValueError(
                "Multi-GPU inference actor layout requires --allow-noset-multi-gpu-inference=true. "
                "Default layout only supports integer single-GPU actors."
            )
        from diffusionrl.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        actual_gpus_per_engine = int(num_gpus_per_actor)
        ray_num_gpus = 0.5  # Fractional claim to satisfy Ray scheduler
        num_actors = total_gpus // actual_gpus_per_engine

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for inference. "
                f"Total GPUs: {total_gpus}, GPUs per engine: {actual_gpus_per_engine}"
            )

        noset_env = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

        logger.info(
            f"Creating {num_actors} inference actors (Slime pattern), "
            f"{actual_gpus_per_engine} GPU(s) per engine, "
            f"ray_num_gpus={ray_num_gpus}"
        )

        group = InferenceActorGroup(
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
                f"Not enough GPUs for inference. "
                f"Total GPUs: {total_gpus}, GPUs per actor: {num_gpus_per_actor}"
            )

        logger.info(
            f"Creating {num_actors} inference actors, "
            f"{num_gpus_per_actor} GPU(s) per actor"
        )

        group = InferenceActorGroup(
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            num_gpus_per_actor=num_gpus_per_actor,
            sampler_engine_type=sampler_engine_type,
        )

    # Initialize actors with domain-split runtime/model schema.
    engine_config = build_inference_engine_config(
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

    num_actors = args.training_num_nodes * args.training_num_gpus_per_node

    # In colocate mode, use fractional GPU allocation to allow sharing
    # Both inference and training actors will claim 0.5 GPU each
    colocate = getattr(args, "colocate_inference_training", False)
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
    )

    config = build_training_actor_init_config(args=args, world_size=num_actors)
    group.init(config)

    return group

__all__ = [
    "create_inference_actor_group",
    "create_training_actor_group",
]
