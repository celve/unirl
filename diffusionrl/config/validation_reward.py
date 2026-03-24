"""Reward and rollout-buffer validation helpers."""

from __future__ import annotations

import logging
from typing import Any

from diffusionrl.config.resolution import derive_global_rollout_batch_size

logger = logging.getLogger(__name__)


def validate_reward_config(args: Any) -> None:
    """Validate reward pool/source configuration consistency."""
    if args.reward.reward_dedicated_gpus_per_actor > 1 and args.reward.reward_dedicated_num_gpus > 0:
        if args.reward.reward_dedicated_num_gpus < args.reward.reward_dedicated_gpus_per_actor:
            raise ValueError(
                f"reward_dedicated_num_gpus ({args.reward.reward_dedicated_num_gpus}) must be >= "
                f"reward_dedicated_gpus_per_actor ({args.reward.reward_dedicated_gpus_per_actor})"
            )
        if args.reward.reward_dedicated_num_gpus % args.reward.reward_dedicated_gpus_per_actor != 0:
            raise ValueError(
                f"reward_dedicated_num_gpus ({args.reward.reward_dedicated_num_gpus}) must be divisible by "
                f"reward_dedicated_gpus_per_actor ({args.reward.reward_dedicated_gpus_per_actor})"
            )

    if args.reward.reward_dedicated_num_nodes > 0 and args.reward.reward_dedicated_num_gpus_per_node <= 0:
        raise ValueError(
            "reward_dedicated_num_gpus_per_node must be > 0 when reward_dedicated_num_nodes > 0"
        )

    if args.reward.reward_dedicated_num_gpus > 0 and args.reward.reward_dedicated_num_nodes > 0:
        raise ValueError(
            "reward_dedicated_num_gpus and reward_dedicated_num_nodes are mutually exclusive. "
            "Use either total dedicated GPUs, or nodes * gpus_per_node."
        )

    has_dedicated_reward_pool = (
        args.reward.reward_dedicated_num_gpus > 0 or args.reward.reward_dedicated_num_nodes > 0
    )
    has_http_reward_urls = bool(
        args.reward.reward_service_url
        or args.reward.reward_service_urls
    )
    has_http_reward = bool(
        args.reward.use_http_reward or has_http_reward_urls
    )
    local_reward_device = str(args.reward.local_reward_device or "cpu").strip().lower()
    reward_location = str(args.reward.reward_location or "manager").strip().lower()
    allow_local_reward_cuda_contention = bool(args.reward.allow_local_reward_cuda_contention)

    if args.reward.use_http_reward and not has_http_reward_urls:
        raise ValueError(
            "use_http_reward=true requires reward_service_url or reward_service_urls."
        )

    if reward_location == "sampling_actor":
        if has_http_reward:
            raise ValueError(
                "reward_location='sampling_actor' cannot be combined with HTTP reward service. "
                "Use reward_location='manager' for HTTP reward."
            )
        if has_dedicated_reward_pool:
            raise ValueError(
                "reward_location='sampling_actor' cannot be combined with dedicated reward actors. "
                "Use reward_location='manager' for reward_dedicated_* modes."
            )

    uses_local_same_process_reward = (
        reward_location == "manager"
        and not has_http_reward
        and not has_dedicated_reward_pool
    )
    if (
        uses_local_same_process_reward
        and local_reward_device == "cuda"
        and not allow_local_reward_cuda_contention
    ):
        raise ValueError(
            "local_reward_device='cuda' in local same-process reward mode can contend with "
            "rollout/training GPUs. Use dedicated reward actors (reward_dedicated_*), "
            "use_http_reward, or set allow_local_reward_cuda_contention=true to force."
        )

    if reward_location == "sampling_actor":
        logger.info(
            "Reward mode: sampling-actor-local worker (local_reward_device=%s)",
            local_reward_device,
        )
    elif has_http_reward:
        logger.info("Reward mode: HTTP (external service)")
    elif has_dedicated_reward_pool:
        total_gpus = args.reward.reward_dedicated_num_gpus
        if args.reward.reward_dedicated_num_nodes > 0:
            total_gpus = args.reward.reward_dedicated_num_nodes * args.reward.reward_dedicated_num_gpus_per_node
        num_actors = total_gpus // args.reward.reward_dedicated_gpus_per_actor
        logger.info(
            "Reward mode: Independent GPU (%s GPUs, %s actors, %s GPUs/actor)",
            total_gpus,
            num_actors,
            args.reward.reward_dedicated_gpus_per_actor,
        )
    else:
        logger.info(
            "Reward mode: Local same-process worker (local_reward_device=%s)",
            local_reward_device,
        )


def validate_reward_and_rollout_buffer_config(args: Any) -> None:
    """Validate reward pool config and rollout-buffer controls."""
    validate_reward_config(args)
    rollout_buffer = args.rollout.buffer

    if bool(rollout_buffer.reassemble_by_group) and rollout_buffer.group_size is not None:
        if bool(rollout_buffer.drop_invalid):
            raise ValueError(
                "rollout.buffer.reassemble_by_group is incompatible with "
                "rollout.buffer.drop_invalid=true. Sample-dropping finite-value "
                "filtering can leave incomplete groups pending forever. Set "
                "rollout.buffer.drop_invalid=false so invalid batches fail fast."
            )
        if rollout_buffer.reward_min is not None or rollout_buffer.reward_max is not None:
            raise ValueError(
                "rollout.buffer.reassemble_by_group is incompatible with "
                "rollout.buffer.reward_min/max. Reward-range filtering drops "
                "samples and breaks the complete-group producer contract."
            )
        target_batch_size = int(derive_global_rollout_batch_size(args))
        if int(rollout_buffer.group_size) > target_batch_size:
            raise ValueError(
                "rollout.buffer.group_size cannot exceed the resolved training batch size. "
                f"Got group_size={rollout_buffer.group_size}, target_batch_size={target_batch_size}."
            )
        if target_batch_size % int(rollout_buffer.group_size) != 0:
            raise ValueError(
                "rollout.buffer.reassemble_by_group requires the resolved training batch size "
                "to be divisible by rollout.buffer.group_size. "
                f"Got target_batch_size={target_batch_size}, group_size={rollout_buffer.group_size}."
            )


__all__ = [
    "validate_reward_and_rollout_buffer_config",
    "validate_reward_config",
]
