"""
Utility functions for advantage computation.

Includes distributed operations and special-purpose transformations.

Copied from unified_grpo/advantages/utils.py
"""

import torch
import torch.distributed as dist


def gather_rewards_across_gpus(rewards: torch.Tensor) -> torch.Tensor:
    """Gather rewards from all GPUs for advantage computation.

    This is needed for flow_grpo/DiffusionNFT style training where advantages
    should be computed across all GPUs' samples.

    Args:
        rewards: Local rewards tensor [local_batch_size]

    Returns:
        Gathered rewards tensor [total_batch_size]
    """
    if not dist.is_initialized():
        return rewards

    world_size = dist.get_world_size()
    if world_size == 1:
        return rewards

    # Gather rewards from all ranks
    gathered_list = [torch.zeros_like(rewards) for _ in range(world_size)]
    dist.all_gather(gathered_list, rewards)

    return torch.cat(gathered_list, dim=0)
