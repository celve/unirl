"""
Core normalization functions for advantage computation.

This module provides the low-level mathematical operations for normalizing rewards.
All functions are stateless and operate on pure tensors.

Copied from unified_grpo/advantages/normalizers.py
"""

from collections import defaultdict
from typing import Dict, List, Optional

import torch


def normalize_grouped(
    rewards: torch.Tensor,
    group_indices: List[List[int]],
    epsilon: float = 1e-8,
    clip_max: Optional[float] = None,
    trimmed_ratio: float = 0.0,
    use_global_std: bool = False,
) -> torch.Tensor:
    """Unified grouped normalization - single implementation for all group-based strategies.

    Normalizes rewards within specified groups, subtracting the group mean and
    dividing by the group (or global) standard deviation.

    Args:
        rewards: Reward values [N]
        group_indices: List of index lists, each defining a group
        epsilon: Small value for numerical stability
        clip_max: If set, clip advantages to [-clip_max, clip_max]
        trimmed_ratio: Ratio of samples to trim from each end when computing stats
        use_global_std: If True, use global std for all groups

    Returns:
        Advantages tensor [N] with per-group normalization
    """
    advantages = torch.zeros_like(rewards)
    batch_std = rewards.std() + epsilon if use_global_std else None

    for indices in group_indices:
        if not indices:
            continue

        group_rewards = rewards[indices]

        if trimmed_ratio > 0 and len(indices) > 2:
            sorted_rewards, _ = torch.sort(group_rewards)
            trim_size = int(len(sorted_rewards) * trimmed_ratio)
            if trim_size > 0 and trim_size * 2 < len(sorted_rewards):
                trimmed = sorted_rewards[trim_size:-trim_size]
            else:
                trimmed = sorted_rewards
            group_mean = trimmed.mean()
            group_std = batch_std if use_global_std else (trimmed.std() + epsilon)
        else:
            group_mean = group_rewards.mean()
            group_std = batch_std if use_global_std else (group_rewards.std() + epsilon)

        advantages[indices] = (group_rewards - group_mean) / group_std

    if clip_max is not None:
        advantages = torch.clamp(advantages, -clip_max, clip_max)

    return advantages


def normalize_global(
    rewards: torch.Tensor,
    epsilon: float = 1e-8,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Global normalization across all samples.

    Args:
        rewards: Reward values [N]
        epsilon: Small value for numerical stability
        clip_max: If set, clip advantages to [-clip_max, clip_max]

    Returns:
        Advantages tensor [N] with global normalization
    """
    mean = rewards.mean()
    std = rewards.std() + epsilon
    advantages = (rewards - mean) / std

    if clip_max is not None:
        advantages = torch.clamp(advantages, -clip_max, clip_max)

    return advantages


def build_fixed_size_groups(total: int, group_size: int) -> List[List[int]]:
    """Build group indices with fixed group size.

    Used by the 'group' strategy where samples are arranged in contiguous groups.

    Args:
        total: Total number of samples
        group_size: Number of samples per group

    Returns:
        List of index lists, each of length group_size
    """
    if group_size <= 0:
        return [[i] for i in range(total)]

    n_groups = total // group_size
    groups = []
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        groups.append(list(range(start_idx, end_idx)))

    return groups


def build_prompt_groups(prompts: List[str]) -> List[List[int]]:
    """Build group indices by prompt content.

    Groups samples that share the same prompt text together.

    Args:
        prompts: List of prompt strings [N]

    Returns:
        List of index lists, grouped by prompt content
    """
    prompt_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        prompt_to_indices[prompt].append(idx)
    return list(prompt_to_indices.values())
