"""
Advantages Module for GRPO Training.

Provides unified advantage computation with clean separation of concerns:
- normalizers: Pure mathematical operations
- strategies: Strategy pattern for different normalization approaches
- utils: Distributed and special-purpose utilities

Usage:
    # Unified function API (recommended)
    from diffusionrl.advantages import compute_advantage
    advantages = compute_advantage(rewards, strategy='group', num_samples_per_prompt=4)
"""

from typing import Dict, List, Optional, Union

import torch

# Core normalizers
from .normalizers import (
    normalize_global,
    normalize_grouped,
    build_fixed_size_groups,
    build_prompt_groups,
)

# Strategy classes
from .strategies import (
    AdvantageConfig,
    GlobalStrategy,
    GroupStrategy,
    PerPromptStrategy,
    STRATEGIES,
    get_strategy,
)

# Utilities
from .utils import (
    gather_rewards_across_gpus,
)

# Cross-batch tracking
from .per_prompt_tracker import (
    PerPromptStatTracker,
)

# Running statistics (for cross-batch global normalization, DanceGRPO)
from .running_stats import (
    RunningMeanStd,
    RunningRewardNormalizer,
)


# =============================================================================
# Unified API
# =============================================================================


def compute_advantage(
    rewards: Union[torch.Tensor, Dict[str, torch.Tensor]],
    *,
    strategy: str = "group",
    # Group strategy parameters
    num_samples_per_prompt: int = 1,
    # Per-prompt strategy parameters
    prompts: Optional[List[str]] = None,
    # Normalization parameters
    epsilon: float = 1e-8,
    clip_max: Optional[float] = 5.0,
    trimmed_ratio: float = 0.0,
    global_std: bool = False,
    # Distributed parameters
    gather_rewards: bool = False,
    # Multi-reward aggregation
    reward_weights: Optional[Dict[str, float]] = None,
    aggregation: str = "advantage",  # "advantage" | "reward"
) -> torch.Tensor:
    """Unified advantage computation function.

    Args:
        rewards: Reward values [N], or dict of {model_name: rewards} for multi-model
        strategy: Normalization strategy - one of "group", "global", "per_prompt"
        num_samples_per_prompt: Number of samples per prompt (for "group" strategy)
        prompts: List of prompts (required for "per_prompt" strategy)
        epsilon: Small value for numerical stability
        clip_max: If set, clip advantages to [-clip_max, clip_max]
        trimmed_ratio: Ratio of samples to trim from each end (for "group" strategy)
        global_std: If True, use global std for normalization (for "per_prompt" strategy)
        gather_rewards: If True, gather rewards from all GPUs before computing
        reward_weights: Weights for multi-reward aggregation (only for dict rewards)
        aggregation: Multi-reward aggregation mode:
            - "advantage": Compute advantage per model, then weighted sum
            - "reward": Weighted sum of rewards first, then compute advantage

    Returns:
        Advantages tensor [N]
    """
    # Handle multi-reward dict input
    if isinstance(rewards, dict):
        if reward_weights is None:
            # Equal weights if not specified
            reward_weights = {k: 1.0 / len(rewards) for k in rewards}

        if aggregation == "reward":
            # Aggregate rewards first, then compute advantage
            aggregated_rewards = sum(
                reward_weights.get(k, 0.0) * v for k, v in rewards.items()
            )
            rewards = aggregated_rewards
        elif aggregation == "advantage":
            # Compute advantage per model, then aggregate
            advantages_dict = {}
            for name, r in rewards.items():
                advantages_dict[name] = _compute_single_advantage(
                    r, strategy, num_samples_per_prompt, prompts,
                    epsilon, clip_max, trimmed_ratio, global_std, gather_rewards
                )
            # Weighted sum of advantages
            return sum(
                reward_weights.get(k, 0.0) * v for k, v in advantages_dict.items()
            )
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation}")

    return _compute_single_advantage(
        rewards, strategy, num_samples_per_prompt, prompts,
        epsilon, clip_max, trimmed_ratio, global_std, gather_rewards
    )


def _compute_single_advantage(
    rewards: torch.Tensor,
    strategy: str,
    num_samples_per_prompt: int,
    prompts: Optional[List[str]],
    epsilon: float,
    clip_max: Optional[float],
    trimmed_ratio: float,
    global_std: bool,
    gather_rewards_flag: bool,
) -> torch.Tensor:
    """Internal function to compute advantage for a single reward tensor."""
    # Optionally gather rewards from all GPUs
    if gather_rewards_flag:
        rewards = gather_rewards_across_gpus(rewards)

    if strategy == "group":
        if num_samples_per_prompt <= 1:
            return normalize_global(rewards, epsilon, clip_max)
        config = AdvantageConfig(
            epsilon=epsilon,
            clip_max=clip_max,
            trimmed_ratio=trimmed_ratio,
            global_std=False,
        )
        strat = GroupStrategy(num_samples_per_prompt=num_samples_per_prompt, config=config)
        return strat.compute(rewards)

    elif strategy == "global":
        return normalize_global(rewards, epsilon, clip_max)

    elif strategy == "per_prompt":
        if prompts is None:
            return normalize_global(rewards, epsilon, clip_max)
        config = AdvantageConfig(
            epsilon=epsilon,
            clip_max=clip_max,
            trimmed_ratio=0.0,
            global_std=global_std,
        )
        strat = PerPromptStrategy(config=config)
        return strat.compute(rewards, prompts=prompts)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


__all__ = [
    # Primary API
    "compute_advantage",
    # Low-level components
    "normalize_global",
    "normalize_grouped",
    "build_fixed_size_groups",
    "build_prompt_groups",
    "AdvantageConfig",
    "GlobalStrategy",
    "GroupStrategy",
    "PerPromptStrategy",
    "STRATEGIES",
    "get_strategy",
    "gather_rewards_across_gpus",
    # Cross-batch tracking (for flow_grpo)
    "PerPromptStatTracker",
    # Running statistics (for DanceGRPO cross-batch global normalization)
    "RunningMeanStd",
    "RunningRewardNormalizer",
]
