"""
Strategy classes for advantage computation.

Each strategy encapsulates a specific normalization approach:
- GlobalStrategy: Normalize across all samples
- GroupStrategy: Normalize within fixed-size groups (MixGRPO/DanceGRPO)
- PerPromptStrategy: Normalize within prompt-based groups

Copied from unified_grpo/advantages/strategies.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import torch

from .normalizers import (
    normalize_global,
    normalize_grouped,
    build_fixed_size_groups,
    build_prompt_groups,
)


@dataclass
class AdvantageConfig:
    """Configuration parameters for advantage computation."""

    epsilon: float = 1e-8
    clip_max: Optional[float] = 5.0
    trimmed_ratio: float = 0.0
    global_std: bool = False


class GlobalStrategy:
    """Global normalization strategy.

    Normalizes rewards using statistics computed across all samples.
    """

    def __init__(self, config: Optional[AdvantageConfig] = None):
        self.config = config or AdvantageConfig()

    def compute(self, rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute advantages using global statistics.

        Args:
            rewards: Reward values [N]
            **kwargs: Reserved for strategy-specific options.

        Returns:
            Advantages tensor [N]
        """
        return normalize_global(
            rewards=rewards,
            epsilon=self.config.epsilon,
            clip_max=self.config.clip_max,
        )


class GroupStrategy:
    """Group-based normalization strategy.

    Normalizes rewards within fixed-size groups. Used by MixGRPO and DanceGRPO
    where multiple samples are generated per prompt.
    """

    def __init__(
        self,
        num_samples_per_prompt: int = 1,
        config: Optional[AdvantageConfig] = None,
    ):
        self.num_samples_per_prompt = num_samples_per_prompt
        self.config = config or AdvantageConfig()

    def compute(self, rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute advantages using per-group statistics.

        Args:
            rewards: Reward values [N]
            **kwargs: May contain 'num_samples_per_prompt' to override

        Returns:
            Advantages tensor [N]
        """
        num_samples = kwargs.get("num_samples_per_prompt", self.num_samples_per_prompt)

        # If only 1 sample per prompt, fall back to global
        if num_samples <= 1:
            return normalize_global(
                rewards=rewards,
                epsilon=self.config.epsilon,
                clip_max=self.config.clip_max,
            )

        group_indices = build_fixed_size_groups(len(rewards), num_samples)
        return normalize_grouped(
            rewards=rewards,
            group_indices=group_indices,
            epsilon=self.config.epsilon,
            clip_max=self.config.clip_max,
            trimmed_ratio=self.config.trimmed_ratio,
            use_global_std=self.config.global_std,
        )


class PerPromptStrategy:
    """Per-prompt normalization strategy.

    Normalizes rewards within groups of samples that share the same prompt.
    """

    def __init__(self, config: Optional[AdvantageConfig] = None):
        self.config = config or AdvantageConfig()

    def compute(
        self,
        rewards: torch.Tensor,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute advantages using per-prompt statistics.

        Args:
            rewards: Reward values [N]
            prompts: List of prompts [N]
            **kwargs: Ignored

        Returns:
            Advantages tensor [N]
        """
        if prompts is None:
            # Fall back to global normalization if no prompts provided
            return normalize_global(
                rewards=rewards,
                epsilon=self.config.epsilon,
                clip_max=self.config.clip_max,
            )

        group_indices = build_prompt_groups(prompts)
        return normalize_grouped(
            rewards=rewards,
            group_indices=group_indices,
            epsilon=self.config.epsilon,
            clip_max=self.config.clip_max,
            trimmed_ratio=self.config.trimmed_ratio,
            use_global_std=self.config.global_std,
        )


# Strategy registry for dynamic instantiation
STRATEGIES: Dict[str, Type] = {
    "global": GlobalStrategy,
    "group": GroupStrategy,
    "per_prompt": PerPromptStrategy,
}


def get_strategy(
    strategy_type: str,
    num_samples_per_prompt: int = 1,
    config: Optional[AdvantageConfig] = None,
):
    """Get a strategy instance by type.

    Args:
        strategy_type: One of "global", "group", "per_prompt"
        num_samples_per_prompt: For group strategy
        config: Optional configuration

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy_type is unknown
    """
    if strategy_type not in STRATEGIES:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    strategy_cls = STRATEGIES[strategy_type]

    if strategy_type == "group":
        return strategy_cls(num_samples_per_prompt=num_samples_per_prompt, config=config)
    else:
        return strategy_cls(config=config)
