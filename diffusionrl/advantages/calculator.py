"""
AdvantageCalculator - unified interface for advantage computation.

.. deprecated::
    This module is deprecated. Advantage computation should be handled by
    algorithm.compute_advantages() in diffusionrl/algorithms/base.py.
    The BaseAlgorithm.compute_advantages() method provides the same functionality
    with better integration into the algorithm abstraction.

Provides a single entry point for computing advantages, used by TrainerEngine
and TrainingStep implementations.

Copied from unified_grpo/advantages/calculator.py
"""
import warnings

from typing import Dict, List, Optional

import torch

from .strategies import AdvantageConfig, get_strategy
from .utils import gather_rewards_across_gpus


class AdvantageCalculator:
    """Unified advantage calculator for TrainerEngine.

    .. deprecated::
        Use algorithm.compute_advantages() instead. This class is kept for
        backward compatibility with unified_grpo but should not be used in
        new code. diffusionrl algorithms compute advantages directly in the
        RolloutManager via algorithm.compute_advantages().

    Wraps strategy selection and optional distributed reward gathering.

    Args:
        advantage_type: One of "group", "global", "per_prompt"
        num_samples_per_prompt: Number of samples per prompt (for group strategy)
        epsilon: Small value for numerical stability
        clip_max: If set, clip advantages to [-clip_max, clip_max]
        trimmed_ratio: Ratio of samples to trim from each end
        global_std: If True, use global std for normalization
        gather_rewards_across_gpus: If True, gather rewards from all GPUs before computing
    """

    def __init__(
        self,
        advantage_type: str = "group",
        num_samples_per_prompt: int = 1,
        epsilon: float = 1e-8,
        clip_max: float = 5.0,
        trimmed_ratio: float = 0.0,
        global_std: bool = False,
        gather_rewards_across_gpus: bool = False,
    ):
        self.advantage_type = advantage_type
        self.num_samples_per_prompt = num_samples_per_prompt
        self.gather_rewards = gather_rewards_across_gpus

        # Create config and strategy
        self.config = AdvantageConfig(
            epsilon=epsilon,
            clip_max=clip_max,
            trimmed_ratio=trimmed_ratio,
            global_std=global_std,
        )
        self.strategy = get_strategy(
            strategy_type=advantage_type,
            num_samples_per_prompt=num_samples_per_prompt,
            config=self.config,
        )

    def compute(
        self,
        rewards: torch.Tensor,
        prompts: Optional[List[str]] = None,
        num_samples_per_prompt: Optional[int] = None,
        reward_dict: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute advantages from rewards.

        Args:
            rewards: Reward tensor [N]
            prompts: Optional list of prompts (for per_prompt strategy)
            num_samples_per_prompt: Override for num_samples_per_prompt
            reward_dict: Optional multi-reward dictionary (ignored, kept for compat)
            **kwargs: Additional arguments passed to strategy

        Returns:
            Advantages tensor [N]
        """
        # Optionally gather rewards from all GPUs
        if self.gather_rewards:
            rewards = gather_rewards_across_gpus(rewards)

        # Prepare kwargs for strategy
        strategy_kwargs = dict(kwargs)
        if prompts is not None:
            strategy_kwargs["prompts"] = prompts
        if num_samples_per_prompt is not None:
            strategy_kwargs["num_samples_per_prompt"] = num_samples_per_prompt

        return self.strategy.compute(rewards, **strategy_kwargs)


def build_advantage_calculator(config) -> AdvantageCalculator:
    """Factory function to create an AdvantageCalculator from config.

    Args:
        config: Training configuration with advantage settings

    Returns:
        Configured AdvantageCalculator instance
    """
    return AdvantageCalculator(
        advantage_type=getattr(config, "advantage_type", "group"),
        num_samples_per_prompt=getattr(config, "num_decodes_per_prompt", 1),
        epsilon=getattr(config, "advantage_epsilon", 1e-8),
        clip_max=getattr(config, "adv_clip_max", 5.0),
        trimmed_ratio=getattr(config, "trimmed_ratio", 0.0),
        global_std=getattr(config, "global_std", False),
        gather_rewards_across_gpus=getattr(config, "gather_rewards_across_gpus", False),
    )
