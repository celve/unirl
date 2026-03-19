"""Reward plugin examples.

Use with:
    --reward-path diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer
"""

from .minimal_reward import MinimalRewardScorer

__all__ = ["MinimalRewardScorer"]
