"""Reward plugin examples.

Use with:
    --reward-path diffusionrl_plugins.rewards.minimal_reward.MinimalRewardWorker
"""

from .minimal_reward import MinimalRewardWorker

__all__ = ["MinimalRewardWorker"]
