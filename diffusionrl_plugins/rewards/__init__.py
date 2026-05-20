"""Reward plugin examples.

Reference from a Hydra config via ``_target_``::

    reward:
      provider:
        _target_: diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer
"""

from .minimal_reward import MinimalRewardScorer

__all__ = ["MinimalRewardScorer"]
