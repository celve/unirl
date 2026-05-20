"""
diffusionrl_plugins — external / third-party extension namespace.

This package is a plain Python namespace. There is no auto-discovery or
auto-registration. Third-party extensions live in their own modules and
are wired in via Hydra ``_target_:`` strings in experiment YAML.

A working example is shipped at
``diffusionrl_plugins/rewards/minimal_reward.py`` — a ``BaseRewardScorer``
subclass that you can reference from a reward config block::

    reward:
      provider:
        _target_: diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer

For shared data types (``RewardRequest``, ``RewardResponse``, segment
shapes, etc.), import from ``diffusionrl.types``.
"""
