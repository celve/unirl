"""Reward subsystem entrypoint.

This package holds typed reward component specs and their runtime
constructors. There is no package-root re-export surface — import from the
appropriate submodule directly:

- ``diffusionrl.reward.base`` — abstract bases (``BaseRewardComponentSpec``,
  ``BaseRewardScorer``, ``BaseRewardExecutor``, ``InProcessRewardExecutor``)
- ``diffusionrl.reward.config`` — ``RewardConfig``
- ``diffusionrl.reward.service`` — ``RewardService``
- ``diffusionrl.reward.pipeline`` — ``RewardPipeline``
- ``diffusionrl.reward.scorers.<name>`` — per-scorer ``<Name>RewardScorer`` + ``<Name>Spec``
"""
