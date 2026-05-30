"""Reward subsystem entrypoint.

This package holds typed reward component specs and their runtime
constructors. There is no package-root re-export surface — import from the
appropriate submodule directly:

- ``diffusionrl.reward.base`` — ``RewardBackend`` + ``BaseRewardComponentSpec``
- ``diffusionrl.reward.service`` — ``RewardService`` (holds one backend)
- ``diffusionrl.reward.remote`` — ``RemoteRewardBackend`` (remote backend)
- ``diffusionrl.reward.local.<name>`` — per-scorer ``<Name>RewardScorer`` + ``<Name>Spec`` (local backends)
"""
