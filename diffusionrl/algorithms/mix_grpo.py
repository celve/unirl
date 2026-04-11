"""Compatibility alias for GRPO configured with custom index schedulers."""

from diffusionrl.algorithms.registry import register_algorithm

from .grpo import GRPOAlgorithm, GRPOAlgorithmConfig


@register_algorithm(component_name="mix_grpo", component_cfg=GRPOAlgorithmConfig)
class MixGRPOAlgorithm(GRPOAlgorithm):
    """Backward-compatible alias.

    MixGRPO behavior is now expressed through GRPO's separate
    ``rollout_scheduler`` and ``training_scheduler`` configs.
    """
    pass
