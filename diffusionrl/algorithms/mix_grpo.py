"""Compatibility alias for GRPO configured with custom index schedulers."""

from .grpo import GRPOAlgorithm


class MixGRPOAlgorithm(GRPOAlgorithm):
    """Backward-compatible alias.

    MixGRPO behavior is now expressed through GRPO's separate
    ``rollout_scheduler`` and ``training_scheduler`` configs.
    """

    pass
