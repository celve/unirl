"""Loss plugin examples.

Use with:
    --loss-type custom
    --loss-path diffusionrl_plugins.losses.minimal_loss.MinimalBackwardLoss
"""

from .minimal_loss import MinimalBackwardLoss, MinimalForwardLoss

__all__ = ["MinimalBackwardLoss", "MinimalForwardLoss"]

