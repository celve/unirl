"""Backward-compatibility shim — import GRPOAlgorithm as GRPOLoss.

.. deprecated::
    Use ``from diffusionrl.algorithms.grpo import GRPOAlgorithm`` instead.
"""

import warnings

warnings.warn(
    "diffusionrl.losses.grpo_loss is deprecated. "
    "Import GRPOAlgorithm from diffusionrl.algorithms.grpo instead.",
    DeprecationWarning,
    stacklevel=2,
)

from diffusionrl.algorithms.grpo import GRPOAlgorithm as GRPOLoss  # noqa: F401
from diffusionrl.algorithms.grpo import _save_training_debug_tensor  # noqa: F401

__all__ = ["GRPOLoss", "_save_training_debug_tensor"]
