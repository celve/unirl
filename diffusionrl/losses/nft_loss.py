"""Backward-compatibility shim — import NFTAlgorithm as NFTLoss.

.. deprecated::
    Use ``from diffusionrl.algorithms.nft import NFTAlgorithm`` instead.
"""

import warnings

warnings.warn(
    "diffusionrl.losses.nft_loss is deprecated. "
    "Import NFTAlgorithm from diffusionrl.algorithms.nft instead.",
    DeprecationWarning,
    stacklevel=2,
)

from diffusionrl.algorithms.nft import NFTAlgorithm as NFTLoss  # noqa: F401

__all__ = ["NFTLoss"]
