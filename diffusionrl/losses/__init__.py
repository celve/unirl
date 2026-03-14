"""Loss functions for diffusionrl training — backward-compatibility shim.

.. deprecated::
    The ``diffusionrl.losses`` module is deprecated.  Loss computation has been
    merged into the algorithm classes in ``diffusionrl.algorithms``.  Import
    ``GRPOAlgorithm`` / ``NFTAlgorithm`` directly from ``diffusionrl.algorithms``
    instead.  The old names ``GRPOLoss`` / ``NFTLoss`` are re-exported here for
    backward compatibility and will be removed in a future release.
"""

import warnings
from typing import Any, Optional

from diffusionrl.algorithms.grpo import GRPOAlgorithm as GRPOLoss
from diffusionrl.algorithms import DEFAULT_ALGORITHM_PATHS

try:
    from diffusionrl.algorithms.nft import NFTAlgorithm as NFTLoss
except ImportError:
    NFTLoss = None  # type: ignore[assignment]

# Legacy mapping — now points to algorithm classes
DEFAULT_LOSS_PATHS = dict(DEFAULT_ALGORITHM_PATHS)


def get_loss(
    loss_type: str = "grpo",
    loss_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a loss instance by dotpath or built-in type name.

    .. deprecated::
        Use ``diffusionrl.algorithms.get_algorithm()`` instead.
    """
    warnings.warn(
        "diffusionrl.losses.get_loss() is deprecated. "
        "Use diffusionrl.algorithms.get_algorithm() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from diffusionrl.algorithms import get_algorithm
    return get_algorithm(algorithm_type=loss_type, algorithm_path=loss_path, **kwargs)


__all__ = [
    "GRPOLoss",
    "NFTLoss",
    "DEFAULT_LOSS_PATHS",
    "get_loss",
]
