"""Loss functions for diffusionrl training."""

from typing import Any, Optional

from .grpo_loss import GRPOLoss

try:
    from .nft_loss import NFTLoss
except ImportError:
    NFTLoss = None  # type: ignore[assignment]

# Default dotpaths for built-in loss classes.
# Used by get_loss() when no explicit loss_path is provided.
DEFAULT_LOSS_PATHS = {
    "grpo": "diffusionrl.losses.grpo_loss.GRPOLoss",
    "nft": "diffusionrl.losses.nft_loss.NFTLoss",
}


def get_loss(
    loss_type: str = "grpo",
    loss_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a loss instance by dotpath or built-in type name.

    For normal training, loss is created via
    ``load_function(loss_path) + cls.from_config(loss_config)``
    inside TrainingActor. This factory is a convenience for
    standalone / testing use.

    Args:
        loss_type: Built-in type name ("grpo", "nft").  Ignored when
            *loss_path* is provided.
        loss_path: Explicit Python dotpath to a loss class
            (e.g. ``"mypackage.MyLoss"``).  Overrides *loss_type*.
        **kwargs: Forwarded to the loss constructor.

    Returns:
        Loss instance.
    """
    from diffusionrl.utils import load_function

    path = loss_path or DEFAULT_LOSS_PATHS.get(loss_type)
    if path is None:
        raise ValueError(
            f"Unknown loss_type: {loss_type!r}. "
            f"Available: {sorted(DEFAULT_LOSS_PATHS)}. "
            f"Or provide a loss_path for custom losses."
        )
    loss_cls = load_function(path)
    return loss_cls(**kwargs)


__all__ = [
    "GRPOLoss",
    "NFTLoss",
    "DEFAULT_LOSS_PATHS",
    "get_loss",
]
