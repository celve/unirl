"""Loss functions for diffusionrl training."""

from typing import Any, Dict, Optional

from .grpo_loss import GRPOLoss

try:
    from .nft_loss import NFTLoss
except ImportError as _nft_import_exc:
    NFTLoss = None  # type: ignore[assignment]
else:
    _nft_import_exc = None

# Loss registry for parameter-driven selection
LOSS_REGISTRY: Dict[str, type] = {
    "grpo": GRPOLoss,
}
if NFTLoss is not None:
    LOSS_REGISTRY["nft"] = NFTLoss


def get_loss(
    loss_type: str,
    loss_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Factory function for creating loss instances.

    Supports two modes:
    1. Parameter-driven (loss_type): Uses LOSS_REGISTRY
    2. Dynamic loading (loss_path): Loads class from path

    Args:
        loss_type: Type of loss ("grpo", "nft")
        loss_path: Optional path for custom loss class (e.g., "mymodule.MyLoss")
        **kwargs: Arguments passed to loss constructor

    Returns:
        Loss instance

    Example:
        # Parameter-driven
        loss = get_loss("grpo", clip_range=1e-4, kl_coef=0.01)

        # Dynamic loading
        loss = get_loss("custom", loss_path="my_losses.CustomLoss", beta=0.2)
    """
    # Dynamic loading if path provided
    if loss_path is not None:
        from diffusionrl.utils import load_class
        loss_cls = load_class(loss_path, base_class=None)
        return loss_cls(**kwargs)

    # Parameter-driven selection
    if loss_type in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_type](**kwargs)

    if loss_type == "nft" and NFTLoss is None:
        raise ImportError(
            "NFTLoss is unavailable because optional dependencies failed to import. "
            f"Original error: {_nft_import_exc}"
        )

    raise ValueError(
        f"Unknown loss_type: {loss_type}. "
        f"Available: {list(LOSS_REGISTRY.keys())}. "
        f"Or provide a loss_path for custom losses."
    )


def register_loss(name: str, loss_cls: type) -> None:
    """
    Register a custom loss class.

    Args:
        name: Name for the loss in registry
        loss_cls: Loss class to register

    Example:
        from diffusionrl.losses import register_loss

        class MyLoss:
            ...

        register_loss("my_loss", MyLoss)
    """
    LOSS_REGISTRY[name] = loss_cls


__all__ = [
    # Implementations
    "GRPOLoss",
    "NFTLoss",
    # Registry
    "LOSS_REGISTRY",
    "get_loss",
    "register_loss",
]
