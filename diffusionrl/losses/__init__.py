"""
Loss functions for GRPO training.

Provides:
- GRPOLoss: Standard GRPO with PPO-style clipping
- NFTLoss: Forward process loss (DiffusionNFT)
- LOSS_REGISTRY: Registry for dynamic loss selection
- get_loss(): Factory function for creating loss instances
"""

from typing import Any, Dict, Optional, Type, Union

from .grpo_loss import GRPOLoss
from .nft_loss import NFTLoss

# Loss registry for parameter-driven selection
LOSS_REGISTRY: Dict[str, type] = {
    "grpo": GRPOLoss,
    "nft": NFTLoss,
}


def get_loss(
    loss_type: str,
    loss_path: Optional[str] = None,
    **kwargs: Any,
) -> Union[GRPOLoss, NFTLoss]:
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
