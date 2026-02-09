"""diffusionrl Models Module."""
from typing import Callable, Dict, List, Type

from .base import ModelBundle
from .forward_plugins import (
    ModelForwardPlugin,
    BaseForwardPlugin,
    FluxForwardPlugin,
    SD3ForwardPlugin,
    HunyuanForwardPlugin,
    DefaultForwardPlugin,
    get_forward_plugin,
    detect_model_type,
    PLUGIN_REGISTRY,
)

__all__ = [
    "ModelBundle",
    # Forward plugins
    "ModelForwardPlugin",
    "BaseForwardPlugin",
    "FluxForwardPlugin",
    "SD3ForwardPlugin",
    "HunyuanForwardPlugin",
    "DefaultForwardPlugin",
    "get_forward_plugin",
    "detect_model_type",
    "PLUGIN_REGISTRY",
    # Lazy import getters
    "get_hunyuan_model_bundle",
    "get_mochi_model_bundle",
    "get_flux_model_bundle",
    "get_sd3_model_bundle",
    # Model bundle registry
    "register_model_bundle",
    "list_model_bundle_types",
    "get_model_bundle_class",
]


# Lazy imports for model implementations
def get_hunyuan_model_bundle():
    """Get HunyuanVideo model bundle class."""
    from .hunyuan import HunyuanModelBundle
    return HunyuanModelBundle


def get_mochi_model_bundle():
    """Get Mochi video model bundle class."""
    from .mochi import MochiModelBundle
    return MochiModelBundle


def get_flux_model_bundle():
    """Get FLUX image model bundle class."""
    from .flux import FluxModelBundle
    return FluxModelBundle


def get_sd3_model_bundle():
    """Get SD3 image model bundle class."""
    from .sd3 import SD3ModelBundle
    return SD3ModelBundle


ModelBundleGetter = Callable[[], Type[ModelBundle]]

_MODEL_BUNDLE_GETTERS: Dict[str, ModelBundleGetter] = {
    "hunyuan": get_hunyuan_model_bundle,
    "mochi": get_mochi_model_bundle,
    "flux": get_flux_model_bundle,
    "sd3": get_sd3_model_bundle,
}


def register_model_bundle(
    model_type: str,
    getter_fn: ModelBundleGetter,
    overwrite: bool = False,
) -> None:
    """Register a model bundle getter for runtime extension."""
    key = model_type.lower()
    if not overwrite and key in _MODEL_BUNDLE_GETTERS:
        raise ValueError(f"Model bundle type already registered: {key}")
    _MODEL_BUNDLE_GETTERS[key] = getter_fn


def list_model_bundle_types() -> List[str]:
    """List available model bundle types."""
    return sorted(_MODEL_BUNDLE_GETTERS.keys())


def get_model_bundle_class(model_type: str):
    """
    Get model bundle class by type.

    Args:
        model_type: Registered model type.

    Returns:
        ModelBundle subclass
    """
    key = model_type.lower()
    getter = _MODEL_BUNDLE_GETTERS.get(key)
    if getter is None:
        available = ", ".join(list_model_bundle_types())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    return getter()
