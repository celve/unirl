"""diffusionrl Models Module."""
from .base import ModelBundle
from .registry import (
    discover_model_bundle_paths,
    resolve_model_bundle_path,
    list_model_types,
)
from .forward_plugins import (
    ModelForwardPlugin,
    BaseForwardPlugin,
    FluxForwardPlugin,
    SD3ForwardPlugin,
    HunyuanForwardPlugin,
    MochiForwardPlugin,
    DefaultForwardPlugin,
)

__all__ = [
    "ModelBundle",
    "discover_model_bundle_paths",
    "resolve_model_bundle_path",
    "list_model_types",
    # Forward plugins
    "ModelForwardPlugin",
    "BaseForwardPlugin",
    "FluxForwardPlugin",
    "SD3ForwardPlugin",
    "HunyuanForwardPlugin",
    "MochiForwardPlugin",
    "DefaultForwardPlugin",
    # Lazy import getters
    "get_hunyuan_model_bundle",
    "get_mochi_model_bundle",
    "get_flux_model_bundle",
    "get_sd3_model_bundle",
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
