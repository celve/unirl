"""diffusionrl Models Module."""
from .base import ModelBundle
from .config import ModelBundleConfig
from .construction import create_model_bundle_from_init_payload
from .forward_plugins import (
    BaseForwardPlugin,
    DefaultForwardPlugin,
    FluxForwardPlugin,
    HunyuanForwardPlugin,
    MochiForwardPlugin,
    ModelForwardPlugin,
    SD3ForwardPlugin,
)
from .registry import (
    discover_model_bundle_paths,
    ensure_builtin_model_registration,
    list_model_types,
    register_model,
    derive_model_bundle_path,
    resolve_model_class,
)

__all__ = [
    "ModelBundle",
    "ModelBundleConfig",
    "create_model_bundle_from_init_payload",
    "discover_model_bundle_paths",
    "ensure_builtin_model_registration",
    "derive_model_bundle_path",
    "resolve_model_class",
    "register_model",
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

ensure_builtin_model_registration()


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
