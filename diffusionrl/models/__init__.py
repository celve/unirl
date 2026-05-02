"""diffusionrl Models Module."""

from .base import ModelBundle
from .config import ModelBundleConfig

__all__ = [
    "ModelBundle",
    "ModelBundleConfig",
    # Lazy import getters
    "get_hunyuan_model_bundle",
    "get_mochi_model_bundle",
    "get_flux_model_bundle",
    "get_sd3_model_bundle",
    "get_wan_model_bundle",
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


def get_wan_model_bundle():
    """Get Wan video model bundle class."""
    from .wan import WAN21ModelBundle

    return WAN21ModelBundle
