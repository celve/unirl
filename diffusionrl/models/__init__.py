"""diffusionrl Models Module."""

from .base import ModelBundle
from .config import ModelBundleConfig

__all__ = [
    "ModelBundle",
    "ModelBundleConfig",
    # Lazy import getters
    "get_hunyuan_video_model_bundle",
    "get_hunyuan_veido1p5_model_bundle",
    "get_mochi_model_bundle",
    "get_flux_model_bundle",
    "get_sd3_model_bundle",
    "get_wan_model_bundle",
    "get_wan22_model_bundle",
]


# Lazy imports for model implementations
def get_hunyuan_video_model_bundle():
    """Get HunyuanVideo (text-to-video) model bundle class."""
    from .hunyuan_video import HunyuanVideoModelBundle

    return HunyuanVideoModelBundle


def get_hunyuan_veido1p5_model_bundle():
    """Get HunyuanVideo-1.5 model bundle class."""
    from .hunyuan_veido1p5 import HunyuanVeido1p5ModelBundle

    return HunyuanVeido1p5ModelBundle


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
    from .wan21 import WAN21ModelBundle

    return WAN21ModelBundle


def get_wan22_model_bundle():
    """Get Wan2.2 dual-transformer video model bundle class."""
    from .wan22 import WAN22ModelBundle

    return WAN22ModelBundle
