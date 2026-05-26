"""Canonical Python package for this repository (`diffusionrl`)."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "0.1.0"

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # shared types
    "RewardRequest": ("diffusionrl.types", "RewardRequest"),
    "RewardResponse": ("diffusionrl.types", "RewardResponse"),
    "RewardType": ("diffusionrl.types", "RewardType"),
    "SamplingRequirements": ("diffusionrl.types.sampling", "SamplingRequirements"),
    # sde
    "get_sigma_schedule": ("diffusionrl.sde", "get_sigma_schedule"),
    # reward
    "BaseRewardScorer": ("diffusionrl.reward.base", "BaseRewardScorer"),
    # utils
    "load_function": ("diffusionrl.utils", "load_function"),
    "set_seed": ("diffusionrl.utils", "set_seed"),
    "configure_logger": ("diffusionrl.utils", "configure_logger"),
}

__all__ = ["__version__", *_LAZY_ATTRS.keys()]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
