"""diffusionrl Algorithms Module.

The algorithms module is the single source of truth for both rollout-side
requirements (sampling, advantages) and training-side gradient computation.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple
from diffusionrl.types.sampling import SamplingRequirements

from .base import BaseAlgorithm
from .construction import create_algorithm_from_init_payload
from .registry import ensure_builtin_algorithm_registration

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "GRPOAlgorithm": ("diffusionrl.algorithms.grpo", "GRPOAlgorithm"),
    "NFTAlgorithm": ("diffusionrl.algorithms.nft", "NFTAlgorithm"),
}

ensure_builtin_algorithm_registration()


__all__ = [
    "BaseAlgorithm",
    "GRPOAlgorithm",
    "NFTAlgorithm",
    "create_algorithm_from_init_payload",
]


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
