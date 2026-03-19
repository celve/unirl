"""diffusionrl Algorithms Module.

The algorithms module is the single source of truth for both rollout-side
requirements (sampling, advantages) and training-side gradient computation.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

from diffusionrl.types.sampling import SamplingRequirements

from .base import BaseAlgorithm
from .registry import DEFAULT_ALGORITHM_PATHS

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "GRPOAlgorithm": ("diffusionrl.algorithms.grpo", "GRPOAlgorithm"),
    "MixGRPOAlgorithm": ("diffusionrl.algorithms.mix_grpo", "MixGRPOAlgorithm"),
    "NFTAlgorithm": ("diffusionrl.algorithms.nft", "NFTAlgorithm"),
}


def get_algorithm(
    algorithm_type: str = "grpo",
    algorithm_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create an algorithm instance by dotpath or built-in type name.

    For normal training, the algorithm is created via
    ``load_function(algorithm_path) + cls.from_config(algorithm_config)``
    inside TrainingActor.  This factory is a convenience for
    standalone / testing use.

    Args:
        algorithm_type: Built-in type name ("grpo", "nft", "mix_grpo").
            Ignored when *algorithm_path* is provided.
        algorithm_path: Explicit Python dotpath to an algorithm class.
            Overrides *algorithm_type*.
        **kwargs: Forwarded to the algorithm constructor.

    Returns:
        Algorithm instance.
    """
    from diffusionrl.utils import load_function

    path = algorithm_path or DEFAULT_ALGORITHM_PATHS.get(algorithm_type)
    if path is None:
        raise ValueError(
            f"Unknown algorithm_type: {algorithm_type!r}. "
            f"Available: {sorted(DEFAULT_ALGORITHM_PATHS)}. "
            f"Or provide an algorithm_path for custom algorithms."
        )
    algorithm_cls = load_function(path)
    return algorithm_cls(**kwargs)


__all__ = [
    "BaseAlgorithm",
    "SamplingRequirements",
    "GRPOAlgorithm",
    "MixGRPOAlgorithm",
    "NFTAlgorithm",
    "DEFAULT_ALGORITHM_PATHS",
    "get_algorithm",
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
