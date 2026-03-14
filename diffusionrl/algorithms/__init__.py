"""diffusionrl Algorithms Module.

The algorithms module is the single source of truth for both rollout-side
requirements (sampling, advantages) and training-side gradient computation.
"""
from typing import Any, Optional

from .base import BaseAlgorithm, SamplingRequirements
from .grpo import GRPOAlgorithm
from .mix_grpo import MixGRPOAlgorithm
from .nft import NFTAlgorithm

# Default dotpaths for built-in algorithm classes.
# Used by get_algorithm() and _normalize_loss_path() for resolution.
DEFAULT_ALGORITHM_PATHS = {
    "grpo": "diffusionrl.algorithms.grpo.GRPOAlgorithm",
    "nft": "diffusionrl.algorithms.nft.NFTAlgorithm",
    "mix_grpo": "diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm",
}


def get_algorithm(
    algorithm_type: str = "grpo",
    algorithm_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create an algorithm instance by dotpath or built-in type name.

    For normal training, the algorithm is created via
    ``load_function(algorithm_path) + cls.from_config(loss_config)``
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
