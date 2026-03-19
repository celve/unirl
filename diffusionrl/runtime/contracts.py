"""Runtime capability and algorithm-contract resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from diffusionrl.config.build_domain_args import build_algorithm_config
from diffusionrl.samplers.engine import get_engine_class_path
from diffusionrl.types.sampling import SamplingRequirements
from diffusionrl.utils.misc import load_function


def _instantiate_algorithm_for_contracts(args: Any) -> Any:
    """Instantiate the algorithm so contracts are read from one runtime surface."""
    algorithm_config = build_algorithm_config(args)
    algorithm_path = algorithm_config.get("algorithm_path")
    if not isinstance(algorithm_path, str) or not algorithm_path.strip():
        raise ValueError("build_algorithm_config() returned an empty algorithm_path.")
    try:
        algorithm_cls = load_function(algorithm_path.strip())
    except Exception as exc:
        raise ValueError(
            "Cannot resolve algorithm class from args.algorithm.algorithm_path="
            f"{algorithm_path!r}."
        ) from exc
    if not hasattr(algorithm_cls, "from_config"):
        raise ValueError(
            f"Algorithm class {algorithm_cls.__name__} must define classmethod from_config(config)."
        )
    return algorithm_cls.from_config(algorithm_config)


def resolve_sampling_requirements(
    args: Any,
    *,
    algorithm: Optional[Any] = None,
) -> SamplingRequirements:
    """Resolve final sampling contract from `algorithm.get_sampling_requirements()`."""
    resolved_algorithm = algorithm if algorithm is not None else _instantiate_algorithm_for_contracts(args)
    requirements = resolved_algorithm.get_sampling_requirements()
    raw_extras = getattr(requirements, "extras", None)
    extras: Dict[str, Any] = dict(raw_extras) if isinstance(raw_extras, Mapping) else {}

    return SamplingRequirements(
        requires_trajectory=bool(getattr(requirements, "requires_trajectory", True)),
        requires_log_prob=bool(getattr(requirements, "requires_log_prob", True)),
        requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
        extras=extras,
    )


def resolve_engine_capabilities(*, engine_type: str) -> Dict[str, bool]:
    """Resolve engine capabilities from engine class declaration."""
    engine_path = get_engine_class_path(engine_type)
    engine_cls = load_function(engine_path)
    declared = getattr(engine_cls, "declared_capabilities", None)
    if not callable(declared):
        raise ValueError(
            f"Engine class {engine_path} must define classmethod declared_capabilities()."
        )
    return dict(declared())


__all__ = [
    "resolve_sampling_requirements",
    "resolve_engine_capabilities",
]
