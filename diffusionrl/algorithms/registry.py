"""Algorithm-family registration and resolution helpers."""

from __future__ import annotations

import importlib
from functools import partial

from diffusionrl.registry import (
    register_component,
    require_subclass,
    resolve_registry_or_dotpath,
)

from .base import BaseAlgorithm

ALGORITHM_COMPONENT_FAMILY = "algorithm"


register_algorithm = partial(
    register_component,
    component_family=ALGORITHM_COMPONENT_FAMILY,
    class_checker=require_subclass(BaseAlgorithm),
)


def ensure_builtin_algorithm_registration() -> None:
    """Import built-in algorithm modules so decorator-based registration runs."""
    importlib.import_module("diffusionrl.algorithms.grpo")
    importlib.import_module("diffusionrl.algorithms.mix_grpo")
    importlib.import_module("diffusionrl.algorithms.nft")


def resolve_algorithm_class(identifier: str) -> Any:
    return resolve_registry_or_dotpath(
        component_family=ALGORITHM_COMPONENT_FAMILY,
        identifier=identifier,
        class_checker=require_subclass(BaseAlgorithm),
    )


__all__ = [
    "ALGORITHM_COMPONENT_FAMILY",
    "ensure_builtin_algorithm_registration",
    "register_algorithm",
    "resolve_algorithm_class",
]
