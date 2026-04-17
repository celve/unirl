"""Algorithm-family registration and resolution helpers."""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any

from diffusionrl.registry import (
    derive_registry_or_dotpath,
    register_component,
    require_subclass,
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
    importlib.import_module("diffusionrl.algorithms.nft")


def derive_algorithm_class(identifier: str) -> Any:
    return derive_registry_or_dotpath(
        component_family=ALGORITHM_COMPONENT_FAMILY,
        identifier=identifier,
        class_checker=require_subclass(BaseAlgorithm),
    )


def derive_algorithm_dotpath(identifier: str) -> str:
    algorithm_cls = derive_algorithm_class(identifier)
    return f"{algorithm_cls.__module__}.{algorithm_cls.__qualname__}"


__all__ = [
    "ALGORITHM_COMPONENT_FAMILY",
    "ensure_builtin_algorithm_registration",
    "register_algorithm",
    "derive_algorithm_class",
    "derive_algorithm_dotpath",
]
