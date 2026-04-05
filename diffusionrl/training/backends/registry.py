"""Shared registry helpers for train backend implementations."""

from __future__ import annotations

from functools import partial
from typing import Any

from diffusionrl.registry import (
    list_registered_component_names,
    register_component,
    require_subclass,
    resolve_registry_or_dotpath,
)
from diffusionrl.training.backends.base import TrainBackend

TRAIN_BACKEND_COMPONENT_FAMILY = "train_backend"

register_train_backend = partial(
    register_component,
    component_family=TRAIN_BACKEND_COMPONENT_FAMILY,
    class_checker=require_subclass(TrainBackend),
)


def ensure_builtin_train_backend_registration() -> None:
    import diffusionrl.training.backends.fsdp  # noqa: F401
    import diffusionrl.training.backends.megatron  # noqa: F401
    import diffusionrl.training.backends.veomni  # noqa: F401


supported_train_backends = partial(
    list_registered_component_names,
    component_family=TRAIN_BACKEND_COMPONENT_FAMILY,
)


def resolve_train_backend_class(identifier: str) -> Any:
    return resolve_registry_or_dotpath(
        component_family=TRAIN_BACKEND_COMPONENT_FAMILY,
        identifier=identifier,
        class_checker=require_subclass(TrainBackend),
    )


__all__ = [
    "TRAIN_BACKEND_COMPONENT_FAMILY",
    "ensure_builtin_train_backend_registration",
    "register_train_backend",
    "resolve_train_backend_class",
    "supported_train_backends",
]
