"""Rollout-engine family registration and resolution helpers."""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any

from diffusionrl.registry import (
    register_component,
    require_subclass,
    resolve_registry_or_dotpath,
)
from diffusionrl.samplers.engine import BaseRolloutEngine

ROLLOUT_ENGINE_COMPONENT_FAMILY = "rollout_engine"

register_rollout_engine = partial(
    register_component,
    component_family=ROLLOUT_ENGINE_COMPONENT_FAMILY,
    class_checker=require_subclass(BaseRolloutEngine),
)


def ensure_builtin_rollout_engine_registration() -> None:
    """Import built-in rollout-engine modules so decorator-based registration runs."""
    importlib.import_module("diffusionrl.samplers.sglang.engine")


def resolve_rollout_engine_class(identifier: str) -> Any:
    return resolve_registry_or_dotpath(
        component_family=ROLLOUT_ENGINE_COMPONENT_FAMILY,
        identifier=identifier,
        class_checker=require_subclass(BaseRolloutEngine),
    )


__all__ = [
    "ROLLOUT_ENGINE_COMPONENT_FAMILY",
    "ensure_builtin_rollout_engine_registration",
    "register_rollout_engine",
    "resolve_rollout_engine_class",
]
