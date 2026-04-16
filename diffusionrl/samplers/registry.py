"""Rollout-engine family registration and resolution helpers."""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any

from diffusionrl.registry import (
    derive_registry_or_dotpath,
    register_component,
    require_subclass,
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


def derive_rollout_engine_class(identifier: str) -> Any:
    return derive_registry_or_dotpath(
        component_family=ROLLOUT_ENGINE_COMPONENT_FAMILY,
        identifier=identifier,
        class_checker=require_subclass(BaseRolloutEngine),
    )


__all__ = [
    "derive_rollout_engine_class",
    "ROLLOUT_ENGINE_COMPONENT_FAMILY",
    "ensure_builtin_rollout_engine_registration",
    "register_rollout_engine",
]
