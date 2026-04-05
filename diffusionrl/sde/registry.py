"""Shared registry helpers for SDE step strategies."""

from __future__ import annotations

from diffusionrl.registry import (
    register_component,
    require_subclass,
    resolve_registry_or_dotpath,
)
from diffusionrl.sde.rules import normalize_sde_type

SDE_STRATEGY_COMPONENT_FAMILY = "sde_strategy"


def _require_sde_strategy_subclass(component_cls) -> None:
    # Import lazily here to avoid the kernels.py <-> registry.py cycle.
    from diffusionrl.sde.kernels import StepStrategy

    require_subclass(StepStrategy)(component_cls)


def register_sde_strategy(*component_names: str):
    """Register an SDE strategy implementation under one or more names."""

    def decorator(component_cls):
        for component_name in component_names:
            register_component(
                component_family=SDE_STRATEGY_COMPONENT_FAMILY,
                component_name=component_name,
                class_checker=_require_sde_strategy_subclass,
            )(component_cls)
        return component_cls

    return decorator


def resolve_sde_strategy_class(identifier: str) -> Any:
    """Resolve an SDE strategy class by registered name or full dot path."""
    if "." not in identifier:
        normalized_identifier = normalize_sde_type(identifier)
    else:
        normalized_identifier = identifier
    return resolve_registry_or_dotpath(
        component_family=SDE_STRATEGY_COMPONENT_FAMILY,
        identifier=normalized_identifier,
        class_checker=_require_sde_strategy_subclass,
    )


__all__ = [
    "SDE_STRATEGY_COMPONENT_FAMILY",
    "register_sde_strategy",
    "resolve_sde_strategy_class",
]
