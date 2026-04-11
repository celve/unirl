"""Shared component registration and resolution keyed by component family."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, TypeVar

from diffusionrl.utils.misc import load_function

T = TypeVar("T")

COMPONENT_REGISTRY: Dict[str, Dict[str, Any]] = defaultdict(dict)
_COMPONENT_CONFIG_ATTR = "__CONFIG_CLASS__"

def require_subclass(base_class: type) -> Callable[[Any], None]:
    """Return a lightweight class checker enforcing subclass inheritance."""

    def _checker(component_cls: Any) -> None:
        if not isinstance(component_cls, type) or not issubclass(
            component_cls, base_class
        ):
            raise TypeError(
                f"Registered component {component_cls!r} must be a subclass of {base_class!r}."
            )

    return _checker


def register_component(
    *,
    component_family: str,
    component_name: str,
    component_cfg: Optional[Any] = None,
    class_checker: Optional[Callable[[Any], None]] = None,
) -> Callable[[T], T]:
    """Register a component implementation under a shared family registry."""
    normalized_family = str(component_family or "").strip().lower()
    normalized_name = str(component_name or "").strip()
    if not normalized_family:
        raise ValueError("component_family must be a non-empty string.")
    if not normalized_name:
        raise ValueError("component_name must be a non-empty string.")

    def decorator(component_cls: T) -> T:
        if class_checker is not None:
            class_checker(component_cls)
        family_registry = COMPONENT_REGISTRY[normalized_family]
        existing = family_registry.get(normalized_name)
        if existing is not None and existing is not component_cls:
            raise ValueError(
                f"Duplicate registration for {normalized_family!r} component "
                f"{normalized_name!r}: {existing!r} vs {component_cls!r}."
            )
        family_registry[normalized_name] = component_cls
        if component_cfg is not None:
            setattr(component_cls, _COMPONENT_CONFIG_ATTR, component_cfg)
        setattr(component_cls, "_component_family", normalized_family)
        setattr(component_cls, "_component_name", normalized_name)
        return component_cls

    return decorator


def resolve_registry_or_dotpath(
    *,
    component_family: str,
    identifier: str,
    class_checker: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Resolve a component class from a family registry or a full dot path."""
    normalized_family = str(component_family or "").strip().lower()
    if not normalized_family:
        raise ValueError("component_family must be a non-empty string.")

    family_registry = COMPONENT_REGISTRY.get(normalized_family)
    if family_registry is None:
        available_families = sorted(COMPONENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown component_family {normalized_family!r}. "
            f"Available families: {available_families}."
        )

    normalized_identifier = str(identifier or "").strip()
    if not normalized_identifier:
        raise ValueError(f"{normalized_family} identifier must be a non-empty string.")

    registry_value = family_registry.get(normalized_identifier)
    if registry_value is not None:
        if class_checker is not None:
            class_checker(registry_value)
        return registry_value

    try:
        resolved = load_function(normalized_identifier)
    except Exception as exc:
        available_names = sorted(str(key) for key in family_registry.keys())
        raise ValueError(
            f"Cannot resolve {normalized_family} implementation {normalized_identifier!r}. "
            f"Available registered names: {available_names}. "
            "Provide a registered name or a valid full dot path."
        ) from exc
    if class_checker is not None:
        class_checker(resolved)
    return resolved


def list_registered_component_names(*, component_family: str) -> tuple[str, ...]:
    normalized_family = str(component_family or "").strip().lower()
    if not normalized_family:
        raise ValueError("component_family must be a non-empty string.")
    family_registry = COMPONENT_REGISTRY.get(normalized_family)
    if family_registry is None:
        return tuple()
    return tuple(sorted(str(key) for key in family_registry.keys()))


__all__ = [
    "COMPONENT_REGISTRY",
    "list_registered_component_names",
    "register_component",
    "require_subclass",
    "resolve_registry_or_dotpath",
]
