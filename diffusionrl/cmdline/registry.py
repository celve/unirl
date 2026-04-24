"""Registry for mapping runtime config classes to cmdline parser functions."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

ParserFn = Callable[[Any], Any]
T = TypeVar("T", bound=ParserFn)

CMDLINE_CONFIG_PARSER_REGISTRY: Dict[type, ParserFn] = {}


def register_cmdline_config_parser(
    config_class: type,
    parser_fn: Optional[ParserFn] = None,
    *,
    type_checking: Optional[str] = "subclass",
) -> Callable[[T], T] | T:
    """Register a cmdline parser for a runtime config class.

    Supports both:
    - direct registration: ``register_cmdline_config_parser(ConfigCls, parser_fn)``
    - decorator form: ``@register_cmdline_config_parser(ConfigCls)``
    """
    if not isinstance(config_class, type):
        raise TypeError(f"config_class must be a class, got {type(config_class)!r}.")
    if type_checking not in (None, "subclass", "exact"):
        raise ValueError(f"type_checking must be one of None/'subclass'/'exact', got {type_checking!r}.")

    def _register(fn: T) -> T:
        if not callable(fn):
            raise TypeError(f"parser_fn must be callable, got {type(fn)!r}.")

        @wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if type_checking == "subclass" and not isinstance(result, config_class):
                raise TypeError(
                    f"Cmdline parser for {config_class.__name__!r} must return "
                    f"an instance of {config_class!r}, got {type(result)!r}."
                )
            if type_checking == "exact" and type(result) is not config_class:
                raise TypeError(
                    f"Cmdline parser for {config_class.__name__!r} must return "
                    f"exactly {config_class!r}, got {type(result)!r}."
                )
            return result

        existing = CMDLINE_CONFIG_PARSER_REGISTRY.get(config_class)
        if existing is not None and getattr(existing, "__wrapped__", None) is not fn:
            raise ValueError(
                f"Duplicate cmdline config parser registration for {config_class!r}: {existing!r} vs {fn!r}."
            )
        CMDLINE_CONFIG_PARSER_REGISTRY[config_class] = _wrapped
        return _wrapped  # type: ignore[return-value]

    if parser_fn is not None:
        return _register(parser_fn)
    return _register


def derive_cmdline_config_parser(config_class: type) -> ParserFn:
    """Derive the registered cmdline parser for a runtime config class."""
    if not isinstance(config_class, type):
        raise TypeError(f"config_class must be a class, got {type(config_class)!r}.")
    parser_fn = CMDLINE_CONFIG_PARSER_REGISTRY.get(config_class)
    if parser_fn is None:
        available = sorted(cls.__name__ for cls in CMDLINE_CONFIG_PARSER_REGISTRY.keys())
        raise ValueError(
            f"No cmdline config parser registered for {config_class.__name__!r}. Available config classes: {available}."
        )
    return parser_fn


def derive_component_cmdline_config_parser(component_class: type) -> ParserFn:
    """Derive a cmdline parser from a component class via its config-class attr."""
    if not isinstance(component_class, type):
        raise TypeError(f"component_class must be a class, got {type(component_class)!r}.")
    config_class = getattr(component_class, "__CONFIG_CLASS__", None)
    if not isinstance(config_class, type):
        raise TypeError(f"Component class {component_class!r} must define __CONFIG_CLASS__.")
    return derive_cmdline_config_parser(config_class)


__all__ = [
    "CMDLINE_CONFIG_PARSER_REGISTRY",
    "register_cmdline_config_parser",
    "derive_component_cmdline_config_parser",
    "derive_cmdline_config_parser",
]
