"""Generic component construction from framework cmdline arguments."""

from __future__ import annotations

from typing import Any, Dict, Optional

from diffusionrl.cmdline.registry import derive_component_cmdline_config_parser
from diffusionrl.construction import (
    ComponentInitPayload,
    create_component_from_init_payload,
)
from diffusionrl.registry import derive_registry_or_dotpath


def build_component_init_payload_from_args(
    *,
    component_family: str,
    identifier: str,
    args: Any,
    parser_kwargs: Optional[Dict[str, Any]] = None,
) -> ComponentInitPayload:
    """Resolve a component class and parse its framework-facing config."""
    component_cls = derive_registry_or_dotpath(
        component_family=component_family,
        identifier=identifier,
    )
    parser_fn = derive_component_cmdline_config_parser(component_cls)
    component_config = parser_fn(args, **dict(parser_kwargs or {}))
    return ComponentInitPayload(
        component_dotpath=f"{component_cls.__module__}.{component_cls.__qualname__}",
        component_config=component_config,
    )


def create_component_from_args(
    *,
    component_family: str,
    identifier: str,
    args: Any,
    parser_kwargs: Optional[Dict[str, Any]] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Resolve, parse config for, and instantiate a component from args."""
    component_init_payload = build_component_init_payload_from_args(
        component_family=component_family,
        identifier=identifier,
        args=args,
        parser_kwargs=parser_kwargs,
    )
    return create_component_from_init_payload(
        component_family=component_family,
        component_init_payload=component_init_payload,
        init_kwargs=init_kwargs,
    )


__all__ = [
    "build_component_init_payload_from_args",
    "create_component_from_args",
    "create_component_from_init_payload",
]
