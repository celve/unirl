"""Generic component construction helpers independent of framework cmdline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from diffusionrl.registry import derive_registry_or_dotpath


@dataclass(frozen=True)
class ComponentInitPayload:
    """Canonical runtime init payload shared by framework-managed components."""

    component_dotpath: str
    component_config: Any


def create_component_from_init_payload(
    *,
    component_family: str,
    component_init_payload: ComponentInitPayload,
    init_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a component from its canonical init payload."""
    if not isinstance(component_init_payload, ComponentInitPayload):
        raise TypeError(
            f"component_init_payload must be a ComponentInitPayload, got: {type(component_init_payload).__name__}"
        )

    component_cls = derive_registry_or_dotpath(
        component_family=component_family,
        identifier=component_init_payload.component_dotpath,
    )
    return component_cls(
        config=component_init_payload.component_config,
        **dict(init_kwargs or {}),
    )


__all__ = [
    "ComponentInitPayload",
    "create_component_from_init_payload",
]
