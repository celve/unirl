"""Algorithm-family construction helpers."""

from __future__ import annotations

from typing import Any
from diffusionrl.construction import (
    ComponentInitPayload,
    create_component_from_init_payload,
)
from .registry import ALGORITHM_COMPONENT_FAMILY


def create_algorithm_from_init_payload(
    algorithm_init_payload: ComponentInitPayload,
    **init_kwargs: Any,
) -> Any:
    """Instantiate an algorithm from the canonical init payload."""
    return create_component_from_init_payload(
        component_family=ALGORITHM_COMPONENT_FAMILY,
        component_init_payload=algorithm_init_payload,
        init_kwargs=init_kwargs or None,
    )

__all__ = [
    "create_algorithm_from_init_payload",
]
