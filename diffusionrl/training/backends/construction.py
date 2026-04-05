"""Train-backend family construction helpers."""

from __future__ import annotations

from typing import Any

from diffusionrl.construction import (
    ComponentInitPayload,
    create_component_from_init_payload,
)
from .registry import TRAIN_BACKEND_COMPONENT_FAMILY


def create_train_backend_from_init_payload(
    train_backend_init_payload: ComponentInitPayload,
    **init_kwargs: Any,
) -> Any:
    return create_component_from_init_payload(
        component_family=TRAIN_BACKEND_COMPONENT_FAMILY,
        component_init_payload=train_backend_init_payload,
        init_kwargs=init_kwargs or None,
    )


__all__ = [
    "create_train_backend_from_init_payload",
]
