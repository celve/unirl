"""Model-bundle family construction helpers."""

from __future__ import annotations

from typing import Any

from diffusionrl.construction import (
    ComponentInitPayload,
    create_component_from_init_payload,
)
from diffusionrl.models.base import ModelBundle
from .registry import MODEL_BUNDLE_COMPONENT_FAMILY


def create_model_bundle_from_init_payload(
    model_init_payload: ComponentInitPayload,
    **init_kwargs: Any,
) -> ModelBundle:
    return create_component_from_init_payload(
        component_family=MODEL_BUNDLE_COMPONENT_FAMILY,
        component_init_payload=model_init_payload,
        init_kwargs=init_kwargs or None,
    )


__all__ = [
    "MODEL_BUNDLE_COMPONENT_FAMILY",
    "create_model_bundle_from_init_payload",
]
