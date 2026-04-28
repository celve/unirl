from __future__ import annotations

import pytest

pytest.importorskip("torch")

from diffusionrl.models import derive_model_bundle_path, list_model_types, resolve_model_class
from diffusionrl.types.forward_context import WAN21ForwardContext, get_forward_context_cls


def test_wan21_model_type_resolves_to_versioned_bundle_and_context() -> None:
    model_types = list_model_types()

    assert "wan21" in model_types
    assert "wan" not in model_types
    assert derive_model_bundle_path("wan21") == "diffusionrl.models.wan.WAN21ModelBundle"
    model_cls = resolve_model_class("wan21")
    assert model_cls.__name__ == "WAN21ModelBundle"
    assert model_cls._component_name == "wan21"
    assert get_forward_context_cls("wan21") is WAN21ForwardContext
