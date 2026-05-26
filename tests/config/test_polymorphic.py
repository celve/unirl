"""Unit tests for ``diffusionrl.config.polymorphic``.

Cover ``polymorphic_field`` metadata attachment and ``expand_polymorphic_fields``
expansion semantics for all three forms (single, dict, list/tuple) — round-trip,
idempotency, error reporting on missing or unknown discriminator values, and
schema enforcement on per-element fields.
"""

from __future__ import annotations

import dataclasses
import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pytest
from omegaconf import OmegaConf

from diffusionrl.config.polymorphic import (
    expand_polymorphic_fields,
    expand_polymorphic_lists,
    polymorphic_field,
    polymorphic_metadata,
)
from diffusionrl.config.registration import register_config

# ---------------------------------------------------------------------------
# Test fixtures: a self-contained polymorphic family registered under a
# dedicated group so tests don't collide with the real reward registrations.
# ---------------------------------------------------------------------------


class _BaseTestSpec(ABC):
    weight: float

    @abstractmethod
    def label(self) -> str: ...


@register_config(
    group="poly_test/spec",
    name="alpha",
    target="tests.test_polymorphic._AlphaSpec",
)
class _AlphaSpec(_BaseTestSpec):
    weight: float = 1.0
    color: str = "red"
    size: int = 1

    def label(self) -> str:
        return f"alpha-{self.color}-{self.size}"


@register_config(
    group="poly_test/spec",
    name="beta",
    target="tests.test_polymorphic._BetaSpec",
)
class _BetaSpec(_BaseTestSpec):
    weight: float = 0.5
    flag: bool = True

    def label(self) -> str:
        return f"beta-{self.flag}"


# --- Tuple form (existing) ---


@register_config(
    group="poly_test/parent",
    name="default",
    target="tests.test_polymorphic._ParentConfig",
)
class _ParentConfig:
    items: Tuple[_BaseTestSpec, ...] = polymorphic_field(
        group="poly_test/spec",
        default_factory=tuple,
    )


@register_config(
    group="poly_test/parent",
    name="list_form",
    target="tests.test_polymorphic._ParentConfigList",
)
class _ParentConfigList:
    items: List[_BaseTestSpec] = polymorphic_field(
        group="poly_test/spec",
        default_factory=list,
    )


# --- Dict form (new) ---


@register_config(
    group="poly_test/parent",
    name="dict_form",
    target="tests.test_polymorphic._ParentConfigDict",
)
class _ParentConfigDict:
    items: Dict[str, _BaseTestSpec] = polymorphic_field(
        group="poly_test/spec",
        default_factory=dict,
    )


# --- Single form (new) ---


@register_config(
    group="poly_test/parent",
    name="single_form",
    target="tests.test_polymorphic._ParentConfigSingle",
)
class _ParentConfigSingle:
    item: _BaseTestSpec = polymorphic_field(
        group="poly_test/spec",
    )


def _structured() -> Any:
    return OmegaConf.structured(_ParentConfig)


def _structured_list() -> Any:
    return OmegaConf.structured(_ParentConfigList)


def _structured_dict() -> Any:
    return OmegaConf.structured(_ParentConfigDict)


def _structured_single() -> Any:
    return OmegaConf.structured(_ParentConfigSingle)


# ---------------------------------------------------------------------------
# Tests — metadata
# ---------------------------------------------------------------------------


class TestPolymorphicFieldMetadata:
    def test_attaches_group_and_default_discriminator(self):
        items_field = next(f for f in dataclasses.fields(_ParentConfig) if f.name == "items")
        meta = polymorphic_metadata(items_field)
        assert meta == {"group": "poly_test/spec", "discriminator": "name"}

    def test_no_metadata_on_plain_field(self):
        weight_field = next(f for f in dataclasses.fields(_AlphaSpec) if f.name == "weight")
        assert polymorphic_metadata(weight_field) is None

    def test_custom_discriminator(self):
        @dataclasses.dataclass
        class _Holder:
            x: Tuple[Any, ...] = polymorphic_field(
                group="some/group",
                discriminator="kind",
                default_factory=tuple,
            )

        x_field = next(f for f in dataclasses.fields(_Holder) if f.name == "x")
        assert polymorphic_metadata(x_field) == {"group": "some/group", "discriminator": "kind"}


# ---------------------------------------------------------------------------
# Tests — list/tuple expansion (existing, must stay green)
# ---------------------------------------------------------------------------


class TestExpandPolymorphicLists:
    def test_basic_expansion_typed_round_trip(self):
        cfg = _structured()
        cfg.items = [
            {"name": "alpha", "color": "blue", "size": 3},
            {"name": "beta", "flag": False},
        ]
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert len(obj.items) == 2
        assert isinstance(obj.items[0], _AlphaSpec)
        assert obj.items[0].color == "blue"
        assert obj.items[0].size == 3
        assert obj.items[0].weight == 1.0  # default kept
        assert isinstance(obj.items[1], _BetaSpec)
        assert obj.items[1].flag is False
        assert obj.items[1].weight == 0.5

    def test_idempotent(self):
        cfg = _structured()
        cfg.items = [{"name": "alpha", "color": "green"}]
        expand_polymorphic_fields(cfg)
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items[0], _AlphaSpec)
        assert obj.items[0].color == "green"

    def test_empty_list_passes_through(self):
        cfg = _structured()
        cfg.items = []
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert obj.items == []

    def test_missing_discriminator_raises(self):
        cfg = _structured()
        cfg.items = [{"color": "blue"}]
        with pytest.raises(ValueError, match="missing 'name'"):
            expand_polymorphic_fields(cfg)

    def test_unknown_name_raises_with_known_list(self):
        cfg = _structured()
        cfg.items = [{"name": "gamma"}]
        with pytest.raises(ValueError, match="unknown 'gamma'"):
            expand_polymorphic_fields(cfg)
        with pytest.raises(ValueError, match=r"alpha.*beta|beta.*alpha"):
            expand_polymorphic_fields(cfg)

    def test_typo_in_per_spec_field_raises_during_merge(self):
        cfg = _structured()
        cfg.items = [{"name": "alpha", "colour": "blue"}]  # British spelling typo
        with pytest.raises(ValueError, match="colour"):
            expand_polymorphic_fields(cfg)

    def test_method_dispatch_on_typed_instances(self):
        cfg = _structured()
        cfg.items = [
            {"name": "alpha", "color": "purple", "size": 7},
            {"name": "beta", "flag": True},
        ]
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        labels = [spec.label() for spec in obj.items]
        assert labels == ["alpha-purple-7", "beta-True"]

    def test_old_name_alias_works(self):
        cfg = _structured()
        cfg.items = [{"name": "alpha"}]
        expand_polymorphic_lists(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items[0], _AlphaSpec)


# ---------------------------------------------------------------------------
# Tests — dict expansion (new)
# ---------------------------------------------------------------------------


class TestExpandPolymorphicDict:
    def test_key_as_discriminator(self):
        cfg = _structured_dict()
        cfg.items = {"alpha": {"color": "blue", "size": 3}}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items["alpha"], _AlphaSpec)
        assert obj.items["alpha"].color == "blue"
        assert obj.items["alpha"].size == 3
        assert obj.items["alpha"].weight == 1.0

    def test_multiple_entries(self):
        cfg = _structured_dict()
        cfg.items = {
            "alpha": {"color": "green"},
            "beta": {"flag": False},
        }
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items["alpha"], _AlphaSpec)
        assert isinstance(obj.items["beta"], _BetaSpec)
        assert obj.items["alpha"].color == "green"
        assert obj.items["beta"].flag is False

    def test_null_value_loads_defaults(self):
        cfg = _structured_dict()
        cfg.items = {"alpha": None}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items["alpha"], _AlphaSpec)
        assert obj.items["alpha"].color == "red"
        assert obj.items["alpha"].weight == 1.0

    def test_string_value_as_config_name(self):
        cfg = _structured_dict()
        cfg.items = {"my_scorer": "alpha"}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items["my_scorer"], _AlphaSpec)
        assert obj.items["my_scorer"].color == "red"

    def test_explicit_discriminator_overrides_key(self):
        cfg = _structured_dict()
        cfg.items = {"my_scorer": {"name": "beta", "flag": False}}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items["my_scorer"], _BetaSpec)
        assert obj.items["my_scorer"].flag is False

    def test_empty_dict_passes_through(self):
        cfg = _structured_dict()
        cfg.items = {}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert obj.items == {}

    def test_idempotent(self):
        cfg = _structured_dict()
        cfg.items = {"alpha": {"color": "blue"}}
        expand_polymorphic_fields(cfg)
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items["alpha"], _AlphaSpec)
        assert obj.items["alpha"].color == "blue"

    def test_unknown_key_raises(self):
        cfg = _structured_dict()
        cfg.items = {"gamma": {}}
        with pytest.raises(ValueError, match="unknown 'gamma'"):
            expand_polymorphic_fields(cfg)

    def test_typo_in_override_raises(self):
        cfg = _structured_dict()
        cfg.items = {"alpha": {"colour": "blue"}}
        with pytest.raises(ValueError, match="colour"):
            expand_polymorphic_fields(cfg)


# ---------------------------------------------------------------------------
# Tests — single expansion (new)
# ---------------------------------------------------------------------------


class TestExpandPolymorphicSingle:
    def test_string_shorthand(self):
        cfg = _structured_single()
        cfg.item = "alpha"
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.item, _AlphaSpec)
        assert obj.item.color == "red"
        assert obj.item.weight == 1.0

    def test_dict_with_discriminator(self):
        cfg = _structured_single()
        cfg.item = {"name": "alpha", "color": "blue", "size": 5}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.item, _AlphaSpec)
        assert obj.item.color == "blue"
        assert obj.item.size == 5

    def test_idempotent(self):
        cfg = _structured_single()
        cfg.item = "beta"
        expand_polymorphic_fields(cfg)
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.item, _BetaSpec)

    def test_unknown_name_raises(self):
        cfg = _structured_single()
        cfg.item = "gamma"
        with pytest.raises(ValueError, match="unknown 'gamma'"):
            expand_polymorphic_fields(cfg)

    def test_dict_missing_discriminator_raises(self):
        cfg = _structured_single()
        cfg.item = {"color": "blue"}
        with pytest.raises(ValueError, match="missing 'name'"):
            expand_polymorphic_fields(cfg)

    def test_typo_in_override_raises(self):
        cfg = _structured_single()
        cfg.item = {"name": "alpha", "colour": "blue"}
        with pytest.raises(ValueError, match="colour"):
            expand_polymorphic_fields(cfg)

    def test_method_dispatch(self):
        cfg = _structured_single()
        cfg.item = {"name": "alpha", "color": "purple", "size": 7}
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert obj.item.label() == "alpha-purple-7"


# ---------------------------------------------------------------------------
# Tests — annotation rewrite
# ---------------------------------------------------------------------------


class TestAnnotationRewrite:
    """``register_config`` rewrites polymorphic annotations so OmegaConf accepts
    raw values at compose time."""

    def test_tuple_form_rewritten_to_tuple_any(self):
        hints = typing.get_type_hints(_ParentConfig)
        assert hints["items"] == Tuple[Any, ...]

    def test_list_form_rewritten_to_list_any(self):
        hints = typing.get_type_hints(_ParentConfigList)
        assert hints["items"] == List[Any]

    def test_dict_form_rewritten_to_dict_str_any(self):
        hints = typing.get_type_hints(_ParentConfigDict)
        assert hints["items"] == Dict[str, Any]

    def test_single_form_rewritten_to_any(self):
        hints = typing.get_type_hints(_ParentConfigSingle)
        assert hints["item"] is Any

    def test_list_form_round_trips_through_expand(self):
        cfg = _structured_list()
        cfg.items = [{"name": "alpha", "color": "purple"}]
        expand_polymorphic_fields(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items[0], _AlphaSpec)
        assert obj.items[0].color == "purple"

    def test_dataclass_field_type_also_rewritten(self):
        items_field = next(f for f in dataclasses.fields(_ParentConfig) if f.name == "items")
        assert items_field.type in (Tuple[Any, ...], "Tuple[Any, ...]")

    def test_dict_dataclass_field_type_also_rewritten(self):
        items_field = next(f for f in dataclasses.fields(_ParentConfigDict) if f.name == "items")
        assert items_field.type in (Dict[str, Any], "Dict[str, Any]")

    def test_single_dataclass_field_type_also_rewritten(self):
        item_field = next(f for f in dataclasses.fields(_ParentConfigSingle) if f.name == "item")
        assert item_field.type is Any or item_field.type == "Any"
