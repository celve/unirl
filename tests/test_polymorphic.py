"""Unit tests for ``diffusionrl.config.polymorphic``.

Cover ``polymorphic_field`` metadata attachment and ``expand_polymorphic_lists``
expansion semantics — round-trip, idempotency, error reporting on missing or
unknown discriminator values, and schema enforcement on per-element fields.
"""

from __future__ import annotations

import dataclasses
import typing
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import pytest
from omegaconf import OmegaConf

from diffusionrl.config.polymorphic import (
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


def _structured() -> Any:
    return OmegaConf.structured(_ParentConfig)


def _structured_list() -> Any:
    return OmegaConf.structured(_ParentConfigList)


# ---------------------------------------------------------------------------
# Tests
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


class TestExpandPolymorphicLists:
    def test_basic_expansion_typed_round_trip(self):
        cfg = _structured()
        cfg.items = [
            {"name": "alpha", "color": "blue", "size": 3},
            {"name": "beta", "flag": False},
        ]
        expand_polymorphic_lists(cfg)
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
        expand_polymorphic_lists(cfg)
        expand_polymorphic_lists(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items[0], _AlphaSpec)
        assert obj.items[0].color == "green"

    def test_empty_list_passes_through(self):
        cfg = _structured()
        cfg.items = []
        expand_polymorphic_lists(cfg)
        obj = OmegaConf.to_object(cfg)
        assert obj.items == []

    def test_missing_discriminator_raises(self):
        cfg = _structured()
        cfg.items = [{"color": "blue"}]
        with pytest.raises(ValueError, match="missing 'name'"):
            expand_polymorphic_lists(cfg)

    def test_unknown_name_raises_with_known_list(self):
        cfg = _structured()
        cfg.items = [{"name": "gamma"}]
        with pytest.raises(ValueError, match="unknown 'gamma'"):
            expand_polymorphic_lists(cfg)
        # The error should also surface the registered alternatives.
        with pytest.raises(ValueError, match=r"alpha.*beta|beta.*alpha"):
            expand_polymorphic_lists(cfg)

    def test_typo_in_per_spec_field_raises_during_merge(self):
        cfg = _structured()
        cfg.items = [{"name": "alpha", "colour": "blue"}]  # British spelling typo
        with pytest.raises(ValueError, match="colour"):
            expand_polymorphic_lists(cfg)

    def test_method_dispatch_on_typed_instances(self):
        cfg = _structured()
        cfg.items = [
            {"name": "alpha", "color": "purple", "size": 7},
            {"name": "beta", "flag": True},
        ]
        expand_polymorphic_lists(cfg)
        obj = OmegaConf.to_object(cfg)
        labels = [spec.label() for spec in obj.items]
        assert labels == ["alpha-purple-7", "beta-True"]


class TestAnnotationRewrite:
    """``register_config`` rewrites polymorphic annotations from ``Tuple[Base, ...]``
    / ``List[Base]`` to ``Tuple[Any, ...]`` / ``List[Any]`` so OmegaConf accepts
    raw dict assignment. The source annotation stays visible to type checkers."""

    def test_tuple_form_rewritten_to_tuple_any(self):
        hints = typing.get_type_hints(_ParentConfig)
        assert hints["items"] == Tuple[Any, ...]

    def test_list_form_rewritten_to_list_any(self):
        hints = typing.get_type_hints(_ParentConfigList)
        assert hints["items"] == List[Any]

    def test_list_form_round_trips_through_expand(self):
        cfg = _structured_list()
        cfg.items = [{"name": "alpha", "color": "purple"}]
        expand_polymorphic_lists(cfg)
        obj = OmegaConf.to_object(cfg)
        assert isinstance(obj.items[0], _AlphaSpec)
        assert obj.items[0].color == "purple"

    def test_dataclass_field_type_also_rewritten(self):
        # OmegaConf reads both ``__annotations__`` and ``Field.type``; both
        # should reflect the rewrite for fields registered after dataclass
        # promotion has already happened (or for round-tripping through
        # ``OmegaConf.structured``).
        items_field = next(f for f in dataclasses.fields(_ParentConfig) if f.name == "items")
        # ``Field.type`` may be a string (forward ref) on Python 3.10+ if the
        # source had ``from __future__ import annotations``; in either case
        # it must not still be the literal Tuple[_BaseTestSpec, ...] type.
        assert items_field.type in (Tuple[Any, ...], "Tuple[Any, ...]")

    def test_unrecognized_annotation_raises(self):
        # ``polymorphic_field`` only makes sense on list-shaped fields; a
        # plain or dict annotation should raise rather than silently coerce.
        with pytest.raises(TypeError, match=r"requires a Tuple\[Base, \.\.\.\] or List\[Base\]"):

            @register_config(
                group="poly_test/parent",
                name="bad_form",
                target="tests.test_polymorphic._BadShapeConfig",
            )
            class _BadShapeConfig:  # pragma: no cover — registration must raise
                items: Any = polymorphic_field(
                    group="poly_test/spec",
                    default_factory=tuple,
                )
