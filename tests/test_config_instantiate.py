"""Tests for ``diffusionrl.config.instantiate.build``.

Cover the two dispatch modes (``expand=False`` / ``expand=True``), schema-only
(no ``_target_``) error path, nested structured configs, classmethod targets,
and runtime-deps pass-through.

Module-level helper classes are used so Hydra's ``_locate`` can import them by
dotpath. Each test constructs its ``DictConfig`` section via
``@register_config`` + ``ConfigStore`` (the production path) rather than
hand-building.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from diffusionrl.config.instantiate import build
from diffusionrl.config.registration import register_config

# ---------------------------------------------------------------------------
# Module-level helpers (must be importable by dotpath for hydra._locate)
# ---------------------------------------------------------------------------


@dataclass
class _SimpleConfig:
    x: int = 1
    y: str = "default"


class _SimpleRuntime:
    """Runtime class using the project's ``(*, config, **kwargs)`` convention."""

    def __init__(self, *, config: _SimpleConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device


@dataclass
class _InnerConfig:
    a: int = 10


@dataclass
class _NestedConfig:
    inner: _InnerConfig = field(default_factory=_InnerConfig)
    b: str = "outer"


class _NestedRuntime:
    def __init__(self, *, config: _NestedConfig) -> None:
        self.config = config


class _ClassmethodRuntime:
    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def from_config(cls, *, config: _SimpleConfig) -> "_ClassmethodRuntime":
        return cls(config.x)


@dataclass
class _FlatConfig:
    x: int = 0
    y: str = "z"


class _FlatRuntime:
    """Runtime class taking flat kwargs — exercises the ``expand=True`` path."""

    def __init__(self, x: int, y: str, device: str = "cpu") -> None:
        self.x = x
        self.y = y
        self.device = device


# Module-level registrations (happen once at import). Keyed by target dotpath,
# which is globally unique to this test module, so ConfigStore pollution is OK.
# Using __name__ rather than a hardcoded string so this works whether pytest
# imports us as ``tests.test_config_instantiate`` or just ``test_config_instantiate``.
_BASE = __name__

register_config(
    _SimpleConfig,
    group="_test_build",
    name="simple",
    target=f"{_BASE}._SimpleRuntime",
)
register_config(
    _NestedConfig,
    group="_test_build",
    name="nested",
    target=f"{_BASE}._NestedRuntime",
)
register_config(
    _SimpleConfig,
    group="_test_build",
    name="cm",
    target=f"{_BASE}._ClassmethodRuntime.from_config",
)
register_config(
    _FlatConfig,
    group="_test_build",
    name="flat",
    target=f"{_BASE}._FlatRuntime",
    expand=True,
)


def _section(group: str, name: str) -> DictConfig:
    """Retrieve a registered ConfigStore node as a DictConfig."""
    entry = ConfigStore.instance().repo[group][f"{name}.yaml"]
    return OmegaConf.create(entry.node)


# ---------------------------------------------------------------------------
# expand=False (default) path
# ---------------------------------------------------------------------------


def test_build_returns_runtime_with_config_attribute():
    cfg = _section("_test_build", "simple")
    result = build(cfg)
    assert isinstance(result, _SimpleRuntime)
    assert isinstance(result.config, _SimpleConfig)
    assert (result.config.x, result.config.y) == (1, "default")
    assert result.device == "cpu"


def test_build_passes_runtime_deps():
    cfg = _section("_test_build", "simple")
    result = build(cfg, device="cuda:3")
    assert result.device == "cuda:3"
    assert (result.config.x, result.config.y) == (1, "default")


def test_build_applies_field_overrides():
    cfg = _section("_test_build", "simple")
    cfg.x = 99
    cfg.y = "overridden"
    result = build(cfg)
    assert (result.config.x, result.config.y) == (99, "overridden")


def test_build_nested_structured_field_materialized():
    cfg = _section("_test_build", "nested")
    # Override a nested field to confirm it survives the merge round-trip.
    cfg.inner.a = 42
    result = build(cfg)
    assert isinstance(result.config, _NestedConfig)
    assert isinstance(result.config.inner, _InnerConfig), (
        "nested structured field must come back as its dataclass, not dict"
    )
    assert result.config.inner.a == 42


def test_build_classmethod_target():
    cfg = _section("_test_build", "cm")
    cfg.x = 7
    result = build(cfg)
    assert isinstance(result, _ClassmethodRuntime)
    assert result.value == 7


# ---------------------------------------------------------------------------
# expand=True path — delegates to hydra.utils.instantiate (flat kwargs)
# ---------------------------------------------------------------------------


def test_build_expand_true_uses_flat_kwargs():
    cfg = _section("_test_build", "flat")
    cfg.x = 5
    cfg.y = "hi"
    result = build(cfg, device="cuda:1")
    assert isinstance(result, _FlatRuntime)
    assert (result.x, result.y, result.device) == (5, "hi", "cuda:1")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_build_raises_on_missing_target():
    cfg = OmegaConf.create({"x": 1, "y": "z"})
    with pytest.raises(ValueError, match="no _target_"):
        build(cfg)


def test_build_returns_none_for_none_cfg():
    assert build(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Decorator contract
# ---------------------------------------------------------------------------


def test_register_config_marks_expand_on_class():
    """``expand`` is recorded as a class attribute on the registered schema,
    not in any module-level registry."""
    simple_cfg = _section("_test_build", "simple")
    simple_cls = OmegaConf.get_type(simple_cfg)
    assert getattr(simple_cls, "_hydra_expand_", None) is False
    assert issubclass(simple_cls, _SimpleConfig)

    flat_cfg = _section("_test_build", "flat")
    flat_cls = OmegaConf.get_type(flat_cfg)
    assert getattr(flat_cls, "_hydra_expand_", None) is True
    assert issubclass(flat_cls, _FlatConfig)


def test_register_config_expand_without_target_raises():
    @dataclass
    class _NoTargetExpand:
        z: int = 0

    with pytest.raises(ValueError, match="expand=True requires a target"):
        register_config(expand=True)(_NoTargetExpand)


def test_register_config_without_target_returns_original_class():
    """Schema-only registrations don't synthesize a subclass; the returned
    class is exactly the input (no ``_target_`` field, no ``_hydra_expand_``)."""

    @dataclass
    class _SchemaOnly:
        q: int = 0

    returned = register_config(name="_schema_only_unique_xyz", target=None)(_SchemaOnly)
    assert returned is _SchemaOnly
    assert not hasattr(_SchemaOnly, "_hydra_expand_")
    del ConfigStore.instance().repo["_schema_only_unique_xyz.yaml"]
