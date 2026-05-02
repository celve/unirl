"""Tests for diffusionrl.config.registration.

Cover every call pattern of ``register_config`` (bare decorator, decorator
factory, positional function call, keyword function call), every
``ConfigStore.store`` kwarg forwarded by both helpers (``group``, ``name``,
``package``, ``provider``), and the validation error cases.

``ConfigStore`` is a process-wide singleton, so each test uses a unique
group/name prefix and cleans up after itself to avoid cross-contamination.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterator

import pytest
from hydra.core.config_store import ConfigStore

from diffusionrl.config.registration import register_config, register_preset


@pytest.fixture
def unique_prefix() -> Iterator[str]:
    """Return a unique prefix and clean up any ConfigStore entries using it."""
    prefix = f"reg_test_{uuid.uuid4().hex[:8]}"
    yield prefix
    repo = ConfigStore.instance().repo
    for key in [k for k in repo if isinstance(k, str) and k.startswith(prefix)]:
        del repo[key]


def _get(group: str | None, name: str):
    repo = ConfigStore.instance().repo
    key = f"{name}.yaml"
    return repo[group][key] if group is not None else repo[key]


# ---------------------------------------------------------------------------
# register_config call-pattern coverage
# ---------------------------------------------------------------------------


def test_bare_decorator_registers_under_class_name():
    @register_config
    @dataclass
    class _BareDirect:
        y: int = 2

    entry = ConfigStore.instance().repo["_BareDirect.yaml"]
    assert entry.node == {"y": 2}
    assert entry.group is None
    del ConfigStore.instance().repo["_BareDirect.yaml"]


def test_factory_no_args_equivalent_to_bare():
    @register_config()
    @dataclass
    class _FactoryEmpty:
        z: int = 3

    entry = ConfigStore.instance().repo["_FactoryEmpty.yaml"]
    assert entry.node == {"z": 3}
    assert entry.group is None
    del ConfigStore.instance().repo["_FactoryEmpty.yaml"]


def test_factory_with_group_derives_schema_name(unique_prefix):
    group = unique_prefix

    @register_config(group=group)
    @dataclass
    class _WithGroup:
        a: int = 7

    entry = _get(group, f"{group}_schema")
    assert entry.group == group
    assert entry.node == {"a": 7}


def test_factory_with_explicit_name_overrides_default(unique_prefix):
    group = unique_prefix
    custom = f"{unique_prefix}_custom"

    @register_config(group=group, name=custom)
    @dataclass
    class _CustomName:
        b: int = 8

    entry = _get(group, custom)
    assert entry.name == f"{custom}.yaml"
    assert entry.group == group


def test_factory_name_only_no_group(unique_prefix):
    name = f"{unique_prefix}_standalone"

    @register_config(name=name)
    @dataclass
    class _NameOnly:
        c: int = 9

    entry = _get(None, name)
    assert entry.group is None
    assert entry.name == f"{name}.yaml"


def test_function_call_positional_cls(unique_prefix):
    group = unique_prefix

    @dataclass
    class _PosCls:
        d: int = 10

    returned = register_config(_PosCls, group=group)
    assert returned is _PosCls

    entry = _get(group, f"{group}_schema")
    assert entry.node == {"d": 10}


def test_function_call_positional_only_cls(unique_prefix):
    @dataclass
    class _FuncPos:
        e: int = 11

    # Bare function call: uses class name as name.
    returned = register_config(_FuncPos)
    assert returned is _FuncPos

    entry = ConfigStore.instance().repo["_FuncPos.yaml"]
    assert entry.node == {"e": 11}
    del ConfigStore.instance().repo["_FuncPos.yaml"]


def test_function_call_all_kwargs(unique_prefix):
    group = unique_prefix
    custom_name = f"{unique_prefix}_full"

    @dataclass
    class _FullKw:
        f: int = 12

    returned = register_config(
        _FullKw,
        group=group,
        name=custom_name,
        package="_global_",
        provider="diffusionrl",
    )
    assert returned is _FullKw

    entry = _get(group, custom_name)
    assert entry.name == f"{custom_name}.yaml"
    assert entry.group == group
    assert entry.package == "_global_"
    assert entry.provider == "diffusionrl"
    assert entry.node == {"f": 12}


def test_cls_is_positional_only(unique_prefix):
    """Passing cls as a keyword must raise TypeError (PEP 570)."""

    @dataclass
    class _KwCls:
        g: int = 13

    with pytest.raises(TypeError):
        register_config(cls=_KwCls, group=unique_prefix)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# register_config error cases
# ---------------------------------------------------------------------------


def test_register_config_auto_promotes_non_dataclass_class():
    """Plain (non-dataclass) classes are auto-promoted via @dataclass so users
    can drop the explicit decorator at registration sites."""
    from dataclasses import is_dataclass

    class Plain:
        x: int = 5

    returned = register_config(name="_plain_unique_xyz")(Plain)
    assert is_dataclass(returned)
    # Cleanup ConfigStore so this doesn't leak into other tests.
    from hydra.core.config_store import ConfigStore

    del ConfigStore.instance().repo["_plain_unique_xyz.yaml"]


def test_register_config_rejects_non_class():
    with pytest.raises(TypeError, match="requires a class"):
        register_config(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# register_preset
# ---------------------------------------------------------------------------


@dataclass
class _PresetDC:
    v: int = 0


def test_register_preset_top_level(unique_prefix):
    name = f"{unique_prefix}_topflat"
    register_preset(name, _PresetDC(v=5))

    entry = _get(None, name)
    assert entry.node == {"v": 5}
    assert entry.group is None


def test_register_preset_under_group(unique_prefix):
    group = unique_prefix
    register_preset("flow", _PresetDC(v=1), group=group)
    register_preset("dance", _PresetDC(v=2), group=group)

    flow = _get(group, "flow")
    dance = _get(group, "dance")
    assert flow.node == {"v": 1}
    assert dance.node == {"v": 2}
    assert flow.group == group == dance.group


def test_register_preset_forwards_package_and_provider(unique_prefix):
    group = unique_prefix
    register_preset(
        "withmeta",
        _PresetDC(v=42),
        group=group,
        package="_global_",
        provider="diffusionrl",
    )

    entry = _get(group, "withmeta")
    assert entry.package == "_global_"
    assert entry.provider == "diffusionrl"


def test_register_preset_rejects_class():
    with pytest.raises(TypeError, match="got class"):
        register_preset("x", _PresetDC)  # type: ignore[arg-type]


def test_register_preset_rejects_non_dataclass_instance():
    with pytest.raises(TypeError, match="dataclass instance"):
        register_preset("x", {"not": "a dataclass"})
