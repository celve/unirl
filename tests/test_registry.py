import pytest

from diffusionrl.algorithms.grpo import GRPOAlgorithm
from diffusionrl.algorithms.registry import (
    derive_algorithm_class,
    derive_algorithm_dotpath,
    ensure_builtin_algorithm_registration,
)
from diffusionrl.registry import (
    COMPONENT_REGISTRY,
    derive_registry_or_dotpath,
    register_component,
    require_subclass,
)


class _TestBase:
    pass


class _TestComponent(_TestBase):
    pass


class _OtherComponent(_TestBase):
    pass


class _ConfigMarker:
    pass


@pytest.fixture
def isolated_test_family():
    family = "unit_test_family"
    COMPONENT_REGISTRY.pop(family, None)
    yield family
    COMPONENT_REGISTRY.pop(family, None)


def test_register_component_stores_class_and_config(isolated_test_family):
    decorator = register_component(
        component_family=isolated_test_family,
        component_name="demo",
        component_cfg=_ConfigMarker,
        class_checker=require_subclass(_TestBase),
    )

    registered_cls = decorator(_TestComponent)

    assert registered_cls is _TestComponent
    assert COMPONENT_REGISTRY[isolated_test_family]["demo"] is _TestComponent
    assert _TestComponent.__CONFIG_CLASS__ is _ConfigMarker
    assert _TestComponent._component_family == isolated_test_family
    assert _TestComponent._component_name == "demo"


def test_register_component_rejects_non_subclass(isolated_test_family):
    class NotASubclass:
        pass

    decorator = register_component(
        component_family=isolated_test_family,
        component_name="bad",
        class_checker=require_subclass(_TestBase),
    )

    with pytest.raises(TypeError, match="must be a subclass"):
        decorator(NotASubclass)


def test_derive_registry_or_dotpath_resolves_registered_name(isolated_test_family):
    decorator = register_component(
        component_family=isolated_test_family,
        component_name="demo",
        class_checker=require_subclass(_TestBase),
    )
    decorator(_TestComponent)

    resolved = derive_registry_or_dotpath(
        component_family=isolated_test_family,
        identifier="demo",
    )

    assert resolved is _TestComponent


def test_derive_registry_or_dotpath_resolves_full_dotpath():
    ensure_builtin_algorithm_registration()

    resolved = derive_registry_or_dotpath(
        component_family="algorithm",
        identifier="diffusionrl.algorithms.grpo.GRPOAlgorithm",
    )

    assert resolved is GRPOAlgorithm


def test_derive_registry_or_dotpath_error_lists_available_names(isolated_test_family):
    register_component(
        component_family=isolated_test_family,
        component_name="demo",
        class_checker=require_subclass(_TestBase),
    )(_TestComponent)
    register_component(
        component_family=isolated_test_family,
        component_name="other",
        class_checker=require_subclass(_TestBase),
    )(_OtherComponent)

    with pytest.raises(ValueError, match="Available registered names: \\['demo', 'other'\\]"):
        derive_registry_or_dotpath(
            component_family=isolated_test_family,
            identifier="missing_component",
        )


def test_algorithm_family_resolves_builtin_registered_name():
    resolved = derive_algorithm_class("grpo")

    assert resolved is GRPOAlgorithm


def test_algorithm_family_resolves_builtin_registered_dotpath():
    resolved = derive_algorithm_dotpath("grpo")

    assert resolved == "diffusionrl.algorithms.grpo.GRPOAlgorithm"


def test_derive_algorithm_class_rejects_non_algorithm_dotpath():
    with pytest.raises(TypeError, match="must be a subclass"):
        derive_algorithm_class("diffusionrl.registry.register_component")
