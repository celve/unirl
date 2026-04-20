import pytest

from diffusionrl.algorithms.grpo import GRPOAlgorithm
from diffusionrl.cmdline.algorithms import build_grpo_algorithm_config_from_args
from diffusionrl.cmdline.registry import (
    CMDLINE_CONFIG_PARSER_REGISTRY,
    derive_cmdline_config_parser,
    derive_component_cmdline_config_parser,
    register_cmdline_config_parser,
)


class _ConfigA:
    pass


class _ConfigASubclass(_ConfigA):
    pass


class _ConfigB:
    pass


@pytest.fixture
def isolated_cmdline_registry():
    old_registry = dict(CMDLINE_CONFIG_PARSER_REGISTRY)
    CMDLINE_CONFIG_PARSER_REGISTRY.clear()
    yield
    CMDLINE_CONFIG_PARSER_REGISTRY.clear()
    CMDLINE_CONFIG_PARSER_REGISTRY.update(old_registry)


def test_register_cmdline_config_parser_direct_registration(isolated_cmdline_registry):
    def parse_config(args):
        del args
        return _ConfigA()

    returned = register_cmdline_config_parser(_ConfigA, parse_config)

    assert returned is derive_cmdline_config_parser(_ConfigA)
    assert isinstance(returned(object()), _ConfigA)


def test_register_cmdline_config_parser_decorator_form(isolated_cmdline_registry):
    @register_cmdline_config_parser(_ConfigA)
    def parse_config(args):
        del args
        return _ConfigA()

    assert derive_cmdline_config_parser(_ConfigA) is parse_config


def test_register_cmdline_config_parser_rejects_duplicate(isolated_cmdline_registry):
    @register_cmdline_config_parser(_ConfigA)
    def parse_config(args):
        del args
        return _ConfigA()

    def parse_other(args):
        del args
        return _ConfigA()

    with pytest.raises(ValueError, match="Duplicate cmdline config parser registration"):
        register_cmdline_config_parser(_ConfigA, parse_other)

    assert derive_cmdline_config_parser(_ConfigA) is parse_config


def test_register_cmdline_config_parser_subclass_checking_by_default(
    isolated_cmdline_registry,
):
    @register_cmdline_config_parser(_ConfigA)
    def parse_config(args):
        del args
        return _ConfigASubclass()

    assert isinstance(parse_config(object()), _ConfigA)


def test_register_cmdline_config_parser_exact_checking(
    isolated_cmdline_registry,
):
    @register_cmdline_config_parser(_ConfigA, type_checking="exact")
    def parse_config(args):
        del args
        return _ConfigASubclass()

    with pytest.raises(TypeError, match="must return exactly"):
        parse_config(object())


def test_derive_cmdline_config_parser_error_lists_available_configs(
    isolated_cmdline_registry,
):
    @register_cmdline_config_parser(_ConfigA)
    def parse_config_a(args):
        del args
        return _ConfigA()

    @register_cmdline_config_parser(_ConfigB)
    def parse_config_b(args):
        del args
        return _ConfigB()

    with pytest.raises(
        ValueError,
        match=r"Available config classes: \['_ConfigA', '_ConfigB'\]",
    ):
        derive_cmdline_config_parser(type("MissingConfig", (), {}))


def test_derive_component_cmdline_config_parser_uses_config_class():
    assert derive_component_cmdline_config_parser(GRPOAlgorithm) is build_grpo_algorithm_config_from_args
