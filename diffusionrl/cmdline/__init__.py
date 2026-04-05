"""Framework cmdline adaptation helpers."""

from .registry import (
    CMDLINE_CONFIG_PARSER_REGISTRY,
    register_cmdline_config_parser,
    resolve_cmdline_config_parser,
    resolve_component_cmdline_config_parser,
)

__all__ = [
    "CMDLINE_CONFIG_PARSER_REGISTRY",
    "register_cmdline_config_parser",
    "resolve_component_cmdline_config_parser",
    "resolve_cmdline_config_parser",
]
