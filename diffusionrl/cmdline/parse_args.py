"""CLI entry point for parsing ``TrainingArguments`` from command line and YAML.

This module owns the argparse-based CLI surface. Config schema definitions,
validation logic, and derived config views remain in
``diffusionrl.cmdline.schema``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from diffusionrl.cmdline.argument_parsing import (
    GROUP_DISPLAY_NAMES,
    build_add_argument_kwargs,
    build_cli_option_strings,
    collect_cli_field_specs,
    collect_explicit_cli_destinations,
    load_yaml_mapping,
    merge_yaml_overrides,
    parse_cli_key_value,
)
from diffusionrl.cmdline.schema import (
    TrainingArguments,
    GROUP_CONFIG_NAMES,
    GROUP_CONFIG_TYPES,
    GROUP_SUBCONFIG_NAMES,
    TOP_LEVEL_FIELD_NAMES,
    print_config_views,
    validate_and_derive_config,
)
from diffusionrl.config.assembly import DerivedConfig


def _merge_algorithm_kwarg_overrides(raw_args: Dict[str, Any]) -> None:
    """Overlay repeated algorithm-specific ``--algorithm.kwarg key=value`` items."""
    overrides = raw_args.pop("_algorithm_kwarg_overrides", None) or []
    if not overrides:
        return

    dest = "algorithm.algorithm_kwargs"
    merged = dict(raw_args.get(dest) or {})
    for item in overrides:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "Internal error: parsed --algorithm.kwarg item must be a (key, value) pair. "
                f"Got: {item!r}"
            )
        key, value = item
        merged[str(key)] = value
    raw_args[dest] = merged


def parse_args_with_derived_config(
    argv: Optional[List[str]] = None,
) -> Tuple[TrainingArguments, DerivedConfig]:
    """Parse command line arguments and return validated args plus derived config."""
    parser = argparse.ArgumentParser(
        description="diffusionrl training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override YAML values.",
    )
    parser.add_argument(
        "--print-derived-config",
        action="store_true",
        help="Print derived config after validation.",
    )
    parser.add_argument(
        "--allow-unknown-config-keys",
        action="store_true",
        help="Allow unknown keys in --config YAML (default is fail-fast).",
    )

    cli_field_specs = collect_cli_field_specs(
        training_args_type=TrainingArguments,
        group_config_names=GROUP_CONFIG_NAMES,
        group_config_types=GROUP_CONFIG_TYPES,
    )

    hidden_cli_destinations = {
        "algorithm.algorithm_kwargs",
    }

    arg_groups: Dict[str, argparse._ArgumentGroup] = {}
    for field_name, field_type, default, help_text, group_key, choices in cli_field_specs:
        dest = f"{group_key}.{field_name}" if group_key else field_name
        if dest in hidden_cli_destinations:
            continue
        if group_key not in arg_groups:
            display_name = GROUP_DISPLAY_NAMES.get(group_key, group_key)
            arg_groups[group_key] = parser.add_argument_group(display_name)
        group = arg_groups[group_key]

        option_strings = build_cli_option_strings(field_name, group_key)
        add_kwargs = build_add_argument_kwargs(
            field_type,
            default,
            help_text,
            field_name=field_name,
            choices=choices,
        )
        add_kwargs["dest"] = dest
        group.add_argument(*option_strings, **add_kwargs)

    algorithm_group = arg_groups.get("algorithm")
    if algorithm_group is None:
        algorithm_group = parser.add_argument_group(
            GROUP_DISPLAY_NAMES.get("algorithm", "Algorithm & Advantage")
        )
        arg_groups["algorithm"] = algorithm_group
    algorithm_group.add_argument(
        "--algorithm.kwarg",
        dest="_algorithm_kwarg_overrides",
        action="append",
        default=[],
        type=parse_cli_key_value,
        metavar="KEY=VALUE",
        help=(
            "Append one algorithm-specific algorithm.algorithm_kwargs override. "
            "Shared framework-owned keys must use dedicated --algorithm.* flags. "
            "Repeat this flag to set multiple extension keys."
        ),
    )

    cli_argv = list(argv) if argv is not None else sys.argv[1:]
    parsed_args = parser.parse_args(cli_argv)
    explicit_cli_keys = collect_explicit_cli_destinations(cli_argv, parser)
    action_by_dest = {
        action.dest: action for action in parser._actions if getattr(action, "dest", None)
    }

    raw_args = vars(parsed_args)
    print_derived_config = bool(raw_args.get("print_derived_config", False))
    allow_unknown_config_keys = bool(raw_args.get("allow_unknown_config_keys", False))

    if raw_args.get("config"):
        defaults: Dict[str, Any] = {}
        for action in parser._actions:
            dest = getattr(action, "dest", None)
            if not dest or dest == "help" or dest in defaults:
                continue
            defaults[dest] = action.default
        defaults.setdefault("algorithm.algorithm_kwargs", {})
        yaml_data = load_yaml_mapping(raw_args["config"])
        merge_yaml_overrides(
            raw_args,
            yaml_data=yaml_data,
            defaults=defaults,
            explicit_cli_keys=explicit_cli_keys,
            action_by_dest=action_by_dest,
            allow_unknown_config_keys=allow_unknown_config_keys,
            top_level_field_names=TOP_LEVEL_FIELD_NAMES,
            group_config_names=GROUP_CONFIG_NAMES,
            group_subconfig_names=GROUP_SUBCONFIG_NAMES,
        )

    raw_args.pop("config", None)
    raw_args.pop("print_derived_config", None)
    raw_args.pop("allow_unknown_config_keys", None)
    _merge_algorithm_kwarg_overrides(raw_args)

    grouped_kwargs: Dict[str, Dict[str, Any]] = {name: {} for name in GROUP_CONFIG_TYPES}
    sub_kwargs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    top_level_kwargs: Dict[str, Any] = {}
    for key, value in raw_args.items():
        if "." in key:
            parts = key.split(".")
            group_name = parts[0]
            if len(parts) == 2 and group_name in GROUP_CONFIG_TYPES:
                grouped_kwargs[group_name][parts[1]] = value
            elif (
                len(parts) == 3
                and group_name in GROUP_CONFIG_TYPES
                and parts[1] in GROUP_SUBCONFIG_NAMES.get(group_name, set())
            ):
                sub_kwargs.setdefault(group_name, {}).setdefault(parts[1], {})[parts[2]] = value
            else:
                top_level_kwargs[key] = value
        else:
            top_level_kwargs[key] = value

    for group_name, group_type in GROUP_CONFIG_TYPES.items():
        kwargs = dict(grouped_kwargs[group_name])
        for info in fields(group_type):
            ft = info.type
            if isinstance(ft, type) and is_dataclass(ft):
                sub_data = sub_kwargs.get(group_name, {}).get(info.name, {})
                kwargs[info.name] = ft(**sub_data)
        top_level_kwargs[group_name] = group_type(**kwargs)

    args = TrainingArguments(**top_level_kwargs)
    args, derived_config = validate_and_derive_config(args)
    print_config_views(
        args=args,
        print_derived_config=print_derived_config,
        derived_config=derived_config,
    )
    return args, derived_config


def parse_args(argv: Optional[List[str]] = None) -> TrainingArguments:
    """Parse command line arguments and return ``TrainingArguments``."""
    args, _ = parse_args_with_derived_config(argv)
    return args
