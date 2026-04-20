"""CLI/YAML parsing helpers for ``TrainingArguments``.

This module only owns input-shape mechanics:
- dataclass schema -> argparse specs
- YAML normalize/merge/coerce
- CLI option formatting and explicit override tracking

It intentionally does not own config semantics or validation policy.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_args, get_origin

GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "": "General",
    "model": "Model Configuration",
    "sampling": "Sampling & Inference",
    "reward": "Reward Configuration",
    "ray": "Ray & Resource Layout",
    "sync": "Weight Sync",
    "algorithm": "Algorithm & Advantage",
    "algorithm.rollout_scheduler": "Rollout Index Scheduler",
    "algorithm.training_scheduler": "Training Index Scheduler",
    "training": "Training & Optimization",
    "precision": "Precision",
    "rollout": "Rollout Configuration",
    "evaluation": "Evaluation",
    "logging": "Logging & Reporting",
    "debug": "Debug Mode & Artifact Saving",
}


def resolve_dataclass_field_default(field_info: Any, *, missing: Any = None) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return missing


def resolve_cli_field_type(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    inner_types = [t for t in get_args(field_type) if t is not type(None)]
    if len(inner_types) == 1:
        return inner_types[0]
    return field_type


def parse_cli_bool(value: Any) -> bool:
    """Parse boolean CLI values with strict validation."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}. Use true/false (or 1/0, yes/no).")


def parse_mapping_object(raw: Any, *, field_name: str) -> Dict[str, Any]:
    """Parse a mapping-like payload without mutating the source."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must decode to a JSON object, got: {type(parsed).__name__}")
        return dict(parsed)
    raise ValueError(
        f"{field_name} must be a JSON object (YAML mapping) or JSON object string, got: {type(raw).__name__}"
    )


def parse_cli_mapping_object(value: Any) -> Dict[str, Any]:
    """Argparse-compatible wrapper around ``parse_mapping_object``."""
    try:
        return parse_mapping_object(value, field_name="CLI JSON object")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_float_or_float_pair(value: Any) -> Any:
    """Parse a single float or a ``(float, float)`` range from CLI/YAML."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise argparse.ArgumentTypeError(f"Expected exactly 2 elements for a range, got {len(value)}")
        return (float(value[0]), float(value[1]))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 1.0
    try:
        parsed = json.loads(text)
    except Exception:
        pass
    else:
        if isinstance(parsed, list):
            if len(parsed) != 2:
                raise argparse.ArgumentTypeError(f"Expected exactly 2 elements for a range, got {len(parsed)}")
            return (float(parsed[0]), float(parsed[1]))
        if isinstance(parsed, (int, float)):
            return float(parsed)
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a float or [float, float], got: {value!r}") from exc


def parse_cli_key_value(value: Any) -> tuple[str, Any]:
    """Parse ``key=value`` CLI overrides with lightweight scalar coercion."""
    text = str(value).strip()
    if not text or "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {value!r}.")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE with a non-empty key, got: {value!r}.")
    raw = raw.strip()
    if raw == "":
        return key, ""

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    if lowered in {"null", "none"}:
        return key, None

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    else:
        return key, parsed

    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return key, float(raw)
        return key, int(raw)
    except Exception:
        return key, raw


def parse_cli_list(value: Any, *, item_type: Any = str) -> List[Any]:
    """Parse comma-separated or JSON list CLI values into typed Python lists."""
    if isinstance(value, list):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise argparse.ArgumentTypeError(f"Expected a list value, got: {value!r}. Error: {exc}") from exc
            if not isinstance(parsed, list):
                raise argparse.ArgumentTypeError(f"Expected a list value, got {type(parsed).__name__}.")
            raw_items = list(parsed)
        else:
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

    normalized_item_type = resolve_cli_field_type(item_type)
    parsed_items: List[Any] = []
    for raw in raw_items:
        if normalized_item_type is bool:
            parsed_items.append(parse_cli_bool(raw))
        elif normalized_item_type is int:
            try:
                parsed_items.append(int(raw))
            except Exception as exc:
                raise argparse.ArgumentTypeError(f"Expected integer list item, got: {raw!r}") from exc
        elif normalized_item_type is float:
            try:
                parsed_items.append(float(raw))
            except Exception as exc:
                raise argparse.ArgumentTypeError(f"Expected float list item, got: {raw!r}") from exc
        else:
            parsed_items.append(str(raw))
    return parsed_items


def _resolve_field_help_text(field_info: Any) -> str:
    help_text = (field_info.metadata or {}).get("help")
    if help_text:
        return help_text
    default = resolve_dataclass_field_default(field_info)
    return f"{field_info.name} (default: {default})"


def _resolve_field_choices(field_info: Any) -> Optional[List[str]]:
    return (field_info.metadata or {}).get("choices")


def _collect_field_specs_from_dataclass(
    config_type: type[Any],
    *,
    group_key: str,
    specs: list[tuple[str, Any, Any, str, str, Optional[List[str]]]],
) -> None:
    for field_info in fields(config_type):
        field_type = field_info.type
        resolved_type = None
        if isinstance(field_type, type) and is_dataclass(field_type):
            resolved_type = field_type

        if resolved_type is not None:
            _collect_field_specs_from_dataclass(
                resolved_type,
                group_key=f"{group_key}.{field_info.name}",
                specs=specs,
            )
            continue

        specs.append(
            (
                field_info.name,
                resolve_cli_field_type(field_info.type),
                resolve_dataclass_field_default(field_info),
                _resolve_field_help_text(field_info),
                group_key,
                _resolve_field_choices(field_info),
            )
        )


def collect_cli_field_specs(
    *,
    training_args_type: type[Any],
    group_config_names: set[str],
    group_config_types: Dict[str, type[Any]],
) -> List[tuple[str, Any, Any, str, str, Optional[List[str]]]]:
    """Return ``(name, type, default, help, group_key, choices)`` for CLI-exposed fields."""
    specs: List[tuple[str, Any, Any, str, str, Optional[List[str]]]] = []

    for field_info in fields(training_args_type):
        if field_info.name in group_config_names:
            continue
        specs.append(
            (
                field_info.name,
                resolve_cli_field_type(field_info.type),
                resolve_dataclass_field_default(field_info),
                _resolve_field_help_text(field_info),
                "",
                _resolve_field_choices(field_info),
            )
        )

    for group_name, config_type in group_config_types.items():
        _collect_field_specs_from_dataclass(
            config_type,
            group_key=group_name,
            specs=specs,
        )

    return specs


def load_yaml_mapping(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return a mapping."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for --config support. Install it with: pip install pyyaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")
    return data


def _flatten_yaml_mapping(
    yaml_data: Dict[str, Any],
    *,
    top_level_field_names: set[str],
    group_config_names: set[str],
    group_subconfig_names: Dict[str, set[str]],
) -> Dict[str, Any]:
    """Flatten nested YAML mapping into parser destination keys.

    Only the canonical nested format is supported::

        model:
          model_type: flux
        algorithm:
          rollout_scheduler:
            timestep_strategy: window
    """
    flat: Dict[str, Any] = {}
    for raw_key, value in yaml_data.items():
        key = str(raw_key).replace("-", "_")
        if key in top_level_field_names:
            flat[key] = value
            continue
        if key in group_config_names and isinstance(value, dict):
            for raw_k2, v2 in value.items():
                k2 = str(raw_k2).replace("-", "_")
                if k2 in group_subconfig_names.get(key, set()) and isinstance(v2, dict):
                    for raw_k3, v3 in v2.items():
                        k3 = str(raw_k3).replace("-", "_")
                        flat[f"{key}.{k2}.{k3}"] = v3
                else:
                    flat[f"{key}.{k2}"] = v2
            continue
        flat[key] = value
    return flat


def _coerce_yaml_value(
    *,
    key: str,
    cli_key: str,
    value: Any,
    action_by_dest: Dict[str, argparse.Action],
) -> Any:
    """Coerce YAML value using argparse converter for the destination key."""
    action = action_by_dest.get(cli_key)
    converter = getattr(action, "type", None) if action is not None else None
    if converter is None or value is None:
        return value
    if converter is str and not isinstance(value, str):
        return value
    try:
        return converter(value)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"Invalid value for YAML key '{key}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid value for YAML key '{key}': {exc}") from exc


def merge_yaml_overrides(
    raw_args: Dict[str, Any],
    *,
    yaml_data: Dict[str, Any],
    defaults: Dict[str, Any],
    explicit_cli_keys: set[str],
    action_by_dest: Dict[str, argparse.Action],
    allow_unknown_config_keys: bool,
    top_level_field_names: set[str],
    group_config_names: set[str],
    group_subconfig_names: Dict[str, set[str]],
) -> None:
    """Apply YAML values to raw_args for keys the user did not set on CLI."""
    flattened_yaml = _flatten_yaml_mapping(
        yaml_data,
        top_level_field_names=top_level_field_names,
        group_config_names=group_config_names,
        group_subconfig_names=group_subconfig_names,
    )
    all_known_keys = set(defaults.keys())
    reported_cli_overrides: set[str] = set()

    for key, value in flattened_yaml.items():
        cli_key = key.replace("-", "_")
        if cli_key not in all_known_keys:
            message = f"Unknown key '{key}' in YAML config (no matching CLI argument)."
            if not allow_unknown_config_keys:
                raise ValueError(
                    message + " Remove/fix the key, or pass --allow-unknown-config-keys to ignore unknown YAML keys."
                )
            warnings.warn(
                message + " Ignoring because --allow-unknown-config-keys is set.",
                stacklevel=3,
            )
            continue

        if cli_key in explicit_cli_keys:
            if cli_key not in reported_cli_overrides:
                warnings.warn(
                    f"YAML key '{key}' ignored because CLI explicitly set '{cli_key}' (CLI takes precedence).",
                    stacklevel=3,
                )
                reported_cli_overrides.add(cli_key)
            continue

        if raw_args.get(cli_key) == defaults.get(cli_key):
            raw_args[cli_key] = _coerce_yaml_value(
                key=key,
                cli_key=cli_key,
                value=value,
                action_by_dest=action_by_dest,
            )


def build_cli_option_strings(field_name: str, group_key: str) -> List[str]:
    """Build CLI option strings."""
    field_opt = field_name.replace("_", "-")
    option = f"--{field_opt}"
    if not group_key:
        return [option]

    dotted_group = ".".join(part.replace("_", "-") for part in group_key.split("."))
    dotted_option = f"--{dotted_group}.{field_opt}"
    return [dotted_option]


def build_add_argument_kwargs(
    field_type: Any,
    default: Any,
    help_text: str,
    *,
    field_name: str = "",
    choices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "default": default,
        "help": help_text,
    }
    if field_name == "timestep_fraction":
        kwargs["type"] = _parse_float_or_float_pair
    elif field_type is bool:
        kwargs["type"] = parse_cli_bool
    elif field_type is int:
        kwargs["type"] = int
    elif field_type is float:
        kwargs["type"] = float
    elif field_type is dict or get_origin(field_type) is dict:
        kwargs["type"] = parse_cli_mapping_object
    elif get_origin(field_type) is list:
        item_type = get_args(field_type)[0] if get_args(field_type) else str
        kwargs["type"] = lambda value, item_type=item_type: parse_cli_list(value, item_type=item_type)
    else:
        kwargs["type"] = str
    if choices:
        kwargs["choices"] = choices
    return kwargs


def collect_explicit_cli_destinations(
    argv: List[str],
    parser: argparse.ArgumentParser,
) -> set[str]:
    """Collect parser destination names explicitly provided via CLI options."""
    explicit: set[str] = set()
    option_to_action = getattr(parser, "_option_string_actions", {})
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_to_action.get(option)
        if action is not None:
            explicit.add(action.dest)
    return explicit


__all__ = [
    "GROUP_DISPLAY_NAMES",
    "build_add_argument_kwargs",
    "build_cli_option_strings",
    "collect_cli_field_specs",
    "collect_explicit_cli_destinations",
    "load_yaml_mapping",
    "merge_yaml_overrides",
    "parse_cli_mapping_object",
    "parse_cli_key_value",
    "parse_mapping_object",
]
