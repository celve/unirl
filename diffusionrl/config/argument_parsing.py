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
    "rollout.topology": "Rollout Topology",
    "rollout.buffer": "Rollout Buffer",
    "rollout.control": "Rollout Control",
    "rollout.artifacts": "Rollout Artifacts",
    "rollout.evaluation": "Rollout Evaluation",
    "rollout.logging": "Rollout Logging",
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
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true/false (or 1/0, yes/no)."
    )


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
            raise ValueError(
                f"{field_name} must decode to a JSON object, got: {type(parsed).__name__}"
            )
        return dict(parsed)
    raise ValueError(
        f"{field_name} must be a JSON object (YAML mapping) or JSON object string, "
        f"got: {type(raw).__name__}"
    )


def parse_cli_mapping_object(value: Any) -> Dict[str, Any]:
    """Argparse-compatible wrapper around ``parse_mapping_object``."""
    try:
        return parse_mapping_object(value, field_name="CLI JSON object")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_cli_timestep_fraction(value: Any) -> Any:
    """Parse timestep_fraction CLI value."""
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise argparse.ArgumentTypeError(
            f"timestep_fraction tuple must have exactly 2 elements, got {len(value)}"
        )
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 1.0
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid timestep_fraction value: {value!r}. Error: {exc}"
            ) from exc
        if isinstance(parsed, list) and len(parsed) == 2:
            return (float(parsed[0]), float(parsed[1]))
        raise argparse.ArgumentTypeError(
            f"timestep_fraction list must have exactly 2 elements, got: {parsed!r}"
        )
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid timestep_fraction value: {value!r}. Error: {exc}"
                ) from exc
        raise argparse.ArgumentTypeError(
            "timestep_fraction comma-separated value must have exactly 2 "
            f"elements, got {len(parts)}"
        )
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid timestep_fraction value: {value!r}. Error: {exc}"
        ) from exc


def parse_cli_key_value(value: Any) -> tuple[str, Any]:
    """Parse ``key=value`` CLI overrides with lightweight scalar coercion."""
    text = str(value).strip()
    if not text or "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE, got: {value!r}."
        )
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE with a non-empty key, got: {value!r}."
        )
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
                raise argparse.ArgumentTypeError(
                    f"Expected a list value, got: {value!r}. Error: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise argparse.ArgumentTypeError(
                    f"Expected a list value, got {type(parsed).__name__}."
                )
            raw_items = list(parsed)
        else:
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

    normalized_item_type = resolve_cli_field_type(item_type)
    parsed_items: List[Any] = []
    for raw in raw_items:
        if normalized_item_type == bool:
            parsed_items.append(parse_cli_bool(raw))
        elif normalized_item_type == int:
            try:
                parsed_items.append(int(raw))
            except Exception as exc:
                raise argparse.ArgumentTypeError(
                    f"Expected integer list item, got: {raw!r}"
                ) from exc
        elif normalized_item_type == float:
            try:
                parsed_items.append(float(raw))
            except Exception as exc:
                raise argparse.ArgumentTypeError(
                    f"Expected float list item, got: {raw!r}"
                ) from exc
        else:
            parsed_items.append(str(raw))
    return parsed_items


def _resolve_field_help_text(field_info: Any) -> str:
    help_text = (field_info.metadata or {}).get("help")
    if help_text:
        return help_text
    default = resolve_dataclass_field_default(field_info)
    return f"{field_info.name} (default: {default})"


def _collect_field_specs_from_dataclass(
    config_type: type[Any],
    *,
    group_key: str,
    specs: list[tuple[str, Any, Any, str, str]],
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
            )
        )


def collect_cli_field_specs(
    *,
    training_args_type: type[Any],
    group_config_names: set[str],
    group_config_types: Dict[str, type[Any]],
) -> List[tuple[str, Any, Any, str, str]]:
    """Return ``(name, type, default, help, group_key)`` for CLI-exposed fields."""
    specs: List[tuple[str, Any, Any, str, str]] = []

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
        raise ImportError(
            "PyYAML is required for --config support. Install it with: pip install pyyaml"
        )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")
    return data


def _is_yaml_container_path(
    parts: List[str],
    *,
    group_config_names: set[str],
    group_subconfig_names: Dict[str, set[str]],
) -> bool:
    if len(parts) == 1 and parts[0] in group_config_names:
        return True
    if len(parts) == 2 and parts[0] in group_subconfig_names:
        return parts[1] in group_subconfig_names[parts[0]]
    return False


def _resolve_yaml_leaf_dest(
    parts: List[str],
    *,
    top_level_field_names: set[str],
    group_config_names: set[str],
    group_subconfig_names: Dict[str, set[str]],
) -> Optional[str]:
    if not parts:
        return None
    if len(parts) == 1:
        key = parts[0]
        if key in top_level_field_names:
            return key
        return None

    group = parts[0]
    if group not in group_config_names:
        return None
    if len(parts) == 2:
        if parts[1] in group_subconfig_names.get(group, set()):
            return None
        return ".".join(parts)
    if len(parts) == 3 and parts[1] in group_subconfig_names.get(group, set()):
        return ".".join(parts)
    return None


def _suggest_grouped_yaml_key(
    parts: List[str],
    *,
    known_destinations: set[str],
) -> Optional[str]:
    if not parts:
        return None
    if len(parts) == 2:
        prefix = f"{parts[0]}."
        suffix = f".{parts[1]}"
        candidates = sorted(
            dest
            for dest in known_destinations
            if "." in dest and dest.startswith(prefix) and dest.endswith(suffix)
            and dest != ".".join(parts)
        )
    elif len(parts) == 1:
        suffix = f".{parts[0]}"
        candidates = sorted(
            dest for dest in known_destinations
            if "." in dest and dest.endswith(suffix)
        )
    else:
        return None
    if not candidates:
        return None
    return candidates[0]


def _flatten_yaml_mapping(
    yaml_data: Dict[str, Any],
    *,
    top_level_field_names: set[str],
    group_config_names: set[str],
    group_subconfig_names: Dict[str, set[str]],
    known_destinations: set[str],
) -> Dict[str, Any]:
    """Flatten nested YAML mapping into parser destination keys."""
    flattened: Dict[str, Any] = {}
    origins: Dict[str, str] = {}

    def _value_repr(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(value)

    def _assign(dest_key: str, value: Any, *, source_path: str) -> None:
        if dest_key in flattened:
            previous_source = origins[dest_key]
            if previous_source != source_path:
                raise ValueError(
                    "Conflicting YAML keys map to the same argument destination "
                    f"'{dest_key}': '{previous_source}'={_value_repr(flattened[dest_key])} "
                    f"and '{source_path}'={_value_repr(value)}. "
                    "Keep only one style (prefer grouped keys)."
                )
        flattened[dest_key] = value
        origins[dest_key] = source_path

    def _walk(node: Dict[str, Any], prefix: List[str]) -> None:
        for raw_key, value in node.items():
            key = str(raw_key).replace("-", "_")
            key_parts = [part for part in key.split(".") if part]
            if not key_parts:
                continue
            parts = prefix + key_parts
            source_path = ".".join(parts)

            if len(parts) == 1:
                suggestion = _suggest_grouped_yaml_key(
                    parts,
                    known_destinations=known_destinations,
                )
                if suggestion is not None and parts[0] not in top_level_field_names:
                    raise ValueError(
                        f"Unsupported flat YAML key '{parts[0]}'. "
                        f"Use grouped YAML key '{suggestion}' instead."
                    )
            elif len(parts) == 2:
                suggestion = _suggest_grouped_yaml_key(
                    parts,
                    known_destinations=known_destinations,
                )
                if suggestion is not None:
                    raise ValueError(
                        f"Unsupported grouped YAML key '{source_path}'. "
                        f"Use nested YAML key '{suggestion}' instead."
                    )

            if isinstance(value, dict):
                leaf_dest = _resolve_yaml_leaf_dest(
                    parts,
                    top_level_field_names=top_level_field_names,
                    group_config_names=group_config_names,
                    group_subconfig_names=group_subconfig_names,
                )
                if leaf_dest is not None and not _is_yaml_container_path(
                    parts,
                    group_config_names=group_config_names,
                    group_subconfig_names=group_subconfig_names,
                ):
                    _assign(leaf_dest, value, source_path=source_path)
                    continue
                if _is_yaml_container_path(
                    parts,
                    group_config_names=group_config_names,
                    group_subconfig_names=group_subconfig_names,
                ):
                    _walk(value, parts)
                    continue
                _assign(".".join(parts), value, source_path=source_path)
                continue

            leaf_dest = _resolve_yaml_leaf_dest(
                parts,
                top_level_field_names=top_level_field_names,
                group_config_names=group_config_names,
                group_subconfig_names=group_subconfig_names,
            )
            if leaf_dest is not None:
                _assign(leaf_dest, value, source_path=source_path)
            else:
                _assign(".".join(parts), value, source_path=source_path)

    _walk(yaml_data, [])
    return flattened


def _coerce_yaml_value(
    *,
    key: str,
    cli_key: str,
    value: Any,
    action_by_dest: Dict[str, argparse.Action],
) -> Any:
    """Coerce YAML value using argparse converter for the destination key."""
    if cli_key in {
        "algorithm.algorithm_kwargs",
        "training.train_backend_kwargs",
    }:
        return parse_mapping_object(
            value,
            field_name=f"YAML key '{key}'",
        )

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
        known_destinations=set(defaults.keys()),
    )
    all_known_keys = set(defaults.keys())
    reported_cli_overrides: set[str] = set()

    for key, value in flattened_yaml.items():
        cli_key = key.replace("-", "_")
        if cli_key not in all_known_keys:
            message = f"Unknown key '{key}' in YAML config (no matching CLI argument)."
            if not allow_unknown_config_keys:
                raise ValueError(
                    message
                    + " Remove/fix the key, or pass --allow-unknown-config-keys "
                    "to ignore unknown YAML keys."
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
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "default": default,
        "help": help_text,
    }
    if field_name == "timestep_fraction":
        kwargs["type"] = parse_cli_timestep_fraction
    elif field_type == bool:
        kwargs["type"] = parse_cli_bool
    elif field_type == int:
        kwargs["type"] = int
    elif field_type == float:
        kwargs["type"] = float
    elif field_type is dict or get_origin(field_type) is dict:
        kwargs["type"] = parse_cli_mapping_object
    elif get_origin(field_type) is list:
        item_type = get_args(field_type)[0] if get_args(field_type) else str
        kwargs["type"] = (
            lambda value, item_type=item_type: parse_cli_list(value, item_type=item_type)
        )
    else:
        kwargs["type"] = str
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
