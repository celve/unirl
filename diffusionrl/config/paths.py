"""Path normalization helpers for configuration parsing and validation."""

from __future__ import annotations

import os
from typing import Any


def _trace_normalize_change(
    args: Any,
    key: str,
    before: Any,
    after: Any,
    *,
    source: str,
) -> None:
    """Emit normalize trace through callback set by arguments.validate_args()."""
    if before == after:
        return
    callback = getattr(args, "_normalize_trace_callback", None)
    if callable(callback):
        callback(key, before, after, source=source)


def repo_root(*, env_repo_root: str) -> str:
    """Resolve repository root from environment override or package-relative path."""
    env_root = os.getenv(env_repo_root)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    # paths.py lives at diffusionrl/config/paths.py.
    # Two levels up resolves to diffusionRL repository root.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_repo_relative_path(path: str, root: str) -> str:
    """Resolve path relative to repository root unless absolute."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(root, expanded))


def looks_like_local_path(path: str, root: str) -> bool:
    """Best-effort check whether a value should be interpreted as local path."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return True
    if any(
        path.startswith(prefix)
        for prefix in ("./", "../", "~", "data/", "models/", "outputs/", "shared_models/")
    ):
        return True
    if os.path.exists(expanded):
        return True
    if os.path.exists(os.path.join(root, expanded)):
        return True
    if path.count("/") >= 2:
        return True
    if path.endswith((".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".json", ".txt")):
        return True
    return False


def normalize_repo_relative_paths(
    args: Any,
    *,
    env_repo_root: str,
    env_data_root: str,
    env_model_root: str,
) -> None:
    """Normalize configured paths relative to repository root."""
    root = repo_root(env_repo_root=env_repo_root)
    data_root_env = os.getenv(env_data_root)
    model_root_env = os.getenv(env_model_root)

    grouped_path_fields = (
        ("rollout", "output_dir"),
        ("rollout", "logging_dir"),
        ("ray", "weight_sync_dir"),
        ("rollout", "resume_from_checkpoint"),
        ("debug", "debug_save_dir"),
        ("debug", "debug_load_path"),
    )
    for group_name, field_name in grouped_path_fields:
        group_obj = getattr(args, group_name)
        value = getattr(group_obj, field_name, None)
        if isinstance(value, str) and value:
            resolved = resolve_repo_relative_path(value, root)
            setattr(group_obj, field_name, resolved)
            _trace_normalize_change(
                args,
                f"{group_name}.{field_name}",
                value,
                resolved,
                source="normalize_repo_relative_paths",
            )

    data_path = getattr(args, "data_path", None)
    if isinstance(data_path, str) and data_path:
        if data_root_env and not os.path.isabs(os.path.expanduser(data_path)):
            trimmed = data_path[5:] if data_path.startswith("data/") else data_path
            resolved_data_path = os.path.abspath(
                os.path.join(os.path.expanduser(data_root_env), trimmed)
            )
            args.data_path = resolved_data_path
        else:
            resolved_data_path = resolve_repo_relative_path(data_path, root)
            args.data_path = resolved_data_path
        _trace_normalize_change(
            args,
            "data_path",
            data_path,
            resolved_data_path,
            source="normalize_repo_relative_paths",
        )

    eval_data_path = getattr(args, "eval_data_path", None)
    if isinstance(eval_data_path, str) and eval_data_path:
        if data_root_env and not os.path.isabs(os.path.expanduser(eval_data_path)):
            trimmed = eval_data_path[5:] if eval_data_path.startswith("data/") else eval_data_path
            resolved_eval_data_path = os.path.abspath(
                os.path.join(os.path.expanduser(data_root_env), trimmed)
            )
            args.eval_data_path = resolved_eval_data_path
        else:
            resolved_eval_data_path = resolve_repo_relative_path(eval_data_path, root)
            args.eval_data_path = resolved_eval_data_path
        _trace_normalize_change(
            args,
            "eval_data_path",
            eval_data_path,
            resolved_eval_data_path,
            source="normalize_repo_relative_paths",
        )

    model_like_fields = (
        ("model", "pretrained_model_saved_path"),
        ("model", "vae_saved_path"),
        ("model", "text_encoder_path"),
        ("reward", "reward_model_saved_path"),
    )
    for group_name, field_name in model_like_fields:
        group_obj = getattr(args, group_name)
        value = getattr(group_obj, field_name, None)
        if not isinstance(value, str) or not value:
            continue
        if not looks_like_local_path(value, root):
            continue
        if model_root_env and not os.path.isabs(os.path.expanduser(value)):
            trimmed = value[7:] if value.startswith("models/") else value
            resolved = os.path.abspath(os.path.join(os.path.expanduser(model_root_env), trimmed))
            setattr(group_obj, field_name, resolved)
        else:
            resolved = resolve_repo_relative_path(value, root)
            setattr(group_obj, field_name, resolved)
        _trace_normalize_change(
            args,
            f"{group_name}.{field_name}",
            value,
            resolved,
            source="normalize_repo_relative_paths",
        )


def is_probably_local_weight_sync_dir(path: str, *, root: str) -> bool:
    """Best-effort guard for local-only paths in multi-node checkpoint sync."""
    if not path:
        return True
    real = os.path.realpath(path)
    for prefix in ("/tmp", "/var/tmp", "/dev/shm"):
        if real == prefix or real.startswith(prefix + os.sep):
            return True
    if real == root or real.startswith(root + os.sep):
        return True
    return False


__all__ = [
    "repo_root",
    "resolve_repo_relative_path",
    "looks_like_local_path",
    "normalize_repo_relative_paths",
    "is_probably_local_weight_sync_dir",
]
