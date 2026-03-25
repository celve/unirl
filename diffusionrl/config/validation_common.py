"""Shared validation primitives and schema-level checks."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional

from diffusionrl.algorithms.construction import resolve_algorithm_path
from diffusionrl.config.resolution import ResolvedModelSpec, derive_model_spec
from diffusionrl.utils.misc import load_function

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"


class PrecisionName(str, Enum):
    BF16 = "bf16"
    BFLOAT16 = "bfloat16"
    FP16 = "fp16"
    FLOAT16 = "float16"
    HALF = "half"
    FP32 = "fp32"
    FLOAT32 = "float32"
    FLOAT = "float"


def repo_root(*, env_repo_root: str) -> str:
    """Resolve repository root from environment override or package-relative path."""
    env_root = os.getenv(env_repo_root)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


def validate_dotpath(path: str, *, label: str) -> None:
    """Fail fast when a configured dotpath is not importable."""
    try:
        load_function(path)
    except Exception as exc:
        raise ValueError(
            f"Invalid {label} path: {path!r}. Import failed: {exc}. "
            f"Check that the module is installed and the dotpath is correct "
            f"(e.g. 'diffusionrl.algorithms.grpo.GRPOAlgorithm')."
        ) from exc


def validate_precision_name(value: Any, *, field_name: str) -> None:
    """Validate precision aliases used by config-facing precision fields."""
    key = str(value or "").strip().lower()
    try:
        PrecisionName(key)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of bf16/fp16/fp32, got: {value!r}"
        ) from exc


def validate_grouped_configs(args: Any) -> None:
    """Run per-group config dataclass validators."""
    args.model.validate()
    args.sampling.validate()
    args.reward.validate()
    args.ray.validate()
    args.sync.validate()
    args.algorithm.validate()
    args.training.validate()
    args.precision.validate()
    args.rollout.validate()
    args.debug.validate()


def validate_dynamic_dotpaths(
    args: Any,
    *,
    resolved_model: Optional[ResolvedModelSpec] = None,
    include_data_source: bool = True,
    include_rollout_buffer_plugins: bool = True,
) -> None:
    """Validate configured dynamic extension dotpaths."""
    if resolved_model is None:
        resolved_model = derive_model_spec(args)
    validate_dotpath(resolved_model.model_path, label="model")
    validate_dotpath(resolved_model.sampler_path, label="sampler")
    validate_dotpath(
        resolve_algorithm_path(
            algorithm_type=args.algorithm.algorithm_type,
            algorithm_path=args.algorithm.algorithm_path,
        ),
        label="algorithm",
    )
    if include_data_source:
        validate_dotpath(args.data_source_path, label="data_source")
    if getattr(args, "rollout_function_path", None):
        validate_dotpath(args.rollout_function_path, label="rollout_function")
    if getattr(args, "eval_function_path", None):
        validate_dotpath(args.eval_function_path, label="eval_function")
    if getattr(args, "reward_hook_path", None):
        validate_dotpath(args.reward_hook_path, label="reward_hook")
    if args.training.train_backend_path:
        validate_dotpath(args.training.train_backend_path, label="train_backend")
    if args.sampling.replay_sampler_path:
        validate_dotpath(args.sampling.replay_sampler_path, label="replay_sampler")
    if include_rollout_buffer_plugins:
        rollout_buffer_plugin_paths = args.rollout.buffer.plugin_paths or ""
        if isinstance(rollout_buffer_plugin_paths, str):
            for plugin_path in [part.strip() for part in rollout_buffer_plugin_paths.split(",") if part.strip()]:
                validate_dotpath(plugin_path, label="rollout_buffer_plugin")


def validate_colocate_fractions(args: Any) -> None:
    """Validate colocate GPU fraction bounds."""
    if args.ray.colocate_training_gpu_fraction <= 0 or args.ray.colocate_rollout_gpu_fraction <= 0:
        raise ValueError(
            "colocate_training_gpu_fraction and colocate_rollout_gpu_fraction must be > 0"
        )
    if args.ray.colocate_training_gpu_fraction + args.ray.colocate_rollout_gpu_fraction > 1.0:
        raise ValueError(
            "colocate_training_gpu_fraction + colocate_rollout_gpu_fraction must be <= 1.0"
        )

__all__ = [
    "ENV_REPO_ROOT",
    "is_probably_local_weight_sync_dir",
    "repo_root",
    "validate_colocate_fractions",
    "validate_dotpath",
    "validate_dynamic_dotpaths",
    "validate_grouped_configs",
    "validate_precision_name",
]
