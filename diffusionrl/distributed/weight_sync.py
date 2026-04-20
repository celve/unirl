"""Weight synchronization config builder for rollout updates.

Replaces the former coordinator class hierarchy with a single config-builder
function.  The training loop calls the runtime directly:

    config = build_weight_sync_config(args, launch_config, mode=..., rollout_runtime=...)
    if config:
        training_runtime.setup_weight_sync(config)
    ...
    training_runtime.sync_weights_to_rollout()
    ...
    training_runtime.teardown_weight_sync()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from diffusionrl.config.assembly import LaunchConfig
from diffusionrl.types.engine import EngineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_target_modules(args: Any) -> list[str]:
    raw = args.sync.target_modules
    if isinstance(raw, (list, tuple)) and raw:
        return [str(name) for name in raw]
    return ["transformer"]


def _resolve_bucket_size_mb(args: Any) -> int:
    bucket_size_mb = int(args.sync.bucket_size)
    if bucket_size_mb < 1:
        raise ValueError(f"sync.bucket_size must be >= 1, got {bucket_size_mb}.")
    return bucket_size_mb


def _resolve_flush_cache(args: Any) -> bool:
    return bool(args.sync.flush_cache)


def _validate_tensor_sync_topology(launch_config: LaunchConfig) -> None:
    """Guard invalid topology values for tensor/distributed sync paths."""
    rollout = launch_config.rollout
    if rollout is None:
        raise ValueError(
            "Tensor/distributed weight sync requires a dedicated rollout launch config."
        )
    engine_config = rollout.engine_init_payload.component_config
    if not isinstance(engine_config, EngineConfig):
        raise ValueError(
            "Tensor/distributed weight sync requires rollout.engine_init_payload.component_config "
            f"to be an EngineConfig, got: {type(engine_config).__name__}"
        )
    tp_size_int = int(engine_config.tp_size or 1)
    if tp_size_int < 1:
        raise ValueError(f"Invalid tp_size={tp_size_int}. Expected tp_size >= 1.")


def _resolve_rollout_tp_payload_count(rollout_runtime: Any) -> int:
    """Resolve payload fan-out count for IPC tensor updates.

    SGLang workers index `serialized_named_tensors` by local TP rank.
    For tp_size>1, provide one slot per TP rank (same payload can be reused).
    """
    if rollout_runtime is None:
        return 1
    topology = rollout_runtime.get_weight_sync_topology()
    if not isinstance(topology, dict):
        raise ValueError(
            f"Invalid rollout weight-sync topology payload: {topology!r}"
        )
    payload_count = int(topology.get("num_gpus_per_actor", 1))
    if payload_count < 1:
        raise ValueError(
            "rollout weight-sync topology must expose num_gpus_per_actor >= 1. "
            f"Got {payload_count}."
        )
    return payload_count


def _select_export_format(launch_config: LaunchConfig) -> str:
    """Select checkpoint export format based on rollout engine and backend capabilities."""
    rollout = launch_config.rollout
    if rollout is None:
        raise ValueError(
            "Checkpoint weight sync requires a dedicated rollout runtime config."
        )
    engine_type = str(rollout.rollout_engine or "").strip().lower()
    if not engine_type:
        raise ValueError(
            "Checkpoint weight sync requires rollout.rollout_engine to be normalized. "
            "Validate args before selecting dedicated rollout checkpoint export format."
        )
    backend_caps = (
        launch_config.training.backend_capabilities.as_dict()
        if launch_config.training.backend_capabilities
        else {}
    )
    if backend_caps:
        preferred_by_engine = backend_caps.get("preferred_weight_export_format_by_rollout_engine")
        preferred_format = None
        if isinstance(preferred_by_engine, dict):
            preferred_format = preferred_by_engine.get(engine_type)
        if not preferred_format:
            preferred_format = backend_caps.get("preferred_weight_export_format")
        supported_formats = set(backend_caps.get("supported_weight_export_formats") or ())
        if preferred_format in supported_formats:
            return str(preferred_format)
    if engine_type == "sglang":
        return "sglang_transformer_safetensors"
    return "state_dict"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_weight_sync_config(
    args: Any,
    launch_config: LaunchConfig,
    *,
    mode: str,
    rollout_runtime: Optional[Any] = None,
) -> dict:
    """Build config dict for training_runtime.setup_weight_sync().

    Returns empty dict for 'disabled' mode (no sync needed).
    """
    if mode == "disabled":
        return {}

    if mode == "tensor_payload":
        _validate_tensor_sync_topology(launch_config)
        if rollout_runtime is None:
            raise RuntimeError("tensor_payload weight sync requires a rollout runtime.")
        rollout_actors = rollout_runtime.get_rollout_actors()
        return {
            "mode": "tensor_payload",
            "rollout_actors": rollout_actors,
            "target_modules": _resolve_target_modules(args),
            "bucket_size_mb": _resolve_bucket_size_mb(args),
            "flush_cache": _resolve_flush_cache(args),
            "rollout_num_gpus_per_engine": _resolve_rollout_tp_payload_count(rollout_runtime),
        }

    if mode == "nccl_broadcast":
        _validate_tensor_sync_topology(launch_config)
        if rollout_runtime is None:
            raise RuntimeError("nccl_broadcast weight sync requires a rollout runtime.")
        rollout_actors = rollout_runtime.get_rollout_actors()
        topology = rollout_runtime.get_weight_sync_topology()
        total_rollout_gpus = int(topology.get("total_gpus", 0)) if isinstance(topology, dict) else 0
        return {
            "mode": "nccl_broadcast",
            "rollout_actors": rollout_actors,
            "target_modules": _resolve_target_modules(args),
            "bucket_size_mb": _resolve_bucket_size_mb(args),
            "flush_cache": _resolve_flush_cache(args),
            "rollout_num_gpus": total_rollout_gpus,
            "rollout_num_gpus_per_engine": int(
                topology.get("num_gpus_per_actor", 1)
            ) if isinstance(topology, dict) else 1,
        }

    if mode == "checkpoint_path":
        if rollout_runtime is None:
            raise RuntimeError("checkpoint_path weight sync requires a rollout runtime.")
        return {
            "mode": "checkpoint_path",
            "rollout_actors": [],
            "rollout_runtime": rollout_runtime,
            "export_format": _select_export_format(launch_config),
            "weight_sync_dir": str(args.sync.dir),
        }

    raise ValueError(
        f"Unsupported sync.protocol={mode}. "
        f"Expected one of: disabled, tensor_payload, nccl_broadcast, checkpoint_path"
    )


__all__ = ["build_weight_sync_config"]
