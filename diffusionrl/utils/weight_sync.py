"""Weight synchronization protocols for rollout updates.

Two-phase lifecycle: setup() -> sync() x N -> teardown().
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import ray
import logging

from diffusionrl.utils.misc import load_function
from diffusionrl.utils.weight_sync_checkpoint import cleanup_published_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_target_modules(args: Any) -> list[str]:
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    if not isinstance(engine_kwargs, dict):
        return ["transformer"]
    raw = engine_kwargs.get("target_modules")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(name) for name in raw]
    return ["transformer"]


def _resolve_bucket_size_mb(args: Any) -> int:
    explicit = getattr(args, "weight_sync_bucket_mb", None)
    if explicit is not None:
        return max(1, int(explicit))
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    if isinstance(engine_kwargs, dict) and engine_kwargs.get("weight_sync_bucket_mb") is not None:
        return max(1, int(engine_kwargs["weight_sync_bucket_mb"]))
    return 256


def _resolve_flush_cache(args: Any) -> bool:
    explicit = getattr(args, "weight_sync_flush_cache", None)
    if explicit is not None:
        return bool(explicit)
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    if isinstance(engine_kwargs, dict) and engine_kwargs.get("weight_sync_flush_cache") is not None:
        return bool(engine_kwargs["weight_sync_flush_cache"])
    return True


def _validate_tensor_sync_topology(args: Any) -> None:
    """Guard invalid topology values for tensor/distributed sync paths."""
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    tp_size = None
    if isinstance(engine_kwargs, dict):
        tp_size = engine_kwargs.get("tp_size")
    if tp_size is None:
        tp_size = getattr(args, "tp_size", 1)
    try:
        tp_size_int = int(tp_size)
    except Exception:
        tp_size_int = 1
    if tp_size_int < 1:
        raise ValueError(f"Invalid tp_size={tp_size_int}. Expected tp_size >= 1.")


def _resolve_rollout_tp_payload_count(rollout_manager: Any) -> int:
    """Resolve payload fan-out count for IPC tensor updates.

    SGLang workers index `serialized_named_tensors` by local TP rank.
    For tp_size>1, provide one slot per TP rank (same payload can be reused).
    """
    topology = ray.get(rollout_manager.get_weight_sync_topology.remote())
    if not isinstance(topology, dict):
        return 1
    try:
        return max(1, int(topology.get("num_gpus_per_actor", 1)))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Result of a weight sync operation."""
    version: int
    mode: str
    rollout_id: int
    elapsed_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WeightSyncProtocol (two-phase base class)
# ---------------------------------------------------------------------------

class WeightSyncProtocol(ABC):
    """Two-phase weight sync: setup() -> sync() x N -> teardown()."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self._version = 0
        self._is_setup = False
        self._training_group = None
        self._rollout_manager = None

    def setup(self, *, training_group: Any, rollout_manager: Any) -> None:
        """Bind runtime handles and perform one-time initialization."""
        self._training_group = training_group
        self._rollout_manager = rollout_manager
        self._do_setup()
        self._is_setup = True

    def sync(self, *, rollout_id: int, **kwargs: Any) -> SyncResult:
        """Synchronize latest policy weights to rollout side.

        Accepts **kwargs for backward compatibility with callers that still
        pass training_group/rollout_manager as keyword arguments.
        """
        if not self._is_setup:
            # Backward compat: auto-setup from kwargs if setup() was not called.
            training_group = kwargs.get("training_group")
            rollout_manager = kwargs.get("rollout_manager")
            if training_group is not None and rollout_manager is not None:
                self.setup(training_group=training_group, rollout_manager=rollout_manager)
            else:
                raise RuntimeError(
                    "WeightSyncProtocol.sync() called before setup(). "
                    "Call setup(training_group=..., rollout_manager=...) first."
                )

        t0 = time.monotonic()
        ray.get(self._rollout_manager.wake_up.remote())
        extra = self._do_sync(rollout_id=rollout_id) or {}
        self._version += 1
        elapsed_ms = (time.monotonic() - t0) * 1000
        result = SyncResult(
            version=self._version,
            mode=self._mode_name,
            rollout_id=rollout_id,
            elapsed_ms=elapsed_ms,
            extra=extra,
        )
        logger.info(
            "Weight sync v%d: %s %.1fms (rollout_id=%d)",
            result.version, result.mode, result.elapsed_ms, result.rollout_id,
        )
        return result

    def teardown(self) -> None:
        """Release resources acquired during setup()."""
        if self._is_setup:
            self._do_teardown()
            self._is_setup = False

    @property
    @abstractmethod
    def _mode_name(self) -> str:
        ...

    def _do_setup(self) -> None:
        """Override for one-time setup work."""

    @abstractmethod
    def _do_sync(self, *, rollout_id: int) -> Optional[Dict[str, Any]]:
        """Core sync logic. Return extra metadata dict or None."""

    def _do_teardown(self) -> None:
        """Override for teardown work."""


# Backward compat alias
WeightSyncStrategy = WeightSyncProtocol


# ---------------------------------------------------------------------------
# TensorPayloadWeightSync (renamed from IPC)
# ---------------------------------------------------------------------------

class TensorPayloadWeightSync(WeightSyncProtocol):
    """Same-node SGLang: serialized tensor via UpdateWeightsFromTensorReqInput."""

    @property
    def _mode_name(self) -> str:
        return "tensor_payload"

    def _do_setup(self) -> None:
        _validate_tensor_sync_topology(self.args)
        self._target_modules = _resolve_target_modules(self.args)
        self._bucket_size_mb = _resolve_bucket_size_mb(self.args)
        self._flush_cache = _resolve_flush_cache(self.args)
        self._tp_payload_count = _resolve_rollout_tp_payload_count(self._rollout_manager)

    def _do_sync(self, *, rollout_id: int) -> Dict[str, Any]:
        stats = self._training_group.sync_weights_to_rollout_ipc(
            rollout_manager=self._rollout_manager,
            target_modules=self._target_modules,
            bucket_size_mb=self._bucket_size_mb,
            flush_cache=self._flush_cache,
            tp_payload_count=self._tp_payload_count,
        )
        return {"stats": stats, "tp_payload_count": self._tp_payload_count}


# Backward compat alias
IPCWeightSync = TensorPayloadWeightSync


# ---------------------------------------------------------------------------
# NCCLBroadcastWeightSync (persistent group)
# ---------------------------------------------------------------------------

class NCCLBroadcastWeightSync(WeightSyncProtocol):
    """Cross-node SGLang: persistent NCCL group, GPU-to-GPU broadcast."""

    @property
    def _mode_name(self) -> str:
        return "nccl_broadcast"

    def _do_setup(self) -> None:
        _validate_tensor_sync_topology(self.args)
        self._target_modules = _resolve_target_modules(self.args)
        self._bucket_size_mb = _resolve_bucket_size_mb(self.args)
        self._flush_cache = _resolve_flush_cache(self.args)
        self._group_name: Optional[str] = None
        self._init_persistent_group()

    def _init_persistent_group(self) -> None:
        topology = ray.get(self._rollout_manager.get_weight_sync_topology.remote())
        if not isinstance(topology, dict):
            raise RuntimeError(f"Invalid rollout weight sync topology payload: {topology!r}")
        total_rollout_gpus = int(topology.get("total_gpus", 0))
        if total_rollout_gpus <= 0:
            logger.warning("No rollout GPUs for NCCL group, skipping init.")
            return

        endpoint = self._training_group.get_rank0_ip_and_free_port()
        master_address = str(endpoint["master_address"])
        master_port = int(endpoint["master_port"])
        world_size = int(total_rollout_gpus + 1)
        self._group_name = f"diffusionrl_wsync_{int(time.time_ns())}"

        # Parallel init on both sides.
        rollout_ref = self._rollout_manager.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=self._group_name,
            backend="nccl",
        )
        train_ref = self._training_group.async_init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=self._group_name,
            backend="nccl",
        )
        try:
            ray.get([rollout_ref, train_ref])
        except Exception:
            logger.exception("NCCL group init failed: %s", self._group_name)
            self._destroy_group_best_effort()
            self._group_name = None
            raise

    def _destroy_group_best_effort(self) -> None:
        if not self._group_name:
            return
        gn = self._group_name
        try:
            self._training_group.destroy_weights_update_group(group_name=gn)
        except Exception:
            logger.debug("Training-side group destroy failed: %s", gn, exc_info=True)
        try:
            ray.get(self._rollout_manager.destroy_weights_update_group.remote(group_name=gn))
        except Exception:
            logger.debug("Rollout-side group destroy failed: %s", gn, exc_info=True)

    def _do_sync(self, *, rollout_id: int) -> Dict[str, Any]:
        if not self._group_name:
            return {"skipped": True}
        try:
            stats = self._training_group.sync_weights_to_rollout_nccl(
                rollout_manager=self._rollout_manager,
                group_name=self._group_name,
                target_modules=self._target_modules,
                bucket_size_mb=self._bucket_size_mb,
                flush_cache=self._flush_cache,
            )
            return {"stats": stats}
        except Exception:
            logger.warning("NCCL sync failed, rebuilding group...", exc_info=True)
            self._destroy_group_best_effort()
            self._group_name = None
            self._init_persistent_group()
            if not self._group_name:
                raise
            stats = self._training_group.sync_weights_to_rollout_nccl(
                rollout_manager=self._rollout_manager,
                group_name=self._group_name,
                target_modules=self._target_modules,
                bucket_size_mb=self._bucket_size_mb,
                flush_cache=self._flush_cache,
            )
            return {"stats": stats, "recovered": True}

    def _do_teardown(self) -> None:
        self._destroy_group_best_effort()
        self._group_name = None


# Backward compat alias
NCCLWeightSync = NCCLBroadcastWeightSync


# ---------------------------------------------------------------------------
# CheckpointWeightSync (renamed from CheckpointPathWeightSync)
# ---------------------------------------------------------------------------

class CheckpointWeightSync(WeightSyncProtocol):
    """Filesystem sync. Suitable for: FSDP engine, SGLang fallback."""

    @property
    def _mode_name(self) -> str:
        return "checkpoint"

    def _do_setup(self) -> None:
        self._export_format = self._select_export_format()

    def _select_export_format(self) -> str:
        engine_type = str(getattr(self.args, "sampler_engine_type", "") or "").lower()
        if engine_type == "sglang":
            return "sglang_transformer_safetensors"
        return "state_dict"

    def _build_weight_checkpoint_path(self, rollout_id: int, *, export_format: str) -> str:
        os.makedirs(self.args.weight_sync_dir, exist_ok=True)
        if export_format == "sglang_transformer_safetensors":
            return os.path.join(
                self.args.weight_sync_dir,
                f"weights_rollout_{rollout_id}_{int(time.time_ns())}",
            )
        return os.path.join(
            self.args.weight_sync_dir,
            f"weights_rollout_{rollout_id}_{int(time.time_ns())}.pt",
        )

    def _do_sync(self, *, rollout_id: int) -> Dict[str, Any]:
        path = self._build_weight_checkpoint_path(
            rollout_id,
            export_format=self._export_format,
        )
        self._training_group.export_weights_to_path(
            path,
            export_format=self._export_format,
        )
        ray.get(self._rollout_manager.update_weights_from_path.remote(path))
        cleanup_published_checkpoint(path)
        return {"checkpoint_path": path}


# Backward compat alias
CheckpointPathWeightSync = CheckpointWeightSync


# ---------------------------------------------------------------------------
# ObjectRefWeightSync (deprecated -> falls back to CheckpointWeightSync)
# ---------------------------------------------------------------------------

class ObjectRefWeightSync(CheckpointWeightSync):
    """Deprecated: use CheckpointWeightSync instead."""

    def __init__(self, args: Any) -> None:
        logger.warning(
            "ObjectRefWeightSync is deprecated, falling back to CheckpointWeightSync"
        )
        super().__init__(args)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

_BUILTIN_PROTOCOLS: Dict[str, Type[WeightSyncProtocol]] = {
    # Primary names
    "tensor_payload": TensorPayloadWeightSync,
    "nccl_broadcast": NCCLBroadcastWeightSync,
    "checkpoint_path": CheckpointWeightSync,
    # Legacy aliases
    "ipc": TensorPayloadWeightSync,
    "nccl": NCCLBroadcastWeightSync,
    "object_ref": ObjectRefWeightSync,
}

# Keep old name for code that imports it directly.
_BUILTIN_WEIGHT_SYNC_STRATEGIES = _BUILTIN_PROTOCOLS


def create_weight_sync_protocol(args: Any) -> WeightSyncProtocol:
    """Create weight sync protocol from runtime args.

    Extension point:
    - If args.weight_sync_strategy_path exists, dynamically load custom strategy.
    - Otherwise resolve built-in protocols from args.weight_sync_mode.
    """
    strategy_path = getattr(args, "weight_sync_strategy_path", None)
    if strategy_path:
        strategy_cls = load_function(strategy_path)
        return strategy_cls(args)

    mode = getattr(args, "weight_sync_mode", "checkpoint_path")
    cls = _BUILTIN_PROTOCOLS.get(mode)
    if cls is None:
        raise ValueError(
            f"Unsupported weight_sync_mode={mode}. "
            f"Expected one of: {sorted(_BUILTIN_PROTOCOLS.keys())}"
        )
    return cls(args)


# Backward compat alias
create_weight_sync_strategy = create_weight_sync_protocol


__all__ = [
    # Primary
    "WeightSyncProtocol",
    "SyncResult",
    "TensorPayloadWeightSync",
    "NCCLBroadcastWeightSync",
    "CheckpointWeightSync",
    "create_weight_sync_protocol",
    # Deprecated aliases
    "WeightSyncStrategy",
    "IPCWeightSync",
    "NCCLWeightSync",
    "CheckpointPathWeightSync",
    "ObjectRefWeightSync",
    "create_weight_sync_strategy",
]
