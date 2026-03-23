"""Weight synchronization coordinators for rollout updates.

Package contract for ``diffusionrl.distributed``:

- owns distributed coordination semantics such as weight-sync strategy,
  persistent-group lifecycle, and export/publish protocols
- may bind to the current runtime transport when a coordination step needs it
  (for example waiting on a Ray async init ref)
- does not own Ray actor classes, group creation, placement, or workflow logic

Two-phase lifecycle: setup() -> sync() x N -> teardown().
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import logging

from diffusionrl.config.rollout_topology import (
    normalize_rollout_service_engine,
    resolve_rollout_service_kwargs,
)
from diffusionrl.distributed.weight_sync_checkpoint import cleanup_published_checkpoint
from diffusionrl.training.backends import resolve_train_backend_capabilities_from_args

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_target_modules(args: Any) -> list[str]:
    raw = getattr(args.sync, "target_modules", None)
    if isinstance(raw, (list, tuple)) and raw:
        return [str(name) for name in raw]
    return ["transformer"]


def _resolve_bucket_size_mb(args: Any) -> int:
    bucket_size_mb = int(getattr(args.sync, "bucket_mb", 256))
    if bucket_size_mb < 1:
        raise ValueError(f"sync.bucket_mb must be >= 1, got {bucket_size_mb}.")
    return bucket_size_mb


def _resolve_flush_cache(args: Any) -> bool:
    return bool(getattr(args.sync, "flush_cache", True))


def _validate_tensor_sync_topology(args: Any) -> None:
    """Guard invalid topology values for tensor/distributed sync paths."""
    engine_kwargs = resolve_rollout_service_kwargs(args)
    tp_size = 1
    if isinstance(engine_kwargs, dict) and engine_kwargs.get("tp_size") is not None:
        tp_size = engine_kwargs.get("tp_size")
    try:
        tp_size_int = int(tp_size)
    except Exception:
        tp_size_int = 1
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
# WeightSyncCoordinator (two-phase base class)
# ---------------------------------------------------------------------------

class WeightSyncCoordinator(ABC):
    """Two-phase weight sync coordinator: setup() -> sync() x N -> teardown()."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self._version = 0
        self._is_setup = False
        self._training_runtime = None
        self._rollout_runtime = None

    def setup(
        self,
        *,
        training_runtime: Any,
        rollout_runtime: Optional[Any] = None,
    ) -> None:
        """Bind runtime handles and perform one-time initialization."""
        self._training_runtime = training_runtime
        self._rollout_runtime = rollout_runtime
        self._do_setup()
        self._is_setup = True

    def sync(self, *, rollout_id: int) -> SyncResult:
        """Synchronize latest policy weights to rollout side."""
        if not self._is_setup:
            raise RuntimeError(
                "WeightSyncCoordinator.sync() called before setup(). "
                "Call setup(training_runtime=..., rollout_runtime=...) first."
            )

        t0 = time.monotonic()
        if self._rollout_runtime is not None:
            self._rollout_runtime.wake_up()
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
# ---------------------------------------------------------------------------
# DisabledWeightSync
# ---------------------------------------------------------------------------

class DisabledWeightSync(WeightSyncCoordinator):
    """No-op coordinator used when sampling runs directly on training actors."""

    @property
    def _mode_name(self) -> str:
        return "disabled"

    def sync(self, *, rollout_id: int) -> SyncResult:
        if not self._is_setup:
            self._is_setup = True
        self._version += 1
        return SyncResult(
            version=self._version,
            mode=self._mode_name,
            rollout_id=rollout_id,
            elapsed_ms=0.0,
            extra={"skipped": True},
        )

    def _do_sync(self, *, rollout_id: int) -> Optional[Dict[str, Any]]:
        return {"skipped": True}


# ---------------------------------------------------------------------------
# TensorPayloadWeightSync (renamed from IPC)
# ---------------------------------------------------------------------------

class TensorPayloadWeightSync(WeightSyncCoordinator):
    """Same-node SGLang: serialized tensor via UpdateWeightsFromTensorReqInput."""

    @property
    def _mode_name(self) -> str:
        return "tensor_payload"

    def _do_setup(self) -> None:
        _validate_tensor_sync_topology(self.args)
        if self._rollout_runtime is None:
            raise RuntimeError("tensor_payload weight sync requires a rollout runtime.")
        self._target_modules = _resolve_target_modules(self.args)
        self._bucket_size_mb = _resolve_bucket_size_mb(self.args)
        self._flush_cache = _resolve_flush_cache(self.args)
        self._rollout_runtime.refresh_weight_update_targets()
        self._tp_payload_count = _resolve_rollout_tp_payload_count(self._rollout_runtime)

    def _do_sync(self, *, rollout_id: int) -> Dict[str, Any]:
        stats = self._training_runtime.sync_weights_to_rollout_ipc(
            rollout_runtime=self._rollout_runtime,
            target_modules=self._target_modules,
            bucket_size_mb=self._bucket_size_mb,
            flush_cache=self._flush_cache,
            tp_payload_count=self._tp_payload_count,
        )
        return {"stats": stats, "tp_payload_count": self._tp_payload_count}
# ---------------------------------------------------------------------------
# NCCLBroadcastWeightSync (persistent group)
# ---------------------------------------------------------------------------

class NCCLBroadcastWeightSync(WeightSyncCoordinator):
    """Cross-node SGLang: persistent NCCL group, GPU-to-GPU broadcast."""

    @property
    def _mode_name(self) -> str:
        return "nccl_broadcast"

    def _do_setup(self) -> None:
        _validate_tensor_sync_topology(self.args)
        if self._rollout_runtime is None:
            raise RuntimeError("nccl_broadcast weight sync requires a rollout runtime.")
        self._target_modules = _resolve_target_modules(self.args)
        self._bucket_size_mb = _resolve_bucket_size_mb(self.args)
        self._flush_cache = _resolve_flush_cache(self.args)
        self._group_name: Optional[str] = None
        self._rollout_runtime.refresh_weight_update_targets()
        self._init_persistent_group()

    def _init_persistent_group(self) -> None:
        topology = self._rollout_runtime.get_weight_sync_topology()
        if not isinstance(topology, dict):
            raise RuntimeError(f"Invalid rollout weight sync topology payload: {topology!r}")
        total_rollout_gpus = int(topology.get("total_gpus", 0))
        if total_rollout_gpus <= 0:
            logger.warning("No rollout GPUs for NCCL group, skipping init.")
            return

        endpoint = self._training_runtime.get_rank0_ip_and_free_port()
        master_address = str(endpoint["master_address"])
        master_port = int(endpoint["master_port"])
        world_size = int(total_rollout_gpus + 1)
        self._group_name = f"diffusionrl_wsync_{int(time.time_ns())}"

        self._rollout_runtime.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=self._group_name,
            backend="nccl",
        )
        train_ref = self._training_runtime.async_init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=self._group_name,
            backend="nccl",
        )
        try:
            # distributed/ is allowed to bind the current transport for
            # coordination handshakes, but actor/group ownership stays in ray/.
            import ray

            ray.get(train_ref)
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
            self._training_runtime.destroy_weights_update_group(gn)
        except Exception:
            logger.debug("Training-side group destroy failed: %s", gn, exc_info=True)
        try:
            self._rollout_runtime.destroy_weights_update_group(gn)
        except Exception:
            logger.debug("Rollout-side group destroy failed: %s", gn, exc_info=True)

    def _do_sync(self, *, rollout_id: int) -> Dict[str, Any]:
        if not self._group_name:
            return {"skipped": True}
        try:
            stats = self._training_runtime.sync_weights_to_rollout_nccl(
                rollout_runtime=self._rollout_runtime,
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
            stats = self._training_runtime.sync_weights_to_rollout_nccl(
                rollout_runtime=self._rollout_runtime,
                group_name=self._group_name,
                target_modules=self._target_modules,
                bucket_size_mb=self._bucket_size_mb,
                flush_cache=self._flush_cache,
            )
            return {"stats": stats, "recovered": True}

    def _do_teardown(self) -> None:
        self._destroy_group_best_effort()
        self._group_name = None
# ---------------------------------------------------------------------------
# CheckpointWeightSync
# ---------------------------------------------------------------------------

class CheckpointWeightSync(WeightSyncCoordinator):
    """Filesystem sync for dedicated rollout engines that consume checkpoints."""

    @property
    def _mode_name(self) -> str:
        return "checkpoint_path"

    def _do_setup(self) -> None:
        if self._rollout_runtime is None:
            raise RuntimeError("checkpoint_path weight sync requires a rollout runtime.")
        self._export_format = self._select_export_format()

    def _select_export_format(self) -> str:
        engine_type = normalize_rollout_service_engine(
            getattr(self.args.rollout, "service_engine", None)
        )
        if not engine_type:
            raise ValueError(
                "Checkpoint weight sync requires rollout.service_engine to be normalized. "
                "Validate args before selecting dedicated rollout checkpoint export format."
            )
        backend_caps = resolve_train_backend_capabilities_from_args(self.args)
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

    def _build_weight_checkpoint_path(self, rollout_id: int, *, export_format: str) -> str:
        os.makedirs(self.args.sync.dir, exist_ok=True)
        if export_format == "sglang_transformer_safetensors":
            return os.path.join(
                self.args.sync.dir,
                f"weights_rollout_{rollout_id}_{int(time.time_ns())}",
            )
        return os.path.join(
            self.args.sync.dir,
            f"weights_rollout_{rollout_id}_{int(time.time_ns())}.pt",
        )

    def _do_sync(self, *, rollout_id: int) -> Dict[str, Any]:
        path = self._build_weight_checkpoint_path(
            rollout_id,
            export_format=self._export_format,
        )
        self._training_runtime.export_weights_to_path(
            path,
            export_format=self._export_format,
        )
        self._rollout_runtime.update_weights_from_path(path)
        cleanup_published_checkpoint(path)
        return {"checkpoint_path": path}


_BUILTIN_COORDINATORS: Dict[str, Type[WeightSyncCoordinator]] = {
    "disabled": DisabledWeightSync,
    "tensor_payload": TensorPayloadWeightSync,
    "nccl_broadcast": NCCLBroadcastWeightSync,
    "checkpoint_path": CheckpointWeightSync,
}


def create_weight_sync(args: Any, *, mode: Optional[str] = None) -> WeightSyncCoordinator:
    """Create a weight-sync coordinator from runtime args.

    Resolve built-in coordinators from the explicit ``sync.protocol`` setting.
    """
    resolved_mode = str(mode if mode is not None else getattr(args.sync, "protocol", "")).strip().lower()
    if not resolved_mode:
        raise ValueError(
            "sync.protocol must be set explicitly before create_weight_sync()."
        )
    cls = _BUILTIN_COORDINATORS.get(resolved_mode)
    if cls is None:
        raise ValueError(
            f"Unsupported sync.protocol={resolved_mode}. "
            f"Expected one of: {sorted(_BUILTIN_COORDINATORS.keys())}"
        )
    return cls(args)


__all__ = [
    "WeightSyncCoordinator",
    "SyncResult",
    "DisabledWeightSync",
    "TensorPayloadWeightSync",
    "NCCLBroadcastWeightSync",
    "CheckpointWeightSync",
    "create_weight_sync",
]
