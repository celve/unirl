"""TensorHandle and TensorMeta — tensor references for the Single Controller.

TensorHandle is the fundamental unit: a reference-counted handle to a single
tensor on a worker's TensorStore. It manages its own GC via weakref.finalize.
It flows between worker and controller via Ray RPC (pickle).

TensorMeta is a controller-side wrapper over List[TensorHandle], representing
a logical tensor = cat(handle[0], handle[1], ..., dim=0). It never appears
on the worker side. It provides [], reshape, permute, local operations.

Lifecycle:
  Worker side:  store.put(tensor) → TensorHandle (bare, no finalizer)
                Ray pickles → controller receives
  Controller:   handle.rebind(actor_handle) → register weakref.finalize
                GC → finalize → remote decref → ref_count=0 → GPU freed
  Controller:   pytree_merge(handles) → TensorMeta (multi-handle)
"""

from __future__ import annotations

import weakref
from typing import Any

import ray
import torch
from torch import Tensor


class TensorHandle:
    """Handle to a single tensor on a worker's GPU.

    Two phases of life:
      1. Worker side (after store.put()): pure data, NO worker_handle.
      2. Controller side (after rebind()): worker_handle set, finalize registered.
    """

    __slots__ = (
        "store_key",
        "worker_id",
        "shape",
        "dtype",
        "device",
        "worker_handle",
        "_finalized",
        "__weakref__",
        "ipc_handle",
        "stride",
        "offset",
        "object_ref",
    )

    def __init__(
        self,
        store_key: str,
        worker_id: str,
        shape: tuple,
        dtype: torch.dtype,
        device: str,
        ipc_handle=None,
        stride=None,
        offset: int = 0,
        object_ref=None,
    ):
        self.store_key = store_key
        self.worker_id = worker_id
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.worker_handle = None
        self._finalized = False
        self.ipc_handle = ipc_handle  # _share_cuda_() result; None for CPU tensors
        self.stride = stride  # tensor.stride(); None for CPU tensors
        self.offset = offset  # tensor.storage_offset()
        self.object_ref = object_ref  # Ray ObjectRef for CPU tensors in plasma store

    # ── Controller-side activation ──

    def rebind(self, worker_handle) -> None:
        """Attach Ray actor handle and register release finalizer.

        Must only be called once. Calling rebind on an already-bound handle
        indicates a bug (e.g. double-processing in _rebind_tree).

        CPU tensors (object_ref is not None): worker_handle is recorded for
        remote_op calls, but no finalizer is registered — lifecycle managed
        by the ObjectRef Python refcount, no decref RPC needed.
        """
        assert not self._finalized, (
            f"TensorHandle {self.store_key!r} is already bound to a worker. rebind() must only be called once."
        )
        self._finalized = True
        self.worker_handle = worker_handle
        if self.object_ref is None:
            # CUDA tensor: register finalizer for decref RPC
            weakref.finalize(self, TensorHandle._release, worker_handle, self.store_key)

    # ── Remote operations ──

    def remote_op_async(self, op: str, *args) -> Any:
        """Fire remote tensor_op, return ObjectRef (does not wait)."""
        assert self.worker_handle is not None, "TensorHandle not bound"
        return self.worker_handle.tensor_op.remote(self, op, *args)

    def remote_op(self, op: str, *args) -> TensorHandle:
        """Execute remote tensor_op synchronously, return new bound TensorHandle."""
        new_h = ray.get(self.remote_op_async(op, *args))
        new_h.rebind(self.worker_handle)
        return new_h

    def local(self) -> Tensor:
        """Fetch the actual tensor to controller CPU."""
        if self.object_ref is not None:
            return ray.get(self.object_ref)
        assert self.worker_handle is not None, "TensorHandle not bound to worker"
        return ray.get(self.worker_handle.get_tensor_cpu.remote(self))

    # ── Copy protocols ──

    def __copy__(self) -> TensorHandle:
        if self.worker_handle is not None and self.object_ref is None:
            # CUDA tensor: explicit incref to keep tensor alive in TensorStore
            self.worker_handle.incref_tensor.remote(self.store_key)
        clone = TensorHandle(
            self.store_key,
            self.worker_id,
            self.shape,
            self.dtype,
            self.device,
            ipc_handle=self.ipc_handle,
            stride=self.stride,
            offset=self.offset,
            object_ref=self.object_ref,
        )
        if self.worker_handle is not None:
            clone.rebind(self.worker_handle)
        return clone

    def __deepcopy__(self, memo) -> TensorHandle:
        clone = self.__copy__()
        memo[id(self)] = clone
        return clone

    # ── Pickle protocol (for Ray RPC) ──

    def __getstate__(self) -> dict:
        return {
            "store_key": self.store_key,
            "worker_id": self.worker_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "ipc_handle": self.ipc_handle,
            "stride": self.stride,
            "offset": self.offset,
            "object_ref": self.object_ref,
        }

    def __setstate__(self, state: dict) -> None:
        self.store_key = state["store_key"]
        self.worker_id = state["worker_id"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.device = state["device"]
        self.worker_handle = None
        self._finalized = False
        self.ipc_handle = state["ipc_handle"]
        self.stride = state["stride"]
        self.offset = state["offset"]
        self.object_ref = state.get("object_ref")  # None for CUDA handles

    # ── Release callback ──

    @staticmethod
    def _release(worker_handle, store_key: str) -> None:
        """GC callback: tell worker to decref this tensor.

        Skipped if Ray is not initialized — happens during interpreter shutdown
        when module-scoped fixtures have already called ray.shutdown().
        """
        try:
            if not ray.is_initialized():
                return
            worker_handle.decref_tensor.remote(store_key)
        except Exception:
            pass  # worker already dead

    def __repr__(self) -> str:
        bound = "bound" if self.worker_handle else "unbound"
        return f"TensorHandle({self.shape}, {self.dtype}, {bound})"
