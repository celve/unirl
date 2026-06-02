"""TensorStore — worker-local tensor registry with ref-counting.

Each worker holds one TensorStore instance. Tensors stay on GPU;
only lightweight TensorHandle refs cross the Ray RPC boundary.

Storage dedup:
  Multiple put() calls for tensors sharing the same CUDA storage
  (views, slices, grad of an IPC-resolved tensor) reuse the same
  store key and IPC handle.  Each put() increments the refcount;
  decref() releases the storage only when the refcount reaches 0.
  The dedup key is storage.data_ptr() (storage start address),
  NOT tensor.data_ptr() which shifts for non-zero-offset views.

IPC inputs:
  _resolve_input returns the IPC view as-is (no eager clone).
  put() detects non-resizable storage (PyTorch marks all IPC-opened storage
  as non-resizable) and clones to local memory before storing.
  Roles that only read the tensor without put() avoid the clone entirely.

Lifecycle:
  put(tensor) → TensorHandle (ref_count=1, or +1 if storage already stored)
  incref(key) → ref_count += 1  (called by TensorHandle.__copy__ on controller)
  decref(key) → ref_count -= 1  (called by TensorHandle._release on controller GC)
                if ref_count == 0 → storage freed, dedup maps cleaned up
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import ray
import torch
import torch.distributed as dist
from torch import Tensor

from unirl.distributed.tensor.backend.tensor_store.handle import TensorHandle
from unirl.distributed.utils import cuda_ipc_needs_clone


class TensorStore:
    """Worker-local GPU tensor registry with reference counting and storage dedup.

    Thread-safe: all mutations protected by a lock for concurrent
    put/get/incref/decref from Ray async calls.
    """

    def __init__(
        self,
        worker_id: str,
        device: str = "cuda:0",
        global_rank: Optional[int] = None,
        global_world_size: Optional[int] = None,
    ):
        self.worker_id = worker_id
        self.device = device
        self.global_rank = global_rank
        self.global_world_size = global_world_size

        # store_key → tensor (holds storage alive)
        self._store: Dict[str, Tensor] = {}
        # store_key → ref count (number of live TensorHandles referencing this key)
        self._ref_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._counter = 0

        # storage.data_ptr() → store_key: dedup map so the same CUDA storage
        # is never entered twice (views/slices share the same key).
        self._storage_ptr_to_key: Dict[int, str] = {}

        # store_key → ipc_handle: cached once per storage, reused for all views.
        # _share_cuda_() is only called once per unique storage allocation.
        self._ipc_handles: Dict[str, tuple] = {}

        # Global ProcessGroup for cross-worker NCCL (initialized lazily)
        self._global_pg = None

    def put(self, tensor: Tensor) -> TensorHandle:
        """Store a tensor and return a lightweight TensorHandle.

        CUDA tensors: stored in TensorStore with IPC handle for same-device siblings.
          Deduplicates by storage.data_ptr(): multiple views of the same storage
          share one store key and IPC handle.
          IPC-opened storage (non-resizable) is cloned to local memory first so
          that _share_cuda_() always operates on process-local allocations.
        CPU tensors: stored in Ray plasma store via ray.put(); not tracked here.
          Lifecycle managed by ObjectRef Python refcount (no decref RPC needed).
        """
        if not tensor.is_cuda:
            return TensorHandle(
                store_key=None,
                worker_id=self.worker_id,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                device=str(tensor.device),
                object_ref=ray.put(tensor.detach()),
            )

        # Clone IPC-opened storage to local memory before storing.
        # PyTorch marks IPC-opened storage as non-resizable; local allocations
        # are always resizable.  This check replaces the old _foreign_storage_ptrs
        # set and is safe even when the tensor was cached across RPC calls.
        if not tensor.untyped_storage().resizable():
            tensor = tensor.clone()

        storage_ptr = tensor.untyped_storage().data_ptr()

        with self._lock:
            existing_key = self._storage_ptr_to_key.get(storage_ptr)
            if existing_key is not None:
                # Same storage already stored — just incref and return new view handle.
                self._ref_counts[existing_key] += 1
                return TensorHandle(
                    store_key=existing_key,
                    worker_id=self.worker_id,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    device=str(tensor.device),
                    ipc_handle=None,  # exported lazily — see ensure_ipc_handle()
                    stride=tensor.stride(),
                    offset=tensor.storage_offset(),
                )

            # New storage — assign a fresh key.
            key = f"{self.worker_id}_{self._counter}"
            self._counter += 1
            self._store[key] = tensor.detach()
            self._ref_counts[key] = 1
            self._storage_ptr_to_key[storage_ptr] = key
            self._ipc_handles[key] = None  # exported lazily — see ensure_ipc_handle()

        # No eager _share_cuda_() here. Exporting a CUDA IPC handle hands the block
        # to a CudaIPCSentData that pins it in the caching allocator until a consumer
        # releases it; a same-process reader never imports it (sibling reads use
        # store.get), so an eager export would dangle forever and never be reclaimed
        # (LIN-361 leak). The handle is exported on demand in ensure_ipc_handle(),
        # called only when a same-device sibling actually opens this tensor over IPC.
        return TensorHandle(
            store_key=key,
            worker_id=self.worker_id,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=str(tensor.device),
            ipc_handle=None,
            stride=tensor.stride(),
            offset=tensor.storage_offset(),
        )

    def ensure_ipc_handle(self, key: str) -> tuple:
        """Lazily export (and cache) the CUDA IPC handle for a stored key.

        This is the ONLY place ``_share_cuda_()`` runs. ``put()`` no longer
        exports eagerly: ``cudaIpcGetMemHandle`` pins the block in the caching
        allocator until a consumer releases it, and in a colocate single-worker
        layout no sibling ever imports the handle, so the export would dangle and
        leak (LIN-361). The controller calls this (via ``Handle._with_ipc``) only
        when a same-device sibling actually needs to open the tensor over IPC.

        Idempotent — returns the cached handle once exported. Preserves the old
        eager behavior, including the expandable-segments (VMM) clone-and-remap so
        the export always operates on a process-local cudaMalloc allocation.
        """
        with self._lock:
            cached = self._ipc_handles.get(key)
            if cached is not None:
                return cached
            tensor = self._store[key]

        probe, needs_clone = cuda_ipc_needs_clone(tensor.untyped_storage())
        if needs_clone:
            old_ptr = tensor.untyped_storage().data_ptr()
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
            tensor = tensor.clone()
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            probe, _ = cuda_ipc_needs_clone(tensor.untyped_storage())
            with self._lock:
                self._storage_ptr_to_key.pop(old_ptr, None)
                self._storage_ptr_to_key[tensor.untyped_storage().data_ptr()] = key
                self._store[key] = tensor.detach()

        with self._lock:
            self._ipc_handles[key] = probe
        return probe

    def get(self, handle: TensorHandle) -> Tensor:
        """Return the tensor described by handle, restoring shape/stride/offset.

        This is the only safe public API for retrieving a stored tensor.
        Always use this — never access the store by key directly.
        """
        with self._lock:
            if handle.store_key not in self._store:
                raise KeyError(f"TensorStore: key '{handle.store_key}' not found")
            base = self._store[handle.store_key]
        return base.as_strided(handle.shape, handle.stride, handle.offset).detach()

    def ref_count(self, key: str) -> int:
        """Return current reference count for a key."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: key '{key}' not found")
            return self._ref_counts[key]

    def incref(self, key: str) -> None:
        """Increment reference count. Called by TensorHandle copy on controller."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: cannot incref unknown key '{key}'")
            self._ref_counts[key] += 1

    def decref(self, key: str) -> None:
        """Decrement reference count. If zero, release the storage.

        Called by TensorHandle._release (GC finalizer on controller side).
        Cleans up all dedup maps atomically so the storage data_ptr can be
        safely reused by a future allocation without false dedup hits.
        """
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: cannot decref unknown key '{key}'")
            self._ref_counts[key] -= 1
            if self._ref_counts[key] <= 0:
                tensor = self._store.pop(key)
                del self._ref_counts[key]
                # Clean up dedup maps before tensor GC so the data_ptr slot is
                # free for reuse — CUDA allocator may return the same address
                # immediately after the tensor Python object is collected.
                storage_ptr = tensor.untyped_storage().data_ptr()
                self._storage_ptr_to_key.pop(storage_ptr, None)
                self._ipc_handles.pop(key, None)
                # tensor goes out of scope here → Python GC → GPU memory freed

    # ── Remote tensor operations ──

    def cat_keys(self, store_keys: list) -> TensorHandle:
        """Cat multiple stored tensors along dim0, store result, return new handle."""
        tensors = [self.get_by_key(k) for k in store_keys]
        result = torch.cat(tensors, dim=0)
        return self.put(result)

    # ── Cross-worker NCCL transfer ──

    def setup_global_pg(self, global_rank: int, global_world_size: int) -> None:
        """Initialize the global ProcessGroup for cross-worker NCCL transfers.

        Always uses ProcessGroupNCCL directly via TCPStore so the global dist
        state (init_process_group) is never touched — each Role group manages
        its own dist process group without interference.

        MASTER_ADDR and MASTER_PORT must be set in the environment (injected by
        DevicePool via Worker runtime_env).
        """
        self.global_rank = global_rank
        self.global_world_size = global_world_size

        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=global_world_size,
            is_master=(global_rank == 0),
            timeout=timedelta(seconds=30),
        )
        self._global_pg = dist.ProcessGroupNCCL(store, global_rank, global_world_size)

    def _nccl_send(self, dst_rank: int, handles: List[TensorHandle]) -> None:
        """Send tensors to dst_rank via NCCL.

        Handles may come from this worker's local store (same worker_id)
        or from a same-device sibling slot (IPC open via ipc_handle).

        Uses ProcessGroupNCCL.send() natively so that privately-created PG
        (not registered in the global dist world) works correctly.
        Each tensor is sent separately so send and recv always stay in sync.
        """
        assert self._global_pg is not None, "Global PG not initialized. Call setup_global_pg first."

        for h in handles:
            assert h.object_ref is None, (
                "CPU tensor (object_ref set) must not go through NCCL. Check _ensure_local._unwrap routing."
            )
            if h.worker_id == self.worker_id:
                tensor = self.get(h).contiguous()
            else:
                # Same-device sibling: open IPC view, ipc_handle already embedded in h.
                # storage stays alive until send().wait() returns.
                storage = torch.UntypedStorage._new_shared_cuda(*h.ipc_handle)
                t = torch.empty(0, dtype=h.dtype, device=self.device)
                t.set_(storage, h.offset, h.shape, h.stride)
                tensor = t.contiguous()
            self._global_pg.send([tensor], dst_rank, 0).wait()

    def _nccl_recv(
        self,
        src_global_rank: int,
        shapes: List[Tuple[int, ...]],
        dtypes: List[torch.dtype],
    ) -> List[TensorHandle]:
        """Receive tensors from another worker via NCCL p2p."""
        assert self._global_pg is not None, "Global PG not initialized. Call setup_global_pg first."

        handles = []
        for shape, dtype in zip(shapes, dtypes):
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._global_pg.recv([buf], src_global_rank, 0).wait()
            handles.append(self.put(buf))
        return handles

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"TensorStore(worker={self.worker_id}, items={len(self)})"
