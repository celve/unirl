"""Worker — physical GPU Ray actor.

Each GPU runs exactly one Worker. It holds a TensorStore and
hosts multiple Remote instances (colocated logical workers).

Worker is the ONLY Ray actor in the system. Remotes are
plain Python objects living inside Worker's process.
"""

from __future__ import annotations

import os
import socket
from typing import Dict, List, Optional, Tuple

import ray
import torch
from torch import Tensor

from diffusionrl.distributed.group.remote import RankInfo, Remote
from diffusionrl.distributed.tensor.backend.tensor_store.handle import TensorHandle
from diffusionrl.distributed.tensor.backend.tensor_store.store import TensorStore
from diffusionrl.distributed.tensor.batch import Batch
from diffusionrl.distributed.utils import collect_leaves


class Worker:
    """Physical worker: one per GPU.

    In Ray mode, this class is wrapped with @ray.remote(num_gpus=1).
    For unit testing, use _init_local() to skip GPU/Ray setup.
    """

    def __init__(self, device_id: int, slot: int = 0, nccl_rank: Optional[int] = None, world_size: int = 1) -> None:
        """Called as Ray remote actor. Auto-initializes GPU + TensorStore.

        Args:
            device_id:  Physical GPU index (same for all slots on a GPU).
            slot:       Slot index on this device (0 = primary, 1+ = colocated).
            nccl_rank:  NCCL rank for slot0 workers (= device_id); None for slot1+.
            world_size: Total number of slot0 workers (= num_devices).
        """
        self.device_id = device_id
        self.slot = slot

        # GPU setup: Ray PlacementGroup sets CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        worker_id = f"dw{device_id}" if slot == 0 else f"dw{device_id}_s{slot}"

        # slot0: participates in NCCL (nccl_rank = device_id)
        # slot1+: IPC only, no NCCL PG
        self.store = TensorStore(
            worker_id=worker_id,
            device=self.device,
            global_rank=nccl_rank,
            global_world_size=world_size if nccl_rank is not None else None,
        )

        self._roles: Dict[str, Remote] = {}

        # Port reservation: held sockets to prevent reuse
        self._reserved_sockets: Dict[int, socket.socket] = {}

    def _init_local(self, device_id: int = 0, slot: int = 0, global_rank: int = 0, world_size: int = 1) -> None:
        """Initialize without GPU for unit testing."""
        self.device_id = device_id
        self.slot = slot
        self.device = "cpu"
        worker_id = f"dw{device_id}" if slot == 0 else f"dw{device_id}_s{slot}"
        self.store = TensorStore(
            worker_id=worker_id,
            device="cpu",
            global_rank=global_rank if slot == 0 else None,
            global_world_size=world_size if slot == 0 else None,
        )
        self._roles = {}
        self._reserved_sockets = {}

    # ── Port reservation ──

    def _reserve_port(self) -> int:
        """Bind a socket to an ephemeral port and hold it open.

        The port is guaranteed unique across calls on this actor
        (Ray actor is single-threaded). Call _release_port() before
        the user's initialize() so init_process_group can bind it.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        port = s.getsockname()[1]
        self._reserved_sockets[port] = s
        return port

    def _release_port(self, port: int) -> None:
        """Release a previously reserved port so init_process_group can use it."""
        s = self._reserved_sockets.pop(port, None)
        if s:
            s.close()

    # ── Role management ──

    def add_remote(
        self, role_name: str, role_cls, rank_info: RankInfo, init_kwargs: dict = None, dist_env: dict = None
    ) -> None:
        """Register a logical worker role on this device.

        ``init_kwargs`` is walked depth-first before ``role_cls`` is
        constructed:

        - ``HandleRef`` leaves → local sibling ``Remote`` from
          ``self._roles``.
        - Nested ``dict`` with ``_target_`` → ``hydra.utils.instantiate``
          after children are resolved. This is what makes the driver a
          config router and the worker the materializer — any object
          with ``_target_`` gets constructed in this worker's process
          (its CUDA context, its placement).
        - ``dict`` / ``list`` / ``tuple`` without ``_target_`` →
          recurse, preserving structure.

        Args:
            role_name:   Unique name for this role.
            role_cls:    Remote subclass.
            rank_info:   Parallelism rank info.
            init_kwargs: Dict of kwargs forwarded to role_cls.__init__
                         after resolution.
            dist_env:    Group-level dist env vars (written to os.environ once).
        """
        resolved_kwargs = self._resolve_init_kwargs(init_kwargs or {})
        role = role_cls(**resolved_kwargs)
        role.setup(
            store=self.store,
            device=self.device,
            rank_info=rank_info,
            dist_env=dist_env,
            get_sibling=lambda name: self._roles[name],
        )
        self._roles[role_name] = role

    def _resolve_init_kwargs(self, obj):
        """Resolve HandleRefs and nested ``_target_`` blocks for one kwarg tree.

        Children resolve before parents, so a HandleRef *inside* a
        nested ``_target_`` block becomes the local Remote before the
        enclosing target is called.

        Uses ``hydra.utils.get_method`` + direct ``cls(**children)`` rather
        than ``hydra.utils.instantiate`` so that already-resolved Python
        objects (Remote instances, constructed siblings) pass through as
        kwargs without OmegaConf coercion. Interpolations and structural
        Hydra features are expected to have been resolved driver-side via
        ``OmegaConf.to_container(resolve=True)`` already.

        Raises clearly if a referenced sibling is not registered on this
        Worker (e.g., the user passed a Handle from a different slab/slot).
        """
        from hydra.utils import get_method

        from diffusionrl.distributed.group.handle import HandleRef

        if isinstance(obj, HandleRef):
            try:
                return self._roles[obj.role_name]
            except KeyError:
                raise RuntimeError(
                    f"Cannot resolve sibling Handle '{obj.role_name}' on "
                    f"Worker {self.store.worker_id}: not registered on this "
                    f"Worker. Likely cause: the sibling lives on a different "
                    f"device slab (separate placement scope) or a different slot."
                )
        if isinstance(obj, dict):
            children = {k: self._resolve_init_kwargs(v) for k, v in obj.items() if k != "_target_"}
            if "_target_" in obj:
                cls = get_method(obj["_target_"])
                return cls(**children)
            return children
        if isinstance(obj, list):
            return [self._resolve_init_kwargs(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._resolve_init_kwargs(v) for v in obj)
        return obj

    def get_rank_info(self, role_name: str) -> RankInfo:
        """Read back rank_info (may have been modified by initialize)."""
        return self._roles[role_name].rank_info

    # ── RPC entry point ──

    def call(self, role_name: str, method_name: str, args: tuple, kwargs: dict, grad_mode: bool = False, call_id=None):
        """Generic RPC entry point.

        Recursively resolves inputs (TensorHandle → Tensor from store) and
        packs outputs (Tensor → TensorHandle via store.put).
        Non-tensor args/kwargs/results are passed through unchanged.

        grad_mode and call_id are passed as dedicated parameters (not via kwargs)
        so dispatch internals remain unaware of grad state.

        When grad_mode=True, additionally:
          - Saves resolved input tensors (as detached leaves) for backward.
          - Saves output tensors (before detach) for backward.
        Both saved under role._grad_inputs[call_id] / role._grad_outputs[call_id].
        """
        role = self._roles[role_name]

        resolved_args = self._transform_tree(args, self._resolve_input)
        resolved_kwargs = self._transform_tree(kwargs, self._resolve_input)

        if grad_mode:
            # _resolve_input already returns .detach() copies, so the resolved
            # tensors are fresh objects that don't alias TensorStore contents.
            # Mark them as grad-tracked leaves directly — no _replace_tensors needed.
            tensors = collect_leaves(resolved_args, Tensor) + collect_leaves(tuple(resolved_kwargs.values()), Tensor)
            for t in tensors:
                t.requires_grad_(True)
                t.retain_grad()
            role._grad_inputs[call_id] = tensors

        result = getattr(role, method_name)(*resolved_args, **resolved_kwargs)

        if grad_mode:
            # Save output tensors BEFORE detach so backward can use grad_fn.
            # All ranks save (not just primary) to support TP/SP backward.
            role._grad_outputs[call_id] = collect_leaves(result, Tensor)

        return self._transform_tree(result, self._pack_output)

    # ── Tree transformation ──

    def _transform_tree(self, obj, leaf_fn):
        """Recursively apply leaf_fn to TensorHandle/Tensor nodes in a nested structure."""
        transformed = leaf_fn(obj)
        if transformed is not obj:
            return transformed
        if isinstance(obj, Batch):
            return obj.map(lambda v: self._transform_tree(v, leaf_fn))
        if isinstance(obj, tuple):
            return tuple(self._transform_tree(item, leaf_fn) for item in obj)
        if isinstance(obj, list):
            return [self._transform_tree(item, leaf_fn) for item in obj]
        if isinstance(obj, dict):
            return {k: self._transform_tree(v, leaf_fn) for k, v in obj.items()}
        return obj

    def _resolve_input(self, obj):
        """Resolve TensorHandle → Tensor.

        Always returns a detached tensor so that TensorStore contents are never
        polluted by grad state set inside any RPC (requires_grad, grad_fn, etc.).

        IPC tensors from sibling workers are returned as-is (non-resizable view).
        If the role passes such a tensor to put(), TensorStore detects the
        non-resizable storage and clones it to local memory before storing.
        Roles that only read the tensor (no put) avoid the clone entirely.
        """
        if isinstance(obj, TensorHandle):
            if obj.object_ref is not None:
                return ray.get(obj.object_ref).detach()
            if obj.worker_id == self.store.worker_id:
                return self.store.get(obj)
            elif obj.ipc_handle is not None:
                storage = torch.UntypedStorage._new_shared_cuda(*obj.ipc_handle)
                t = torch.empty(0, dtype=obj.dtype, device=self.device)
                t.set_(storage, obj.offset, obj.shape, obj.stride)
                return t
            else:
                raise RuntimeError(
                    f"Unresolved TensorHandle from worker '{obj.worker_id}': "
                    f"not local and has no ipc_handle. "
                    f"_ensure_local should have handled cross-device transfers."
                )
        return obj

    def _pack_output(self, obj):
        """Pack Tensor → TensorHandle into store.

        CPU tensors are stored in Ray plasma store (object_ref), not TensorStore.
        CUDA tensors must be on this worker's device; wrong device raises.
        """
        if isinstance(obj, Tensor):
            if obj.is_cuda and str(obj.device) != self.device:
                raise RuntimeError(
                    f"Worker {self.store.worker_id} device={self.device}, "
                    f"but output tensor is on {obj.device}. "
                    f"Move the tensor to the correct device before returning."
                )
            return self.store.put(obj)
        return obj

    # ── GC helpers (called remotely by TensorHandle finalizer) ──

    def decref_tensor(self, key: str) -> None:
        self.store.decref(key)

    def incref_tensor(self, key: str) -> None:
        self.store.incref(key)

    def ref_count_tensor(self, key: str) -> int:
        return self.store.ref_count(key)

    def get_tensor_cpu(self, handle: TensorHandle) -> Tensor:
        """Fetch tensor described by handle to CPU, restoring the exact view."""
        return self.store.get(handle).cpu()

    # ── Tensor operations (called remotely by TensorHandle/TensorMeta ops) ──

    def tensor_op(self, handle: TensorHandle, op: str, *op_args) -> TensorHandle:
        """Execute a tensor operation on a stored tensor, return new TensorHandle.

        Accepts TensorHandle (may be CPU via object_ref or CUDA via store_key).
        Uses store.get() to reconstruct the exact view before applying the op.
        """
        if handle.object_ref is not None:
            t = ray.get(handle.object_ref)
        else:
            t = self.store.get(handle)

        if op == "getitem":
            result = t[op_args[0]]
        elif op == "reshape":
            result = t.reshape(op_args[0])
        elif op == "permute":
            result = t.permute(op_args[0])
        else:
            raise ValueError(f"Unknown tensor_op: '{op}'")

        return self.store.put(result)

    def cat_tensors(self, handles: List[TensorHandle]) -> TensorHandle:
        """Cat multiple tensors along dim0, return new TensorHandle.

        Handles may be local (same worker_id), CPU (object_ref),
        or from a same-device sibling (opened via CUDA IPC).
        """
        tensors = []
        for h in handles:
            if h.object_ref is not None:
                tensors.append(ray.get(h.object_ref))
            elif h.worker_id == self.store.worker_id:
                tensors.append(self.store.get(h))
            else:
                storage = torch.UntypedStorage._new_shared_cuda(*h.ipc_handle)
                t = torch.empty(0, dtype=h.dtype, device=self.device)
                t.set_(storage, h.offset, h.shape, h.stride)
                tensors.append(t)
        return self.store.put(torch.cat(tensors, dim=0))

    # ── NCCL (private, called by WorkerGroup internally) ──

    def _nccl_send(self, dst_rank: int, handles: List[TensorHandle]) -> None:
        self.store._nccl_send(dst_rank, handles)

    def _nccl_recv(
        self,
        src_global_rank: int,
        shapes: List[Tuple[int, ...]],
        dtypes: List[torch.dtype],
    ) -> List[TensorHandle]:
        return self.store._nccl_recv(src_global_rank, shapes, dtypes)

    def setup_global_pg(self) -> None:
        """Initialize NCCL process group for cross-worker transfers (slot0 only)."""
        assert self.slot == 0, "setup_global_pg must only be called on slot0 workers"
        self.store.setup_global_pg(self.store.global_rank, self.store.global_world_size)

    # ── Info queries ──

    def get_gpu_count(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def get_cuda_visible_devices(self) -> str:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "")

    def get_node_ip(self) -> str:
        return ray.util.get_node_ip_address()
