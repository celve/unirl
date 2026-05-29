"""Miscellaneous utilities for the distributed controller.

Network helpers used by DevicePool and RoleProxy, plus batch-size
inference primitives used by the dispatch layer.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import fields as dc_fields
from typing import List, Optional, Tuple, Type

import numpy as np
import torch

from diffusionrl.distributed.tensor.backend.tensor_store.handle import TensorHandle
from diffusionrl.distributed.tensor.batch import Batch
from diffusionrl.distributed.tensor.transport import TensorMeta

logger = logging.getLogger(__name__)


# ── CUDA IPC compatibility ──

# cudaMalloc-backed IPC handles always have a 66-byte device handle (CUDA_IPC_HANDLE_SIZE=64
# + 2 bytes padding).  Expandable-segments (VMM) allocations produce shorter handles
# because they use pidfd_getfd for cross-process sharing instead of the legacy IPC path.
_CUDA_IPC_HANDLE_BYTES = 66


def cuda_ipc_needs_clone(storage: torch.UntypedStorage) -> tuple:
    """Return (ipc_handle, needs_clone) for a CUDA storage.

    needs_clone is True when the storage was allocated by the expandable-segments
    (VMM) allocator: its IPC handle is not a standard cudaMalloc handle and cannot
    be opened by other processes on kernels lacking pidfd_getfd (Linux < 5.6).
    The caller should clone to a cudaMalloc-backed allocation before sharing.

    Returns the probe handle so the caller avoids a second _share_cuda_() call.
    """
    handle = storage._share_cuda_()
    return handle, len(handle[1]) != _CUDA_IPC_HANDLE_BYTES


# ── Broadcast wrapper ──


class Broadcast:
    """Mark a value as broadcast — it will NOT be split across workers.

    Usage::

        role.forward(images, lr=Broadcast(0.01), config=Broadcast(cfg_dict))

    Most callers no longer need to wrap manually: the dispatch layer
    auto-detects per-rollout metadata (tensors / lists whose ``shape[0]``
    differs from the canonical batch size) and treats them as broadcast.
    Use ``Broadcast(...)`` only when you want to suppress that
    heuristic — e.g. when a real per-sample tensor happens to be smaller
    than other per-sample fields in the same call.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return f"Broadcast({self.value!r})"


# ── Generic tree traversal ──


def collect_leaves(x, leaf_type: Type) -> list:
    """Depth-first collect all instances of leaf_type from an arbitrary structure.

    Traversal rules:
      - isinstance(x, leaf_type) -> append and stop recursing
      - Batch / dict -> recurse into values, keys sorted alphabetically
      - list / tuple  -> recurse in positional order
      - everything else -> skip

    Sorting dict/Batch keys ensures deterministic ordering across call sites,
    which is critical for aligning controller-side collect_leaves(TensorMeta)
    with worker-side collect_leaves(torch.Tensor) by index.
    """
    result = []
    if isinstance(x, leaf_type):
        result.append(x)
    elif isinstance(x, Batch):
        for f in sorted(dc_fields(x), key=lambda f: f.name):
            v = getattr(x, f.name)
            if v is not None:
                result.extend(collect_leaves(v, leaf_type))
    elif isinstance(x, dict):
        for k in sorted(x.keys()):
            result.extend(collect_leaves(x[k], leaf_type))
    elif isinstance(x, (list, tuple)):
        for v in x:
            result.extend(collect_leaves(v, leaf_type))
    return result


# ── Batch-size inference ──


def _collect_batch_sizes(value, sizes: list) -> None:
    """Recursively collect batch dimension sizes from a value.

    Rules:
      - Broadcast → skip
      - Tensor/ndarray → shape[0]
      - TensorHandle/TensorMeta → shape[0]
      - list → len (treated as per-sample batch)
      - tuple → recurse into elements (same as dict/Batch)
      - dict/Batch → recurse into values
      - other (int/float/str/None) → skip (broadcast)

    Use Broadcast(x) to explicitly prevent a value from contributing batch size.
    """
    if isinstance(value, Broadcast):
        return
    elif isinstance(value, (torch.Tensor, np.ndarray, TensorHandle, TensorMeta)):
        if len(value.shape) > 0:  # skip 0-dim scalars (no batch dimension)
            sizes.append(value.shape[0])
    elif isinstance(value, list):
        sizes.append(len(value))
    elif isinstance(value, tuple):
        for v in value:
            _collect_batch_sizes(v, sizes)
    elif isinstance(value, Batch):
        for f in dc_fields(value):
            _collect_batch_sizes(getattr(value, f.name), sizes)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_batch_sizes(v, sizes)


# Module-level state for warn-once-per-pattern: avoids spamming the log
# when production training triggers the same outlier set every step.
auto_broadcast_warned: set[tuple] = set()


def infer_and_validate_batch_size(args: tuple, kwargs: dict) -> Optional[int]:
    """Infer the canonical batch size from args/kwargs.

    Walks the call payload and finds every indexable field's
    ``shape[0]`` (or ``len`` for lists). When sizes disagree the
    **largest** wins as the canonical per-sample batch size; any field
    with a different size is treated as **per-rollout metadata** and
    will be replicated (not split) by :func:`split_value` in
    ``diffusionrl.distributed.group.dispatch``.

    A warning is logged the first time each outlier-set is encountered
    so silent misclassification is auditable. Callers who want to
    suppress the heuristic can wrap a value explicitly in
    :class:`Broadcast`.

    Returns ``None`` if no indexable field is found (pure broadcast call).
    """
    sizes: List[int] = []
    for v in args:
        _collect_batch_sizes(v, sizes)
    for v in kwargs.values():
        _collect_batch_sizes(v, sizes)

    if not sizes:
        return None

    canonical = max(sizes)
    outliers = tuple(sorted({s for s in sizes if s != canonical}))
    if outliers:
        signature = (canonical, outliers)
        if signature not in auto_broadcast_warned:
            auto_broadcast_warned.add(signature)
            logger.info(
                "Auto-broadcasting fields with shape[0] in %s "
                "(canonical batch size = %d). These tensors / lists "
                "will be replicated across DP shards instead of split. "
                "Wrap explicitly in Broadcast(...) to suppress this "
                "heuristic.",
                outliers,
                canonical,
            )
    return canonical


# ── Network helpers ──


def get_open_port() -> int:
    """Find an available TCP port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_node_ip_and_port(pg, bundle_index: int = 0) -> Tuple[str, int]:
    """Get IP and an open port on the node where a PG bundle landed.

    Both values come from the same worker node, avoiding
    controller-vs-worker mismatch. Uses a lightweight probe actor.
    """
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    @ray.remote(num_cpus=0)
    class _Probe:
        def info(self):
            ip = ray.util.get_node_ip_address()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
            return ip, port

    probe = _Probe.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
        ),
    ).remote()
    result = ray.get(probe.info.remote())
    ray.kill(probe)
    return result


def get_node_ip(pg, bundle_index: int = 0) -> str:
    """Get IP of the node where a PG bundle landed."""
    ip, _ = get_node_ip_and_port(pg, bundle_index)
    return ip
