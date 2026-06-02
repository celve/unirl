"""Dispatch modes, dispatch/collect functions, and @distributed decorator.

All dispatch/collect logic lives here — single source of truth.
Handle imports from this module and uses DISPATCH_MODE_REGISTRY.

Design:
  - Dispatch enum: declares how input flows to workers
  - Execute enum: declares which workers run
  - Each Dispatch mode is paired with dispatch_fn + collect_fn in DISPATCH_MODE_REGISTRY
  - dispatch/collect functions take (wg, args, kwargs, batch_size) to access rank_info, dp_size, etc.
  - @distributed decorator marks Remote methods with their dispatch/execute modes

DP-aware dispatch (DP_SCATTER, DP_SCATTER_HEAD):
  - Input is split by dp_size (not world_size) using recursive split_value
  - Workers in the same DP group (varying TP/PP/SP rank) receive the SAME shard
  - Collect filters: only tp_rank==0, pp_last_stage, sp_rank==0 results are kept
  - Kept results are merged via pytree_merge to reconstruct the full batch
"""

from __future__ import annotations

from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from unirl.distributed.tensor.backend.tensor_store.handle import TensorHandle
from unirl.distributed.tensor.batch import Batch
from unirl.distributed.tensor.transport import TensorMeta
from unirl.distributed.utils import Broadcast


def _unwrap_broadcast(args: tuple, kwargs: dict):
    """Strip top-level Broadcast wrappers from args and kwargs.

    Broadcast is a controller-side dispatch annotation: it must be consumed
    here and never reach workers. Only top-level args/kwargs values can be
    Broadcast — nesting is not supported.
    """
    clean_args = tuple(v.value if isinstance(v, Broadcast) else v for v in args)
    clean_kwargs = {k: (v.value if isinstance(v, Broadcast) else v) for k, v in kwargs.items()}
    return clean_args, clean_kwargs


# ── Enums ──


class Dispatch(Enum):
    """How to distribute input to workers."""

    BROADCAST = auto()  # Same data to every worker
    SCATTER = auto()  # Split N ways across world (one shard per worker)
    DP_SCATTER = auto()  # Chunk by dp_size; all ranks in DP group get the same shard; collect merge
    DP_SCATTER_HEAD = auto()  # Chunk by dp_size; only DP head gets shard, others empty; collect merge


class Execute(Enum):
    """Which workers execute."""

    ALL = auto()  # All workers execute
    RANK_ZERO = auto()  # Only rank 0 executes


# ── split_value ──


def split_value(value, dp_size: int, batch_size: int) -> list:
    """Recursively split a value into dp_size shards.

    Rules:
      - Broadcast(x) → [x] * dp_size  (explicit opt-out of splitting)
      - Tensor → chunk along dim0 (must be divisible)
      - ndarray → split along axis0 (must be divisible)
      - TensorMeta → chunk (requires len(shards) == dp_size)
      - list → slice into equal parts (must be divisible)
      - tuple → recurse element-wise, reassemble per-shard tuples
      - dict → recurse into values, reassemble per-shard dicts
      - Batch → split each field, reassemble per-shard Batch objects
      - other (int/float/str/None) → [value] * dp_size (broadcast)

    To prevent a value inside a tuple/dict/Batch from being split,
    wrap it in Broadcast(x).
    """
    if isinstance(value, Broadcast):
        return [value.value] * dp_size

    elif isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return [value] * dp_size  # 0-dim scalar → broadcast
        if value.shape[0] != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        chunk_size = batch_size // dp_size
        return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(dp_size)]

    elif isinstance(value, np.ndarray):
        if value.shape[0] != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        chunk_size = batch_size // dp_size
        return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(dp_size)]

    elif isinstance(value, TensorMeta):
        total = value.shape[0]
        if total != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        n_refs = len(value.refs)
        if n_refs % dp_size != 0:
            raise ValueError(f"TensorMeta has {n_refs} refs, not divisible by dp_size={dp_size}")
        refs_per_shard = n_refs // dp_size
        parts = []
        for i in range(dp_size):
            start = i * refs_per_shard
            end = start + refs_per_shard
            shard_refs = value.refs[start:end]
            shard_sizes = value.sizes[start:end]
            shard_total = sum(shard_sizes)
            if len(shard_refs) == 1:
                parts.append(shard_refs[0])
            else:
                parts.append(
                    TensorMeta(
                        refs=shard_refs,
                        sizes=shard_sizes,
                        shape=(shard_total, *value.shape[1:]) if value.shape else None,
                        dtype=value.dtype,
                        device=value.device,
                    )
                )
        return parts

    elif isinstance(value, list):
        if len(value) != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        chunk_size = batch_size // dp_size
        return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(dp_size)]

    elif isinstance(value, dict):
        # Recurse into each value, then reassemble per-shard dicts
        split_dict = {k: split_value(v, dp_size, batch_size) for k, v in value.items()}
        return [{k: split_dict[k][i] for k in value} for i in range(dp_size)]

    elif isinstance(value, tuple):
        # Recurse element-wise, reassemble per-shard tuples.
        # To prevent splitting, wrap individual elements in Broadcast(x).
        split_elems = [split_value(v, dp_size, batch_size) for v in value]
        return [tuple(split_elems[j][i] for j in range(len(value))) for i in range(dp_size)]

    elif isinstance(value, Batch):
        if value.batch_size != batch_size:
            return [value] * dp_size  # not batch-aligned with dispatch → broadcast
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        # Batch.chunk delegates to slice → field-kind-aware (CONCAT/PACKED/SHARED)
        # and propagates cu_seqlens; inverse of the collect-side Batch.concat.
        return value.chunk(dp_size)

    else:
        # int, float, str, None, etc. → broadcast
        return [value] * dp_size


# ── pytree_merge ──


def pytree_merge(results: list) -> Any:
    """Recursively merge multiple same-structure results from workers.

    Rules:
      - Tensor → torch.cat along dim0
      - TensorHandle → combine into TensorMeta (multi-handle)
      - TensorMeta → merge all handles into one TensorMeta
      - ndarray → np.concatenate along axis0
      - list → flatten (concatenate lists)
      - tuple → recurse element-wise, return tuple
      - dict → recurse per-key, return dict
      - Batch → Batch.cat
      - None → return None
      - scalar (int/float/str/...) → return first (all should be identical)
    """
    if not results:
        return None

    first = results[0]

    if first is None:
        return None
    elif isinstance(first, torch.Tensor):
        return torch.cat(results, dim=0)
    elif isinstance(first, TensorHandle):
        return TensorMeta.from_handles(list(results))
    elif isinstance(first, TensorMeta):
        all_refs = []
        all_sizes = []
        for m in results:
            all_refs.extend(m.refs)
            all_sizes.extend(m.sizes)
        total = sum(all_sizes)
        return TensorMeta(
            refs=all_refs,
            sizes=all_sizes,
            shape=(total, *first.shape[1:]) if first.shape else None,
            dtype=first.dtype,
            device=first.device,
        )
    elif isinstance(first, np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(first, list):
        return sum(results, [])
    elif isinstance(first, tuple):
        return tuple(pytree_merge([r[i] for r in results]) for i in range(len(first)))
    elif isinstance(first, dict):
        return {k: pytree_merge([r[k] for r in results]) for k in first}
    elif isinstance(first, Batch):
        return type(first).concat(results)
    else:
        # Scalar: all DP ranks should return the same value, take first
        return first


# ── Dispatch functions (wg, args, kwargs, batch_size) → List[(args_i, kwargs_i)] ──


def _dispatch_broadcast(wg, args: tuple, kwargs: dict, batch_size: Optional[int]) -> List:
    """Broadcast same args/kwargs to all workers."""
    args, kwargs = _unwrap_broadcast(args, kwargs)
    return [(args, kwargs)] * wg.world_size


def _dispatch_scatter(wg, args: tuple, kwargs: dict, batch_size: Optional[int]) -> List:
    """Split args/kwargs by world_size (treat every worker as its own DP rank).

    Equivalent to DP_SCATTER with dp_size == world_size.
    """
    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs)] * wg.world_size

    split_args = tuple(split_value(v, wg.world_size, batch_size) for v in args)
    split_kwargs = {k: split_value(v, wg.world_size, batch_size) for k, v in kwargs.items()}

    return [
        (tuple(split_args[j][i] for j in range(len(args))), {k: split_kwargs[k][i] for k in kwargs})
        for i in range(wg.world_size)
    ]


def _dispatch_dp_scatter(wg, args: tuple, kwargs: dict, batch_size: Optional[int]) -> List:
    """Split args/kwargs by dp_size, assign by dp_rank.

    Workers in the same DP group (different TP/PP/SP ranks) receive
    the identical data shard. Each worker is responsible for internal
    slicing (TP slices hidden dim, PP runs its own layers, etc.).

    If batch_size is None (all broadcast), replicate to all workers.
    """
    dp_size = wg.dp_size

    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs)] * wg.world_size

    # Split into dp_size shards
    split_args = tuple(split_value(v, dp_size, batch_size) for v in args)
    split_kwargs = {k: split_value(v, dp_size, batch_size) for k, v in kwargs.items()}

    # Build per-dp-rank shards
    dp_shards = []
    for dp_rank in range(dp_size):
        shard_args = tuple(split_args[j][dp_rank] for j in range(len(args)))
        shard_kwargs = {k: split_kwargs[k][dp_rank] for k in kwargs}
        dp_shards.append((shard_args, shard_kwargs))

    # Map each worker to its DP shard
    return [dp_shards[wg.rank_infos[i].dp_rank] for i in range(wg.world_size)]


def _dispatch_dp_scatter_head(wg, args: tuple, kwargs: dict, batch_size: Optional[int]) -> List:
    """Like DP_SCATTER, but non-head ranks receive empty args/kwargs.

    DP head rank per group: tp_rank==0, pp_rank==0, sp_rank==0.
    This saves RPC bandwidth when workers broadcast data internally.
    """
    dp_size = wg.dp_size

    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs) if _is_dp_head(wg.rank_infos[i]) else ((), {}) for i in range(wg.world_size)]

    # Split into dp_size shards
    split_args = tuple(split_value(v, dp_size, batch_size) for v in args)
    split_kwargs = {k: split_value(v, dp_size, batch_size) for k, v in kwargs.items()}

    dp_shards = []
    for dp_rank in range(dp_size):
        shard_args = tuple(split_args[j][dp_rank] for j in range(len(args)))
        shard_kwargs = {k: split_kwargs[k][dp_rank] for k in kwargs}
        dp_shards.append((shard_args, shard_kwargs))

    return [
        dp_shards[wg.rank_infos[i].dp_rank] if _is_dp_head(wg.rank_infos[i]) else ((), {}) for i in range(wg.world_size)
    ]


def _is_dp_head(ri) -> bool:
    return ri.tp_rank == 0 and ri.pp_rank == 0 and ri.sp_rank == 0


# ── Collect functions (wg, results) → collected ──


def _collect_passthrough(wg, results: List) -> List:
    """Return all results as list (raw)."""
    return results


def _collect_dp_merge(wg, results: List) -> Any:
    """Collect only DP-head results per DP group, then merge.

    DP head: tp_rank==0, is_pipeline_last_stage, sp_rank==0.
    Returns the pytree_merge'd result across DP ranks.

    Handles Execute.RANK_ZERO case where len(results) < world_size.
    """
    dp_results = []
    for i in range(len(results)):
        ri = wg.rank_infos[i]
        if ri.tp_rank == 0 and ri.is_pipeline_last_stage and ri.sp_rank == 0:
            dp_results.append(results[i])

    if not dp_results:
        return None
    if len(dp_results) == 1:
        return dp_results[0]

    return pytree_merge(dp_results)


# ── Registry: Dispatch mode → paired (dispatch_fn, collect_fn) ──

DISPATCH_MODE_REGISTRY: Dict[Dispatch, Dict[str, Callable]] = {
    Dispatch.BROADCAST: {"dispatch_fn": _dispatch_broadcast, "collect_fn": _collect_passthrough},
    Dispatch.SCATTER: {"dispatch_fn": _dispatch_scatter, "collect_fn": _collect_passthrough},
    Dispatch.DP_SCATTER: {"dispatch_fn": _dispatch_dp_scatter, "collect_fn": _collect_dp_merge},
    Dispatch.DP_SCATTER_HEAD: {"dispatch_fn": _dispatch_dp_scatter_head, "collect_fn": _collect_dp_merge},
}


# ── Backward dispatch mode resolution ────────────────────────────────────────


def resolve_backward_dispatch_mode(
    method_name: str,
    fwd_dispatch_mode: Dispatch,
    rank_infos: list,
) -> Dispatch:
    """Return the dispatch mode for the backward RPC, or raise if unsupported.

    Rules:
      DP_SCATTER  + pp_size==1 → DP_SCATTER   (grad shards align with output shards)
      DP_SCATTER_HEAD + pp_size==1 → DP_SCATTER  (all ranks must participate in backward)
      DP_SCATTER / DP_SCATTER_HEAD + pp_size>1 → Error (autograd graph broken across PP)
      BROADCAST → Error
      SCATTER → Error

    !! IMPORTANT — adding a new Dispatch variant !!
    Update this function to decide whether DP_SCATTER backward is correct,
    or a hard error is needed.  Also check Remote._auto_backward's dispatch_mode.
    """
    if fwd_dispatch_mode in (Dispatch.BROADCAST, Dispatch.SCATTER):
        raise ValueError(
            f"Method '{method_name}' uses dispatch_mode={fwd_dispatch_mode.name}, "
            f"which does not support auto-backward (no shared batch dimension). "
            f"Do not call this method inside enable_grad()."
        )

    pp_sizes = {ri.pp_size for ri in rank_infos}
    if any(pp > 1 for pp in pp_sizes):
        raise ValueError(
            f"Method '{method_name}' has pp_size>1. "
            f"Auto-backward cannot propagate gradients across pipeline stages. "
            f"Do not call this method inside enable_grad()."
        )

    # DP_SCATTER_HEAD → DP_SCATTER (all ranks must participate in backward)
    # DP_SCATTER   → DP_SCATTER (unchanged)
    return Dispatch.DP_SCATTER


# ── @distributed decorator ──

DISTRIBUTED_CONFIG_ATTR = "_distributed_config"


def distributed(
    _func: Callable = None,
    *,
    dispatch_mode: Dispatch = Dispatch.DP_SCATTER,
    execute_mode: Execute = Execute.ALL,
) -> Callable:
    """Declare SPMD dispatch/execute mode on a Role method.

    Handle scans for this attribute and auto-generates proxy methods.
    Default dispatch mode is DP_SCATTER.

    Usage:
        class DiffusionRemote(Remote):
            @distributed
            def rollout(self, samples, prompts):
                ...

            @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
            def get_metrics(self):
                ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(
            wrapper,
            DISTRIBUTED_CONFIG_ATTR,
            {
                "dispatch_mode": dispatch_mode,
                "execute_mode": execute_mode,
            },
        )
        return wrapper

    if _func is not None:
        # Called as @distributed without parentheses
        return decorator(_func)
    return decorator
