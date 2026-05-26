"""Transportable: ``Batched`` dataclass that round-trips through TransferQueue.

Subclasses tag fields with ``transport=True`` (via the generic ``field(...)``
constructor in ``utils/batched.py``) to route them through efficient transport.
Recursion is gated on the same flag — a tagged field whose value is itself a
``Transportable`` is descended into; an untagged field never is. Reachable
leaves get a *dotted-path* key derived from the field-attr names along the way
(e.g. ``segment.latents``), unique by construction.

Two concentric levels of API:

- **Per-instance roundtrip** — :meth:`Transportable.dehydrate` /
  :meth:`Transportable.hydrate`. ``dehydrate`` drains the container's tensors
  into TQ and replaces each leaf with a :class:`TqMeta` reference; ``hydrate``
  fetches them back. The four lower-level methods
  (``to_tensordict`` / ``replace_with_meta`` / ``collect_remote_metas`` /
  ``restore_from_tensordict``) stay public for partial-flow callers and tests.

- **Decorator surface** — :func:`tqbridge` wraps an actor method so that
  arguments are hydrated before the call and / or the return value is
  dehydrated after. The decorator walks the value tree (lists / tuples / dicts
  / Batched containers) to find top-level ``Transportable`` subtrees.

Wire-key wiring: keys are computed at *put* time via the dotted-path walker
and frozen into ``TqMeta._data_key``. ``hydrate`` / ``restore_from_tensordict``
look up by ``wrapper._data_key`` (not by the receiver's walker path), so
sender and receiver containers can have different field-name shapes — the
wire is keyed by the sender's structure, end of story.
"""

from __future__ import annotations

import copy
import inspect
import logging
import os
import time
import uuid as _uuid
from dataclasses import MISSING, Field
from dataclasses import fields as dc_fields
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar

import torch

from diffusionrl.distributed.transfer_queue.meta import TqMeta
from diffusionrl.distributed.transfer_queue.runtime import (
    _DEFAULT_PARTITION_ID,
    TransferQueueRuntime,
    _run_async_in_temp_loop,
)
from diffusionrl.utils.batched import Batched

if TYPE_CHECKING:
    from tensordict import TensorDict
    from transfer_queue import BatchMeta


logger = logging.getLogger(__name__)
_DEBUG = int(os.getenv("TRANSFER_QUEUE_UTILS_DEBUG", 0))

T = TypeVar("T", bound="Transportable")


def _tq_enabled() -> bool:
    """True iff a TransferQueueRuntime with a live client is bound to this process."""
    runtime = TransferQueueRuntime.current()
    return runtime is not None and runtime.is_enabled()


def _stack_data(data_ret: Any) -> Any:
    """Re-stack Mooncake's zero-copy list return into a single tensor.

    With ``manager_merge_to_tensordict=False`` Mooncake returns ``list[Tensor]``
    for tensor fields; stack along dim 0. Pass-through for already-stacked
    tensors and for non-tensor lists (which represent ``NonTensorData`` payloads).
    """
    if isinstance(data_ret, list):
        if data_ret and all(isinstance(t, torch.Tensor) for t in data_ret):
            return torch.stack(data_ret, dim=0)
        return data_ret
    if isinstance(data_ret, torch.Tensor):
        return data_ret
    raise ValueError(f"unexpected payload shape from TQ: {type(data_ret).__name__}")


def _ensure_element_eid(elem: "Transportable") -> str:
    """Lazily stamp a transient UUID on a ``list[Transportable]`` element.

    The eid becomes a wire-key segment so each element's transport leaves get
    globally unique ``_data_key``s — required because list positions are not
    identities, so two shards' element-0s must not share a wire key. Stored via
    ``object.__setattr__`` to bypass any frozen-dataclass restriction.
    """
    eid = getattr(elem, "__transport_eid__", None)
    if eid is None:
        eid = _uuid.uuid4().hex[:12]
        object.__setattr__(elem, "__transport_eid__", eid)
    return eid


# =========================================================
# Transportable mixin
# =========================================================


class Transportable(Batched):
    """``Batched`` subclass with TransferQueue round-tripping baked in.

    Tag a field via the generic ``field(...)`` constructor::

        latents: Tensor = field(kind=FieldKind.CONCAT, transport=True)
        segment: Optional[Segment] = field(
            kind=FieldKind.CONCAT, default=None, transport=True,
        )

    Recursion enters a tagged field when its value is one of:
    - a ``Transportable`` — recurse with the field name as the next path segment;
    - a ``list[Transportable]`` (homogeneous) — recurse into each element using
      a per-element transient UUID as the path segment, so cross-shard merges
      don't collide on positional indices;
    - a ``dict[str, Transportable]`` (homogeneous values) — recurse into each
      value using the dict key as the path segment. Same key across shards is
      by design; ``_concat_value`` for dicts merges same-keyed values.

    Other tagged values are leaves and the visitor sees them. Leaf wire-keys
    are dotted paths assembled from the segments along the way (e.g.
    ``segment.latents``, ``conditions.<key>.embeds``).
    """

    # ---- per-leaf walker (visitor-as-map) ------------------------------------

    def _walk_leaves(
        self,
        fn: Callable[["Transportable", Field, Any, str], Any],
        prefix: str = "",
    ) -> None:
        """Apply ``fn`` to each reachable transport leaf.

        Recursion is entered iff a field is tagged ``transport=True`` AND its
        value is a ``Transportable``, a homogeneous ``list[Transportable]``,
        or a homogeneous ``dict[str, Transportable]``. Other tagged fields are
        leaves and the visitor sees them.

        ``fn`` receives ``(owner, field, value, dotted_key)`` and returns the
        value to leave in the slot. The walker writes back via ``setattr`` only
        if the returned value is not the same object — visitors never write
        ``setattr`` themselves.
        """
        for f in dc_fields(self):
            if not f.metadata.get("transport"):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            key = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(v, Transportable):
                v._walk_leaves(fn, key)
            elif isinstance(v, list) and v and all(isinstance(x, Transportable) for x in v):
                for elem in v:
                    eid = _ensure_element_eid(elem)
                    elem._walk_leaves(fn, f"{key}.{eid}")
            elif isinstance(v, dict) and v and all(isinstance(x, Transportable) for x in v.values()):
                for dk, elem in v.items():
                    if not isinstance(dk, str) or "." in dk:
                        raise ValueError(
                            f"dict-of-Transportable key must be a string without '.'; "
                            f"got {dk!r} on {type(self).__name__}.{f.name}"
                        )
                    elem._walk_leaves(fn, f"{key}.{dk}")
            else:
                new_v = fn(self, f, v, key)
                if new_v is not v:
                    setattr(self, f.name, new_v)

    # ---- light (metadata-only) copy ------------------------------------------

    def metadata_only(self: T) -> T:
        """Return a shallow copy with every ``transport=True`` field reset to its
        default — the light metadata, with the heavy transport payload dropped.

        The ``transport=True`` tag is the single source of truth for "what is
        heavy" (the same tag :meth:`to_tensordict` / :meth:`dehydrate` route on),
        so adding/removing a transport field needs no matching edit here. Light
        (non-transport) fields are carried over by reference.

        This is a *top-level-field* drop. A container whose own light data lives
        *inside* a transport field (e.g. ``RolloutResp.tracks``) must override
        this to recurse, since dropping the field here would discard that data.
        """
        light = copy.copy(self)
        for f in dc_fields(self):
            if not f.metadata.get("transport"):
                continue
            if f.default_factory is not MISSING:
                default: Any = f.default_factory()
            elif f.default is not MISSING:
                default = f.default
            else:
                raise ValueError(
                    f"metadata_only() cannot drop transport field "
                    f"{type(self).__name__}.{f.name}: it declares no default to reset to"
                )
            setattr(light, f.name, default)
        return light

    # ---- four roundtrip methods (consume the walker) -------------------------

    def to_tensordict(self) -> Optional["TensorDict"]:
        """Collect transport-tagged populated values into a flat TensorDict.

        Tensor values become TensorDict entries directly; ``list`` values are
        wrapped in ``NonTensorData``. Returns ``None`` when no leaf is
        populated. Raises on duplicate keys (paths are unique by construction
        so this can only fire if a class-author misuses ``transport=``).
        """
        from tensordict import NonTensorData, TensorDict

        d: dict = {}
        bs: Optional[int] = None

        def collect(owner: "Transportable", f: Field, v: Any, key: str) -> Any:
            nonlocal bs
            if isinstance(v, TqMeta):
                return v
            if key in d:
                raise ValueError(f"duplicate transport key {key!r} on {type(owner).__name__}.{f.name}")
            if isinstance(v, torch.Tensor):
                d[key] = v
                if bs is None:
                    bs = int(v.shape[0])
            elif isinstance(v, list):
                d[key] = NonTensorData(v)
                if bs is None:
                    bs = len(v)
            else:
                raise TypeError(
                    f"transport field {type(owner).__name__}.{f.name} has type "
                    f"{type(v).__name__}; expected Tensor or list"
                )
            return v

        self._walk_leaves(collect)
        if not d:
            return None
        return TensorDict(d, batch_size=bs).cpu()

    def replace_with_meta(self, batch_meta: "BatchMeta") -> None:
        """Pure transform: swap each tensor / list leaf with a ``TqMeta`` ref.

        Slots already holding a ``TqMeta`` (or non-tensor / non-list types) are
        left alone. The walker handles the ``setattr``.
        """

        def transform(owner: "Transportable", f: Field, v: Any, key: str) -> Any:
            if isinstance(v, TqMeta) or not isinstance(v, (torch.Tensor, list)):
                return v
            shape = v.shape if isinstance(v, torch.Tensor) else (len(v),)
            return TqMeta(
                batch_meta=batch_meta.select_fields([key]),
                data=None,
                _shape=shape,
                _data_key=key,
            )

        self._walk_leaves(transform)

    def collect_remote_metas(self) -> List["BatchMeta"]:
        """Return the BatchMetas held by every ``TqMeta`` leaf."""
        metas: List["BatchMeta"] = []

        def collect(owner: "Transportable", f: Field, v: Any, key: str) -> Any:
            if isinstance(v, TqMeta) and v.batch_meta is not None:
                metas.append(v.batch_meta)
            return v

        self._walk_leaves(collect)
        return metas

    def restore_from_tensordict(self, td: "TensorDict") -> None:
        """Pure transform: replace each ``TqMeta`` leaf with the fetched tensor.

        Lookup uses the wrapper's stored ``_data_key`` (frozen at put time), not
        the walker's dotted path — sender / receiver shapes can differ.
        """

        def transform(owner: "Transportable", f: Field, v: Any, key: str) -> Any:
            if isinstance(v, TqMeta) and v._data_key in td:
                return _stack_data(td[v._data_key])
            return v

        self._walk_leaves(transform)

    # ---- high-level: full TQ roundtrip --------------------------------------

    async def dehydrate(
        self,
        client: Any,
        partition_id: str = _DEFAULT_PARTITION_ID,
    ) -> None:
        """Send transport leaves to TQ; swap each in-place with a ``TqMeta`` ref.

        No-op when no leaf is populated. After this the container is "light":
        heavy bytes live in TQ; only references travel over Ray.
        """
        td = self.to_tensordict()
        if td is None:
            return
        meta = await client.async_put(data=td, partition_id=partition_id)
        self.replace_with_meta(meta)

    async def hydrate(self, client: Any) -> None:
        """Resolve all ``TqMeta`` references via one ``async_get_data`` call.

        Unions every leaf's ``BatchMeta`` so the whole container fetches in one
        round-trip. No-op when nothing is currently remote.
        """
        metas = self.collect_remote_metas()
        if not metas:
            return
        union = metas[0]
        for m in metas[1:]:
            union = union.union(m)
        td = await client.async_get_data(union)
        self.restore_from_tensordict(td)


# =========================================================
# value-tree walker — finds top-level Transportable subtrees
# =========================================================


def _walk_subtrees(data: Any, fn: Callable[[Transportable], None]) -> None:
    """Recurse through lists / tuples / dicts / Batched values; call ``fn`` on
    each ``Transportable`` encountered, then stop descending into it (its own
    ``_walk_leaves`` handles the rest of its subtree).
    """
    if data is None or isinstance(data, (int, float, str, bytes, bool)):
        return
    if isinstance(data, Transportable):
        fn(data)
        return
    if isinstance(data, (list, tuple)):
        for x in data:
            _walk_subtrees(x, fn)
        return
    if isinstance(data, dict):
        for v in data.values():
            _walk_subtrees(v, fn)
        return
    if isinstance(data, Batched):
        for f in dc_fields(data):
            _walk_subtrees(getattr(data, f.name), fn)
        return
    # Plain tensors and other scalars: nothing to do.


# =========================================================
# profiler (optional debug-timing wrapper)
# =========================================================


def tq_profiler(name: str = ""):
    def decorator(func):
        if not _DEBUG:
            return func

        @wraps(func)
        def wrapper_inner(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"[{name}] {func.__name__} Runtime: {end_time - start_time:.6f} s, timestamp: f{end_time}")
            return result

        @wraps(func)
        async def wrapper_async_inner(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"[{name}] {func.__name__} Runtime: {end_time - start_time:.6f} s, timestamp: f{end_time}")
            return result

        return wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner

    return decorator


# =========================================================
# bridge orchestration over arbitrary value trees
# =========================================================


@tq_profiler(name="TQ utils")
async def _tq_dehydrate_async(data: Any, partition_id: str = _DEFAULT_PARTITION_ID) -> Any:
    """Dehydrate every ``Transportable`` subtree found in ``data``.

    Each subtree's ``dehydrate`` does its own ``to_tensordict`` + ``async_put``
    + ``replace_with_meta``; no cross-subtree multiplexing is needed because
    the typical sender boundary is a single subtree per response.
    """
    runtime = TransferQueueRuntime.current()
    if runtime is None or runtime.client is None:
        return data
    client = runtime.client
    subtrees: List[Transportable] = []
    _walk_subtrees(data, subtrees.append)
    for subtree in subtrees:
        await subtree.dehydrate(client, partition_id)
    return data


@tq_profiler(name="TQ utils")
async def _tq_hydrate_async(data: Any) -> Any:
    """Hydrate every ``Transportable`` subtree found in ``data``.

    Unions metas across all subtrees so the entire data tree fetches in one
    ``async_get_data`` call — important when the receiver is a ``Batched``
    container with multiple parallel ``Transportable`` children. (When the
    receiver is itself a single ``Transportable``, this path also works and
    issues exactly one round-trip.)
    """
    runtime = TransferQueueRuntime.current()
    if runtime is None or runtime.client is None:
        return data
    client = runtime.client
    subtrees: List[Transportable] = []
    _walk_subtrees(data, subtrees.append)

    all_metas: List["BatchMeta"] = []
    for subtree in subtrees:
        all_metas.extend(subtree.collect_remote_metas())
    if not all_metas:
        return data

    union = all_metas[0]
    for m in all_metas[1:]:
        union = union.union(m)
    td = await client.async_get_data(union)
    for subtree in subtrees:
        subtree.restore_from_tensordict(td)
    return data


def _tq_dehydrate(data: Any, partition_id: str = _DEFAULT_PARTITION_ID) -> Any:
    return _run_async_in_temp_loop(_tq_dehydrate_async, data, partition_id)


def _tq_hydrate(data: Any) -> Any:
    return _run_async_in_temp_loop(_tq_hydrate_async, data)


# =========================================================
# decorator
# =========================================================


def tqbridge(get: bool = False, put: bool = False):
    """Wrap a function so its inputs are hydrated and / or its output dehydrated.

    ``get=True``: each arg is walked; any ``TqMeta`` references in its
    ``Transportable`` subtrees are resolved into tensors before the wrapped
    function runs.

    ``put=True``: the return value is walked and each ``Transportable``
    subtree's tagged tensors are sent to TQ and replaced with references.

    No-ops when no TQ client is initialized.
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if get and _tq_enabled():
                args = [_tq_hydrate(a) for a in args]
                kwargs = {k: _tq_hydrate(v) for k, v in kwargs.items()}
            output = func(*args, **kwargs)
            if put and _tq_enabled():
                output = _tq_dehydrate(output)
            return output

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if get and _tq_enabled():
                args = [await _tq_hydrate_async(a) for a in args]
                kwargs = {k: await _tq_hydrate_async(v) for k, v in kwargs.items()}
            output = await func(*args, **kwargs)
            if put and _tq_enabled():
                output = await _tq_dehydrate_async(output)
            return output

        return async_inner if inspect.iscoroutinefunction(func) else inner

    return decorator


@tqbridge(get=True, put=False)
def resolve_batch_from_tq(data):
    """Resolve any TQ-referenced tensors held in ``data`` and return it."""
    return data


__all__ = [
    "Transportable",
    "resolve_batch_from_tq",
    "tq_profiler",
    "tqbridge",
]
