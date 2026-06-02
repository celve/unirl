"""Handle — controller-side SPMD handle for a group of logical workers.

Handle owns a set of device slots (by device_ids), registers a Remote
on each Worker, and binds @distributed-decorated methods as handle functions.

Cross-GPU TensorHandle transfer is handled automatically: when a shard contains
TensorHandle from a foreign worker, _ensure_local() triggers NCCL send/recv
before execution. Users never call NCCL directly.

Usage:
    pool = DevicePool(num_gpus=8)
    pool.setup()

    # Basic
    handle = pool.create_remote(DiffusionRemote, device_ids=[0,1,2,3])
    handle.initialize(model_path="/models/sd", tp_size=2)

    # With constructor args
    handle = pool.create_remote(ScalerRemote, device_ids=[0,1,2,3], init_kwargs={"scale": 3.0})

    # Separated: tensor transfer is automatic
    actor = pool.create_remote(ActorRemote, device_ids=[0,1,2,3])
    reward = pool.create_remote(RewardRemote, device_ids=[4,5,6,7])
    samples = actor.rollout(prompts=prompts)
    rewards = reward.score(samples)  # auto NCCL from gpu 0-3 to gpu 4-7
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type

import ray

from diffusionrl.distributed.group.dispatch import (
    DISPATCH_MODE_REGISTRY,
    DISTRIBUTED_CONFIG_ATTR,
    Dispatch,
    Execute,
    resolve_backward_dispatch_mode,
)
from diffusionrl.distributed.group.remote import RankInfo, Remote
from diffusionrl.distributed.tensor.backend.tensor_store.handle import TensorHandle
from diffusionrl.distributed.tensor.batch import Batch
from diffusionrl.distributed.tensor.grad_context import (
    RPCBackwardNode,
    current_grad_context,
)
from diffusionrl.distributed.tensor.transport import TensorMeta
from diffusionrl.distributed.utils import (
    collect_leaves,
    infer_and_validate_batch_size,
)

if TYPE_CHECKING:
    from diffusionrl.distributed.group.device_pool import DevicePool


# ── Module-level counter for unique role_name generation ─────────────────────
_role_name_counter: Dict[str, int] = {}


def _owning_class(role_cls) -> Type[Remote]:
    """Return the class for Handle method-binding and role naming.

    ``role_cls`` may be a class (normal case), a bound classmethod
    factory (e.g. ``SD3Bundle.from_config``), or a plain function.
    For classmethods we use the owning class so ``_bind_methods`` finds
    its ``@distributed`` methods and ``_make_role_name`` produces a
    meaningful base name. For everything else, fall back to ``role_cls``
    itself.
    """
    import inspect

    if inspect.ismethod(role_cls) and isinstance(role_cls.__self__, type):
        return role_cls.__self__
    return role_cls


def _make_role_name(role_cls) -> str:
    """Generate a unique role_name from the worker class name.

    Always appends counter suffix for deterministic names.
    """
    base = _owning_class(role_cls).__name__
    count = _role_name_counter.get(base, 0)
    _role_name_counter[base] = count + 1
    return f"{base}_{count}"


def reset_role_name_counter() -> None:
    """Reset the role name counter. For testing only."""
    _role_name_counter.clear()


def _check_batch_divisibility(dispatch_mode: Dispatch, batch_size: Optional[int], dp_size: int) -> None:
    """Raise if a DP-split batch can't be divided evenly across dp ranks.

    Only DP_SCATTER / DP_SCATTER_HEAD split the per-sample batch by dp_size, so only they
    require divisibility. BROADCAST broadcasts and SCATTER splits by
    world_size — both ignore this precondition — so the check must NOT apply to
    them (it would spuriously reject valid broadcast calls, e.g.
    set_rollout_targets(rollout.workers, ...) when len(workers) % dp_size != 0).
    """
    if (
        dispatch_mode in (Dispatch.DP_SCATTER, Dispatch.DP_SCATTER_HEAD)
        and batch_size is not None
        and batch_size % dp_size != 0
    ):
        raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")


@dataclass(frozen=True)
class HandleRef:
    """Serializable marker for a Handle.

    When a ``Handle`` is passed as a kwarg to ``remote(...)``, the framework
    substitutes a ``HandleRef`` so the Worker can resolve it to the local
    ``Remote`` instance with this ``role_name`` (looked up in
    ``Worker._roles``) before constructing the new role.

    Only resolves on the same Worker as the referenced role — i.e. when the
    sibling lives on the same device slab and slot.
    """

    role_name: str


class Handle:
    """Controller-side SPMD handle.

    Creates logical workers on Workers and binds @distributed methods.

    Args:
        role_cls:      Remote subclass to register.
        pool:          DevicePool managing Workers.
        device_ids:    Explicit GPU indices. If None, auto-allocate via n_gpus.
        n_gpus:        Number of GPUs to auto-allocate (used when device_ids=None).
        role_name:     Optional role name. If None, auto-generated from class name.
        init_kwargs:   Dict of kwargs forwarded to role_cls.__init__.
    """

    def __init__(
        self,
        role_cls: Type[Remote],
        pool: DevicePool,
        device_ids: Optional[List[int]] = None,
        n_gpus: Optional[int] = None,
        role_name: Optional[str] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        slot_id: int = 0,
    ) -> None:  # noqa: D107 (args documented in class docstring)
        self.role_cls = role_cls
        self.pool = pool
        self.role_name = role_name or _make_role_name(role_cls)
        self.slot_id = slot_id

        # GPU allocation
        if device_ids is not None:
            self.device_ids = list(device_ids)  # support range, tuple, etc.
        elif n_gpus is not None:
            self.device_ids = pool.allocate(n_gpus)
        else:
            raise ValueError("Must provide device_ids or n_gpus")

        self.world_size = len(self.device_ids)
        self.workers = pool.get_workers(self.device_ids, slot=slot_id)

        # worker_ids for this group (used in _ensure_local)
        self.worker_ids = [f"dw{d}" if slot_id == 0 else f"dw{d}_s{slot_id}" for d in self.device_ids]

        # Reserve a port on rank 0's node for this group's sub-PG.
        # Held by socket until initialize() releases it.
        self._group_port = ray.get(self.workers[0]._reserve_port.remote())
        self._group_master_addr = ray.get(self.workers[0].get_node_ip.remote())

        self._dist_env_base = {
            "MASTER_ADDR": self._group_master_addr,
            "MASTER_PORT": str(self._group_port),
            "WORLD_SIZE": str(self.world_size),
            "GROUP_NAME": self.role_name,
        }

        # Register role on each Worker with dist_env
        self.rank_infos = [
            RankInfo(rank=i, world_size=self.world_size, dp_rank=i, dp_size=self.world_size)
            for i in range(self.world_size)
        ]
        ray.get(
            [
                w.add_remote.remote(
                    self.role_name,
                    role_cls,
                    self.rank_infos[i],
                    init_kwargs=init_kwargs or {},
                    dist_env={"RANK": str(i), **self._dist_env_base},
                )
                for i, w in enumerate(self.workers)
            ]
        )

        # Bind @distributed methods as handle functions
        self._bind_methods(role_cls)

        # Counter for unique call_id generation within enable_grad contexts.
        # Single-threaded training loop assumption: no concurrent handle calls.
        self._grad_call_counter = count()

    @property
    def dp_size(self) -> int:
        """Number of data-parallel groups."""
        return self.rank_infos[0].dp_size if self.rank_infos else self.world_size

    # ── User-facing initialize ──

    def initialize(self, *args, **kwargs) -> None:
        """Call role.initialize(*args, **kwargs) on all workers.

        Releases the reserved port first so init_process_group can bind it,
        then reads back (possibly modified) rank_infos.
        """
        # Release port so init_process_group can use it
        ray.get(self.workers[0]._release_port.remote(self._group_port))

        # Forward to all workers via generic call
        ray.get([w.call.remote(self.role_name, "initialize", args, kwargs) for w in self.workers])

        # Read back rank_infos (user may have modified them in initialize)
        self.rank_infos = ray.get([w.get_rank_info.remote(self.role_name) for w in self.workers])

    # ── Method binding ──

    def _bind_methods(self, role_cls) -> None:
        """Scan role_cls for @distributed methods and create handle functions.

        For classmethod ``role_cls`` (e.g. ``SD3Bundle.from_config``)
        we scan the owning class instead — the constructed instance is
        of that class, so its ``@distributed`` methods are the ones
        callers will dispatch through this Handle.
        """
        role_cls = _owning_class(role_cls)
        for name in dir(role_cls):
            method = getattr(role_cls, name, None)
            if method is None:
                continue
            config = getattr(method, DISTRIBUTED_CONFIG_ATTR, None)
            if config is None:
                continue

            fns = DISPATCH_MODE_REGISTRY[config["dispatch_mode"]]
            dispatch_fn = fns["dispatch_fn"]
            collect_fn = fns["collect_fn"]

            if config["execute_mode"] == Execute.ALL:
                execute_fn = self._execute_all
            else:
                execute_fn = self._execute_rank_zero

            bound = self._make_handle_fn(name, config["dispatch_mode"], dispatch_fn, collect_fn, execute_fn)
            setattr(self, name, bound)

    def _make_handle_fn(
        self,
        method_name: str,
        dispatch_mode: Dispatch,
        dispatch_fn: Callable,
        collect_fn: Callable,
        execute_fn: Callable,
    ) -> Callable:
        """Create handle method: dispatch → ensure_local → execute → collect → rebind.

        When a GradContext is active, wraps the call to record input/output
        TensorMetas and append an RPCBackwardNode for later auto-backward.
        grad_mode and call_id are passed as dedicated parameters to Worker.call
        (not via kwargs) so dispatch internals remain unaware of grad state.
        """

        def handle_fn(*args, **kwargs):
            ctx = current_grad_context()

            # ── enable_grad: validate backward support, record input TensorMetas ──
            call_id = None
            input_metas = []
            bwd_dispatch_mode = None
            if ctx is not None:
                bwd_dispatch_mode = resolve_backward_dispatch_mode(method_name, dispatch_mode, self.rank_infos)
                call_id = f"{method_name}_{next(self._grad_call_counter)}"
                input_metas = collect_leaves(args, TensorMeta) + collect_leaves(tuple(kwargs.values()), TensorMeta)

            batch_size = infer_and_validate_batch_size(args, kwargs)
            _check_batch_divisibility(dispatch_mode, batch_size, self.dp_size)

            shards = dispatch_fn(self, args, kwargs, batch_size)
            shards = self._ensure_local(shards)
            # grad_mode/call_id passed as dedicated args, not mixed into kwargs
            refs = execute_fn(method_name, shards, grad_mode=ctx is not None, call_id=call_id)
            results = ray.get(refs)

            # Rebind before collect: results[i] comes from workers[i],
            # so worker attribution is unambiguous at this point.
            # Also wraps bare TensorHandle into TensorMeta.
            results = [self._rebind_tree(r, self.workers[i]) for i, r in enumerate(results)]

            # Collect: merge DP-head rank results
            collected = collect_fn(self, results)

            if ctx is not None:
                output_metas = collect_leaves(collected, TensorMeta)
                ctx.nodes.append(
                    RPCBackwardNode(
                        role_proxy=self,
                        call_id=call_id,
                        dispatch_mode=bwd_dispatch_mode,
                        input_metas=input_metas,
                        output_metas=output_metas,
                    )
                )

            return collected

        handle_fn.__name__ = method_name
        handle_fn.__doc__ = f"SPMD handle: {method_name} (dispatch={dispatch_fn.__name__})"
        return handle_fn

    # ── Execute strategies ──

    def _execute_all(self, method_name: str, shards: List, grad_mode: bool = False, call_id=None) -> List:
        """Send RPC to all Workers."""
        return [
            w.call.remote(self.role_name, method_name, s_args, s_kwargs, grad_mode, call_id)
            for w, (s_args, s_kwargs) in zip(self.workers, shards)
        ]

    def _execute_rank_zero(self, method_name: str, shards: List, grad_mode: bool = False, call_id=None) -> List:
        """Send RPC to rank 0 only."""
        return [
            self.workers[0].call.remote(self.role_name, method_name, shards[0][0], shards[0][1], grad_mode, call_id)
        ]

    # ── TensorHandle rebinding ──

    def _rebind_tree(self, obj, worker_handle):
        """Recursively rebind TensorHandle and wrap into TensorMeta.

        Returns the transformed tree where every TensorHandle is:
          1. Rebound with the given worker_handle (registers GC finalizer)
          2. Wrapped into a single-handle TensorMeta
        """
        if isinstance(obj, TensorHandle):
            obj.rebind(worker_handle)
            return TensorMeta.from_handles([obj])
        elif isinstance(obj, TensorMeta):
            for h in obj.refs:
                h.rebind(worker_handle)
            return obj
        elif isinstance(obj, Batch):
            return obj.map(lambda v: self._rebind_tree(v, worker_handle))
        elif isinstance(obj, tuple):
            return tuple(self._rebind_tree(item, worker_handle) for item in obj)
        elif isinstance(obj, list):
            return [self._rebind_tree(item, worker_handle) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._rebind_tree(v, worker_handle) for k, v in obj.items()}
        return obj

    # ── Prepare shards for execution: unwrap TensorMeta + NCCL transfer ──

    def _ensure_local(self, shards: List) -> List:
        """Prepare shards for worker execution.

        Pass 1 (_unwrap): TensorMeta → TensorHandle (single) or TensorMeta
            (multi-handle kept as-is for later cat). Collect foreign handles.
        NCCL: batch-transfer all foreign handles to their dst workers.
        Pass 2 (_substitute): rebuild trees replacing routing handles with
            NCCL recv handles.
        Pass 3 (_cat_multi): for any multi-handle TensorMeta remaining in a
            shard, remote-cat their handles on the dst worker → single TensorHandle.
        """
        # foreign: (src_device_id, dst_device_id) → [TensorHandle, ...]
        foreign: Dict[Tuple[int, int], List] = defaultdict(list)

        unwrapped_shards = []
        for shard_idx, (s_args, s_kwargs) in enumerate(shards):
            dst_device_id = self.device_ids[shard_idx]
            dst_worker_id = self.worker_ids[shard_idx]
            new_args = self._unwrap(s_args, dst_worker_id, dst_device_id, foreign)
            new_kwargs = self._unwrap(s_kwargs, dst_worker_id, dst_device_id, foreign)
            unwrapped_shards.append((new_args, new_kwargs))

        if foreign:
            # Batch NCCL transfer for each (src, dst) pair
            group_keys = list(foreign.keys())
            all_send_refs = []
            all_recv_refs = []
            for src_id, dst_id in group_keys:
                handles = foreign[(src_id, dst_id)]
                src_slot = self.pool.slot_of(handles[0].worker_id)
                src_worker = (
                    self.pool.slot0_worker(src_id) if src_slot > 0 else self.pool.get_worker(handles[0].worker_id)
                )
                dst_worker = self.pool.slot0_worker(dst_id)
                all_send_refs.append(src_worker._nccl_send.remote(dst_id, handles))
                all_recv_refs.append(
                    dst_worker._nccl_recv.remote(src_id, [h.shape for h in handles], [h.dtype for h in handles])
                )

            ray.get(all_send_refs)
            recv_results = ray.get(all_recv_refs)

            # Build substitution map: id(routing_handle) → new_handle
            subs: Dict[int, TensorHandle] = {}
            for (src_id, dst_id), new_handles in zip(group_keys, recv_results):
                dst_worker = self.pool.slot0_worker(dst_id)
                old_handles = foreign[(src_id, dst_id)]
                for old_h, new_h in zip(old_handles, new_handles):
                    new_h.rebind(dst_worker)
                    subs[id(old_h)] = new_h

            # Pass 2: rebuild each shard substituting routing handles
            unwrapped_shards = [
                (self._substitute(args, subs), self._substitute(kwargs, subs)) for args, kwargs in unwrapped_shards
            ]

        # Pass 3: remote-cat any remaining multi-handle TensorMeta on dst worker
        final_shards = []
        for shard_idx, (s_args, s_kwargs) in enumerate(unwrapped_shards):
            dst_worker = self.workers[shard_idx]
            new_args = self._cat_multi(s_args, dst_worker)
            new_kwargs = self._cat_multi(s_kwargs, dst_worker)
            final_shards.append((new_args, new_kwargs))

        return final_shards

    def _cat_multi(self, obj, dst_worker):
        """Pass 3: remote-cat multi-handle TensorMeta → single TensorHandle.

        After NCCL, all handles in a TensorMeta shard are guaranteed to be on
        dst_worker. Cat them into one tensor on dst_worker and return the handle.
        Single-handle TensorMeta and plain TensorHandle are passed through.
        """
        if isinstance(obj, TensorMeta):
            if len(obj.refs) == 1:
                return obj.refs[0]
            new_h = ray.get(dst_worker.cat_tensors.remote(obj.refs))
            new_h.rebind(dst_worker)
            return new_h
        if isinstance(obj, TensorHandle):
            return obj
        if isinstance(obj, Batch):
            return obj.map(lambda v: self._cat_multi(v, dst_worker))
        if isinstance(obj, tuple):
            return tuple(self._cat_multi(item, dst_worker) for item in obj)
        if isinstance(obj, list):
            return [self._cat_multi(item, dst_worker) for item in obj]
        if isinstance(obj, dict):
            return {k: self._cat_multi(v, dst_worker) for k, v in obj.items()}
        return obj

    def _with_ipc(self, handle: TensorHandle) -> TensorHandle:
        """Return a copy of *handle* with its CUDA IPC handle populated.

        ``TensorStore.put()`` no longer exports IPC handles eagerly — that pinned
        every stored block in the caching allocator and leaked on the colocate hot
        path (LIN-361). When a foreign handle is about to be opened over IPC by a
        same-device sibling, fetch the handle from the owning worker on demand. A
        handle that already carries one, or a CPU (``object_ref``) handle, passes
        through unchanged.
        """
        if handle.ipc_handle is not None or handle.object_ref is not None:
            return handle
        ipc = ray.get(self.pool.get_worker(handle.worker_id).get_ipc_handle.remote(handle.store_key))
        return TensorHandle(
            handle.store_key,
            handle.worker_id,
            handle.shape,
            handle.dtype,
            handle.device,
            ipc_handle=ipc,
            stride=handle.stride,
            offset=handle.offset,
        )

    def _unwrap(self, obj, dst_worker_id: str, dst_device_id: int, foreign: dict):
        """Pass 1: unwrap TensorMeta → TensorHandle or multi-handle TensorMeta.

        Single-handle TensorMeta: unwrapped to its TensorHandle.
        Multi-handle TensorMeta: each handle processed individually; returned
            as TensorMeta(routing_handles) so _cat_multi can later cat on dst.
        Foreign handles get routing copies; local handles pass through.
        """
        if isinstance(obj, TensorMeta):
            if len(obj.refs) == 1:
                return self._unwrap(obj.refs[0], dst_worker_id, dst_device_id, foreign)
            routed = [self._unwrap(h, dst_worker_id, dst_device_id, foreign) for h in obj.refs]
            return TensorMeta.from_handles(routed)
        if isinstance(obj, TensorHandle):
            if obj.object_ref is not None:
                return obj
            if obj.worker_id != dst_worker_id:
                src_device_id = self.pool.device_id_of(obj.worker_id)
                if src_device_id == dst_device_id:
                    # Same physical GPU, different slot → the consumer opens this
                    # tensor over CUDA IPC. put() no longer exports eagerly (it
                    # leaked — LIN-361), so export the handle lazily now.
                    return self._with_ipc(obj)
                # Cross-device → NCCL. When the source lives on a sibling slot, the
                # slot0 sender opens it over IPC and needs the lazily-exported
                # handle; a slot0 source is read by its own worker via store.get.
                src = self._with_ipc(obj) if self.pool.slot_of(obj.worker_id) > 0 else obj
                routing = TensorHandle(
                    src.store_key,
                    src.worker_id,
                    src.shape,
                    src.dtype,
                    src.device,
                    ipc_handle=src.ipc_handle,
                    stride=src.stride,
                    offset=src.offset,
                )
                foreign[(src_device_id, dst_device_id)].append(routing)
                return routing
            return obj
        if isinstance(obj, Batch):
            return obj.map(lambda v: self._unwrap(v, dst_worker_id, dst_device_id, foreign))
        if isinstance(obj, tuple):
            return tuple(self._unwrap(item, dst_worker_id, dst_device_id, foreign) for item in obj)
        if isinstance(obj, list):
            return [self._unwrap(item, dst_worker_id, dst_device_id, foreign) for item in obj]
        if isinstance(obj, dict):
            return {k: self._unwrap(v, dst_worker_id, dst_device_id, foreign) for k, v in obj.items()}
        return obj

    def _substitute(self, obj, subs: dict):
        """Pass 2: rebuild tree replacing routing handles (by id) with new handles."""
        if isinstance(obj, TensorHandle):
            return subs.get(id(obj), obj)
        if isinstance(obj, TensorMeta):
            new_handles = [subs.get(id(h), h) for h in obj.refs]
            return TensorMeta.from_handles(new_handles)
        if isinstance(obj, Batch):
            return obj.map(lambda v: self._substitute(v, subs))
        if isinstance(obj, tuple):
            return tuple(self._substitute(item, subs) for item in obj)
        if isinstance(obj, list):
            return [self._substitute(item, subs) for item in obj]
        if isinstance(obj, dict):
            return {k: self._substitute(v, subs) for k, v in obj.items()}
        return obj
