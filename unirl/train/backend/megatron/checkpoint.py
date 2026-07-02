"""Native mcore dist_checkpointing behind UniRL's save/load seam (M0).

The trainer touches only ``backend.save(path, step, mode)`` and
``backend.load(path) -> int`` (both ``@distributed(BROADCAST)``); this module
supplies the mcore-native bodies. mcore's ``save_checkpoint`` consumes
``model.sharded_state_dict()`` (NOT a DTensor dict) and writes ``iter_NNNNNNN``
dirs, so there is no drop-in at the ``dcp.save`` call site — the divergence is at
the envelope. ``sharded_state.py`` / ``sharded_load.py`` stay FSDP-only, untouched.

Ported from slime ``checkpoint.py`` (native save/load + the ShardedTensor
validation-bypass monkeypatch + the ``iter_\\d{7}`` marker). The rank-0
``metadata.pt`` sidecar carries only ``{step, save_mode}`` — mcore's
``OptimizerParamScheduler`` persists itself inside its own checkpoint, so we do NOT
double-store scheduler state. ``load`` returns the ROLLOUT step (not mcore's
internal iteration — different axis).

VERIFY every mcore symbol against the pinned version; M0 is sync-only.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional

import torch


def apply_sharded_tensor_validation_bypass() -> None:
    """Skip torch's slow cross-rank shard-overlap validation on load (slime's
    monkeypatch). Guarded — a no-op if the internal API moved in the pinned torch.
    """
    try:
        from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
        from torch.distributed._shard.sharded_tensor import ShardedTensor  # noqa: F401
    except ImportError:
        return

    def _post_init(self):  # bypass the O(shards^2) overlap check
        pass

    # VERIFY: slime also replaces ShardedTensor._init_from_local_shards_and_global_metadata;
    # port that classmethod too if the pinned torch still validates there.
    EnumerableShardingSpec.__post_init__ = _post_init  # type: ignore[method-assign]


def is_megatron_checkpoint(path: str) -> bool:
    """A dir is an mcore checkpoint if it has ``latest_checkpointed_iteration.txt``
    or is itself an ``iter_NNNNNNN`` dir (slime ``_is_megatron_checkpoint``).
    """
    return os.path.isfile(os.path.join(path, "latest_checkpointed_iteration.txt")) or bool(
        re.fullmatch(r"iter_\d{7}", os.path.basename(os.path.normpath(path)))
    )


def disable_forward_pre_hook(model_chunks: List[Any], *, param_sync: bool = True) -> None:
    for chunk in model_chunks:
        chunk.disable_forward_pre_hook(param_sync=param_sync)  # VERIFY: mcore DDP API


def enable_forward_pre_hook(model_chunks: List[Any]) -> None:
    for chunk in model_chunks:
        chunk.enable_forward_pre_hook()  # VERIFY: mcore DDP API


def save_native(path: str, step: Optional[int], save_mode: str, *, model_chunks: List[Any],
                optimizer: Any, scheduler: Any, megatron_args: Any, rank: int) -> None:
    """Set ``args.save=path`` on every rank, save, restore the forward-pre-hook,
    write the rank-0 metadata sidecar.

    The forward-pre-hook (dist-optimizer overlapped param-gather) MUST be re-enabled
    in a ``finally`` — if it stays disabled the next forward runs on un-gathered
    params -> wrong log-probs, no crash. Single place, no nesting.
    """
    os.makedirs(path, exist_ok=True)
    megatron_args.save = path  # mcore save_checkpoint reads get_args().save

    from megatron.training.checkpointing import save_checkpoint  # VERIFY import path

    disable_forward_pre_hook(model_chunks)
    try:
        # VERIFY the exact save_checkpoint signature (iteration, model, optimizer,
        # opt_param_scheduler, num_floating_point_operations_so_far, ...).
        save_checkpoint(step or 0, model_chunks, optimizer, scheduler, 0)
    finally:
        enable_forward_pre_hook(model_chunks)

    if rank == 0:
        # Sidecar at path/ (mcore payload sits at path/iter_NNNNNNN/). step is the
        # ROLLOUT axis; scheduler state is deliberately NOT stored (mcore owns it).
        torch.save({"step": step, "save_mode": save_mode}, os.path.join(path, "metadata.pt"))


def load_native(path: str, *, model_chunks: List[Any], optimizer: Any, scheduler: Any,
                megatron_args: Any) -> int:
    """Set ``args.load=path``, load, return the ROLLOUT step from the sidecar
    (0 if absent) — never mcore's internal iteration.
    """
    megatron_args.load = path

    from megatron.training.checkpointing import load_checkpoint  # VERIFY import path

    # VERIFY the load_checkpoint signature/return against the pin.
    load_checkpoint(model_chunks, optimizer, scheduler)

    meta_path = os.path.join(path, "metadata.pt")
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        return int(meta.get("step") or 0)
    return 0
