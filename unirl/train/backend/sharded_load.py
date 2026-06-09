"""Shared sharded-weight loader for the FSDP and VeOmni backends.

Both backends materialize a meta-built, FSDP2-sharded trainable module and
broadcast real weights into it from rank 0.  The mechanics are identical — the
only backend difference is *timing*: VeOmni's ``parallelize`` already
``to_empty``-materializes the module before this runs, while FSDP's wrap leaves
it on meta.  :func:`_load_state_dict_sharded`'s meta-gate absorbs that
difference (it ``to_empty``s only if params are still on meta), so the same
loader is correct on both.

This module imports ``torch`` / ``safetensors`` at module level and MUST stay
out of the ``veomni`` package's import graph — it is imported only from inside
``backend.py`` — so the selective-import audit (``tests/test_compat_import.py``)
and the torch-free package-import check (``tests/test_recipe_compose.py``) stay
green.
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import torch
from torch import nn

StateDict = Dict[str, object]


def load_sharded(
    module: nn.Module,
    weights_dir: str,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Materialize ``module`` from a (diffusers-layout) safetensors directory.

    Rank 0 reads every ``*.safetensors`` shard under ``weights_dir``; the
    weights are broadcast into the sharded module.  See
    :func:`_load_state_dict_sharded` for the per-rank mechanics.  This is the
    common path for single-module trainables whose weights live in a dedicated
    directory (diffusion ``<ckpt>/transformer``, AR ``<ckpt>`` root).
    """
    state_dict = _read_safetensors_dir(weights_dir) if _current_rank() == 0 else {}
    _load_state_dict_sharded(module, state_dict, device=device, strict=strict)


def _load_state_dict_sharded(
    module: nn.Module,
    state_dict: StateDict,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Allocate storage for any meta params, then broadcast-load ``state_dict``.

    ``state_dict`` is the rank-0 full state dict (empty ``{}`` on other ranks).
    Steps:

    1. ``to_empty(device)`` any submodule still on meta — gated, so it is a
       no-op when the wrap already materialized the module (VeOmni's
       ``parallelize``) and the allocator that the FSDP-meta path needs when it
       did not.
    2. rank 0: insert the ``base_layer`` hop for LoRA-injected modules.
    3. ``set_model_state_dict(..., broadcast_from_rank0=True, strict=strict)``
       — DTensor-aware; handles FSDP2 shards + plain params in one collective.
    """
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    if _module_has_meta_param(module):
        module.to_empty(device=device)

    if _current_rank() == 0:
        state_dict = _remap_lora_base_keys(state_dict, module)

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
        strict=strict,
    )
    try:
        set_model_state_dict(module, state_dict, options=options)
    except TypeError:
        set_model_state_dict(module, state_dict)


def _module_has_meta_param(module: nn.Module) -> bool:
    """True if any parameter of ``module`` (recursing into children) is on the
    meta device.  Used to gate the per-shard ``to_empty`` call."""
    return any(p.is_meta for p in module.parameters(recurse=True))


def _read_safetensors_dir(weights_dir: str) -> StateDict:
    """Merge all ``*.safetensors`` shards in a directory.

    Loading every shard makes the index json unnecessary and covers both
    single-file and sharded checkpoints."""
    from safetensors.torch import load_file

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"sharded_load: transformer weights dir not found: {weights_dir!r}. "
            "HF repo IDs are not supported here — point the recipe's checkpoint "
            "path at a local download."
        )
    shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"sharded_load: no *.safetensors files under {weights_dir!r}")
    state_dict: StateDict = {}
    for shard in shards:
        state_dict.update(load_file(shard, device="cpu"))
    return state_dict


def _remap_lora_base_keys(state_dict: StateDict, model: nn.Module) -> StateDict:
    """Translate base-checkpoint keys for LoRA-injected modules.

    ``peft.inject_adapter_in_model`` (via ``unirl.train.lora`` /
    ``unirl.train.ema``) rewires target Linears in place, so their original
    weight moves to ``<module>.base_layer.weight``.  The base checkpoint still
    uses the original key — insert the ``base_layer`` hop where (and only
    where) the model expects it."""
    model_keys = {n for n, _ in model.named_parameters()}
    model_keys.update(n for n, _ in model.named_buffers())
    remapped: StateDict = {}
    for key, value in state_dict.items():
        if key not in model_keys:
            stem, _, leaf = key.rpartition(".")
            candidate = f"{stem}.base_layer.{leaf}" if stem else key
            if candidate in model_keys:
                remapped[candidate] = value
                continue
        remapped[key] = value
    return remapped


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _build_state_dict_options(**kwargs: object) -> object:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    candidates = [
        dict(kwargs),
        {k: v for k, v in kwargs.items() if k != "strict"},
        {k: v for k, v in kwargs.items() if k not in {"strict", "broadcast_from_rank0"}},
        {k: v for k, v in kwargs.items() if k in {"full_state_dict", "cpu_offload"}},
        {},
    ]
    for candidate in candidates:
        try:
            return StateDictOptions(**candidate)
        except TypeError:
            continue
    return StateDictOptions()


__all__ = ["load_sharded", "_load_state_dict_sharded", "StateDict"]
