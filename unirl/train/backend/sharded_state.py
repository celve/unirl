"""FSDP2-generic sharded-state helpers shared by every train backend.

These operate on any module whose params are ``DTensor``s over a device mesh —
they are agnostic to *how* the module was sharded (torch-native ``fully_shard``
or VeOmni's ``parallelize``), so both :class:`~unirl.train.backend.fsdp.FSDPBackend`
and :class:`~unirl.train.backend.veomni.VeOmniBackend` consume them verbatim via
their package ``state.py`` re-exports.  Engine-specific bits (grad clipping,
offload/onload) live in the per-package ``state.py`` next to the backend.

This module imports ``torch`` at module level and MUST stay out of the
``veomni`` package's import graph — it is imported only from inside ``backend.py``
(directly and via the package ``state.py`` re-export, itself reached only through
``backend.py``).  Same discipline as ``sharded_load.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]


# ------------------------------------------------------------------
# Model state dict (DCP, full-state-dict, rank-0 gather / rank-0 broadcast)
# ------------------------------------------------------------------


def gather_state_dict(model: nn.Module) -> StateDict:
    """Rank-0 DCP gather.  Returns full state on rank 0, empty on others."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    try:
        full = dict(get_model_state_dict(model, options=options))
    except TypeError:
        full = dict(get_model_state_dict(model))

    if _current_rank() != 0:
        return {}
    return _to_cpu_state_dict(full)


def load_model_state_dict(model: nn.Module, state_dict: StateDict, *, strict: bool = True) -> None:
    """Load a full state dict, broadcasting from rank 0 across ranks.

    ``strict=False`` loads a partial dict (adapter-only checkpoints, or the
    backend's post-parallelize weight load where injected adapter params are
    legitimately absent): keys absent from ``state_dict`` keep the model's
    current weights.
    """
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
        strict=strict,
    )
    try:
        set_model_state_dict(model, state_dict, options=options)
    except TypeError:
        set_model_state_dict(model, state_dict)


# ------------------------------------------------------------------
# Optimizer state dict (DCP) — used by FSDPBackend; VeOmni uses plain state_dict()
# ------------------------------------------------------------------


def gather_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Rank-0 DCP gather of optimizer state.  Full state on rank 0, empty on others.

    All ranks must call this (the gather is a collective).  Values are full
    (unsharded) CPU tensors keyed by parameter FQN — symmetric with
    :func:`gather_state_dict`.  ``optimizer.state_dict()`` is NOT a substitute:
    under FSDP2 its values are this rank's local DTensor shards.
    """
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    try:
        full = dict(get_optimizer_state_dict(model, optimizer, options=options))
    except TypeError:
        full = dict(get_optimizer_state_dict(model, optimizer))

    if _current_rank() != 0:
        return {}
    return full


def load_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer, state_dict: StateDict) -> None:
    """Load a full optimizer state dict, broadcasting from rank 0 across ranks.

    Pass the rank-0 dict from :func:`gather_optimizer_state_dict`; other ranks
    pass ``{}`` (their input is ignored — tensors broadcast from rank 0 and
    re-shard into each rank's local state).
    """
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    try:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=options)
    except TypeError:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: object) -> None:
    """Move every tensor in the optimizer state to ``device`` (the on/offload loop).

    ``device`` may be a ``torch.device`` or a string (``"cpu"`` for offload, the
    live device for onload).  Non-tensor state (step counts, etc.) is left alone.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# ------------------------------------------------------------------
# Adapter export filters
# ------------------------------------------------------------------


def lora_state_dict(
    model: nn.Module,
    full_sd: Optional[StateDict] = None,
) -> StateDict:
    """Adapter-only state for inference export.

    All ranks must call this (the DCP gather is a collective).  Returns
    the filtered dict on rank 0, empty dict on other ranks.
    """
    if full_sd is None:
        full_sd = gather_state_dict(model)
    if _current_rank() != 0:
        return {}
    return {k: v for k, v in full_sd.items() if _is_lora_key(k)}


def nft_state_dict(
    model: nn.Module,
    full_sd: Optional[StateDict] = None,
    shadow_adapter: str = "old",
) -> StateDict:
    """Export the shadow ('old') adapter state for DiffusionNFT checkpoint.

    All ranks must call this (the DCP gather is a collective).  Returns
    the filtered dict on rank 0, empty dict on other ranks.
    """
    if full_sd is None:
        full_sd = gather_state_dict(model)
    if _current_rank() != 0:
        return {}
    token = f".{shadow_adapter}."
    return {k: v for k, v in full_sd.items() if ("lora_A" in k or "lora_B" in k) and token in k}


# ------------------------------------------------------------------
# Tensor / module utilities
# ------------------------------------------------------------------


def local_view(tensor: Tensor) -> Tensor:
    """DTensor -> local shard.  Identity for non-DTensors."""
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    return tensor


def is_materialized(model: nn.Module) -> bool:
    return not any(p.is_meta for p in model.parameters())


def trainable_params(model: nn.Module) -> Iterator[Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def infer_device(model: nn.Module) -> torch.device:
    """First non-meta parameter's device, else current cuda, else cpu."""
    for param in model.parameters():
        if param.is_meta:
            continue
        return param.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _is_lora_key(key: str) -> bool:
    """True for default-adapter LoRA keys (excludes shadow/old adapter)."""
    return ("lora_A" in key or "lora_B" in key) and ".old." not in key


def _build_state_dict_options(**kwargs: object) -> object:
    """Construct ``StateDictOptions`` degrading gracefully on older torch.

    The fallback ladder drops the newest kwargs first (``strict``, then
    ``broadcast_from_rank0``) so a torch whose ``StateDictOptions`` predates them
    still constructs.  On supported torch the first rung wins (full fidelity).
    """
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


def _maybe_dtensor_to_tensor(value: object) -> object:
    if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
        try:
            return value.full_tensor()
        except Exception:
            return value
    return value


def _to_cpu_state_dict(state_dict: StateDict) -> StateDict:
    converted: StateDict = {}
    for key, value in state_dict.items():
        tensor_or_obj = _maybe_dtensor_to_tensor(value)
        if isinstance(tensor_or_obj, torch.Tensor):
            converted[key] = tensor_or_obj.detach().cpu()
        else:
            converted[key] = tensor_or_obj
    return converted


__all__ = [
    "StateDict",
    "gather_state_dict",
    "load_model_state_dict",
    "gather_optimizer_state_dict",
    "load_optimizer_state_dict",
    "move_optimizer_state",
    "lora_state_dict",
    "nft_state_dict",
    "local_view",
    "is_materialized",
    "trainable_params",
    "infer_device",
]
