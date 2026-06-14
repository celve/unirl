"""Sharded-state helpers for the VeOmni backend.

Same public surface as ``unirl.train.backend.fsdp.state`` so ``backend.py``
reads identically across the two packages.  The DCP state-dict helpers are
copied verbatim (they are FSDP2-generic — they operate on any module whose
params are DTensors); grad clipping and offload/onload delegate to VeOmni's
implementations via the :mod:`._compat` shim, which under EP (Phase 2) are
the variants that understand VeOmni's extra-parallel placements.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]


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

    ``strict=False`` is used by the backend's post-parallelize weight load,
    where injected adapter params (LoRA) are legitimately absent from the
    base checkpoint.
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


def trainable_params(model: nn.Module) -> Iterator[Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def clip_grad_norm(model: nn.Module, max_norm: float) -> Tensor:
    """Gradient clipping via VeOmni's FSDP2 clip (EP-aware under Phase 2).

    Takes the *model* (not a param list) — VeOmni's clip dispatches on
    model attributes (``_extra_parallel_param_groups``, CPU-offload flags)
    that a bare param list cannot carry.
    """
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.fsdp2 import clip_grad_norm as _veomni_clip_grad_norm

    result = _veomni_clip_grad_norm(model, max_norm)
    return _maybe_dtensor_to_tensor(result)


def veomni_offload(model: nn.Module) -> None:
    """Move the parallelized model to CPU via VeOmni (reshards the root first).

    VeOmni's offload calls ``model.cpu()``, which cannot handle meta tensors —
    v1 supports fully-materialized trainables only (the qwen-image pilot is;
    aux components like VAE live on the bundle, outside this module)."""
    meta_names = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_names:
        raise RuntimeError(
            f"veomni_offload: {len(meta_names)} params still on meta "
            f"(e.g. {meta_names[:4]}); VeOmniBackend v1 requires a fully-"
            "materialized trainable module."
        )
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.offloading import offload_model_to_cpu

    offload_model_to_cpu(model)
    logger.debug("veomni_offload: offloaded params/grads to CPU")


def veomni_onload(model: nn.Module, device: torch.device) -> None:
    """Move the parallelized model back to ``device`` via VeOmni."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.offloading import load_model_to_gpu

    load_model_to_gpu(model, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.debug("veomni_onload: onloaded params/grads to %s", device)


# ------------------------------------------------------------------
# Internal helpers (verbatim from unirl.train.backend.fsdp.state)
# ------------------------------------------------------------------


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
    "clip_grad_norm",
    "gather_state_dict",
    "load_model_state_dict",
    "trainable_params",
    "veomni_offload",
    "veomni_onload",
]
