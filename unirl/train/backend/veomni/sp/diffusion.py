"""Ulysses SP for diffusers transformers — universal kernel-dispatch patch.

The trick that avoids per-model attention reimplementation: diffusers routes
every transformer's attention through a module-level ``dispatch_attention_fn``.
We patch it to wrap the attention KERNEL with the Ulysses all-to-all
(gather seq / scatter heads -> attention -> scatter seq / gather heads). This is
model-AGNOSTIC — the model's own processor still does projection / QK-norm / RoPE
/ stream-concat on the sliced streams, and full (non-causal) attention is
order-invariant over the gathered key set, so the joint all-to-all is correct as
long as RoPE was applied before the kernel (it always is).

So the only per-model code is a small **boundary spec**: slice the
sequence-carrying inputs + the RoPE freqs at entry, gather the hidden states at
exit. A few lines per architecture (see :data:`FORWARD_WRAPPERS`).

Built on VeOmni primitives + the folded mesh (no sp_size grad compensation;
docs/usp-derisk/sp_fsdp.py). v1 requires each stream's sequence divisible by
sp_size (no padding — full attention can't tolerate unmasked padding).
"""

from __future__ import annotations

import functools
import inspect
import logging
import sys
from typing import Any, Callable, Dict

from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _sp():
    """Lazy VeOmni handles (after the selective-import shim is installed)."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import slice_input_tensor
    from veomni.distributed.sequence_parallel.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    return get_parallel_state, slice_input_tensor, gather_seq_scatter_heads, gather_heads_scatter_seq


# ---------------------------------------------------------------------------
# 1. Universal: Ulysses all-to-all around the attention kernel
# ---------------------------------------------------------------------------

def _make_ulysses_dispatch(orig_dispatch: Callable) -> Callable:
    get_parallel_state, _, gather_seq_scatter_heads, gather_heads_scatter_seq = _sp()

    @functools.wraps(orig_dispatch)
    def ulysses_dispatch(query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any):
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return orig_dispatch(query, key, value, *args, **kwargs)
        attn_mask = kwargs.get("attn_mask", args[0] if args else None)
        if attn_mask is not None:
            raise NotImplementedError(
                "diffusion SP v1 does not support an attention mask under Ulysses "
                "(full-attention padding needs permutation-aware masking); ensure each "
                "stream length is divisible by sp_size so no mask is built."
            )
        g = ps.sp_group
        # q/k/v: (B, S_joint_local, H, D_h) -> gather seq, scatter heads -> full joint, local heads
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2, group=g)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2, group=g)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2, group=g)
        out = orig_dispatch(query, key, value, *args, **kwargs)
        # (B, S_joint_full, H/sp, D_h) -> scatter seq, gather heads -> local joint, full heads
        return gather_heads_scatter_seq(out, head_dim=2, seq_dim=1, group=g)

    ulysses_dispatch._unirl_sp_patched = True
    return ulysses_dispatch


def _patch_attention_dispatch(model: nn.Module) -> None:
    """Patch ``dispatch_attention_fn`` in the model's defining module (idempotent).

    Diffusers processors call the ``dispatch_attention_fn`` bound in their own
    module's namespace, so we patch it there.
    """
    module = sys.modules.get(type(model).__module__)
    if module is None or not hasattr(module, "dispatch_attention_fn"):
        raise RuntimeError(
            f"diffusion SP: {type(model).__module__} has no dispatch_attention_fn to patch "
            "(unexpected diffusers layout)."
        )
    if getattr(module.dispatch_attention_fn, "_unirl_sp_patched", False):
        return
    module.dispatch_attention_fn = _make_ulysses_dispatch(module.dispatch_attention_fn)
    logger.info("diffusion SP: patched %s.dispatch_attention_fn with Ulysses all-to-all", module.__name__)


# ---------------------------------------------------------------------------
# 2. Per-model boundary hooks (slice inputs + RoPE in, gather hidden out)
# ---------------------------------------------------------------------------

def _wrap_qwen_image(model: nn.Module, sp_group: Any) -> None:
    """qwen-image boundary: slice [image, text] streams + per-stream RoPE in,
    gather the image stream out. temb is global (per-sample) so it is untouched.
    """
    get_parallel_state, slice_input_tensor, _, _ = _sp()
    from veomni.distributed.sequence_parallel import gather_outputs

    state: Dict[str, int] = {}

    def pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        hs = kwargs.get("hidden_states")
        if hs is not None:
            state["img_len"] = hs.shape[1]
            kwargs["hidden_states"] = slice_input_tensor(hs, dim=1, group=sp_group)
        ehs = kwargs.get("encoder_hidden_states")
        if ehs is not None:
            kwargs["encoder_hidden_states"] = slice_input_tensor(ehs, dim=1, group=sp_group)
        return args, kwargs

    def rope_hook(_m, _inp, out):
        if not get_parallel_state().ulysses_enabled:
            return out
        vid, txt = out  # (S_img, fd), (S_txt, fd) complex
        return (slice_input_tensor(vid, dim=0, group=sp_group), slice_input_tensor(txt, dim=0, group=sp_group))

    def norm_out_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        hs = args[0]
        hs = gather_outputs(hs, gather_dim=1, group=sp_group)
        true_len = state.get("img_len")
        if true_len is not None and hs.shape[1] > true_len:
            hs = hs[:, :true_len]
        return (hs, *args[1:]), kwargs

    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.pos_embed.register_forward_hook(rope_hook)
    model.norm_out.register_forward_pre_hook(norm_out_pre, with_kwargs=True)
    logger.info("diffusion SP: qwen-image boundary hooks installed (slice img/text + RoPE, gather at norm_out)")


# class name -> boundary-hook installer
FORWARD_WRAPPERS: Dict[str, Callable[[nn.Module, Any], None]] = {
    "QwenImageTransformer2DModel": _wrap_qwen_image,
}


def apply_diffusion_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Patch the attention dispatch (universal) + install the per-model boundary hooks."""
    get_parallel_state, *_ = _sp()
    sp_group = get_parallel_state().sp_group

    _patch_attention_dispatch(model)

    cls_name = type(model).__name__
    wrapper = FORWARD_WRAPPERS.get(cls_name)
    if wrapper is None:
        raise NotImplementedError(
            f"diffusion SP: attention dispatch patched, but no boundary spec for {cls_name} "
            f"(slice inputs/RoPE in, gather out). Add one to FORWARD_WRAPPERS in "
            f"unirl.train.backend.veomni.sp.diffusion."
        )
    wrapper(model, sp_group)
    logger.info("diffusion SP installed for %s (sp_size=%d)", cls_name, sp_size)


def is_diffusers_transformer(model: nn.Module) -> bool:
    """Has at least one diffusers ``Attention`` module (set_processor interface)."""
    return any(hasattr(m, "set_processor") and hasattr(m, "get_processor") for m in model.modules())


__all__ = [
    "apply_diffusion_sequence_parallelism",
    "is_diffusers_transformer",
    "FORWARD_WRAPPERS",
]
