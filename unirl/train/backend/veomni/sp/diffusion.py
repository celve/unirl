"""Ulysses SP for diffusers transformers.

Two attention-SP mechanisms (auto-detected per model), plus a small per-model
boundary spec (slice the sequence-carrying inputs + RoPE in, gather hidden out):

* **dispatch-patch** (newer models that route attention through
  ``dispatch_attention_fn`` — qwen-image / flux2 / wan): wrap that kernel call
  with the Ulysses all-to-all (gather seq / scatter heads -> attention ->
  scatter seq / gather heads). Model-agnostic — the model's own processor still
  does projection / QK-norm / RoPE / stream-concat on the sliced streams, and
  full (non-causal) attention is order-invariant over the gathered set, so the
  joint all-to-all is correct (RoPE is applied before the kernel). Cross-attention
  (Wan text branch: sliced image query vs full text K/V) is detected by unequal
  q/k seq length and skipped.

* **processor-injection** (older models whose processor calls SDPA directly,
  e.g. SD3's ``JointAttnProcessor2_0``): replace the attention processor with
  :class:`SPAttentionProcessor` (port of mmrl), which does the per-stream
  all-to-all itself.

Built on VeOmni primitives + the folded mesh (no sp_size grad compensation;
docs/usp-derisk/sp_fsdp.py). v1 requires each stream's sequence divisible by
sp_size (no padding — full attention can't tolerate unmasked padding).
Validated: qwen-image (dispatch) relerr 2e-7.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _sp():
    """Lazy VeOmni handles (after the selective-import shim is installed)."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
    from veomni.distributed.sequence_parallel.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    return (get_parallel_state, slice_input_tensor, gather_outputs,
            gather_seq_scatter_heads, gather_heads_scatter_seq)


# ---------------------------------------------------------------------------
# Mechanism 1: Ulysses all-to-all around dispatch_attention_fn (newer models)
# ---------------------------------------------------------------------------

def _make_ulysses_dispatch(orig_dispatch: Callable) -> Callable:
    import functools

    get_parallel_state, _, _, gather_seq_scatter_heads, gather_heads_scatter_seq = _sp()

    @functools.wraps(orig_dispatch)
    def ulysses_dispatch(query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any):
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return orig_dispatch(query, key, value, *args, **kwargs)
        # Cross-attention (e.g. Wan text branch): sliced image query vs full text
        # K/V -> different seq lengths; each rank attends locally, no all-to-all.
        if query.shape[1] != key.shape[1]:
            return orig_dispatch(query, key, value, *args, **kwargs)
        attn_mask = kwargs.get("attn_mask", args[0] if args else None)
        if attn_mask is not None:
            raise NotImplementedError(
                "diffusion SP v1 does not support an attention mask under Ulysses; "
                "ensure each stream length is divisible by sp_size so no mask is built."
            )
        g = ps.sp_group
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2, group=g)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2, group=g)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2, group=g)
        out = orig_dispatch(query, key, value, *args, **kwargs)
        return gather_heads_scatter_seq(out, head_dim=2, seq_dim=1, group=g)

    ulysses_dispatch._unirl_sp_patched = True
    return ulysses_dispatch


def _patch_attention_dispatch(model: nn.Module) -> bool:
    """Patch ``dispatch_attention_fn`` in the model's module. Returns True if patched."""
    module = sys.modules.get(type(model).__module__)
    if module is None or not hasattr(module, "dispatch_attention_fn"):
        return False
    if not getattr(module.dispatch_attention_fn, "_unirl_sp_patched", False):
        module.dispatch_attention_fn = _make_ulysses_dispatch(module.dispatch_attention_fn)
        logger.info("diffusion SP: patched %s.dispatch_attention_fn with Ulysses all-to-all", module.__name__)
    return True


# ---------------------------------------------------------------------------
# Mechanism 2: SP attention processor (older models, e.g. SD3 JointAttnProcessor2_0)
# ---------------------------------------------------------------------------

def apply_rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    """RoPE for diffusers Q/K, ``(B, S, H, D_h)``. Flux: cos/sin ``(S, D_h)`` (2D);
    Wan: ``(1, S, 1, D_h)`` (4D interleaved). Ported from mmrl."""
    cos, sin = freqs_cis
    cos, sin = cos.to(x.device), sin.to(x.device)
    if cos.ndim == 2:
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    x1, x2 = x.to(torch.float64).unflatten(-1, (-1, 2)).unbind(-1)
    fc, fs = cos[..., 0::2].to(torch.float64), sin[..., 1::2].to(torch.float64)
    out = torch.empty_like(x, dtype=torch.float64)
    out[..., 0::2] = x1 * fc - x2 * fs
    out[..., 1::2] = x1 * fs + x2 * fc
    return out.to(x.dtype)


def _sdpa(q: Tensor, k: Tensor, v: Tensor, scale: float, attn_mask: Tensor | None) -> Tensor:
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, scale=scale
    )
    return o.transpose(1, 2)


class SPAttentionProcessor:
    """SP-aware diffusers attention processor (Ulysses), port of mmrl.

    dual-stream joint (``add_q_proj``): all-to-all each [encoder, image] stream,
    re-cat in the original processor's order, attend, inverse; cross-attention:
    skip all-to-all; self-attention: full all-to-all.
    """

    def __init__(self, sp_group: Any, original_processor: Any = None):
        self.sp_group = sp_group
        self.original_processor = original_processor
        _, _, _, gsh, ghs = _sp()
        self._gather_seq, self._gather_heads = gsh, ghs

    def _a2a_sh(self, x):  # scatter heads, gather seq
        return self._gather_seq(x, seq_dim=1, head_dim=2, group=self.sp_group)

    def _a2a_hs(self, x):  # scatter seq, gather heads
        return self._gather_heads(x, head_dim=2, seq_dim=1, group=self.sp_group)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None, rotary_emb=None, **kwargs):
        if image_rotary_emb is None and rotary_emb is not None:
            image_rotary_emb = rotary_emb
        is_cross = encoder_hidden_states is not None
        has_added_kv = hasattr(attn, "add_q_proj") and attn.add_q_proj is not None and is_cross
        b = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        kv_in = hidden_states if (has_added_kv or not is_cross) else encoder_hidden_states
        key, value = attn.to_k(kv_in), attn.to_v(kv_in)
        inner = query.shape[-1]
        hd = inner // attn.heads
        nq = getattr(attn, "norm_q", None) is not None
        nk = getattr(attn, "norm_k", None) is not None
        pre = bool(nq and hasattr(attn.norm_q, "normalized_shape") and attn.norm_q.normalized_shape == (inner,))
        if pre:
            if nq: query = attn.norm_q(query)
            if nk: key = attn.norm_k(key)
        query = query.view(b, -1, attn.heads, hd)
        key = key.view(b, -1, attn.heads, hd)
        value = value.view(b, -1, attn.heads, hd)
        if not pre:
            if nq: query = attn.norm_q(query)
            if nk: key = attn.norm_k(key)

        if has_added_kv:
            eq = attn.add_q_proj(encoder_hidden_states).view(b, -1, attn.heads, hd)
            ek = attn.add_k_proj(encoder_hidden_states).view(b, -1, attn.heads, hd)
            ev = attn.add_v_proj(encoder_hidden_states).view(b, -1, attn.heads, hd)
            if getattr(attn, "norm_added_q", None) is not None: eq = attn.norm_added_q(eq)
            if getattr(attn, "norm_added_k", None) is not None: ek = attn.norm_added_k(ek)
            query, key, value = torch.cat([eq, query], 1), torch.cat([ek, key], 1), torch.cat([ev, value], 1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        scale = attn.scale if hasattr(attn, "scale") else hd ** -0.5

        if has_added_kv:
            encoder_first = type(self.original_processor).__name__ != "JointAttnProcessor2_0"
            el = encoder_hidden_states.shape[1]
            eq, iq = query[:, :el], query[:, el:]
            ek, ik = key[:, :el], key[:, el:]
            ev, iv = value[:, :el], value[:, el:]
            eq, ek, ev = self._a2a_sh(eq), self._a2a_sh(ek), self._a2a_sh(ev)
            iq, ik, iv = self._a2a_sh(iq), self._a2a_sh(ik), self._a2a_sh(iv)
            if encoder_first:
                q, k, v = torch.cat([eq, iq], 1), torch.cat([ek, ik], 1), torch.cat([ev, iv], 1)
            else:
                q, k, v = torch.cat([iq, eq], 1), torch.cat([ik, ek], 1), torch.cat([iv, ev], 1)
            out = _sdpa(q, k, v, scale, attention_mask)
            fe, fi = eq.shape[1], iq.shape[1]
            enc_out, img_out = (out[:, :fe], out[:, fe:]) if encoder_first else (out[:, fi:], out[:, :fi])
            enc_out, img_out = self._a2a_hs(enc_out), self._a2a_hs(img_out)
            hs = attn.to_out[0](img_out.reshape(b, -1, inner))
            if len(attn.to_out) > 1: hs = attn.to_out[1](hs)
            enc = enc_out.reshape(b, -1, inner)
            if not getattr(attn, "context_pre_only", False): enc = attn.to_add_out(enc)
            return hs, enc

        if is_cross:
            out = _sdpa(query, key, value, scale, attention_mask)
        else:
            query, key, value = self._a2a_sh(query), self._a2a_sh(key), self._a2a_sh(value)
            out = self._a2a_hs(_sdpa(query, key, value, scale, attention_mask))
        hs = out.reshape(b, -1, inner)
        if getattr(attn, "to_out", None) is not None:
            hs = attn.to_out[0](hs)
            if len(attn.to_out) > 1: hs = attn.to_out[1](hs)
        return hs


def inject_sp_processors(model: nn.Module, sp_group: Any) -> int:
    import torch.distributed as dist

    sp_size = dist.get_world_size(sp_group)
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_processor") and hasattr(module, "get_processor"):
            heads = getattr(module, "heads", None)
            if heads is not None and heads % sp_size != 0:
                raise ValueError(f"SP requires num_heads % sp_size == 0, but '{name}' has {heads} heads, sp_size={sp_size}.")
            module.set_processor(SPAttentionProcessor(sp_group, original_processor=module.get_processor()))
            count += 1
    logger.info("diffusion SP: injected SPAttentionProcessor into %d attention modules", count)
    return count


# ---------------------------------------------------------------------------
# Per-model boundary hooks: slice streams + RoPE in, gather hidden out
# ---------------------------------------------------------------------------

def _install_boundary_hooks(model, sp_group, blocks_attr, norm_out_attr, rope_hook=None,
                            rope_attr="pos_embed", slice_encoder=True):
    """Slice the image (+ optionally text) stream at the first block, gather the
    image stream at the output norm. ``slice_encoder=False`` keeps the text full
    (Wan cross-attention). Handles kwargs (qwen/sd3) and positional (wan) calls.
    """
    get_parallel_state, slice_input_tensor, gather_outputs, _, _ = _sp()
    state: Dict[str, int] = {}

    def block0_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        new_args = list(args)
        if "hidden_states" in kwargs:
            state["img_len"] = kwargs["hidden_states"].shape[1]
            kwargs["hidden_states"] = slice_input_tensor(kwargs["hidden_states"], dim=1, group=sp_group)
        elif new_args:
            state["img_len"] = new_args[0].shape[1]
            new_args[0] = slice_input_tensor(new_args[0], dim=1, group=sp_group)
        if slice_encoder:
            if "encoder_hidden_states" in kwargs:
                kwargs["encoder_hidden_states"] = slice_input_tensor(kwargs["encoder_hidden_states"], dim=1, group=sp_group)
            elif len(new_args) >= 2:
                new_args[1] = slice_input_tensor(new_args[1], dim=1, group=sp_group)
        return tuple(new_args), kwargs

    def norm_out_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        hs = gather_outputs(args[0], gather_dim=1, group=sp_group)
        tl = state.get("img_len")
        if tl is not None and hs.shape[1] > tl:
            hs = hs[:, :tl]
        return (hs, *args[1:]), kwargs

    getattr(model, blocks_attr)[0].register_forward_pre_hook(block0_pre, with_kwargs=True)
    getattr(model, norm_out_attr).register_forward_pre_hook(norm_out_pre, with_kwargs=True)
    if rope_hook is not None:
        getattr(model, rope_attr).register_forward_hook(rope_hook)


def _make_qwen_rope_hook(sp_group):
    get_parallel_state, slice_input_tensor, *_ = _sp()

    def hook(_m, _inp, out):
        # pos_embed runs after text_seq_len is read full, so RoPE is full-length;
        # slice each stream's freqs (vid: image positions, txt: text positions).
        if not get_parallel_state().ulysses_enabled:
            return out
        vid, txt = out
        return (slice_input_tensor(vid, dim=0, group=sp_group), slice_input_tensor(txt, dim=0, group=sp_group))

    return hook


def _wrap_qwen_image(model, sp_group):
    _install_boundary_hooks(model, sp_group, "transformer_blocks", "norm_out",
                            rope_hook=_make_qwen_rope_hook(sp_group), rope_attr="pos_embed")
    logger.info("diffusion SP: qwen-image boundary hooks installed")


def _wrap_sd3(model, sp_group):
    # SD3 has learned positional embeddings baked into the patches (no RoPE).
    _install_boundary_hooks(model, sp_group, "transformer_blocks", "norm_out")
    logger.info("diffusion SP: sd3 boundary hooks installed")


def _make_wan_rope_hook(sp_group):
    get_parallel_state, slice_input_tensor, *_ = _sp()

    def hook(_m, _inp, out):
        # Wan rotary: (cos, sin), each (1, S_img, 1, D); slice the image seq dim.
        if not get_parallel_state().ulysses_enabled:
            return out
        if isinstance(out, (tuple, list)):
            return type(out)(slice_input_tensor(t, dim=1, group=sp_group) for t in out)
        return slice_input_tensor(out, dim=1, group=sp_group)

    return hook


def _wrap_wan(model, sp_group):
    # Wan: image self-attn (slice image) + text cross-attn (text stays FULL; the
    # dispatch cross-attn guard skips its all-to-all). Block call is positional.
    _install_boundary_hooks(model, sp_group, "blocks", "norm_out",
                            rope_hook=_make_wan_rope_hook(sp_group), rope_attr="rope",
                            slice_encoder=False)
    logger.info("diffusion SP: wan boundary hooks installed")


def _wrap_flux2(model, sp_group):
    # flux2: dual->single blocks + text-strip (hidden[:, num_txt_tokens:]). Because
    # num_txt_tokens == encoder_hidden_states.shape[1] and img_ids/txt_ids are
    # forward ARGS (not derived from the sliced tensors), we can slice both streams
    # at the MODEL input (the strip then removes the LOCAL text, leaving image-local)
    # and gather the image-only output at the MODEL exit. pos_embed is called per
    # stream (img_ids, txt_ids); slice each (cos, sin) so the in-forward cat is the
    # joint sliced RoPE.
    get_parallel_state, slice_input_tensor, gather_outputs, _, _ = _sp()

    def model_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        if kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = slice_input_tensor(kwargs["hidden_states"], dim=1, group=sp_group)
        if kwargs.get("encoder_hidden_states") is not None:
            kwargs["encoder_hidden_states"] = slice_input_tensor(kwargs["encoder_hidden_states"], dim=1, group=sp_group)
        return args, kwargs

    def rope_hook(_m, _inp, out):
        if not get_parallel_state().ulysses_enabled:
            return out
        if isinstance(out, (tuple, list)):
            return type(out)(slice_input_tensor(t, dim=0, group=sp_group) for t in out)
        return slice_input_tensor(out, dim=0, group=sp_group)

    def model_post(_m, _args, _kwargs, out):
        if not get_parallel_state().ulysses_enabled:
            return out
        sample = out.sample if hasattr(out, "sample") else out[0]
        sample = gather_outputs(sample, gather_dim=1, group=sp_group)
        if hasattr(out, "sample"):
            out.sample = sample
            return out
        return (sample, *out[1:])

    model.register_forward_pre_hook(model_pre, with_kwargs=True)
    model.pos_embed.register_forward_hook(rope_hook)
    model.register_forward_hook(model_post, with_kwargs=True)
    logger.info("diffusion SP: flux2 boundary hooks installed (model-level slice/gather)")


FORWARD_WRAPPERS: Dict[str, Callable[[nn.Module, Any], None]] = {
    "QwenImageTransformer2DModel": _wrap_qwen_image,
    "SD3Transformer2DModel": _wrap_sd3,
    "WanTransformer3DModel": _wrap_wan,
    "Flux2Transformer2DModel": _wrap_flux2,
}


def apply_diffusion_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Attention SP (dispatch-patch or processor-injection, auto-detected) + boundary hooks."""
    get_parallel_state, *_ = _sp()
    sp_group = get_parallel_state().sp_group

    if not _patch_attention_dispatch(model):       # newer models
        inject_sp_processors(model, sp_group)      # older models (SD3)

    cls = type(model).__name__
    wrapper = FORWARD_WRAPPERS.get(cls)
    if wrapper is None:
        raise NotImplementedError(
            f"diffusion SP: attention SP wired, but no boundary spec for {cls}. "
            f"Add one to FORWARD_WRAPPERS in unirl.train.backend.veomni.sp.diffusion."
        )
    wrapper(model, sp_group)
    logger.info("diffusion SP installed for %s (sp_size=%d)", cls, sp_size)


def is_diffusers_transformer(model: nn.Module) -> bool:
    return any(hasattr(m, "set_processor") and hasattr(m, "get_processor") for m in model.modules())


__all__ = ["apply_diffusion_sequence_parallelism", "is_diffusers_transformer", "FORWARD_WRAPPERS"]
