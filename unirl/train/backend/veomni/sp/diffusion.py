"""Ulysses SP for diffusers transformers (qwen-image / wan / sd3 / flux2).

Diffusers models don't go through HF's ``ALL_ATTENTION_FUNCTIONS``, so unlike the
AR path there is no registered attention to reuse — we inject an SP-aware
attention *processor* (the diffusers ``set_processor`` interface) and re-wrap each
model's ``forward`` to slice the latent/text streams in and gather them out.

Ported from the validated mmrl implementation
(``LIN-308/mmrl/mmrl/models/diffusers/sp_patch.py``), but built on VeOmni's
all-to-all primitives + the folded FSDP mesh — so, unlike mmrl's separate-group
primitives, NO sp_size gradient compensation is applied (validated in
docs/usp-derisk/sp_fsdp.py). Primitive mapping:

    mmrl SeqAllToAll.apply(g, x, scatter=2, gather=1)  ==  gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)
    mmrl SeqAllToAll.apply(g, x, scatter=1, gather=2)  ==  gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)

Per-model ``forward`` wrappers live in :data:`FORWARD_WRAPPERS` (Phase 2.2).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def apply_rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    """RoPE for diffusers Q/K, shape ``(B, S, H, D_h)``.

    Flux convention: cos/sin are ``(S, D_h)`` (2D) — Llama-style. Wan convention:
    cos/sin are ``(1, S, 1, D_h)`` (4D) — interleaved even/odd. Ported verbatim
    from mmrl ``sp_patch.apply_rotary_emb``.
    """
    cos, sin = freqs_cis
    cos = cos.to(x.device)
    sin = sin.to(x.device)
    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    x1, x2 = x.to(torch.float64).unflatten(-1, (-1, 2)).unbind(-1)
    freq_cos = cos[..., 0::2].to(torch.float64)
    freq_sin = sin[..., 1::2].to(torch.float64)
    out = torch.empty_like(x, dtype=torch.float64)
    out[..., 0::2] = x1 * freq_cos - x2 * freq_sin
    out[..., 1::2] = x1 * freq_sin + x2 * freq_cos
    return out.to(x.dtype)


def _sdpa(q: Tensor, k: Tensor, v: Tensor, scale: float, attn_mask: Tensor | None) -> Tensor:
    """SDPA on ``(B, S, H, D_h)`` tensors (transpose to ``(B, H, S, D_h)``).

    Diffusion attention is full (non-causal); SDPA gives exact, backend-stable
    results for parity. Flash can replace this later for throughput.
    """
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, scale=scale
    )
    return o.transpose(1, 2)


class SPAttentionProcessor:
    """SP-aware diffusers attention processor (Ulysses all-to-all).

    Three paths (port of mmrl ``SPAttentionProcessor``):

    * **dual-stream joint** (Flux/SD3/qwen-image MMDiT, ``add_q_proj`` present):
      all-to-all each of the [encoder, image] streams, re-cat in the *original
      processor's* order, attend, inverse all-to-all, project both streams.
    * **cross-attention** (``encoder_hidden_states`` but no ``add_q_proj``):
      skip all-to-all — local Q chunk attends full K/V (correct, queries are
      partition-independent).
    * **self-attention**: full Ulysses all-to-all on Q/K/V.
    """

    def __init__(self, sp_group: Any, original_processor: Any = None):
        self.sp_group = sp_group
        self.original_processor = original_processor
        from unirl.train.backend.veomni import _compat
        _compat.ensure_installed()
        from veomni.distributed.sequence_parallel.ulysses import (
            gather_heads_scatter_seq,
            gather_seq_scatter_heads,
        )
        self._gather_seq = gather_seq_scatter_heads   # scatter heads, gather seq
        self._gather_heads = gather_heads_scatter_seq  # scatter seq, gather heads

    def _a2a_seq_to_heads(self, x: Tensor) -> Tensor:
        return self._gather_seq(x, seq_dim=1, head_dim=2, group=self.sp_group)

    def _a2a_heads_to_seq(self, x: Tensor) -> Tensor:
        return self._gather_heads(x, head_dim=2, seq_dim=1, group=self.sp_group)

    def __call__(
        self,
        attn: Any,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ):
        if image_rotary_emb is None and rotary_emb is not None:
            image_rotary_emb = rotary_emb
        is_cross_attention = encoder_hidden_states is not None
        has_added_kv = (
            hasattr(attn, "add_q_proj") and attn.add_q_proj is not None and encoder_hidden_states is not None
        )
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        kv_input = hidden_states if (has_added_kv or not is_cross_attention) else encoder_hidden_states
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads
        has_norm_q = getattr(attn, "norm_q", None) is not None
        has_norm_k = getattr(attn, "norm_k", None) is not None
        norm_before_reshape = bool(
            has_norm_q and hasattr(attn.norm_q, "normalized_shape") and attn.norm_q.normalized_shape == (inner_dim,)
        )

        if norm_before_reshape:
            if has_norm_q:
                query = attn.norm_q(query)
            if has_norm_k:
                key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if not norm_before_reshape:
            if has_norm_q:
                query = attn.norm_q(query)
            if has_norm_k:
                key = attn.norm_k(key)

        if has_added_kv:
            enc_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim)
            enc_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim)
            enc_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim)
            if getattr(attn, "norm_added_q", None) is not None:
                enc_q = attn.norm_added_q(enc_q)
            if getattr(attn, "norm_added_k", None) is not None:
                enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale = attn.scale if hasattr(attn, "scale") else head_dim ** -0.5

        if has_added_kv:
            # cat order must mirror the original processor (Flux=encoder-first,
            # SD3 JointAttnProcessor2_0=image-first) for bit-exact parity.
            encoder_first = type(self.original_processor).__name__ != "JointAttnProcessor2_0"
            enc_len = encoder_hidden_states.shape[1]
            eq, iq = query[:, :enc_len], query[:, enc_len:]
            ek, ik = key[:, :enc_len], key[:, enc_len:]
            ev, iv = value[:, :enc_len], value[:, enc_len:]
            eq, ek, ev = self._a2a_seq_to_heads(eq), self._a2a_seq_to_heads(ek), self._a2a_seq_to_heads(ev)
            iq, ik, iv = self._a2a_seq_to_heads(iq), self._a2a_seq_to_heads(ik), self._a2a_seq_to_heads(iv)
            if encoder_first:
                q, k, v = (torch.cat([eq, iq], 1), torch.cat([ek, ik], 1), torch.cat([ev, iv], 1))
            else:
                q, k, v = (torch.cat([iq, eq], 1), torch.cat([ik, ek], 1), torch.cat([iv, ev], 1))
            out = _sdpa(q, k, v, scale, attention_mask)
            full_enc, full_img = eq.shape[1], iq.shape[1]
            if encoder_first:
                enc_out, img_out = out[:, :full_enc], out[:, full_enc:]
            else:
                img_out, enc_out = out[:, :full_img], out[:, full_img:]
            enc_out = self._a2a_heads_to_seq(enc_out)
            img_out = self._a2a_heads_to_seq(img_out)
            hidden_states = attn.to_out[0](img_out.reshape(batch_size, -1, inner_dim))
            if len(attn.to_out) > 1:
                hidden_states = attn.to_out[1](hidden_states)
            enc = enc_out.reshape(batch_size, -1, inner_dim)
            if not getattr(attn, "context_pre_only", False):
                enc = attn.to_add_out(enc)
            return hidden_states, enc

        if is_cross_attention:
            out = _sdpa(query, key, value, scale, attention_mask)  # skip all-to-all
        else:
            query, key, value = (self._a2a_seq_to_heads(query), self._a2a_seq_to_heads(key), self._a2a_seq_to_heads(value))
            out = _sdpa(query, key, value, scale, attention_mask)
            out = self._a2a_heads_to_seq(out)

        hidden_states = out.reshape(batch_size, -1, inner_dim)
        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            if len(attn.to_out) > 1:
                hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def inject_sp_processors(model: nn.Module, sp_group: Any) -> int:
    """Swap every diffusers ``Attention`` module's processor for the SP one.

    Duck-types the diffusers interface (``get_processor`` / ``set_processor``);
    validates ``heads % sp_size == 0``. Returns the count injected.
    """
    import torch.distributed as dist

    sp_size = dist.get_world_size(sp_group)
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_processor") and hasattr(module, "get_processor"):
            heads = getattr(module, "heads", None)
            if heads is not None and heads % sp_size != 0:
                raise ValueError(
                    f"SP requires num_heads % sp_size == 0, but '{name}' has {heads} heads "
                    f"and sp_size={sp_size}."
                )
            module.set_processor(SPAttentionProcessor(sp_group, original_processor=module.get_processor()))
            count += 1
    logger.info("diffusion SP: injected SPAttentionProcessor into %d attention modules", count)
    return count


# Per-model forward wrappers (slice latent/text streams in, gather out). Phase 2.2.
FORWARD_WRAPPERS: Dict[str, Callable[[nn.Module, Any], None]] = {}


def apply_diffusion_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Inject SP processors + install the per-model forward wrapper."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    sp_group = get_parallel_state().sp_group
    inject_sp_processors(model, sp_group)

    cls_name = type(model).__name__
    wrapper = FORWARD_WRAPPERS.get(cls_name)
    if wrapper is None:
        raise NotImplementedError(
            f"diffusion SP: attention processors injected, but no forward wrapper for "
            f"{cls_name} (input slice / output gather). Register one in "
            f"unirl.train.backend.veomni.sp.diffusion.FORWARD_WRAPPERS (Phase 2.2)."
        )
    wrapper(model, sp_group)
    logger.info("diffusion SP installed for %s (sp_size=%d)", cls_name, sp_size)


def is_diffusers_transformer(model: nn.Module) -> bool:
    """Has at least one diffusers ``Attention`` module (set_processor interface)."""
    return any(
        hasattr(m, "set_processor") and hasattr(m, "get_processor") for m in model.modules()
    )


__all__ = [
    "SPAttentionProcessor",
    "apply_diffusion_sequence_parallelism",
    "apply_rotary_emb",
    "inject_sp_processors",
    "is_diffusers_transformer",
    "FORWARD_WRAPPERS",
]
