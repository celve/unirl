"""Ulysses SP for HunyuanImage-3.0 (HI3) under the VeOmni backend.

HI3 is an ~80B MoE composite whose shared decoder (``HunyuanImage3Model``) runs
both gen_text (AR) and gen_image (DiT) through ONE attention class,
``HunyuanImage3SDPAAttention`` — a hand-rolled ``F.scaled_dot_product_attention``
over a dense 4D causal-image mask, resolved from HI3's own
``Hunyuan_ATTENTION_CLASSES`` dict (it bypasses transformers'
``ALL_ATTENTION_FUNCTIONS``). So the AR path's "set ``_attn_implementation``"
re-dispatch cannot attach, and HI3 is not a diffusers transformer. This adapter
supplies both halves directly:

1. **Attention** — monkey-patch ``HunyuanImage3SDPAAttention.forward`` to insert
   the Ulysses all-to-all (gather sequence / scatter heads -> full-seq attention
   -> scatter sequence / gather heads) around the SDPA, after RoPE + qk_norm and
   before ``repeat_kv``. Gated on ``ulysses_enabled`` (a true no-op at sp<=1), so
   safe to install unconditionally. ONE patch covers both modes (shared
   attention). The full 4D ``[B,1,L,L]`` mask is kept and broadcasts over the
   head-sharded SDPA — no mask slicing (this is why HI3 avoids the AR path's
   B=1 cu_seqlens machinery).

2. **Boundary** — wrap the decoder (``HunyuanImage3Model``) ``forward`` to slice
   the sequence-carrying inputs (``input_ids`` / ``inputs_embeds`` /
   ``position_ids`` / ``custom_pos_emb`` = the 2D-RoPE ``(cos, sin)``) across SP
   ranks at entry and ``gather_outputs`` the hidden states at exit. The outer
   ``HunyuanImage3ForCausalMM`` builds inputs_embeds (``model.wte`` + image
   scatter) before, and applies ``model.ln_f`` / ``ragged_final_layer`` after —
   all on full-length tensors, untouched (the gathered output restores full L).

GQA 32 Q / 8 KV (kv_groups 4): ``num_key_value_heads % sp`` is the binding
constraint -> sp in {2,4,8}. The packed sequence length must be a multiple of sp
(pad in the collate); validated at the boundary. No manual sp_size gradient
compensation (folded FSDP mesh; same as the AR path).

Empirically validated: sp=1 vs sp in {2,4} parity to ~1e-6 fp32 on the real
attention + full decoder layer (``docs/hi3-sp-veomni-feasibility.md``,
``scripts/hi3-sp-feasibility/``).
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

_HI3_ATTN_CLASS = "HunyuanImage3SDPAAttention"


def is_hunyuan_image3_model(model: nn.Module) -> bool:
    """True if ``model`` is HI3's decoder (carries ``HunyuanImage3SDPAAttention``).

    Probes for the attention submodules rather than the root class name, so it is
    robust to the FSDP2 root-class rename.
    """
    return any(type(m).__name__ == _HI3_ATTN_CLASS for m in model.modules())


def apply_hi3_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Patch HI3 attention with the Ulysses all-to-all + wrap the decoder forward."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()

    cfg = getattr(model, "config", None)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    if n_heads is None:
        raise ValueError("HI3 SP: decoder has no config.num_attention_heads")
    if n_heads % sp_size != 0 or n_kv % sp_size != 0:
        raise ValueError(
            f"HI3 SP: num_attention_heads={n_heads} and num_key_value_heads={n_kv} "
            f"must both be divisible by sp_size={sp_size} (8 KV heads -> sp in {{2,4,8}})."
        )

    _patch_hi3_attention(model)
    _wrap_hi3_decoder_forward(model)

    if os.environ.get("UNIRL_HI3_GATHER_MOE"):
        # Isolation / reference fix: run the WHOLE MoE (gate + experts + shared) on the
        # full gathered sequence so every expert sees M_e^full -> per-expert cuBLAS ==
        # the no-SP baseline, bit-exact. Subsumes grouped + gate-pad + minvariant-shared
        # (the entire MoE runs at full L, so none of the M-invariance hacks are needed).
        # Requires per-expert ModuleList (backend skips the grouped restructure).
        from unirl.train.backend.veomni.sp.hi3_moe import patch_gathered_moe_forward

        patch_gathered_moe_forward(model)
        logger.info(
            "HI3 SP installed: attention all-to-all + decoder wrapper + gathered MoE (sp_size=%d)",
            sp_size,
        )
        return

    # MoE experts: when restructured to stacked params (backend pre-FSDP hook / bisect),
    # run them through veomni's M-invariant grouped-GEMM. Falls back to per-expert
    # F.linear routing if not restructured.
    grouped = any(getattr(m, "_grouped", False) for m in model.modules())
    if grouped:
        from unirl.train.backend.veomni.sp.hi3_moe import patch_grouped_moe_forward

        patch_grouped_moe_forward(model)

    # The router gate (fp32, routing-sensitive) and the shared MLP's moe_inter-dim
    # GEMM still go through F.linear; route them M-invariantly. Expert GEMMs (when
    # grouped) bypass F.linear entirely.
    moe_inter = _moe_inter_dims(cfg)
    _install_minvariant_linear(moe_inter)
    logger.info(
        "HI3 SP installed: attention all-to-all + decoder wrapper + %s MoE (sp_size=%d)",
        "grouped-GEMM" if grouped else "per-expert",
        sp_size,
    )


def _moe_inter_dims(cfg) -> frozenset:
    """The moe/shared-MLP intermediate dim(s) (N or K of the GEMMs that carry the
    bf16 M-dependence). Used to shape-gate the M-invariant F.linear routing."""
    dims = set()
    for attr in ("moe_intermediate_size", "intermediate_size"):
        v = getattr(cfg, attr, None)
        if isinstance(v, (list, tuple)):
            dims.update(int(x) for x in v)
        elif isinstance(v, int):
            dims.add(int(v))
    return frozenset(dims)


def _install_minvariant_linear(moe_inter: "frozenset[int]") -> None:
    """Make the remaining (non-grouped) ``F.linear`` calls M-invariant under SP.

    The bf16 GEMM divergence that breaks the GRPO ratio is cuBLAS M-dependence: a
    different kernel (K-reduction order) at M=L/sp vs M=L. The expert GEMMs (the main
    offender) go through veomni's M-invariant grouped-GEMM (see ``hi3_moe``). Two
    cases remain on ``F.linear`` and are routed here, shape-gated:

    * **router gate** (tiny N = num_experts, and/or fp32): its output drives a DISCRETE
      top-k selection, hypersensitive to a 1-ULP logit change. Triton ``!=`` cuBLAS by
      a hair would flip routing vs generation, so reproduce cuBLAS-M=L *exactly* by
      padding the local L/sp tokens up to L and keeping the local rows (gate is tiny,
      so the sp x padding is negligible).
    * **shared MLP** ``down_proj`` (K = moe_intermediate): carries the same bf16
      M-dependence as the experts; route through the fixed-tiling kernel (== cuBLAS
      large-M in bf16).

    Everything else (qkv/o_proj/lm_head — empirically M-stable above M~256) stays on
    cuBLAS. Global patch so it also covers activation-checkpointing recompute.
    Idempotent.
    """
    import torch.nn.functional as F

    if getattr(F.linear, "_unirl_minvariant", False):
        return

    from veomni.distributed.parallel_state import get_parallel_state

    from unirl.utils.triton_mm import minvariant_linear

    _orig_linear = F.linear

    def _patched_linear(x, weight, bias=None, _orig=_orig_linear, _dims=moe_inter):
        if x.dim() >= 2:
            try:
                ps = get_parallel_state()
                if ps.ulysses_enabled and int(getattr(ps, "ulysses_size", 1)) > 1:
                    n = int(weight.shape[0])
                    k = int(weight.shape[-1])
                    if n <= 256 or x.dtype == torch.float32:
                        sp = int(getattr(ps, "ulysses_size", 1))
                        m = 1
                        for s in x.shape[:-1]:
                            m *= int(s)
                        xf = x.reshape(-1, x.shape[-1])
                        xp = torch.cat([xf, xf.new_zeros(m * (sp - 1), xf.shape[-1])], dim=0)
                        yp = _orig(xp, weight, bias)
                        return yp[:m].reshape(*x.shape[:-1], n)
                    if x.dtype in (torch.bfloat16, torch.float16) and (n in _dims or k in _dims):
                        # shared-MLP moe_inter-dim GEMMs (experts go via the grouped path);
                        # qkv/o_proj stay on cuBLAS (M-stable). Real-weight probes confirmed
                        # both are bit-exact, so route only the moe_inter shapes for speed.
                        return minvariant_linear(x, weight, bias)
            except Exception:
                pass
        return _orig(x, weight, bias)

    _patched_linear._unirl_minvariant = True  # type: ignore[attr-defined]
    F.linear = _patched_linear
    logger.info("HI3 SP: M-invariant F.linear routing installed (gate pad + moe_inter dims %s)", sorted(moe_inter))


def _patch_hi3_attention(model: nn.Module) -> None:
    """Monkey-patch ``HunyuanImage3SDPAAttention.forward`` (class-level, idempotent).

    The patched forward inserts the Ulysses all-to-all around the SDPA; it is a
    no-op (delegates to the original) unless ``ulysses_enabled``. RoPE + qk_norm
    run on the rank-local ``L/sp`` q/k (the boundary wrapper sliced ``custom_pos_emb``
    to match), then the all-to-all gathers to full ``L`` / scatters the 32 Q / 8 KV
    heads; ``repeat_kv`` runs on the local ``H/sp`` heads; the full 4D mask drives
    SDPA; the inverse all-to-all returns the local sequence shard.
    """
    attn_cls = next((type(m) for m in model.modules() if type(m).__name__ == _HI3_ATTN_CLASS), None)
    if attn_cls is None:
        raise ValueError("HI3 SP: no HunyuanImage3SDPAAttention module found in the decoder")
    if getattr(attn_cls.forward, "_unirl_hi3_sp", False):
        return

    mod = sys.modules[attn_cls.__module__]
    apply_rotary_pos_emb = mod.apply_rotary_pos_emb
    repeat_kv = mod.repeat_kv

    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    orig_forward = attn_cls.forward

    @functools.wraps(orig_forward)
    def sp_attn_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        custom_pos_emb=None,
        **kwargs,
    ):
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return orig_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                **kwargs,
            )
        g = ps.ulysses_group
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim
        )
        q, k, v = torch.split(qkv, [self.num_key_value_groups, 1, 1], dim=3)
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        # Ulysses: seq-sharded [B,H,L/sp,d] -> head-sharded [B,H/sp,L,d].
        q = gather_seq_scatter_heads(q, seq_dim=2, head_dim=1, group=g)
        k = gather_seq_scatter_heads(k, seq_dim=2, head_dim=1, group=g)
        v = gather_seq_scatter_heads(v, seq_dim=2, head_dim=1, group=g)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        if q.device.type == "cuda" and attention_mask is not None:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        # inverse: head-sharded [B,H/sp,L,d] -> seq-sharded [B,H,L/sp,d].
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2, group=g)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    sp_attn_forward._unirl_hi3_sp = True
    attn_cls.forward = sp_attn_forward


def _extend_mask(mask: torch.Tensor, true_len: int, padded_len: int) -> torch.Tensor:
    """Extend a ``[B,H,L,L]`` 4D mask to ``[B,H,L',L']`` for ``L'``-padded SP.

    The few right-pad tokens (``true_len:padded_len``, < sp of them, added only to
    make the sequence divisible by sp) must be NaN-safe under SDPA: a fully-masked
    query row softmaxes to NaN, which would survive into the (later-trimmed) pad
    rows. So each pad query attends ONLY itself (its output is trimmed after the
    gather), real queries never attend pad keys, and the real ``[L,L]`` region
    keeps the original mask. Handles BOTH a bool mask (``True`` = attend) and a
    float additive mask (``0`` = attend, large-negative = masked): HI3 uses a bool
    mask for gen_image and an additive float mask for the gen_text AR replay.
    """
    b, h = mask.shape[0], mask.shape[1]
    dev = mask.device
    idx = torch.arange(true_len, padded_len, device=dev)
    if mask.dtype == torch.bool:
        out = torch.zeros((b, h, padded_len, padded_len), dtype=torch.bool, device=dev)
        out[:, :, :true_len, :true_len] = mask
        out[:, :, idx, idx] = True  # each pad query attends itself (NaN-safe)
        return out
    out = torch.full((b, h, padded_len, padded_len), torch.finfo(mask.dtype).min, dtype=mask.dtype, device=dev)
    out[:, :, :true_len, :true_len] = mask
    out[:, :, idx, idx] = 0  # each pad query attends itself (NaN-safe)
    return out


def _wrap_hi3_decoder_forward(decoder: nn.Module) -> None:
    """Wrap the decoder forward: slice seq inputs + ``(cos, sin)`` in, gather hidden out.

    Keeps the 4D ``attention_mask`` FULL (it broadcasts over the head-sharded
    SDPA — no slicing). When the packed length is not a multiple of sp (the
    common ``B=1`` micro-batch, which HI3's collate does not pad), pads the
    sequence up to the next multiple of sp here — inputs, ``(cos, sin)``, and a
    NaN-safe mask extension (bool or float additive) — then trims the gathered
    hidden back to the true length. Idempotent; a run-time no-op unless ``ulysses_enabled``.
    Replacing ``.forward`` composes with FSDP2 (its hooks fire on ``__call__``).
    """
    if getattr(decoder.forward, "_unirl_hi3_sp_wrapped", False):
        return

    orig = decoder.forward

    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

    def _pad_seq(t: torch.Tensor, pad: int, dim: int = 1) -> torch.Tensor:
        if pad == 0:
            return t
        shape = list(t.shape)
        shape[dim] = pad
        return torch.cat([t, t.new_zeros(shape)], dim=dim)

    @functools.wraps(orig)
    def sp_forward(*args: Any, **kwargs: Any):
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return orig(*args, **kwargs)
        spg = ps.sp_group
        sp = int(getattr(ps, "ulysses_size", 1))

        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        ref = inputs_embeds if inputs_embeds is not None else input_ids
        if ref is None:
            return orig(*args, **kwargs)
        true_len = ref.shape[1]
        _n = getattr(sp_forward, "_memprobe_n", 0)
        if _n < 4:  # diagnostic: memory at decoder entry (after the composite's gen_image scatter)
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.warning(
                    "[MEMPROBE sp_forward#%d] mode=%s B=%d L=%d sp=%d | allocated=%.1fGB reserved=%.1fGB",
                    _n, kwargs.get("mode", "?"), ref.shape[0], true_len, sp,
                    torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9,
                )
            sp_forward._memprobe_n = _n + 1
        pad = (-true_len) % sp  # tokens to add to reach a multiple of sp (0 if already divisible)

        if input_ids is not None:
            kwargs["input_ids"] = slice_input_tensor(_pad_seq(input_ids, pad), dim=1, group=spg)
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = slice_input_tensor(_pad_seq(inputs_embeds, pad), dim=1, group=spg)
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            kwargs["position_ids"] = slice_input_tensor(
                _pad_seq(position_ids, pad, dim=position_ids.dim() - 1), dim=position_ids.dim() - 1, group=spg
            )
        custom_pos_emb = kwargs.get("custom_pos_emb")
        if custom_pos_emb is not None:
            cos, sin = custom_pos_emb
            kwargs["custom_pos_emb"] = (
                slice_input_tensor(_pad_seq(cos, pad), dim=1, group=spg),
                slice_input_tensor(_pad_seq(sin, pad), dim=1, group=spg),
            )
        attention_mask = kwargs.get("attention_mask")
        if pad and attention_mask is not None:
            kwargs["attention_mask"] = _extend_mask(attention_mask, true_len, true_len + pad)
        # (no pad -> mask stays full; it broadcasts over the head-sharded SDPA)

        out = orig(*args, **kwargs)

        def _finish(hidden: torch.Tensor) -> torch.Tensor:
            hidden = gather_outputs(hidden, gather_dim=1, group=spg)
            return hidden[:, :true_len, :] if pad else hidden

        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = _finish(out.last_hidden_state)
            return out
        return (_finish(out[0]),) + tuple(out[1:])

    sp_forward._unirl_hi3_sp_wrapped = True
    decoder.forward = sp_forward


__all__ = ["apply_hi3_sequence_parallelism", "is_hunyuan_image3_model"]
