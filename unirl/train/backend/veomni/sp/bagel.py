"""Ulysses SP for BAGEL-7B-MoT (vendored ``Qwen2ForCausalLM`` packed-MoT model).

BAGEL is neither a plain HF causal-LM nor a diffusers transformer: its trainable
module is the vendored :class:`Qwen2ForCausalLM` (``unirl.models.bagel.vendor``),
whose RL forward is ``Bagel._forward_flow -> language_model.forward_inference``
(``mode="gen"``) and whose attention calls ``flash_attn_varlen_func`` *directly*
(packed varlen, ``causal=False``, queries attend to a replicated text KV cache).
So neither the AR patch (it keys on ``input_ids``/``inputs_embeds``) nor the
diffusion patch (it rejects masks / has no ``set_processor``) applies — this is a
third SP family, dispatched ahead of the AR check in :mod:`..sp`.

Two parts, both gated at run time on ``ulysses_enabled`` AND ``mode=="gen"`` (the
text-context-build forward is a strict passthrough — its short cache stays
replicated full on every rank):

1. **Boundary** — wrap ``Qwen2ForCausalLM.forward_inference``: slice the QUERY
   stream (``packed_query_sequence`` + ``packed_query_position_ids``) across SP
   ranks, rebuild the per-token index tensors for the local slice
   (``packed_query_indexes``, ``query_lens``, and the MoT routing
   ``packed_text_indexes`` / ``packed_vae_token_indexes``); keep the context
   (``past_key_values`` / ``key_values_lens`` / ``packed_key_value_indexes``)
   full. Gather the hidden states back to full length at exit (before ``llm2vae``).
   RoPE needs no separate hook — it is computed per-token from the sliced
   ``packed_query_position_ids`` inside ``Qwen2Model.forward_inference``.

2. **Attention** — wrap the module-level ``flash_attn_varlen_func`` symbol: each
   layer's merged K/V is context-first contiguous ``[context | local-query]``
   (``prepare_vae_latent``). All-to-all gather the local query Q/K/V to the full
   query length scattered over heads (``gather_seq_scatter_heads``), head-slice
   the replicated context K/V to match (``tensor_split`` — byte-matches the
   all-to-all split), attend over ``[context | full-query]`` with **real** lengths
   (no unmasked padding under ``causal=False``), then scatter the output back
   (``gather_heads_scatter_seq``). GQA (28 q / 4 kv heads) is preserved because the
   contiguous head-scatter keeps rank ``r``'s q-heads ``[7r:7r+7]`` aligned with
   its kv-head ``r`` (sp=4) / ``[2r:2r+2]`` (sp=2).

Scope: B=1 single packed sequence (the recipe; ``forward_batch_size=1``) -> the
varlen ``cu_seqlens`` are single-segment ``[0, L]``. ``sp_size ∈ {2,4}`` (28 % sp
== 0 and 4 % sp == 0; sp=8 is rejected). Folded ``dp_shard × ulysses`` mesh needs
no manual grad compensation (docs/usp-derisk; tests/distributed/parallel/sp_fsdp.py):
only the grad-bearing query stream is sliced; the context K/V is built under
``no_grad`` (``requires_grad=False``), so it contributes no double-counted gradient.
"""

from __future__ import annotations

import contextvars
import functools
import logging
from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


class _GenSP(NamedTuple):
    """SP metadata stashed by the boundary wrapper, read by the flash wrapper."""

    group: Any
    sp: int
    rank: int
    l_q_real: int  # true query length (pre-padding)
    l_ctx: int     # context (cached) length, per the original key_values_lens


# Set inside the boundary wrapper around the original ``forward_inference`` call;
# read by the ``flash_attn_varlen_func`` wrapper deep inside the (synchronous)
# layer loop. ``None`` => not in an SP gen forward => both wrappers pass through.
_GEN_SP: "contextvars.ContextVar[Optional[_GenSP]]" = contextvars.ContextVar("_bagel_gen_sp", default=None)


def is_bagel_mot(model: nn.Module) -> bool:
    """True iff ``model`` is the BAGEL packed-MoT LM (its stacks are
    ``Qwen2MoTDecoderLayer``). MRO-walked because after ``veomni_parallelize`` the
    layer's class is a dynamic FSDP2 subclass whose base is the real class. Keyed
    on the exact class name (NOT ``use_moe`` / ``'Mo' in layer_module``, which also
    match ``Qwen2MoEDecoderLayer``)."""
    for m in model.modules():
        for klass in type(m).__mro__:
            if klass.__name__ == "Qwen2MoTDecoderLayer":
                return True
    return False


def _sp_handles():
    """Lazy VeOmni handles (after the selective-import shim is installed)."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
    from veomni.distributed.sequence_parallel.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    return (
        get_parallel_state,
        slice_input_tensor,
        gather_outputs,
        gather_seq_scatter_heads,
        gather_heads_scatter_seq,
    )


def apply_bagel_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Install BAGEL Ulysses SP on ``model`` (the vendored ``Qwen2ForCausalLM``)."""
    cfg = model.config
    nq = int(cfg.num_attention_heads)
    nkv = int(getattr(cfg, "num_key_value_heads", nq))
    if nq % sp_size != 0 or nkv % sp_size != 0:
        valid = [s for s in (1, 2, 4, 8) if nq % s == 0 and nkv % s == 0]
        raise ValueError(
            f"BAGEL SP: sp_size={sp_size} must divide num_attention_heads={nq} AND "
            f"num_key_value_heads={nkv}. Valid sp for this checkpoint: {valid} "
            f"(GQA caps it — sp=8 is impossible for the 28q/4kv config)."
        )
    _install_flash_attn_ulysses()
    _wrap_forward_inference(model)
    logger.info(
        "BAGEL SP installed: query-stream slice/gather + flash_attn_varlen Ulysses "
        "(sp_size=%d, %dq/%dkv heads)",
        sp_size,
        nq,
        nkv,
    )


def _wrap_forward_inference(model: nn.Module) -> None:
    """Boundary half: slice the query stream in / gather the hidden states out.

    Replaces ``model.forward_inference`` (the method ``Bagel._forward_flow`` calls
    directly — NOT ``forward``). Idempotent. No-op unless ``ulysses_enabled`` and
    ``mode == "gen"``. Composes with FSDP2, whose unshard hooks fire per
    ``Qwen2MoTDecoderLayer.__call__`` inside the layer loop, independent of this
    root-level wrap.
    """
    orig = model.forward_inference
    if getattr(orig, "_unirl_bagel_sp", False):
        return

    get_parallel_state, slice_input_tensor, gather_outputs, _, _ = _sp_handles()

    @functools.wraps(orig)
    def sp_forward_inference(**kwargs: Any):
        ps = get_parallel_state()
        if not ps.ulysses_enabled or kwargs.get("mode") != "gen":
            return orig(**kwargs)

        spg = ps.sp_group
        sp = dist.get_world_size(spg)
        r = dist.get_rank(spg)

        seq = kwargs["packed_query_sequence"]
        pos = kwargs["packed_query_position_ids"]
        kv_lens = kwargs["key_values_lens"]
        l_q = seq.shape[0]
        l_ctx = int(kv_lens.sum().item())
        unit = (l_q + sp - 1) // sp  # local (padded) query length per rank

        # Slice the grad-bearing query stream; context stays full/replicated.
        kwargs["packed_query_sequence"] = slice_input_tensor(seq, dim=0, group=spg)
        kwargs["packed_query_position_ids"] = slice_input_tensor(pos, dim=0, group=spg)

        # Rebuild per-token index tensors for the local slice. The local merged
        # K/V is [context (L_ctx) | local-query (unit)] (context-first), so the
        # query rows map to merged positions [L_ctx, L_ctx+unit).
        dev = seq.device
        kwargs["packed_query_indexes"] = torch.arange(
            l_ctx, l_ctx + unit, device=dev, dtype=kwargs["packed_query_indexes"].dtype
        )
        kwargs["query_lens"] = torch.tensor([unit], device=kv_lens.device, dtype=kwargs["query_lens"].dtype)
        lo = r * unit
        kwargs["packed_text_indexes"] = _rebase_local(kwargs.get("packed_text_indexes"), lo, unit)
        kwargs["packed_vae_token_indexes"] = _rebase_local(kwargs.get("packed_vae_token_indexes"), lo, unit)
        # key_values_lens / packed_key_value_indexes (context) are left FULL.

        token = _GEN_SP.set(_GenSP(group=spg, sp=sp, rank=r, l_q_real=l_q, l_ctx=l_ctx))
        try:
            out = orig(**kwargs)
        finally:
            _GEN_SP.reset(token)

        # Gather the hidden states back to full query length (drop SP pad), so the
        # downstream full-length packed_vae_token_indexes / llm2vae are unchanged.
        out.packed_query_sequence = gather_outputs(
            out.packed_query_sequence, gather_dim=0, padding_dim=0, unpad_dim_size=l_q, group=spg
        )
        return out

    sp_forward_inference._unirl_bagel_sp = True
    model.forward_inference = sp_forward_inference


def _rebase_local(idx: Optional[torch.Tensor], lo: int, unit: int) -> Optional[torch.Tensor]:
    """Select the indexes falling in this rank's contiguous query slice [lo, lo+unit)
    and re-base them to [0, unit). Returns an empty LongTensor (never ``None``) for a
    rank whose slice holds no token of that type (e.g. all-VAE slice -> no text),
    which the vendored advanced-indexing routing treats as a valid no-op."""
    if idx is None:
        return None
    mask = (idx >= lo) & (idx < lo + unit)
    return (idx[mask] - lo).contiguous()


def _install_flash_attn_ulysses() -> None:
    """Attention half: wrap the module-level ``flash_attn_varlen_func`` symbol in
    the vendored ``qwen2_navit`` with the Ulysses all-to-all. Idempotent. A true
    passthrough unless the boundary wrapper has stashed ``_GEN_SP`` (i.e. inside an
    SP ``mode=="gen"`` forward), so the und/non-SP paths are untouched.
    """
    from unirl.models.bagel.vendor.modeling.bagel import qwen2_navit as qn

    orig_flash = qn.flash_attn_varlen_func
    if getattr(orig_flash, "_unirl_bagel_sp", False):
        return

    _, _, _, gather_seq_scatter_heads, gather_heads_scatter_seq = _sp_handles()

    @functools.wraps(orig_flash)
    def ulysses_flash(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, causal=False, **kw):
        st = _GEN_SP.get()
        if st is None:
            return orig_flash(
                q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=causal, **kw,
            )

        spg, sp, r, l_q_real, l_ctx = st
        # q: [unit, nq, D] (local query). k/v: [L_ctx + unit, nkv, D], context-first.
        assert k.shape[0] >= l_ctx, f"BAGEL SP: merged K len {k.shape[0]} < context {l_ctx}"
        ctx_k, qry_k = k[:l_ctx], k[l_ctx:]
        ctx_v, qry_v = v[:l_ctx], v[l_ctx:]

        # All-to-all the local query Q/K/V -> full query length, heads scattered;
        # unpadded_dim_size strips the SP divisibility padding after the gather, so
        # only REAL tokens reach the (mask-free, causal=False) kernel.
        q_f = gather_seq_scatter_heads(q, seq_dim=0, head_dim=1, unpadded_dim_size=l_q_real, group=spg)
        k_f = gather_seq_scatter_heads(qry_k, seq_dim=0, head_dim=1, unpadded_dim_size=l_q_real, group=spg)
        v_f = gather_seq_scatter_heads(qry_v, seq_dim=0, head_dim=1, unpadded_dim_size=l_q_real, group=spg)

        # Head-slice the replicated context K/V to match the all-to-all head split
        # exactly (tensor_split, the same op _SeqAllToAll uses on the head dim).
        ck = ctx_k.tensor_split(sp, dim=1)[r].contiguous()
        cv = ctx_v.tensor_split(sp, dim=1)[r].contiguous()
        merged_k = torch.cat([ck, k_f], dim=0)
        merged_v = torch.cat([cv, v_f], dim=0)

        l_k = l_ctx + l_q_real
        cu_q = torch.tensor([0, l_q_real], device=q.device, dtype=torch.int32)
        cu_k = torch.tensor([0, l_k], device=q.device, dtype=torch.int32)
        out = orig_flash(
            q_f, merged_k, merged_v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=l_q_real, max_seqlen_k=l_k, causal=causal, **kw,
        )
        # out: [l_q_real, nq/sp, D] -> back to local [unit, nq, D].
        return gather_heads_scatter_seq(out, head_dim=1, seq_dim=0, group=spg)

    ulysses_flash._unirl_bagel_sp = True
    qn.flash_attn_varlen_func = ulysses_flash


__all__ = ["apply_bagel_sequence_parallelism", "is_bagel_mot"]
