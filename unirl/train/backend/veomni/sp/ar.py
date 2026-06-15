"""Ulysses SP for HF autoregressive causal-LMs (e.g. Qwen3).

Installed by the VeOmni backend after ``veomni_parallelize``. Two pieces:

1. **Attention** — set ``config._attn_implementation`` to VeOmni's registered
   ``veomni_flash_attention_2_with_sp`` so the decoder's attention runs the
   Ulysses all-to-all (gather sequence / scatter heads → full-seq attention →
   scatter sequence / gather heads). The wrapper self-disables when
   ``ulysses_size == 1``, so this is safe to set unconditionally.

2. **Boundary** — wrap the *decoder* ``forward`` (``model.model``) to slice the
   sequence across SP ranks at entry and ``gather_outputs`` the hidden states at
   exit. The CausalLM head + replay log-prob code
   (:mod:`unirl.models.qwen3.ar`) then run unchanged on full-length hidden — no
   model/stage edits. Because unirl's train-side causal-LM is only ever driven
   teacher-forced (rollout is the decoupled engine's job), gating on
   ``ulysses_enabled`` is sufficient; we never slice a decode step.

Verified design: slice-in / gather-out + folded FSDP mesh needs no manual
sp_size gradient compensation (docs/usp-derisk/sp_fsdp.py).
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

SP_ATTN_IMPL = "veomni_flash_attention_2_with_sp"


def is_ar_causal_lm(model: nn.Module) -> bool:
    """HF causal-LM shape: a decoder (``.model``) + ``.lm_head`` + ``.config``."""
    return hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model, "config")


def apply_ar_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Route attention through VeOmni Ulysses + wrap the decoder forward."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_attention_patch_installed()
    _install_b1_dense_attn_patch()

    # Set the SP attn impl on the model config and every sub-config that carries
    # one (transformers resolves the attention fn per-forward via this field, so
    # setting it on the already-built model re-dispatches).
    _set_attn_impl(model.config)
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            _set_attn_impl(cfg)

    _wrap_decoder_forward(model.model)
    logger.info(
        "AR SP installed: attn_implementation=%s + decoder slice/gather wrapper (sp_size=%d)",
        SP_ATTN_IMPL,
        sp_size,
    )


def _install_b1_dense_attn_patch() -> None:
    """Force the dense flash-attn path for B=1 under Ulysses.

    VeOmni gathers q/k/v to the FULL sequence inside its SP flash-attention, but
    the decoder still hands down the per-rank *local* ``position_ids``. With a
    2D mask absent (our B=1 pad-strip drops it), transformers' flash-attn derives
    the varlen ``cu_seqlens`` from that local ``position_ids`` (length S/sp) and
    applies it to the full gathered q (length S) -> reshape mismatch off by
    exactly sp. After the wrapper's strip, B=1 is a single dense causal sequence,
    so we drop the packing hints for B=1 and let the kernel use the dense path
    (``query_length`` = full seq). B>1 is untouched (keeps varlen). Idempotent.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    orig = ALL_ATTENTION_FUNCTIONS[SP_ATTN_IMPL]
    if getattr(orig, "_unirl_b1_dense", False):
        return

    _PACK_KEYS = (
        "position_ids", "cu_seq_lens_q", "cu_seq_lens_k",
        "max_length_q", "max_length_k", "max_seqlen_q", "max_seqlen_k",
    )

    @functools.wraps(orig)
    def _b1_dense(*args: Any, **kwargs: Any):
        query = args[1] if len(args) > 1 else kwargs.get("query")
        if query is not None and query.shape[0] == 1:
            for k in _PACK_KEYS:
                kwargs.pop(k, None)
        return orig(*args, **kwargs)

    _b1_dense._unirl_b1_dense = True
    ALL_ATTENTION_FUNCTIONS.register(SP_ATTN_IMPL, _b1_dense)


def _set_attn_impl(cfg: Any) -> None:
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = SP_ATTN_IMPL
    # Some HF configs nest a text sub-config (VLMs); set there too if present.
    get_text = getattr(cfg, "get_text_config", None)
    if callable(get_text):
        try:
            tcfg = get_text()
            if tcfg is not None and tcfg is not cfg and hasattr(tcfg, "_attn_implementation"):
                tcfg._attn_implementation = SP_ATTN_IMPL
        except Exception:  # noqa: BLE001 — best-effort; absence is fine
            pass


def _wrap_decoder_forward(decoder: nn.Module) -> None:
    """Wrap ``decoder.forward``: slice seq-dim inputs in, gather hidden out.

    Idempotent. No-op at run time unless ``ulysses_enabled``. Replacing
    ``.forward`` composes with FSDP2 (its hooks fire on ``__call__``), exactly
    like the dual-mode forward in unirl.models.qwen3.ar.
    """
    if getattr(decoder.forward, "_unirl_sp_wrapped", False):
        return

    orig = decoder.forward

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

    @functools.wraps(orig)
    def sp_forward(*args: Any, **kwargs: Any):
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return orig(*args, **kwargs)
        spg = ps.sp_group

        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        position_ids = kwargs.get("position_ids")
        attention_mask = kwargs.get("attention_mask")

        if inputs_embeds is not None:
            true_len = inputs_embeds.shape[1]
        elif input_ids is not None:
            true_len = input_ids.shape[1]
        else:
            return orig(*args, **kwargs)  # nothing to slice (decode-style call)

        batch = (inputs_embeds if inputs_embeds is not None else input_ids).shape[0]
        mask2d = attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None

        # B=1 + padding (the train micro geometry, micro_batch_size=1): a single
        # left/right-padded sample's cumsum position_ids carries repeated zeros.
        # VeOmni's SP attn gathers q/k/v to the full seq but forwards the *sliced*
        # position_ids to flash-attn, whose varlen inference then reads those
        # resets as bogus sequence boundaries -> wrong cu_seqlens -> corrupted
        # logprobs. Strip the pad to a dense, monotonic-position_ids sequence
        # (the validated no-pad path), then re-pad the hidden states so the
        # downstream lm_head/logprob slice (which indexes the original padded
        # layout) is unchanged. B>1 keeps the plain slice/gather path.
        if batch == 1 and mask2d is not None and int(mask2d.sum().item()) < true_len:
            nz = mask2d[0].nonzero(as_tuple=False).flatten()
            s, e = int(nz[0].item()), int(nz[-1].item()) + 1  # contiguous real span [s, e)
            real_len = e - s
            sp = max(1, int(getattr(ps, "ulysses_size", 1)))
            # Round the real span UP to a multiple of sp and right-pad it
            # ourselves with a MONOTONIC arange position_ids. We cannot let
            # slice_input_tensor do the divisibility padding: it pads
            # position_ids with zeros, and a trailing 0 reads as a sequence
            # reset to flash-attn's varlen inference -> bogus cu_seqlens ->
            # reshape error (off by exactly sp). With mask=None + monotonic
            # position_ids the kernel takes the dense causal path; the few
            # right-pad tokens attend causally and are dropped before re-pad,
            # so they never affect the real tokens' hidden states.
            target = ((real_len + sp - 1) // sp) * sp
            pad = target - real_len
            ref = input_ids if input_ids is not None else inputs_embeds
            if input_ids is not None:
                seq = input_ids[:, s:e]
                if pad:
                    seq = torch.cat([seq, seq.new_zeros((batch, pad))], dim=1)
                kwargs["input_ids"] = slice_input_tensor(seq, dim=1, group=spg)
            if inputs_embeds is not None:
                emb = inputs_embeds[:, s:e]
                if pad:
                    emb = torch.cat([emb, emb.new_zeros((batch, pad, emb.shape[-1]))], dim=1)
                kwargs["inputs_embeds"] = slice_input_tensor(emb, dim=1, group=spg)
            pos_full = torch.arange(target, device=ref.device).unsqueeze(0).expand(batch, target).contiguous()
            kwargs["position_ids"] = slice_input_tensor(pos_full, dim=1, group=spg)
            kwargs["attention_mask"] = None  # dense causal, pad stripped
            kwargs.pop("cache_position", None)
            out = orig(*args, **kwargs)
            h = gather_outputs(out.last_hidden_state, gather_dim=1, group=spg)
            h = h[:, :real_len, :]  # drop right-pad
            full = h.new_zeros((batch, true_len, h.shape[-1]))
            full[:, s:e, :] = h
            out.last_hidden_state = full
            return out

        # No padding (or B>1): plain slice in / gather out. attention_mask is left
        # untouched (full) so it matches the all-to-all-gathered q/k/v.
        if input_ids is not None:
            kwargs["input_ids"] = slice_input_tensor(input_ids, dim=1, group=spg)
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, group=spg)
        if position_ids is not None:
            kwargs["position_ids"] = slice_input_tensor(position_ids, dim=position_ids.dim() - 1, group=spg)
        # The decoder rebuilds cache_position for the local chunk length; a stale
        # full-length one would mismatch the sliced hidden states.
        kwargs.pop("cache_position", None)

        out = orig(*args, **kwargs)
        hidden = gather_outputs(out.last_hidden_state, gather_dim=1, group=spg)
        if hidden.shape[1] > true_len:  # drop SP divisibility padding
            hidden = hidden[:, :true_len, :]
        out.last_hidden_state = hidden
        return out

    sp_forward._unirl_sp_wrapped = True
    decoder.forward = sp_forward


__all__ = ["apply_ar_sequence_parallelism", "is_ar_causal_lm", "SP_ATTN_IMPL"]
