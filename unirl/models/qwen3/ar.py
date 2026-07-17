"""Qwen3 AR stage: typed params + per-token kernel + rollout-level stage.

Three classes:

- ``Qwen3ARParams`` — typed request-shape knobs (max_tokens / temperature
  / top_p / top_k / stop_token_ids).
- ``Qwen3ARStep`` — per-token sampling kernel (reads logits, returns
  ``(token_id, log_prob)``). Verbatim mechanics from
  :class:`unirl.models.hunyuan_image3.ar.HunyuanImage3ARStep`.
- ``Qwen3ARStage`` — implements ``ARStage[Qwen3ARConditions]``. Drives
  HF :class:`AutoModelForCausalLM` through a per-token loop with KV
  cache, packs the results into a varlen :class:`TextSegment` with
  per-step full-softmax log-probs. ``replay`` recomputes per-token
  log-probs from stored rollout tokens via a single teacher-forced
  forward — the GRPO/PPO substitution point.
"""

from __future__ import annotations

import functools
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MethodType
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep, left_pad_prompt
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Qwen3Bundle
from .conditions import Qwen3ARConditions
from .replay_layout import (
    build_packed_replay_layout,
    build_padded_replay_layout,
    pack_padded_token_outputs,
)

logger = logging.getLogger(__name__)

_SPARSE_PACKED_ATTN = ("flex_attention", "flash_attention_2", "flash_attention_3", "flash_attention_4")


@functools.lru_cache(maxsize=None)
def _warn_packed_disabled(attn_impl: str) -> None:
    """One-time warning (per distinct backend) when packed replay is skipped.

    Fires when packing WOULD apply (B > 1) but the attention backend is not a
    sparse-block kernel, so replay uses the slower padded path instead.
    """
    logger.warning(
        "packed-varlen replay disabled: attn_implementation=%r is not a "
        "sparse-block kernel; using the padded replay path. Set "
        "attn_implementation='flex_attention' (or 'flash_attention_2' with "
        "flash_attn installed) to enable packed replay.",
        attn_impl,
    )


def _packed_replay_supported(attn_impl: Optional[str]) -> bool:
    """Feature-detect the packed varlen replay prerequisites (review #43).

    1. A sparse-block attention backend (flex_attention or flash_attention_2);
       on plain sdpa packed attention is full O((sum L)^2) and can regress, so
       require a sparse backend (checked first; warns once on fallback).
    2. transformers building a block-causal mask from restarting position_ids
       (masking_utils.find_packed_sequence_indices, transformers >= 4.53); on
       older versions the forward would silently attend ACROSS sequence
       boundaries (wrong logps, no error), so fall back to the dense path.
    """
    if attn_impl not in _SPARSE_PACKED_ATTN:
        _warn_packed_disabled(str(attn_impl))
        return False
    try:
        from transformers.masking_utils import find_packed_sequence_indices  # noqa: F401
    except Exception:
        return False
    return True


def _replay_aware_forward(
    self: Any,
    *,
    response_tokens: Optional[torch.Tensor] = None,
    prompt_len: Optional[int] = None,
    temperature: float = 1.0,
    autocast_dtype: Optional[torch.dtype] = None,
    packed_predict_index: Optional[torch.Tensor] = None,
    **kw: Any,
) -> Any:
    """Dual-mode ``forward`` installed on the Qwen3 CausalLM instance.

    Without ``response_tokens``: delegate to the stock class forward (decode /
    generate). With it: run the decoder body only and return the padded
    ``[B, T_max]`` FP32 per-token log-probs via a chunked
    ``x[tok] - logsumexp(x)`` over ``lm_head`` — the full ``[B, L, vocab]``
    logits are never materialized. Running inside ``forward`` keeps replay
    valid under FSDP2 root wrap: the root pre-forward gathers the leftover
    embed/norm/lm_head group and does not reshard it after forward.
    """
    if response_tokens is None:
        # Resolve the real class forward through the MRO (skips this instance
        # override; FSDPModule defines no forward).
        for klass in type(self).__mro__:
            f = klass.__dict__.get("forward")
            if f is not None and f is not _replay_aware_forward:
                return f(self, **kw)
        raise RuntimeError("_replay_aware_forward: no class-level forward found in the MRO")

    # cuDNN's fused SDPA backward (ScaledDotProductCudnnAttentionBackward0) returns
    # NaN grads on some bf16 sequences while the forward stays finite (confirmed via
    # torch.autograd.detect_anomaly): it floods every parameter grad and forces the
    # optimizer to skip the whole step (~half of them, observed on Qwen3-4B-Base AR
    # RL). Disable the cuDNN SDPA backend so PyTorch uses the stable flash /
    # mem-efficient attention kernels for the replay forward.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    # Body in autocast (matmul bf16, norms fp32); the caller passes None off-CUDA.
    autocast_ctx = (
        torch.autocast("cuda", autocast_dtype) if autocast_dtype in (torch.float16, torch.bfloat16) else nullcontext()
    )
    with autocast_ctx:
        hidden = self.model(**kw, use_cache=False, return_dict=True).last_hidden_state  # [B, L, H]

    # Chunked lm_head outside autocast: FP32 logp matching SGLang's
    # log_softmax(logits/T). The chunk scales inversely with batch so the
    # [B, chunk, vocab] FP32 transient stays ~1.2 GiB, and each chunk is
    # gradient-checkpointed (recomputed in backward rather than held).
    T = float(temperature) if float(temperature) > 0.0 else 1.0

    if packed_predict_index is not None:
        # Packed varlen replay: ``hidden`` is one packed row [1, L_total, H]
        # holding every sequence back-to-back (block-causal mask built by
        # transformers from the restarting position_ids). Predictions for the
        # FLAT response stream (``response_tokens`` = segment-order targets,
        # [T_total]) live at ``packed_predict_index`` — gather those hidden
        # states and run the same chunked fp32 ``x[tok] - logsumexp(x)`` over
        # the flat stream. No pad tokens exist anywhere on this path.
        h_pred = hidden[0].index_select(0, packed_predict_index)  # [T_total, H]
        targets = response_tokens

        def _flat_logp_chunk(h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
            lf = self.lm_head(h).float() / T  # [chunk, vocab] FP32
            return lf.gather(-1, tok.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1)

        flat_parts: List[torch.Tensor] = []
        flat_chunk = 2048
        for s in range(0, int(h_pred.size(0)), flat_chunk):
            h = h_pred[s : s + flat_chunk]
            tok = targets[s : s + flat_chunk]
            if torch.is_grad_enabled() and h.requires_grad:
                flat_parts.append(checkpoint(_flat_logp_chunk, h, tok, use_reentrant=False))
            else:
                flat_parts.append(_flat_logp_chunk(h, tok))
        if not flat_parts:
            return hidden.new_zeros((0,), dtype=torch.float32)
        return torch.cat(flat_parts, dim=0)
    T_max = int(response_tokens.size(1))
    resp_hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + T_max, :]

    def _logp_chunk(h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        lf = self.lm_head(h).float() / T  # [B, chunk, vocab] FP32
        chosen = lf.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        return chosen - torch.logsumexp(lf, dim=-1)

    bsz = resp_hidden.size(0)
    chunk = max(64, 2048 // max(1, bsz))
    parts: List[torch.Tensor] = []
    for s in range(0, T_max, chunk):
        h = resp_hidden[:, s : s + chunk, :]
        tok = response_tokens[:, s : s + chunk]
        if torch.is_grad_enabled() and h.requires_grad:
            parts.append(checkpoint(_logp_chunk, h, tok, use_reentrant=False))
        else:
            parts.append(_logp_chunk(h, tok))
    if not parts:
        return resp_hidden.new_zeros((bsz, 0), dtype=torch.float32)  # T_max == 0
    return torch.cat(parts, dim=1)


# Attention backends with a sparse packed kernel (skip cross-sequence blocks):
# flex via BlockMask, flash_attention_2 via flash_attn_varlen + cu_seqlens. Under
# either, packed replay is always a win; on any other backend (sdpa/eager) it is
# gated on length variance (see packed_replay).
@dataclass
class Qwen3ARParams:
    """Per-request AR-mode knobs for Qwen3.

    ``stop_token_ids`` is unioned with ``tokenizer.eos_token_id`` inside
    the stage so callers don't need to repeat the EOS id.
    """

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class Qwen3ARStep(ARStep):
    """Per-token sampling kernel.

    Implements the ``ARStep`` Protocol: given logits over the vocabulary
    at the current position, sample the next token and return its
    elementwise log-probability under the *temperature-scaled* full
    softmax (computed before top-k/top-p truncation), so it matches the
    replay-time ``log_softmax(logits / T)`` without filter masking.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"Qwen3ARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        if self.temperature <= 0.0:
            # Greedy: argmax under the full (untempered) softmax.
            log_probs_full = F.log_softmax(logits.float(), dim=-1)
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = logits.float() / self.temperature

        # Behavior log-prob under the temperature-scaled distribution, computed
        # BEFORE top-k/top-p truncation so it matches Qwen3ARStage.replay's
        # log_softmax(logits / T): old_logp == replay new_logp at the same
        # weights. Storing the untempered log_softmax(logits) only matched when
        # T == 1 (mirrors the QwenVL behavior-logprob fix in #165).
        log_probs_full = F.log_softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < scaled.shape[-1]:
            topk_vals, _ = torch.topk(scaled, self.top_k, dim=-1)
            kth = topk_vals[..., -1, None]
            scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

        if self.top_p < 1.0:
            sorted_vals, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
            cumprob = torch.softmax(sorted_vals, dim=-1).cumsum(dim=-1)
            cutoff = (cumprob > self.top_p).float()
            cutoff = torch.cat([torch.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], dim=-1)
            mask = cutoff > 0
            sorted_vals = sorted_vals.masked_fill(mask, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_vals)

        probs = F.softmax(scaled, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, log_prob


class Qwen3ARStage(ARStage[Qwen3ARConditions]):
    """Rollout-level AR stage for Qwen3.

    Drives :class:`AutoModelForCausalLM` through
    ``prepare_inputs_for_generation`` + per-token forward with KV cache,
    samples via :class:`Qwen3ARStep`, and packs per-sample
    ``(tokens, log_probs)`` into a varlen :class:`TextSegment`.
    """

    def __init__(
        self,
        *,
        model: Qwen3Bundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        # ``replay`` runs the transformer forward under an explicit autocast
        # scope so softmax / layer_norm stay FP32 (mirrors SD3DiffusionStage);
        # ``logprob_dtype`` then forces the per-token log-prob into FP32 so
        # the GRPO ratio / clip math starts from FP32.
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="Qwen3ARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="Qwen3ARStage.logprob_precision")
        # Install the dual-mode forward (see ``_replay_aware_forward``) as an
        # INSTANCE attribute: it wins over the class forward, survives the
        # FSDP2 class swap (only ``__class__`` changes) and LoRA injection,
        # and is idempotent via the ``__func__`` identity check.
        transformer = model.transformer
        if getattr(transformer.forward, "__func__", None) is not _replay_aware_forward:
            transformer.forward = MethodType(_replay_aware_forward, transformer)

    def trainable_module(self) -> "torch.nn.Module":
        """Return the HF causal LM module — the FSDP/LoRA wrap target.

        Required by the Policy-chain contract (LoRAPolicy / FSDPPolicy
        call ``source.trainable_module()`` to find the module they wrap).
        The Qwen3Bundle composes a transformer + tokenizer; the
        transformer is the only trainable component.
        """
        return self.model.transformer

    def autoregress(
        self,
        conditions: Qwen3ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[Qwen3ARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation. Returns a varlen-packed ``TextSegment``."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError(
                "Qwen3ARStage.autoregress: requires conditions.prompt.input_ids — "
                "produced by Qwen3ChatTemplateStage.embed(...)."
            )
        if conditions.prompt.attention_mask is None:
            raise ValueError(
                "Qwen3ARStage.autoregress: requires conditions.prompt.attention_mask — "
                "produced by Qwen3ChatTemplateStage.embed(...)."
            )

        transformer = self.model.transformer
        input_ids: torch.Tensor = conditions.prompt.input_ids
        attention_mask: torch.Tensor = conditions.prompt.attention_mask
        device = input_ids.device

        # Qwen3ChatTemplateStage right-pads prompts to the in-batch max. The
        # decode loop below reads ``logits[:, -1, :]`` and appends each new token
        # at the end, which is only correct when the last column is a row's last
        # *real* token — i.e. for an equal-length batch. Re-pad to LEFT here so
        # mixed-length batches decode correctly too (no-op when already equal
        # length, e.g. the same-prompt-group recipe). HF derives the right
        # ``position_ids`` from the left-padded ``attention_mask``.
        pad_id = self.model.tokenizer.pad_token_id or 0
        input_ids, attention_mask = left_pad_prompt(input_ids, attention_mask, pad_id)
        batch_size = int(input_ids.shape[0])

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = Qwen3ARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_new_tokens)

        # HF transformers >= 4.47 require ``cache_position`` to be present in
        # model_kwargs across the per-token loop (``_update_model_kwargs_for_generation``
        # reads model_kwargs["cache_position"][-1:] and bumps it by num_new_tokens).
        # Mirror what ``GenerationMixin._get_initial_cache_position`` would do.
        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": None,
            "cache_position": torch.arange(int(input_ids.shape[1]), device=device, dtype=torch.long),
        }
        cur_input_ids = input_ids

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        # NOTE: this decode loop calls the ROOT module every token, so FSDP
        # hooks fire normally in both wrap modes. Per-block groups re-gather
        # per step according to ``reshard_after_forward``; under the default
        # root wrap the leftover group (embed / final norm / lm_head) stays
        # gathered after the first step — the root group never reshards after
        # forward — so only the blocks pay per-token gather traffic. The
        # transformer's ``forward`` is the patched dual-mode function
        # (``_replay_aware_forward``); without ``response_tokens`` it delegates
        # to the stock class forward, so this loop is unchanged.
        for _ in range(max_new):
            model_inputs = transformer.prepare_inputs_for_generation(
                cur_input_ids,
                past_key_values=model_kwargs.get("past_key_values"),
                attention_mask=model_kwargs.get("attention_mask"),
                cache_position=model_kwargs.get("cache_position"),
                use_cache=True,
            )
            with torch.no_grad():
                out = transformer(**model_inputs, return_dict=True)
            logits = out.logits
            next_logits = logits[:, -1, :]
            if next_logits.device != device:
                next_logits = next_logits.to(device)

            token_id, log_prob = step.step(next_logits)
            for b in range(batch_size):
                if finished[b]:
                    continue
                tid = int(token_id[b].item())
                generated_tokens[b].append(tid)
                per_token_logps[b].append(float(log_prob[b].item()))
                if tid in stop_ids:
                    finished[b] = True
            # Synchronize finished status across all FSDP ranks. If any rank
            # still has unfinished samples, all ranks must keep running forward
            # passes (FSDP AllGather requires every rank to participate), so the
            # collective runs every step under a real multi-rank group. With no
            # process group (single-process / tests) the local view is final.
            local_done = all(finished)
            if dist.is_initialized() and dist.get_world_size() > 1:
                done = torch.tensor([1 if local_done else 0], device=device)
                dist.all_reduce(done, op=dist.ReduceOp.MIN)
                local_done = done.item() == 1
            if local_done:
                break

            cur_input_ids = torch.cat([cur_input_ids, token_id.unsqueeze(-1)], dim=1)
            model_kwargs = transformer._update_model_kwargs_for_generation(out, model_kwargs)
            model_kwargs["use_cache"] = True
        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    def replay(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Per-token log-prob replay over a stored rollout segment.

        Branch: prefer :meth:`packed_replay` (packed-varlen, zero padding, B > 1)
        and fall back to :meth:`padding_replay` (the dense ``[B, P_max + T_max]``
        padded path) when packing does not apply. Returns packed varlen
        ``[total_tokens]`` aligned with ``segment.log_probs``; caller controls
        grad / ``.train()`` scope. ``temperature`` divides logits before
        ``log_softmax`` to match SGLang's sampler (``1.0`` is a no-op).
        """
        attn_impl = getattr(getattr(self.model.transformer, "config", None), "_attn_implementation", None)
        if _packed_replay_supported(attn_impl):
            packed = self.packed_replay(conditions, segment=segment, temperature=temperature)
            if packed is not None:
                return packed
        return self.padding_replay(conditions, segment=segment, temperature=temperature)

    def packed_replay(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Packed-varlen replay (B > 1): zero padding anywhere.

        Concatenate every sample's REAL prompt tokens + its flat response tokens
        into ONE row; position_ids restart at 0 per sequence, and with
        ``attention_mask=None`` transformers' masking_utils detects the packed
        layout from the restarting positions and builds the block-causal mask
        (verl remove_padding equivalence). Per-sequence RoPE positions equal the
        standalone layout, so logp semantics match :meth:`padding_replay`.
        Returns ``None`` (→ padding fallback) when packing does not apply or
        would not pay: single sample, no per-sample lengths, no packed-mask
        support (:func:`_packed_replay_supported`), or no sparse-block attention
        kernel in use (only flex_attention / flash_attention_2 make packed a win).
        """
        if conditions.prompt is None or conditions.prompt.input_ids is None or conditions.prompt.attention_mask is None:
            return None
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            return None
        device = next(self.model.transformer.parameters()).device
        pad_id = self.model.tokenizer.pad_token_id or 0
        attn_impl = getattr(getattr(self.model.transformer, "config", None), "_attn_implementation", None)
        layout = build_packed_replay_layout(
            prompt_ids=conditions.prompt.input_ids,
            prompt_mask=conditions.prompt.attention_mask,
            segment=segment,
            device=device,
            pad_id=pad_id,
            caller="Qwen3ARStage.packed_replay",
            pad_to_multiple=1024 if attn_impl == "flex_attention" else None,
        )
        if layout is None:
            return None
        per_token_flat = self.model.transformer(
            input_ids=layout.input_ids,
            attention_mask=None,
            position_ids=layout.position_ids,
            response_tokens=layout.response_tokens,
            packed_predict_index=layout.predict_index,
            prompt_len=0,
            temperature=temperature,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )
        return per_token_flat.to(dtype=self.logprob_dtype)

    def padding_replay(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Dense ``[B, P_max + T_max]`` padded replay — the default / fallback path.

        One teacher-forced forward over padded ``prompt + response``; gather
        log-probs at the predicting positions per response token; return packed
        varlen ``[total_tokens]``. ``temperature`` divides logits before
        ``log_softmax`` to match SGLang's sampler (``1.0`` is a no-op).
        """
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3ARStage.replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3ARStage.replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "Qwen3ARStage.replay: segment requires tokens with framework-managed "
                "cu_seqlens (construct via TextSegment.pack)"
            )

        device = next(self.model.transformer.parameters()).device
        pad_id = self.model.tokenizer.pad_token_id or 0
        layout = build_padded_replay_layout(
            prompt_ids=conditions.prompt.input_ids,
            prompt_mask=conditions.prompt.attention_mask,
            segment=segment,
            device=device,
            pad_id=pad_id,
            caller="Qwen3ARStage.padding_replay",
        )

        # Replay goes through the patched dual-mode ``forward`` (see
        # ``_replay_aware_forward``): the decoder body + chunked lm_head run
        # INSIDE the root forward, so this is topology-independent (FSDP2
        # root-wrapped or plain) and never materializes [B, L, vocab] logits.
        # The cuda-vs-cpu autocast decision lives here; dtype validity and the
        # autocast scope live in the patched forward.
        per_token = self.model.transformer(
            input_ids=layout.input_ids,
            attention_mask=layout.attention_mask,
            position_ids=layout.position_ids,
            response_tokens=layout.response_tokens,
            prompt_len=layout.prompt_len,
            temperature=temperature,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )  # [B, T_max] FP32
        return pack_padded_token_outputs(per_token, layout.lengths).to(dtype=self.logprob_dtype)

    def _resolve_stop_ids(
        self,
        params: Optional[Qwen3ARParams],
        sampling_params: ARSamplingParams,
    ) -> List[int]:
        ids: List[int] = []
        if params is not None and params.stop_token_ids:
            ids.extend(int(t) for t in params.stop_token_ids)
        if sampling_params.stop_token_id is not None:
            ids.append(int(sampling_params.stop_token_id))
        eos = self.model.tokenizer.eos_token_id
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                ids.extend(int(t) for t in eos)
            else:
                ids.append(int(eos))
        # Deduplicate while preserving order.
        seen: set = set()
        out: List[int] = []
        for t in ids:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def _pack_text_segment(
    generated_tokens: List[List[int]],
    per_token_logps: List[List[float]],
    *,
    device: torch.device,
) -> TextSegment:
    """Pack per-sample lists of tokens / log-probs into a varlen ``TextSegment``."""
    return TextSegment.pack(
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["Qwen3ARParams", "Qwen3ARStage", "Qwen3ARStep"]
