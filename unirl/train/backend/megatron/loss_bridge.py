"""Megatron forward_step loss bridge (M0).

The genuinely-fresh, UniRL-specific glue: it makes the mcore ``forward_step`` path
produce per-token log-probs in the SAME packed ``[total_tokens]`` layout that
``ARStage.replay`` produces on the FSDP path, so the shared ``algorithm.compute_loss``
clip math is byte-interchangeable between the two worlds.

Faithful to two references:
  * ``unirl/models/qwen3/ar.py::packed_replay`` (:469-522) — the prompt+response
    stream assembly and the ``predict_index`` construction.
  * ``unirl/models/qwen3/ar.py::_replay_aware_forward`` (:141-159, packed path) —
    the exact per-token logp kernel: ``lf = logits.float()/T; lf.gather(-1,tok) -
    logsumexp(lf,-1)`` chunked at 2048. Reproduced here on the mcore GPTModel's
    logits (which already ran the output layer) instead of ``hidden`` + ``lm_head``.
  * slime ``get_batch`` / ``loss_function`` — the mcore THD ``PackedSeqParams``
    contract and the ``(loss, normalizer, log_dict)`` return shape.

M0 is tp=pp=ep=1: the vocab is un-sharded, so the logp math needs no TP
vocab-parallel all-reduce (that is M1 — marked below). VERIFY every mcore symbol
(``PackedSeqParams``, model forward kwargs) against the pinned mcore version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Tuple

import torch

if TYPE_CHECKING:
    from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
    from unirl.types.rollout_resp import RolloutTrack

_LOGP_CHUNK = 2048


class _MicroDataIterator:
    """Yield one micro per Megatron microbatch call.

    UniRL ``micros`` are already contiguous ``(start, end)`` ranges (the planner
    sort-then-slices), so this is a thin cursor: ``num_microbatches = len(micros)``
    drives how many times ``forward_step`` pulls ``next_micro()``.
    """

    def __init__(self, track: "RolloutTrack", micros: List[Tuple[int, int]], *, update_total: int, bs: int) -> None:
        self.track = track
        self.micros = micros
        self.i = 0
        self.update_total = update_total
        self.single = len(micros) == 1 and micros[0] == (0, bs)

    def next_micro(self) -> Tuple["RolloutTrack", float]:
        start, end = self.micros[self.i]
        self.i += 1
        micro = self.track if self.single else self.track.slice(start, end)
        loss_scale = (end - start) / float(self.update_total)  # == TrainStack loss_scale
        return micro, loss_scale


def build_packed_micro_batch(micro_track: "RolloutTrack", *, pad_token_id: int, tp_pad: int = 1) -> Mapping[str, Any]:
    """Assemble the mcore THD packed batch from UniRL's split prompt/response.

    UniRL packs the RESPONSE varlen (``segment.tokens [total_resp]`` +
    ``segment.cu_seqlens``) but keeps the PROMPT padded-separate
    (``conditions.prompt.input_ids [B, P_max]`` + ``attention_mask``). This
    reconstructs the full ``prompt+response`` stream per sample exactly as
    ``packed_replay`` does, then emits mcore's THD contract (``cu_seqlens`` +
    ``PackedSeqParams``) instead of HF restarting position_ids.

    ``predict_index`` is in segment order and aligns 1:1 with
    ``segment.log_probs`` — this is what keeps the extracted ``new_logp``
    byte-layout-identical to ``stage.replay``'s output.
    """
    seg = micro_track.segment
    prompt = micro_track.conditions["prompt"]
    device = seg.tokens.device
    prompt_ids = prompt.input_ids.to(device)
    real_prompt_lens = prompt.attention_mask.to(device).long().sum(dim=-1)  # [B], right-padded
    cu_p = [int(c) for c in seg.cu_seqlens.tolist()]
    lengths = [int(n) for n in seg.lengths.tolist()]
    flat_resp = seg.tokens.to(dtype=torch.long)
    batch_size = int(prompt_ids.shape[0])

    streams: List[torch.Tensor] = []
    pred_parts: List[torch.Tensor] = []
    cu = [0]
    offset = 0
    for b in range(batch_size):
        n_p = int(real_prompt_lens[b].item())
        n_r = lengths[b]
        # (offset + n_p - 1) assumes >=1 real prompt token; n_p == 0 would gather the
        # PRIOR stream's last position -> silent cross-sequence logp corruption.
        assert n_p >= 1, "build_packed_micro_batch: stream has 0 real prompt tokens"
        seq = torch.cat([prompt_ids[b, :n_p], flat_resp[cu_p[b] : cu_p[b] + n_r]])
        streams.append(seq)
        if n_r > 0:
            pred_parts.append(torch.arange(offset + n_p - 1, offset + n_p - 1 + n_r, device=device))
        cu.append(cu[-1] + int(seq.numel()))
        offset += int(seq.numel())

    tokens = torch.cat(streams)  # [T]
    predict_index = torch.cat(pred_parts) if pred_parts else torch.zeros(0, dtype=torch.long, device=device)

    # Optional TE alignment pad: append filler as an isolated trailing "sequence"
    # (own cu entry, carries no predict_index -> contributes no logp). At tp=1
    # tp_pad is typically 1 (no-op). Keep the fillers out of predict_index.
    total = int(tokens.numel())
    pad = (tp_pad - total % tp_pad) % tp_pad if tp_pad > 1 else 0
    if pad:
        tokens = torch.cat([tokens, torch.full((pad,), pad_token_id, dtype=tokens.dtype, device=device)])
        cu.append(cu[-1] + pad)

    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    seqlens = (cu_t[1:] - cu_t[:-1]).max()
    max_seqlen = int(seqlens.item()) if cu_t.numel() > 1 else 0

    # VERIFY: PackedSeqParams import path + field names against the pinned mcore.
    from megatron.core.packed_seq_params import PackedSeqParams

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_t,
        cu_seqlens_kv=cu_t,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
    return {
        "tokens": tokens.unsqueeze(0),  # [1, T]
        "packed_seq_params": packed_seq_params,
        "predict_index": predict_index,  # [total_resp_tokens], segment order
        "response_tokens": flat_resp,  # [total_resp_tokens] targets, segment order
        "old_logp": seg.log_probs,  # for the step-0 ratio assert
    }


def extract_packed_new_logp(logits: torch.Tensor, *, predict_index: torch.Tensor,
                            response_tokens: torch.Tensor, temperature: float) -> torch.Tensor:
    """mcore logits [1,T,V] (or [T,1,V]) -> per-token new_logp [total_tokens], fp32.

    Byte-faithful to ``ar.py:_replay_aware_forward`` packed path: gather the logits
    at ``predict_index``, then chunked fp32 ``(lf/T)[tok] - logsumexp(lf/T)``.

    M0 (tp=1): the vocab is un-sharded so ``logits`` are full and this is exact.
    M1 (tp>1): ``logits`` are vocab-parallel (column-sharded); ``.gather`` and
    ``logsumexp`` must then run under an all-reduce over
    ``mpu.get_tensor_model_parallel_group()`` — the ONLY change needed here.
    """
    logits = logits.squeeze(0).float().contiguous()  # [T, V]
    h = logits.index_select(0, predict_index)  # [total_tokens, V]
    T = float(temperature) if float(temperature) > 0.0 else 1.0
    parts: List[torch.Tensor] = []
    for s in range(0, int(h.size(0)), _LOGP_CHUNK):
        lf = h[s : s + _LOGP_CHUNK] / T
        tok = response_tokens[s : s + _LOGP_CHUNK]
        parts.append(lf.gather(-1, tok.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1))
    if not parts:
        return logits.new_zeros((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def assert_ratio_near_one(new_logp: torch.Tensor, old_logp: torch.Tensor, *, tol: float = 0.25) -> None:
    """Step-0 on-policy sanity: train weights == rollout weights on the first
    update, so ``new_logp`` (mcore bridge) must ≈ ``old_logp`` (rollout emission).
    Loose ``tol`` tolerates the genuine mcore-vs-SGLang engine gap while catching
    gross bridge drift (predict_index off-by-one, temperature, THD RoPE, fp32/vocab
    layout) which produce Δ ≫ 1. The sharpest silent-ratio-drift guard the bridge has.
    """
    if new_logp.numel() == 0:
        return
    old = old_logp.to(dtype=new_logp.dtype, device=new_logp.device)
    absdiff = (new_logp - old).abs()
    mean_d = float(absdiff.mean())
    if mean_d > tol:
        j = int(absdiff.argmax())
        raise AssertionError(
            f"[megatron loss_bridge] step-0 mean|new_logp-old_logp|={mean_d:.4f} > {tol} "
            f"(max={float(absdiff.max()):.4f} at token {j}: new={float(new_logp[j]):.4f} "
            f"old={float(old[j]):.4f}). Likely bridge drift: predict_index off-by-one, "
            "temperature mismatch, THD RoPE/position, or fp32/vocab layout."
        )


def make_loss_closure(*, algorithm: "StageAlgorithm", micro_track: "RolloutTrack", batch: Mapping[str, Any],
                      training_progress: float, loss_scale: float, num_microbatches: int,
                      temperature: float, first_update: bool, sink: List["AlgorithmStepResult"]):
    """Build the mcore ``loss_function``-contract closure that get_forward_backward_func calls.

    It extracts ``new_logp``, (step-0) asserts the ratio, delegates the clip math to
    ``algorithm.compute_loss``, records the per-micro ``AlgorithmStepResult`` into
    ``sink`` (pp=1: same-rank, byte-simplest), and returns the mcore 3-tuple.

    loss_scale reconciliation (see the M0 plan): the closure returns
    ``loss · loss_scale · num_microbatches``. The ``· num_microbatches`` cancels
    mcore's internal ``÷ num_microbatches`` (calculate_per_token_loss=False), leaving
    the per-micro backward = ``loss · loss_scale`` and the DP-average intact — exactly
    the FSDP reference. Do NOT apply slime's ``· dp_world_size / step_global_batch_size``.
    """
    from unirl.algorithms import AlgorithmStepResult

    def loss_closure(logits: torch.Tensor):
        new_logp = extract_packed_new_logp(
            logits, predict_index=batch["predict_index"],
            response_tokens=batch["response_tokens"], temperature=temperature,
        )
        if first_update:
            assert_ratio_near_one(new_logp, batch["old_logp"])

        loss, metrics, n = algorithm.compute_loss(
            forward_output=new_logp,
            conditions=micro_track.conditions,
            segment=micro_track.segment,
            advantages=micro_track.advantages,
            training_progress=training_progress,
        )
        sink.append(AlgorithmStepResult(
            loss=float(loss.detach().item()), metrics=metrics, num_steps_or_tokens=n, has_backward=True,
        ))

        scaled = loss * loss_scale * num_microbatches
        if new_logp.numel() == 0:  # keep the empty micro grad-connected
            scaled = scaled + 0.0 * logits.sum()
        # mcore 3-tuple: (loss, normalizer, log_dict). normalizer=1 == not per-token;
        # metrics ride `sink` at M0, so log_dict carries only the placeholder mcore expects.
        return scaled, torch.tensor(1, device=logits.device), {"keys": [], "values": logits.new_zeros(1)}

    return loss_closure
