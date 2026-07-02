"""Megatron forward_step loss bridge (M0, padded forward).

Makes the mcore forward produce per-token log-probs in UniRL's packed
``[total_tokens]`` layout (segment order, aligned 1:1 with ``segment.log_probs``),
so the shared ``algorithm.compute_loss`` clip math is byte-interchangeable between
the FSDP ``stage.replay`` path and the mcore path.

Padded (not THD) because flash-attn varlen is unavailable in this env: build a
padded ``[B, L]`` batch (real prompt + response, right-padded), plain causal mask
(trailing pad is never attended by real queries), gather logits at the positions
that predict each response token. The per-token logp kernel mirrors
``unirl/models/qwen3/ar.py`` (``lf = logits.float()/T; lf.gather(-1,tok) -
logsumexp``). Forward correctness validated vs HF (probe stage3, argmax 1.000).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Tuple

import torch

if TYPE_CHECKING:
    from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
    from unirl.types.rollout_resp import RolloutTrack

_LOGP_CHUNK = 2048


class _MicroDataIterator:
    """One micro per Megatron microbatch call (``num_microbatches = len(micros)``)."""

    def __init__(self, track: "RolloutTrack", micros: List[Tuple[int, int]], *, update_total: int, bs: int) -> None:
        self.track = track
        self.micros = micros
        self.i = 0
        self.update_total = update_total
        self.single = len(micros) == 1 and micros[0] == (0, bs)

    def __next__(self) -> Tuple["RolloutTrack", float]:
        start, end = self.micros[self.i]
        self.i += 1
        micro = self.track if self.single else self.track.slice(start, end)
        return micro, (end - start) / float(self.update_total)


def build_padded_micro_batch(micro_track: "RolloutTrack", *, pad_token_id: int) -> Mapping[str, Any]:
    """Assemble a padded ``[B, L]`` mcore batch from UniRL's split prompt/response.

    Prompt is padded ``conditions.prompt.input_ids [B, P_max]`` + attention_mask;
    response is packed ``segment.tokens [total_resp]`` + ``cu_seqlens``. Per sample:
    ``[real prompt tokens, response tokens, pad...]``. Records per-sample
    ``(prompt_len, resp_len)`` so the logp gather aligns with ``segment.log_probs``.
    """
    seg = micro_track.segment
    prompt = micro_track.conditions["prompt"]
    device = seg.tokens.device
    prompt_ids = prompt.input_ids.to(device)
    prompt_lens = prompt.attention_mask.to(device).long().sum(dim=-1)  # [B], right-padded
    cu_p = [int(c) for c in seg.cu_seqlens.tolist()]
    lengths = [int(n) for n in seg.lengths.tolist()]
    flat_resp = seg.tokens.to(dtype=torch.long)
    B = int(prompt_ids.shape[0])

    rows: List[torch.Tensor] = []
    pls, rls = [], []
    for b in range(B):
        n_p = int(prompt_lens[b].item())
        n_r = lengths[b]
        assert n_p >= 1, "build_padded_micro_batch: stream has 0 real prompt tokens"
        rows.append(torch.cat([prompt_ids[b, :n_p], flat_resp[cu_p[b] : cu_p[b] + n_r]]))
        pls.append(n_p)
        rls.append(n_r)

    L = max(int(r.numel()) for r in rows)
    ids = torch.full((B, L), pad_token_id, dtype=torch.long, device=device)
    for b, r in enumerate(rows):
        ids[b, : r.numel()] = r
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    mask = torch.triu(torch.ones(1, 1, L, L, device=device, dtype=torch.bool), diagonal=1)
    return {
        "ids": ids,
        "pos": pos,
        "mask": mask,
        "prompt_lens": pls,
        "resp_lens": rls,
        "response_tokens": flat_resp,  # segment order, aligned to segment.log_probs
        "old_logp": seg.log_probs,
    }


def extract_padded_new_logp(logits: torch.Tensor, *, prompt_lens: List[int], resp_lens: List[int],
                            response_tokens: torch.Tensor, temperature: float) -> torch.Tensor:
    """mcore logits -> per-token new_logp ``[total_tokens]`` (fp32, segment order).

    Response token j of sample b (abs pos ``prompt_len_b + j``) is predicted by the
    logit at ``prompt_len_b - 1 + j``. Gather that slice per sample, chunked fp32
    ``(lf/T)[tok] - logsumexp(lf/T)`` — byte-identical to the FSDP replay kernel.
    """
    # mcore GPTModel returns [B, L, V] or [L, B, V]; normalize to [B, L, V].
    if logits.dim() == 3 and logits.shape[0] != len(prompt_lens):
        logits = logits.transpose(0, 1)
    T = float(temperature) if float(temperature) > 0.0 else 1.0
    off = 0
    parts: List[torch.Tensor] = []
    for b, (pl, rl) in enumerate(zip(prompt_lens, resp_lens)):
        if rl <= 0:
            continue
        sl = pl - 1
        h = logits[b, sl : sl + rl].float()  # [rl, V]
        tok = response_tokens[off : off + rl]
        off += rl
        for s in range(0, rl, _LOGP_CHUNK):
            lf = h[s : s + _LOGP_CHUNK] / T
            t = tok[s : s + _LOGP_CHUNK]
            parts.append(lf.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1))
    if not parts:
        return logits.new_zeros((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def assert_ratio_near_one(new_logp: torch.Tensor, old_logp: torch.Tensor, *, tol: float = 0.35) -> None:
    """Step-0 on-policy sanity: new_logp (mcore) must ≈ old_logp (rollout). Loose
    tol tolerates the mcore-vs-SGLang engine gap while catching gross bridge drift.
    """
    if new_logp.numel() == 0:
        return
    old = old_logp.to(dtype=new_logp.dtype, device=new_logp.device)
    mean_d = (new_logp - old).abs().mean().item()
    if mean_d > tol:
        raise AssertionError(
            f"[megatron loss_bridge] step-0 mean|new_logp-old_logp|={mean_d:.4f} > {tol}: "
            "likely bridge drift (predict-index offset, temperature, or logp layout)."
        )


def make_loss_closure(*, algorithm: "StageAlgorithm", micro_track: "RolloutTrack", batch: Mapping[str, Any],
                      training_progress: float, loss_scale: float, num_microbatches: int,
                      temperature: float, first_update: bool, sink: List["AlgorithmStepResult"]):
    """mcore loss_function-contract closure. Returns ``loss·loss_scale·num_microbatches``
    (cancels mcore's ÷num_microbatches, keeps the DP-average → matches the FSDP grad).
    """
    from unirl.algorithms import AlgorithmStepResult

    def loss_closure(logits: torch.Tensor):
        new_logp = extract_padded_new_logp(
            logits, prompt_lens=batch["prompt_lens"], resp_lens=batch["resp_lens"],
            response_tokens=batch["response_tokens"], temperature=temperature,
        )
        if first_update:
            assert_ratio_near_one(new_logp, batch["old_logp"])
        loss, metrics, n = algorithm.compute_loss(
            forward_output=new_logp, conditions=micro_track.conditions,
            segment=micro_track.segment, advantages=micro_track.advantages,
            training_progress=training_progress,
        )
        sink.append(AlgorithmStepResult(
            loss=float(loss.detach().item()), metrics=metrics, num_steps_or_tokens=n, has_backward=True))
        scaled = loss * loss_scale * num_microbatches
        if new_logp.numel() == 0:
            scaled = scaled + 0.0 * logits.sum()
        return scaled, torch.tensor(1, device=logits.device), {"keys": [], "values": logits.new_zeros(1)}

    return loss_closure
