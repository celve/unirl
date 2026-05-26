"""Qwen3 AR stage: typed params + per-token kernel + rollout-level stage.

Three classes:

- ``Qwen3ARParams`` — typed request-shape knobs (max_tokens / temperature
  / top_p / top_k / stop_token_ids).
- ``Qwen3ARStep`` — per-token sampling kernel (reads logits, returns
  ``(token_id, log_prob)``). Verbatim mechanics from
  :class:`diffusionrl.models.hunyuan_image3.ar.HunyuanImage3ARStep`.
- ``Qwen3ARStage`` — implements ``ARStage[Qwen3ARConditions]``. Drives
  HF :class:`AutoModelForCausalLM` through a per-token loop with KV
  cache, packs the results into a varlen :class:`TextSegment` with
  per-step full-softmax log-probs. ``replay`` recomputes per-token
  log-probs from stored rollout tokens via a single teacher-forced
  forward — the GRPO/PPO substitution point.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusionrl.models.types.ar import ARSamplingParams, ARStage, ARStep
from diffusionrl.types.segments import TextSegment

from .bundle import Qwen3Bundle
from .conditions import Qwen3ARConditions


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
    elementwise log-probability under the *full* softmax (so it's
    directly comparable to a replay-time full-softmax log-prob without
    filter masking).
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

        log_probs_full = F.log_softmax(logits.float(), dim=-1)

        if self.temperature <= 0.0:
            # Greedy: argmax under the full softmax, log-prob from the same.
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = logits.float() / self.temperature

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

    def __init__(self, *, model: Qwen3Bundle) -> None:
        self.model = model

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
            if all(finished):
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
    ) -> torch.Tensor:
        """Per-token log-prob replay over a stored rollout segment.

        One teacher-forced forward over ``prompt + response`` (no KV
        cache), gather full-softmax log-probs at the predicting positions
        for each response token, return packed varlen ``[total_tokens]``
        aligned with ``segment.log_probs``. Caller controls grad / no_grad
        scope and ``.train()`` mode. Empty-response samples contribute
        zero tokens to the output.
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

        prompt_ids = conditions.prompt.input_ids
        prompt_mask = conditions.prompt.attention_mask
        device = prompt_ids.device
        batch_size = int(prompt_ids.shape[0])
        prompt_len = int(prompt_ids.shape[1])

        lengths = [int(n) for n in segment.lengths.tolist()]
        T_max = max(lengths) if lengths else 0
        pad_id = self.model.tokenizer.pad_token_id or 0
        response_tokens = torch.full((batch_size, T_max), pad_id, dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        if T_max > 0:
            full_ids = torch.cat([prompt_ids, response_tokens], dim=1)
            full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        out = self.model.transformer(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits

        if T_max == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)

        # logits[:, prompt_len - 1 + t, :] predicts response_tokens[:, t].
        pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + T_max, :]
        log_probs_full = F.log_softmax(pred_logits.float(), dim=-1)
        per_token = log_probs_full.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)

        flat: List[torch.Tensor] = []
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            flat.append(per_token[b, :n])
        if not flat:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.cat(flat, dim=0)

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
    n = len(generated_tokens)
    return TextSegment.pack(
        sample_indices=torch.arange(n, dtype=torch.long, device=device),
        positions=torch.zeros(n, dtype=torch.long, device=device),
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["Qwen3ARParams", "Qwen3ARStage", "Qwen3ARStep"]
