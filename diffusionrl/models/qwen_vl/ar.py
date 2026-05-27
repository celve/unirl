from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusionrl.models.types.ar import ARSamplingParams, ARStage, ARStep
from diffusionrl.types.segments import TextSegment

from .bundle import QwenVLBundle
from .conditions import QwenVLARConditions


@dataclass
class QwenVLARParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class QwenVLARStep(ARStep):
    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"QwenVLARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        log_probs_full = F.log_softmax(logits.float(), dim=-1)

        if self.temperature <= 0.0:
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


class QwenVLARStage(ARStage[QwenVLARConditions]):
    def __init__(self, *, model: QwenVLBundle) -> None:
        self.model = model

    def trainable_module(self) -> "torch.nn.Module":
        return self.model.transformer

    def autoregress(
        self,
        conditions: QwenVLARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[QwenVLARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("QwenVLARStage.autoregress: requires conditions.prompt.input_ids")
        if conditions.prompt.attention_mask is None:
            raise ValueError("QwenVLARStage.autoregress: requires conditions.prompt.attention_mask")

        transformer = self.model.transformer
        input_ids: torch.Tensor = conditions.prompt.input_ids
        attention_mask: torch.Tensor = conditions.prompt.attention_mask
        device = input_ids.device
        batch_size = int(input_ids.shape[0])

        # Reset stale rope_deltas from any prior forward/generate call
        transformer.model.rope_deltas = None

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = QwenVLARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_new_tokens)

        model_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": None,
            "cache_position": torch.arange(int(input_ids.shape[1]), device=device, dtype=torch.long),
        }

        if conditions.pixel_values is not None:
            model_kwargs["pixel_values"] = conditions.pixel_values
        if conditions.image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = conditions.image_grid_thw
        if conditions.mm_token_type_ids is not None:
            model_kwargs["mm_token_type_ids"] = conditions.mm_token_type_ids

        cur_input_ids = input_ids

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        is_first_step = True

        for _ in range(max_new):
            prep_kwargs: Dict[str, Any] = {
                "past_key_values": model_kwargs.get("past_key_values"),
                "attention_mask": model_kwargs.get("attention_mask"),
                "cache_position": model_kwargs.get("cache_position"),
                "use_cache": True,
            }
            if is_first_step:
                if "pixel_values" in model_kwargs:
                    prep_kwargs["pixel_values"] = model_kwargs["pixel_values"]
                if "image_grid_thw" in model_kwargs:
                    prep_kwargs["image_grid_thw"] = model_kwargs["image_grid_thw"]
                if "mm_token_type_ids" in model_kwargs:
                    prep_kwargs["mm_token_type_ids"] = model_kwargs["mm_token_type_ids"]
                prep_kwargs["is_first_iteration"] = True
            else:
                prep_kwargs["is_first_iteration"] = False

            model_inputs = transformer.prepare_inputs_for_generation(cur_input_ids, **prep_kwargs)

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
            is_first_step = False

        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    def replay(
        self,
        conditions: QwenVLARConditions,
        *,
        segment: TextSegment,
    ) -> torch.Tensor:
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("QwenVLARStage.replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("QwenVLARStage.replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("QwenVLARStage.replay: segment requires tokens with cu_seqlens")

        prompt_ids = conditions.prompt.input_ids
        prompt_mask = conditions.prompt.attention_mask
        device = prompt_ids.device
        batch_size = int(prompt_ids.shape[0])
        prompt_len = int(prompt_ids.shape[1])

        # Reset stale rope_deltas — critical for correct M-RoPE position IDs
        self.model.transformer.model.rope_deltas = None

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

        full_mm_type_ids = None
        if conditions.mm_token_type_ids is not None:
            if T_max > 0:
                response_types = torch.zeros(
                    (batch_size, T_max), dtype=conditions.mm_token_type_ids.dtype, device=device
                )
                full_mm_type_ids = torch.cat([conditions.mm_token_type_ids, response_types], dim=1)
            else:
                full_mm_type_ids = conditions.mm_token_type_ids

        forward_kwargs: Dict[str, Any] = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if conditions.pixel_values is not None:
            forward_kwargs["pixel_values"] = conditions.pixel_values
        if conditions.image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = conditions.image_grid_thw
        if full_mm_type_ids is not None:
            forward_kwargs["mm_token_type_ids"] = full_mm_type_ids

        # Compute correct 4D position_ids for M-RoPE.
        # Rollout uses prepare_inputs_for_generation which produces [4, bs, seq]:
        #   row 0 = text positions (for causal mask), rows 1-3 = M-RoPE (temporal, height, width).
        # Direct forward with position_ids=None only produces [3, bs, seq] (no text_position_ids),
        # causing incorrect causal mask for multimodal inputs.
        # Fix: call get_rope_index ourselves and prepend text_positions.
        vision_pos, _ = self.model.transformer.model.get_rope_index(
            full_ids,
            image_grid_thw=conditions.image_grid_thw,
            attention_mask=full_mask,
        )  # [3, bs, seq]
        text_pos = full_mask.long().cumsum(-1) - 1
        text_pos.masked_fill_(full_mask == 0, 1)
        forward_kwargs["position_ids"] = torch.cat([text_pos[None], vision_pos], dim=0)  # [4, bs, seq]

        out = self.model.transformer(**forward_kwargs)
        logits = out.logits

        if T_max == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)

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
        params: Optional[QwenVLARParams],
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
    n = len(generated_tokens)
    return TextSegment.pack(
        sample_indices=torch.arange(n, dtype=torch.long, device=device),
        positions=torch.zeros(n, dtype=torch.long, device=device),
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["QwenVLARParams", "QwenVLARStage", "QwenVLARStep"]
