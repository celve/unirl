"""HunyuanImage3 AR stage: typed params + per-token kernel + rollout-level stage.

Three classes:

- ``HunyuanImage3ARParams`` — typed request-shape knobs (bot_task /
  max_tokens / temperature / top_p / top_k / stop_token_ids /
  cot_text / taylor_cache_*).
- ``HunyuanImage3ARStep`` — per-token sampling kernel (reads logits,
  returns ``(token_id, log_prob)``).
- ``HunyuanImage3ARStage`` — implements
  ``ARStage[HunyuanImage3ARConditions]``. Calls the shared backbone in
  ``mode="gen_text"`` to generate token sequences, packs the results
  into a varlen ``TextSegment`` with ``cu_seqlens`` + per-step
  ``log_probs``.

PR 3 lands the **single-pass** AR autoregress. The multi-pass chain
(``bot_task ∈ {think, recaption, think_recaption, img_ratio}``) lands
in PR 4 — its outer-loop logic mirrors
``modeling_hunyuan_image_3.py:3237-3396``. Image-vocab token spans
emitted by the AR stage (the ``<img>`` splice handled at upstream
``modeling_hunyuan_image_3.py:3111``) ride in the same ``tokens``
tensor; the consumer (the diffusion stage in t2i / it2i) extracts them
via ``TextSegment.as_condition_with(reembed)`` — also wired in PR 4.

``replay()`` recomputes per-token log-probs for a stored rollout's
response tokens via a single teacher-forced forward over
``prompt + response`` (no KV cache). Used by GRPO/PPO-style training to
get gradient-flowing ``π_θ(token_t | prefix)``. Rollout's stored
log-probs (``segment.log_probs``) are full-softmax (ar.py:101-102), so
they're directly comparable to replay's output for π_old / π_θ
substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusionrl.models_new.types.ar import ARSamplingParams, ARStage, ARStep
from diffusionrl.types.segments import TextSegment

from .bundle import HunyuanImage3Bundle
from .conditions import HunyuanImage3ARConditions


@dataclass
class HunyuanImage3ARParams:
    """Per-request AR-mode knobs for HunyuanImage 3.0.

    Sampling defaults match the vllm-omni stage configs at
    ``vllm-omni/vllm_omni/model_executor/stage_configs/hunyuan_image3_*.yaml``.

    ``system_prompt`` / ``use_system_prompt`` mirror upstream
    ``HunyuanImage3ForCausalMM.generate_image``'s
    ``get_system_prompt(use_system_prompt, bot_task, system_prompt)``
    flow. ``use_system_prompt`` selects a built-in preset
    (``en_vanilla`` / ``en_recaption`` / ``en_think_recaption`` / ``dynamic``
    / ``None``) and ``system_prompt`` is the explicit string used when
    ``use_system_prompt='custom'`` (or as a fallback under
    ``use_system_prompt='dynamic'``). The bare HunyuanImage3 model is
    not a chat model -- gen_text without a t2i-shaped system prompt
    produces incoherent / repetitive output.
    """

    bot_task: str = "auto"  # auto | think | recaption | think_recaption | img_ratio
    max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 1024
    stop_token_ids: List[int] = dc_field(default_factory=list)
    cot_text: Optional[str] = None

    # System-prompt knobs -- see class docstring.
    system_prompt: Optional[str] = None
    use_system_prompt: Optional[str] = None  # None -> read gen_config default

    # Taylor-cache acceleration knobs (forwarded to the model when supported).
    taylor_cache_interval: Optional[int] = None
    taylor_cache_order: Optional[int] = None


class HunyuanImage3ARStep(ARStep):
    """Per-token sampling kernel.

    Implements the ``ARStep`` Protocol: given logits over the vocabulary
    at the current position, sample the next token and return its
    elementwise log-probability. Honors ``temperature``, ``top_p``,
    ``top_k`` from the construction args.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one token from a ``[B, vocab]`` logits tensor.

        Returns ``(token_id [B], log_prob [B])``. ``log_prob`` is the
        post-filter log-probability of the sampled token under the
        full softmax (so it's directly comparable to a replay-time
        full-softmax log-prob without filter masking).
        """
        if logits.dim() != 2:
            raise ValueError(f"HunyuanImage3ARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        log_probs_full = F.log_softmax(logits.float(), dim=-1)
        scaled = logits.float() / max(self.temperature, 1e-6)

        # top-k filtering
        if self.top_k > 0 and self.top_k < scaled.shape[-1]:
            topk_vals, _ = torch.topk(scaled, self.top_k, dim=-1)
            kth = topk_vals[..., -1, None]
            scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

        # top-p filtering
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


class HunyuanImage3ARStage(ARStage[HunyuanImage3ARConditions]):
    """Rollout-level AR stage: ``HunyuanImage3ARConditions → TextSegment``.

    Calls the shared HunyuanImage3 backbone with ``mode="gen_text"`` to
    perform autoregressive token generation. The unified-sequence input
    comes from ``conditions.fused.input_ids`` (the chat-template-built
    token sequence), with optional cond-image scatter for i2t / it2i via
    ``conditions.cond_vit`` + ``conditions.fused.cond_vit_image_mask``.

    PR 3 ships **single-pass** generation only — the ``bot_task`` knob
    in ``HunyuanImage3ARParams`` is read for stop-token selection but
    no multi-pass orchestration is performed. PR 4 lands the full
    ``think → recaption → img_ratio`` chain.
    """

    def __init__(
        self,
        *,
        model: HunyuanImage3Bundle,
        step: Optional[HunyuanImage3ARStep] = None,
    ) -> None:
        self.model = model
        # The Step is rebuilt per-call from the request's sampling params
        # (so each request gets its own temperature / top-p / top-k);
        # ``step`` here is a class-level fallback used only when the
        # caller doesn't pass an ARSamplingParams.
        self._default_step = step

    def autoregress(
        self,
        conditions: HunyuanImage3ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[HunyuanImage3ARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation. Returns a varlen-packed ``TextSegment``.

        Drives the upstream chat-template-built sequence (carried in
        ``conditions.fused``) through ``transformer.prepare_inputs_for_generation``
        + ``_update_model_kwargs_for_generation`` per token. Required for
        ``tencent/HunyuanImage-3.0`` weights — the model expects
        ``input_ids`` in ``mode="gen_text"``, not ``inputs_embeds``.

        Stop-token policy: any token in ``params.stop_token_ids`` (if
        provided) terminates that sample's generation. Falls back to
        ``sampling_params.stop_token_id`` otherwise.
        """
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            raise ValueError(
                "HunyuanImage3ARStage.autoregress: requires "
                "conditions.fused.input_ids — produced by "
                "HunyuanImage3TextEmbedStage.embed_for_ar(...)."
            )
        if fused.attention_mask is None or fused.position_ids is None or fused.rope_cache is None:
            raise ValueError(
                "HunyuanImage3ARStage.autoregress: input_ids path requires "
                "fused.attention_mask / position_ids / rope_cache to be set "
                "by HunyuanImage3TextEmbedStage.embed_for_ar(...)."
            )

        transformer = self.model.transformer
        input_ids: torch.Tensor = fused.input_ids  # [B, L_prompt] long
        device = input_ids.device
        batch_size = int(input_ids.shape[0])

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = HunyuanImage3ARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_tokens)

        # Pre-build a ``HunyuanStaticCache`` sized for prompt + max_new_tokens,
        # mirroring upstream ``_prepare_model_inputs`` (hunyuan.py:2326-2333).
        # Falls back to ``None`` (HF default DynamicCache) when the upstream
        # symbol isn't accessible -- e.g. fake-bundle unit tests.
        prompt_len = int(input_ids.shape[1])
        past_kv_initial = self._build_kv_cache(transformer, batch_size=batch_size, max_cache_len=prompt_len + max_new)

        # i2t / it2i cond-vit fields — None for t2t. Reconstruct
        # ``vit_kwargs`` from the typed ``ImageEmbedCondition`` (the
        # upstream ViT module expects this dict shape).
        cond_vit = conditions.cond_vit
        cond_vit_images = cond_vit.embeds if cond_vit is not None else None
        vit_kwargs: Optional[Dict[str, Any]] = None
        if cond_vit is not None and (cond_vit.spatial_shapes is not None or cond_vit.attn_mask is not None):
            vit_kwargs = {
                "spatial_shapes": cond_vit.spatial_shapes,
                "attention_mask": cond_vit.attn_mask,
            }

        # Standard HF-style ``model_kwargs`` carried across the per-token
        # loop. Carries the rope tables, the 4D attention mask, and the
        # opaque tokenizer_output for the prefill ``_update_model_kwargs``
        # hook (which derives ``position_ids`` from ``real_pos`` for
        # right-padded batches).
        model_kwargs: Dict[str, Any] = {
            "mode": "gen_text",
            "attention_mask": fused.attention_mask,  # [B, 1, L, L] bool
            "position_ids": fused.position_ids,  # [B, L] long
            "custom_pos_emb": fused.rope_cache,  # ([B, L, D], [B, L, D])
            "use_cache": True,
            "past_key_values": past_kv_initial,
            "cond_vit_images": cond_vit_images,
            "cond_vit_image_mask": fused.cond_vit_image_mask,
            "vit_kwargs": vit_kwargs,
        }
        if conditions.tokenizer_output is not None:
            model_kwargs["tokenizer_output"] = conditions.tokenizer_output

        cur_input_ids = input_ids  # [B, T] long; T grows per step

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for step_idx in range(max_new):
            model_inputs = transformer.prepare_inputs_for_generation(
                cur_input_ids,
                past_key_values=model_kwargs.get("past_key_values"),
                attention_mask=model_kwargs.get("attention_mask"),
                tokenizer_output=model_kwargs.get("tokenizer_output"),
                position_ids=model_kwargs["position_ids"],
                custom_pos_emb=model_kwargs["custom_pos_emb"],
                mode="gen_text",
                use_cache=True,
                cond_vit_images=model_kwargs.get("cond_vit_images"),
                cond_vit_image_mask=model_kwargs.get("cond_vit_image_mask"),
                vit_kwargs=model_kwargs.get("vit_kwargs"),
            )
            with torch.no_grad():
                out = transformer(**model_inputs, first_step=(step_idx == 0))
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, dict):
                logits = out.get("logits")
            if logits is None:
                raise RuntimeError("HunyuanImage3ARStage.autoregress: model output has no .logits in mode='gen_text'.")

            # Under ``device_map="auto"`` the model's lm_head returns
            # logits on whichever shard owns it (often cuda:N≠0). Gather
            # the predicting-position slice on logits' own device, then
            # move the small ``[B, vocab]`` slice to the AR loop's home
            # device so subsequent sampling ops live on a single device.
            logits_device = logits.device
            if step_idx == 0 and conditions.tokenizer_output is not None:
                real_pos = getattr(conditions.tokenizer_output, "real_pos", None)
                if real_pos is not None:
                    # ``real_pos`` is the *next* write position (one past
                    # the last valid input token under right-padding); the
                    # last valid input position is ``real_pos - 1``.
                    real_pos_t = real_pos.to(device=logits_device, dtype=torch.long)
                    if real_pos_t.dim() == 2:
                        real_pos_t = real_pos_t[:, -1]
                    last_valid = (real_pos_t - 1).clamp(min=0, max=logits.shape[1] - 1)
                    next_logits = logits[torch.arange(batch_size, device=logits_device), last_valid]
                else:
                    next_logits = logits[:, -1, :]
            else:
                next_logits = logits[:, -1, :]  # [B, vocab]
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

            # Append the sampled token to the running input_ids, then have
            # the upstream helper advance position_ids / past_key_values.
            cur_input_ids = torch.cat([cur_input_ids, token_id.unsqueeze(-1)], dim=1)
            updated = transformer._update_model_kwargs_for_generation(out, model_kwargs)
            # Replace model_kwargs entirely. Upstream's
            # ``_update_model_kwargs_for_generation`` returns a *new* dict
            # that intentionally drops ``attention_mask`` and
            # ``tokenizer_output`` -- carrying the prompt's [B, 1, L, L]
            # 4D mask into decode steps would mismatch SDPA's expected
            # [B, H, q_len=1, kv_len] shape. Keep the cond_* / vit_kwargs
            # i2t/it2i pass-throughs alive across steps.
            new_kwargs: Dict[str, Any] = dict(updated)
            for carry in ("cond_vit_images", "cond_vit_image_mask", "vit_kwargs"):
                if carry not in new_kwargs and carry in model_kwargs:
                    new_kwargs[carry] = model_kwargs[carry]
            new_kwargs["use_cache"] = True
            model_kwargs = new_kwargs

        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    @staticmethod
    def _resolve_stop_ids(
        params: Optional[HunyuanImage3ARParams],
        sampling_params: ARSamplingParams,
    ) -> List[int]:
        if params is not None and params.stop_token_ids:
            return list(params.stop_token_ids)
        if sampling_params.stop_token_id is not None:
            return [int(sampling_params.stop_token_id)]
        return []

    @staticmethod
    def _build_kv_cache(transformer, *, batch_size: int, max_cache_len: int):
        """Pre-build a ``HunyuanStaticCache`` for the AR loop.

        Mirrors upstream ``hunyuan.py:2326-2333`` for ``mode="gen_text"``:
        ``dynamic=True`` (the cache slot count grows as new tokens land,
        bounded by ``max_cache_len``), ``dtype=bf16``. Falls back to
        ``None`` (HF default DynamicCache) when the upstream
        ``HunyuanStaticCache`` symbol isn't reachable -- e.g. fake-bundle
        unit tests where the transformer is just a stub ``nn.Module``.
        """
        import sys as _sys

        upstream_mod = _sys.modules.get(type(transformer).__module__)
        cache_cls = getattr(upstream_mod, "HunyuanStaticCache", None)
        if cache_cls is None:
            return None
        config = getattr(transformer, "config", None)
        if config is None:
            return None
        try:
            return cache_cls(
                config=config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                dtype=torch.bfloat16,
                dynamic=True,
            )
        except Exception:  # noqa: BLE001 -- fall back to HF default cache
            return None

    def replay(
        self,
        conditions: HunyuanImage3ARConditions,
        *,
        segment: TextSegment,
    ) -> torch.Tensor:
        """Per-token log-prob replay over a stored rollout segment.

        One teacher-forced forward over ``prompt + response`` (no KV
        cache, no incremental loop), gather full-softmax log-probs at
        the predicting positions for each response token, return packed
        varlen ``[total_tokens]`` aligned with ``segment.log_probs``.

        Builds ``inputs_embeds`` for the forward by looking up the chat-
        template ``input_ids`` in the model's shared embedding table —
        for t2t this is exact; for i2t / it2i replay with cond-image
        conditioning the cond_vit scatter is *not* re-applied here (a
        future training-side enhancement).

        Caller controls grad / no_grad scope and ``.train()`` mode.
        Empty-response samples contribute zero tokens to the output.
        """
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            raise ValueError("HunyuanImage3ARStage.replay: conditions.fused.input_ids is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "HunyuanImage3ARStage.replay: segment requires tokens with "
                "framework-managed cu_seqlens (construct via TextSegment.pack)"
            )

        prompt_ids = fused.input_ids
        device = prompt_ids.device
        batch_size = int(prompt_ids.shape[0])
        prompt_len = int(prompt_ids.shape[1])

        # 1. Pad varlen response tokens into [B, T_max].
        lengths = [int(n) for n in segment.lengths.tolist()]
        T_max = max(lengths) if lengths else 0
        response_tokens = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        # 2. Concat prompt + response into a single token stream.
        # Drive the forward with ``input_ids`` (not ``inputs_embeds``)
        # because ``HunyuanImage3ForCausalMM.forward`` (the outer wrapper)
        # does NOT expose ``inputs_embeds`` in its signature. It internally
        # calls ``self.model.wte(input_ids)`` and handles rope + lm_head.
        if T_max > 0:
            full_ids = torch.cat([prompt_ids, response_tokens], dim=1)
        else:
            full_ids = prompt_ids
        L_full = int(full_ids.shape[1])

        # 3. Build the 4D attention mask the wrapper expects (``bsz x 1 x
        # seqlen x seqlen``). Lower-triangular causal mask over L_full with
        # padded response positions masked out.
        wte_weight = self.model.transformer.model.wte.weight
        param_dtype = wte_weight.dtype
        neg_inf = torch.finfo(param_dtype).min
        full_mask_4d = torch.full(
            (batch_size, 1, L_full, L_full),
            neg_inf,
            dtype=param_dtype,
            device=device,
        )
        causal = torch.tril(torch.ones((L_full, L_full), dtype=torch.bool, device=device))
        full_mask_4d.masked_fill_(causal.unsqueeze(0).unsqueeze(0), 0.0)
        if T_max > 0:
            valid_pos = torch.ones((batch_size, L_full), dtype=torch.bool, device=device)
            valid_pos[:, prompt_len:] = response_mask.bool()
            invalid_q = ~valid_pos  # [B, L_full]
            full_mask_4d.masked_fill_(invalid_q.unsqueeze(1).unsqueeze(-1), neg_inf)
            full_mask_4d.masked_fill_(invalid_q.unsqueeze(1).unsqueeze(2), neg_inf)

        # 4. Set the runtime attributes the wrapper forward reads off
        # ``self``. MUST be unconditional overwrite — DiffusionGRPO
        # predict_noise earlier in the same train step sets
        # num_image_tokens=4096 (image gen). If we only set-when-missing,
        # the AR replay inherits 4096 even though full_ids is pure text
        # → image-aware attention / rope indexing OOB → 100% NaN logits.
        transformer = self.model.transformer
        transformer.post_token_len = None
        transformer.num_special_tokens = None
        transformer.num_image_tokens = 0
        transformer.use_taylor_cache = False
        # Force cached_rope rebuild for text-only seq. Do NOT set
        # cached_rope=None — CachedRoPE is a callable nn.Module, setting
        # it to None causes TypeError. Reset internal cache fields instead.
        if hasattr(transformer, "cached_rope") and transformer.cached_rope is not None:
            for _rope_attr in ("seq_len", "rope_image_info", "cos_cache", "sin_cache"):
                if hasattr(transformer.cached_rope, _rope_attr):
                    setattr(transformer.cached_rope, _rope_attr, None)

        # 5. One teacher-forced forward through the outer wrapper.
        out = transformer(
            input_ids=full_ids,
            attention_mask=full_mask_4d,
            mode="gen_text",
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("HunyuanImage3ARStage.replay: model output has no .logits")

        # 6. logits[:, prompt_len - 1 + t, :] predicts response_tokens[:, t]
        if T_max == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)
        pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + T_max, :]
        log_probs_full = F.log_softmax(pred_logits.float(), dim=-1)
        per_token = log_probs_full.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T_max], fp32

        # 7. Pack to varlen [total_tokens] aligned with segment.log_probs.
        flat: List[torch.Tensor] = []
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            flat.append(per_token[b, :n])
        if not flat:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.cat(flat, dim=0)


def _pack_text_segment(
    generated_tokens: List[List[int]],
    per_token_logps: List[List[float]],
    *,
    device: torch.device,
) -> TextSegment:
    """Pack per-sample lists of tokens / log-probs into a varlen ``TextSegment``.

    Delegates to :meth:`TextSegment.pack`, which packs the per-sample tensor
    lists along dim 0 and derives the framework-managed ``cu_seqlens``
    metadata. ``sample_indices`` / ``positions`` are one entry per *segment*
    (length ``B`` with one AR segment per sample). ``tokens`` / ``log_probs``
    are packed per *token* across all segments, length ``sum(lengths)``.
    """
    n = len(generated_tokens)
    return TextSegment.pack(
        sample_indices=torch.arange(n, dtype=torch.long, device=device),
        positions=torch.zeros(n, dtype=torch.long, device=device),
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["HunyuanImage3ARParams", "HunyuanImage3ARStage", "HunyuanImage3ARStep"]
