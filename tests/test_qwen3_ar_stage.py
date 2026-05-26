"""Tests for ``Qwen3ARStage.autoregress`` and ``replay`` with a fake transformer.

CPU-only, no real Qwen3 weights. The fake transformer is a Markov-1
model whose logits at position ``p`` deterministically depend on
``input_ids[:, p]`` — so greedy rollout and teacher-forced replay see
identical predictions, and their per-token log-probs should match
exactly.

The fake also implements the HF-style ``prepare_inputs_for_generation``
+ ``_update_model_kwargs_for_generation`` hooks the AR stage drives, so
the loop runs end-to-end without depending on the real ``transformers``
package's internals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn

from diffusionrl.models.qwen3.ar import Qwen3ARParams, Qwen3ARStage
from diffusionrl.models.qwen3.bundle import Qwen3Bundle
from diffusionrl.models.qwen3.conditions import Qwen3ARConditions
from diffusionrl.models.types.ar import ARSamplingParams
from diffusionrl.types.conditions import TextTokenCondition

# A deliberately impossible eos id so the loop runs full ``max_tokens``
# unless a test-provided stop id triggers earlier.
_VOCAB = 16
_EOS = 99
_PAD = 0


class _FakeTokenizer:
    """Minimal stand-in for ``AutoTokenizer`` — just the fields the stage reads."""

    eos_token_id = _EOS
    pad_token_id = _PAD


class _FakeQwen3Transformer(nn.Module):
    """Deterministic Markov-1 fake.

    Logits at position ``p`` are a function of ``input_ids[:, p]``:
    ``argmax = (input_ids[:, p] * 7 + 3) % vocab``, with the winning
    logit at +10 and all others at 0. This makes greedy autoregress
    fully predictable AND ensures rollout-time and replay-time
    predictions of the same response token match bit-for-bit (both see
    the same ``(b, p, value)``-keyed logits).
    """

    def __init__(self, vocab_size: int = _VOCAB) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.config = SimpleNamespace(vocab_size=self.vocab_size)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **_kwargs: Any,
    ) -> dict:
        if past_key_values is None:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None,
            }
        # Cached step: only the new last token enters the forward.
        return {
            "input_ids": input_ids[:, -1:],
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def _update_model_kwargs_for_generation(self, out: Any, model_kwargs: dict) -> dict:
        new_kwargs = dict(model_kwargs)
        new_kwargs["past_key_values"] = getattr(out, "past_key_values", "present")
        am = new_kwargs.get("attention_mask")
        if am is not None:
            new_kwargs["attention_mask"] = torch.cat(
                [am, torch.ones(am.shape[0], 1, dtype=am.dtype, device=am.device)], dim=1
            )
        return new_kwargs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **_kwargs: Any,
    ) -> Any:
        assert input_ids is not None, "fake transformer expects input_ids"
        b, length = input_ids.shape
        targets = (input_ids * 7 + 3) % self.vocab_size  # [B, L]
        logits = torch.zeros(b, length, self.vocab_size, dtype=torch.float32)
        logits.scatter_(2, targets.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits, past_key_values="present")


def _make_bundle() -> Qwen3Bundle:
    return Qwen3Bundle(
        transformer=_FakeQwen3Transformer(),
        tokenizer=_FakeTokenizer(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        pretrained_path="fake",
    )


def _make_conditions(prompt_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Qwen3ARConditions:
    if attention_mask is None:
        attention_mask = torch.ones_like(prompt_ids)
    return Qwen3ARConditions(
        prompt=TextTokenCondition(input_ids=prompt_ids, attention_mask=attention_mask),
    )


def test_autoregress_returns_packed_text_segment_with_correct_shapes():
    bundle = _make_bundle()
    stage = Qwen3ARStage(model=bundle)
    prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    conds = _make_conditions(prompt_ids)
    params = Qwen3ARParams(max_tokens=5, temperature=0.0)
    sp = ARSamplingParams(max_new_tokens=params.max_tokens, temperature=params.temperature)

    seg = stage.autoregress(conds, sampling_params=sp, params=params)

    assert seg.tokens is not None and seg.log_probs is not None
    assert seg.cu_seqlens is not None and seg.lengths is not None
    assert int(seg.lengths.shape[0]) == 2
    # Greedy with no stop hits should generate exactly ``max_tokens`` per sample.
    assert seg.lengths.tolist() == [5, 5]
    assert int(seg.tokens.shape[0]) == 10
    assert int(seg.log_probs.shape[0]) == 10
    assert seg.cu_seqlens.tolist() == [0, 5, 10]


def test_autoregress_stop_token_halts_generation():
    bundle = _make_bundle()
    stage = Qwen3ARStage(model=bundle)
    # With seed prompt id=1, greedy outputs are deterministic:
    # f(x) = (x * 7 + 3) % 16:  1 → 10, 10 → 9, 9 → 2, 2 → 1, 1 → 10, ...
    # Sample 0 expected sequence (max 6 tokens): [10, 9, 2, 1, 10, 9]
    # Sample 1 expected sequence with prompt id=0: 0 → 3, 3 → 8, 8 → 11, 11 → 0, ...
    prompt_ids = torch.tensor([[1], [0]], dtype=torch.long)
    conds = _make_conditions(prompt_ids)
    params = Qwen3ARParams(max_tokens=6, temperature=0.0, stop_token_ids=[2])
    sp = ARSamplingParams(max_new_tokens=params.max_tokens, temperature=params.temperature)

    seg = stage.autoregress(conds, sampling_params=sp, params=params)

    # Sample 0 hits stop id 2 at position 2 (tokens: [10, 9, 2]) — length 3.
    # Sample 1 never hits id 2 within the window — length 6.
    assert seg.lengths.tolist() == [3, 6]
    sample_0 = seg.tokens[: seg.lengths[0].item()].tolist()
    assert sample_0 == [10, 9, 2]


def test_replay_log_probs_match_rollout_log_probs_under_greedy():
    """End-to-end parity: greedy rollout's stored log_probs must equal replay's
    teacher-forced log_probs token-for-token. This is the GRPO substitution
    invariant — π_old (rollout) and π_θ (replay with no grad change) must
    agree on a frozen model."""
    bundle = _make_bundle()
    stage = Qwen3ARStage(model=bundle)
    prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    conds = _make_conditions(prompt_ids)
    params = Qwen3ARParams(max_tokens=4, temperature=0.0)
    sp = ARSamplingParams(max_new_tokens=params.max_tokens, temperature=params.temperature)

    seg = stage.autoregress(conds, sampling_params=sp, params=params)
    replay_logps = stage.replay(conds, segment=seg)

    assert replay_logps.shape == seg.log_probs.shape
    assert torch.allclose(replay_logps, seg.log_probs, atol=1e-5)


def test_replay_handles_empty_response_samples():
    """Mixed batch where one sample produces zero tokens (max_tokens=0)."""
    bundle = _make_bundle()
    stage = Qwen3ARStage(model=bundle)
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    conds = _make_conditions(prompt_ids)
    params = Qwen3ARParams(max_tokens=0, temperature=0.0)
    sp = ARSamplingParams(max_new_tokens=params.max_tokens, temperature=params.temperature)

    seg = stage.autoregress(conds, sampling_params=sp, params=params)
    assert seg.lengths.tolist() == [0]
    replay_logps = stage.replay(conds, segment=seg)
    assert replay_logps.numel() == 0


def test_autoregress_requires_input_ids():
    bundle = _make_bundle()
    stage = Qwen3ARStage(model=bundle)
    conds = Qwen3ARConditions(prompt=TextTokenCondition(input_ids=None, attention_mask=None))
    sp = ARSamplingParams(max_new_tokens=4)
    try:
        stage.autoregress(conds, sampling_params=sp)
    except ValueError as e:
        assert "input_ids" in str(e)
    else:
        raise AssertionError("autoregress should raise when input_ids is None")
