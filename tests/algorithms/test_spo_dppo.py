"""Tests for ARSPODPPO: trust-region masks + sampling-temperature wiring.

The TV/KL mask helpers are pure functions over ``[total_tokens]`` tensors. The
sampling-temperature test pins the #170 fix: ARSPODPPO must forward the rollout
sampling temperature to ``stage.replay`` (so replay's log-softmax matches the
sampling distribution), mirroring ARGRPO.
"""

from __future__ import annotations

from typing import Mapping

import pytest
import torch
import torch.nn as nn

from diffusionrl.algorithms.spo_dppo import (
    ARSPODPPO,
    _ar_spo_dppo_kl_loss,
    _ar_spo_dppo_tv_loss,
)
from diffusionrl.types.conditions import Condition
from diffusionrl.types.segments.text import TextSegment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeARStage:
    """Minimal AR stage: ``replay`` records the temperature it was called with
    and returns a leaf-param-wired ``[total_tokens]`` log-prob tensor."""

    def __init__(self, *, init_value: float = 0.5) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))
        self.recorded_temperature: float | None = None

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.recorded_temperature = temperature
        assert segment.tokens is not None
        return self.param * (segment.tokens.float() + 1.0)


class _FakePipeline:
    def __init__(self) -> None:
        self.ar = _FakeARStage()


def _make_text_segment(*, batch_size: int = 2, tokens_per_sample: int = 3) -> TextSegment:
    return TextSegment.pack(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        tokens=[
            torch.arange(k * tokens_per_sample, (k + 1) * tokens_per_sample, dtype=torch.long)
            for k in range(batch_size)
        ],
        log_probs=[torch.full((tokens_per_sample,), -2.0, dtype=torch.float32) for _ in range(batch_size)],
    )


# ---------------------------------------------------------------------------
# Sampling-temperature wiring (the #170 fix)
# ---------------------------------------------------------------------------


def test_sampling_temperature_is_forwarded_to_replay() -> None:
    pipe = _FakePipeline()
    alg = ARSPODPPO(pipeline=pipe, stage_attr="ar", variant="tv", sampling_temperature=0.7, conditions_cls=None)
    result = alg.compute_loss_and_backward(
        conditions={},
        segment=_make_text_segment(),
        advantages=torch.tensor([1.0, -1.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert pipe.ar.recorded_temperature == 0.7
    assert result.has_backward
    assert pipe.ar.param.grad is not None  # gradient flows through replay's new_logp


def test_sampling_temperature_defaults_to_arsamplingparams() -> None:
    from diffusionrl.types.sampling import ARSamplingParams

    alg = ARSPODPPO(pipeline=_FakePipeline(), stage_attr="ar", conditions_cls=None)
    assert alg.sampling_temperature == float(ARSamplingParams.__dataclass_fields__["temperature"].default)


def test_requires_pipeline() -> None:
    with pytest.raises(ValueError, match="pipeline"):
        ARSPODPPO(stage_attr="ar")


def test_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        ARSPODPPO(pipeline=_FakePipeline(), variant="bogus")


# ---------------------------------------------------------------------------
# TV mask
# ---------------------------------------------------------------------------


def test_tv_mask_blocks_large_prob_increase_for_positive_adv() -> None:
    # token 0: prob jumps 0.1 -> 0.9 (delta 0.8 > clip_high) with adv>0 -> masked.
    # token 1: prob unchanged (delta 0) with adv>0 -> kept.
    losses, metrics = _ar_spo_dppo_tv_loss(
        new_logp=torch.log(torch.tensor([0.9, 0.5])),
        old_logp=torch.log(torch.tensor([0.1, 0.5])),
        advantages=torch.tensor([1.0, 1.0]),
        clip_divergence_low=0.2,
        clip_divergence_high=0.2,
        clip_ratio_c=20.0,
    )
    assert losses[0].item() == 0.0
    assert losses[1].item() != 0.0
    assert metrics["valid_fraction"].item() == 0.5


# ---------------------------------------------------------------------------
# KL mask
# ---------------------------------------------------------------------------


def test_kl_mask_allows_conservative_update_against_advantage() -> None:
    # adv>0 but prob DECREASED (ratio<1, conservative) -> kept even though KL is
    # far above the threshold.
    _losses, metrics = _ar_spo_dppo_kl_loss(
        new_logp=torch.log(torch.tensor([0.1])),
        old_logp=torch.log(torch.tensor([0.9])),
        advantages=torch.tensor([1.0]),
        clip_divergence_low=0.01,
        clip_divergence_high=0.01,
        clip_ratio_c=20.0,
    )
    assert metrics["valid_fraction"].item() == 1.0


def test_kl_mask_blocks_large_nonconservative_increase_for_positive_adv() -> None:
    # adv>0 and prob INCREASED far (ratio>1, KL >> threshold) -> masked.
    losses, metrics = _ar_spo_dppo_kl_loss(
        new_logp=torch.log(torch.tensor([0.9])),
        old_logp=torch.log(torch.tensor([0.1])),
        advantages=torch.tensor([1.0]),
        clip_divergence_low=0.01,
        clip_divergence_high=0.01,
        clip_ratio_c=20.0,
    )
    assert losses[0].item() == 0.0
    assert metrics["valid_fraction"].item() == 0.0
