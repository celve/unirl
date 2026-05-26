"""Tests for typed pipeline-stage protocols (EncodeStage / EmbedStage / DiffusionStage / ARStage / DecodeStage) and step kernels."""

from __future__ import annotations

from typing import Tuple

import torch

from diffusionrl.models.types import (
    ARSamplingParams,
    ARStage,
    ARStep,
    DecodeStage,
    DiffusionStage,
    DiffusionStep,
    EmbedStage,
    EncodeStage,
    ReplayResult,
)
from diffusionrl.types.conditions import (
    Conditions,
    ImageLatentCondition,
    TextEmbedCondition,
)
from diffusionrl.types.primitives import Image, Text
from diffusionrl.types.segments import LatentSegment, TextSegment

# ---------------------------------------------------------------------------
# Pipeline-stage protocols (runtime structural conformance)
# ---------------------------------------------------------------------------


def test_encode_stage_protocol_is_satisfied_by_method_presence():
    class FakeImageEncoder:
        def encode(self, p: Image) -> ImageLatentCondition:
            return ImageLatentCondition(latents=p.pixels.unsqueeze(0))

    encoder = FakeImageEncoder()
    assert isinstance(encoder, EncodeStage)


def test_decode_stage_protocol_is_satisfied():
    class FakeImageDecoder:
        def decode(self, s: LatentSegment) -> Image:
            return Image(pixels=s.latents[0, -1])

    decoder = FakeImageDecoder()
    assert isinstance(decoder, DecodeStage)


def test_embed_stage_protocol_is_satisfied():
    class FakeTextEmbedder:
        def embed(self, p: Text) -> TextEmbedCondition:
            return TextEmbedCondition(embeds=torch.zeros(1, 8, 16))

    embedder = FakeTextEmbedder()
    assert isinstance(embedder, EmbedStage)


def test_diffusion_stage_protocol_is_satisfied():
    class FakeDiffusionStage:
        def diffuse(self, conditions: Conditions, *, schedule, params=None) -> LatentSegment:
            B = conditions["text"].embeds.shape[0]
            return LatentSegment(
                sample_indices=torch.arange(B),
                positions=torch.zeros(B, dtype=torch.long),
                latents=torch.zeros(B, 4, 16, 8, 8),
                sigmas=schedule,
            )

        def replay(
            self,
            conditions: Conditions,
            *,
            segment: LatentSegment,
            params=None,
            step_indices=None,
        ) -> ReplayResult:
            B = segment.latents.shape[0]
            S = (
                len(step_indices)
                if step_indices is not None
                else (int(segment.sde_indices.shape[0]) if segment.sde_indices is not None else 0)
            )
            return ReplayResult(
                log_probs=torch.zeros(B, S),
                prev_sample_means=torch.zeros(B, S, 4, 16, 8, 8),
            )

    stage = FakeDiffusionStage()
    assert isinstance(stage, DiffusionStage)
    out = stage.diffuse(
        {"text": TextEmbedCondition(embeds=torch.zeros(2, 8, 16))},
        schedule=torch.linspace(1.0, 0.0, 5),
    )
    assert out.batch_size == 2
    assert out.latents.shape == (2, 4, 16, 8, 8)
    # replay returns ReplayResult with [B, S] log-probs.
    seg = LatentSegment(
        sample_indices=torch.arange(2),
        positions=torch.zeros(2, dtype=torch.long),
        latents=torch.zeros(2, 5, 4, 8, 8),
        indices=torch.arange(5),
        sigmas=torch.linspace(1.0, 0.0, 5),
        sde_indices=torch.tensor([0, 1, 2]),
    )
    result = stage.replay(
        {"text": TextEmbedCondition(embeds=torch.zeros(2, 8, 16))},
        segment=seg,
    )
    assert isinstance(result, ReplayResult)
    assert result.log_probs.shape == (2, 3)
    assert result.prev_sample_means is not None
    assert result.prev_sample_means.shape[:2] == (2, 3)


def test_ar_stage_protocol_is_satisfied():
    class FakeARStage:
        def autoregress(
            self,
            conditions: Conditions,
            *,
            sampling_params: ARSamplingParams,
            **kwargs,
        ) -> TextSegment:
            B = conditions["text"].embeds.shape[0]
            return TextSegment.pack(
                sample_indices=torch.arange(B),
                positions=torch.zeros(B, dtype=torch.long),
                tokens=[torch.zeros(sampling_params.max_new_tokens, dtype=torch.long) for _ in range(B)],
            )

        def replay(
            self,
            conditions: Conditions,
            *,
            segment: TextSegment,
        ) -> torch.Tensor:
            return torch.zeros(int(segment.tokens.shape[0]), dtype=torch.float32)

    stage = FakeARStage()
    assert isinstance(stage, ARStage)
    out = stage.autoregress(
        {"text": TextEmbedCondition(embeds=torch.zeros(3, 8, 16))},
        sampling_params=ARSamplingParams(max_new_tokens=4),
    )
    assert out.tokens.shape == (12,)
    assert out.cu_seqlens.tolist() == [0, 4, 8, 12]
    # replay returns [total_tokens] aligned with segment.log_probs.
    logp = stage.replay({"text": TextEmbedCondition(embeds=torch.zeros(3, 8, 16))}, segment=out)
    assert logp.shape == (12,)


# ---------------------------------------------------------------------------
# Step kernels
# ---------------------------------------------------------------------------


def test_ar_step_typed_signature():
    class FakeARStep:
        def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            token = logits.argmax(dim=-1)
            logp = torch.zeros_like(token, dtype=torch.float)
            return token, logp

    s = FakeARStep()
    assert isinstance(s, ARStep)

    logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 2.0, 4.0]])
    token, logp = s.step(logits)
    assert token.tolist() == [1, 2]
    assert logp.shape == token.shape


def test_diffusion_step_protocol_is_satisfied_by_existing_kernels():
    # Existing DiffusionStep implementations live in diffusionrl/sde/kernels.
    # We assert structural conformance with a minimal stub instead so the test
    # doesn't depend on the concrete kernel module.
    # ``step_with_logp`` returns a 3-tuple (prev_sample, log_prob,
    # prev_sample_mean) — the third value powers KL-penalty consumption.
    class FakeDiffusionStep:
        def forward(self, **kwargs):
            return torch.zeros(1), None, None

        def step(self, *args, **kwargs):
            return torch.zeros(1), None, None

        def step_with_logp(self, *args, **kwargs):
            return torch.zeros(1), None, None

    assert isinstance(FakeDiffusionStep(), DiffusionStep)


# ---------------------------------------------------------------------------
# SampleStage is removed
# ---------------------------------------------------------------------------


def test_sample_stage_is_removed():
    import diffusionrl.models.types as mt

    assert not hasattr(mt, "SampleStage")
