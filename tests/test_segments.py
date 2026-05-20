"""Tests for Segment types (LatentSegment, TextSegment) and promotion."""

from __future__ import annotations

import pytest
import torch

from diffusionrl.types.conditions import (
    ImageLatentCondition,
    Modality,
    TextEmbedCondition,
)
from diffusionrl.types.segments import (
    LatentSegment,
    Segment,
    TextSegment,
    make_audio_segment,
    make_image_segment,
    make_video_segment,
)

# ---------------------------------------------------------------------------
# LatentSegment
# ---------------------------------------------------------------------------


def test_latent_segment_basic_fields():
    seg = LatentSegment(
        sample_indices=torch.tensor([0, 0, 1]),
        positions=torch.tensor([0, 1, 0]),
        latents=torch.zeros(3, 4, 16, 8, 8),
    )
    assert seg.modality is Modality.IMAGE
    assert seg.batch_size == 3


def test_latent_segment_concat_preserves_sample_indices_and_positions():
    a = LatentSegment(
        sample_indices=torch.tensor([0, 1]),
        positions=torch.tensor([0, 0]),
        latents=torch.zeros(2, 4, 16, 8, 8),
    )
    b = LatentSegment(
        sample_indices=torch.tensor([0]),
        positions=torch.tensor([1]),
        latents=torch.ones(1, 4, 16, 8, 8),
    )
    merged = LatentSegment.concat([a, b])
    # No remap at this level — that's RolloutResp's job.
    assert merged.sample_indices.tolist() == [0, 1, 0]
    assert merged.positions.tolist() == [0, 0, 1]
    assert merged.latents.shape == (3, 4, 16, 8, 8)


def test_modality_factories_set_modality():
    img_seg = make_image_segment(latents=torch.zeros(1, 4, 16, 8, 8))
    assert img_seg.modality is Modality.IMAGE

    vid_seg = make_video_segment(latents=torch.zeros(1, 4, 16, 8, 8))
    assert vid_seg.modality is Modality.VIDEO

    aud_seg = make_audio_segment(latents=torch.zeros(1, 4, 16, 8, 8))
    assert aud_seg.modality is Modality.AUDIO


# ---------------------------------------------------------------------------
# Segment.as_condition() / as_condition_with()
# ---------------------------------------------------------------------------


def test_base_segment_as_condition_returns_none_by_default():
    # Use a minimal Segment-shaped subclass without overrides.
    @torch.jit.ignore
    def _build():
        class _Plain(Segment):
            modality = Modality.TEXT  # placeholder

        return _Plain(sample_indices=torch.tensor([0]), positions=torch.tensor([0]))

    seg = _build()
    assert seg.as_condition() is None


def test_base_segment_as_condition_with_raises():
    @torch.jit.ignore
    def _build():
        class _Plain(Segment):
            modality = Modality.TEXT

        return _Plain(sample_indices=torch.tensor([0]), positions=torch.tensor([0]))

    seg = _build()
    with pytest.raises(NotImplementedError):
        seg.as_condition_with(lambda x: x)


def test_image_latent_segment_promotes_to_image_latent_condition():
    latents = torch.randn(2, 4, 16, 8, 8)
    seg = LatentSegment(
        sample_indices=torch.tensor([0, 1]),
        positions=torch.tensor([0, 0]),
        latents=latents,
    )
    cond = seg.as_condition()
    assert isinstance(cond, ImageLatentCondition)
    # Last step's latent.
    assert torch.equal(cond.latents, latents[:, -1])


def test_video_segment_does_not_auto_promote():
    seg = make_video_segment(
        sample_indices=torch.tensor([0]),
        positions=torch.tensor([0]),
        latents=torch.zeros(1, 4, 16, 8, 8),
    )
    assert seg.as_condition() is None


# ---------------------------------------------------------------------------
# TextSegment
# ---------------------------------------------------------------------------


def test_text_segment_packed_cu_seqlens():
    seg = TextSegment.pack(
        sample_indices=torch.tensor([0, 1]),
        positions=torch.tensor([0, 0]),
        tokens=[
            torch.tensor([10, 11, 12]),
            torch.tensor([20, 21]),
        ],
    )
    # Framework derives cu_seqlens and lengths from the per-sample lists.
    assert seg.tokens.tolist() == [10, 11, 12, 20, 21]
    assert seg.cu_seqlens.tolist() == [0, 3, 5]
    assert seg.lengths.tolist() == [3, 2]
    # Tokens for sample 0 are tokens[0:3]; sample 1 is tokens[3:5].
    assert seg.tokens[0:3].tolist() == [10, 11, 12]
    assert seg.tokens[3:5].tolist() == [20, 21]


def test_text_segment_promotes_via_encoder():
    seg = TextSegment.pack(
        sample_indices=torch.tensor([0, 1]),
        positions=torch.tensor([0, 0]),
        tokens=[
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
        ],
    )
    embed_dim = 16

    def fake_encoder(t):
        return torch.zeros(t.shape[0], embed_dim)

    cond = seg.as_condition_with(fake_encoder)
    assert isinstance(cond, TextEmbedCondition)
    assert cond.embeds.shape == (5, embed_dim)
