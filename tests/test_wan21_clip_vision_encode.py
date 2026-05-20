"""CPU tests for the optional CLIP-vision encode stage on the WAN 2.1 I2V path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from diffusionrl.models_new.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from diffusionrl.types.conditions import ImageEmbedCondition
from diffusionrl.types.primitives import Images


class _FakeImageProcessor:
    """Stand-in for ``CLIPImageProcessor`` — returns a deterministic pixel tensor."""

    def __init__(self, num_patches: int = 257, dim: int = 1024) -> None:
        self.num_patches = num_patches
        self.dim = dim
        self.last_pils = None

    def __call__(self, *, images, return_tensors: str = "pt") -> SimpleNamespace:  # noqa: ARG002
        self.last_pils = list(images)
        # CLIP expects [B, 3, 224, 224]; the actual values don't matter — the
        # fake vision encoder reads only the batch dim.
        return SimpleNamespace(pixel_values=torch.zeros(len(self.last_pils), 3, 224, 224))


class _FakeVisionEncoder:
    """Stand-in for ``CLIPVisionModel`` — emits canned hidden_states.

    The penultimate hidden state ``hidden_states[-2]`` is shape
    ``[B, num_patches, dim]``; the last is a distractor with different
    shape so any code that grabs the wrong index gets caught.
    """

    def __init__(self, num_patches: int = 257, dim: int = 1024) -> None:
        self.num_patches = num_patches
        self.dim = dim
        self.last_call_pixels = None

    def __call__(self, pixel_values: torch.Tensor, *, output_hidden_states: bool = False):
        assert output_hidden_states, "stage must request output_hidden_states=True"
        self.last_call_pixels = pixel_values
        batch_size = int(pixel_values.shape[0])
        penultimate = torch.full((batch_size, self.num_patches, self.dim), 0.42, dtype=pixel_values.dtype)
        # last_hidden_state is intentionally a different shape — selecting
        # the wrong index would surface as a shape mismatch downstream.
        last = torch.zeros(batch_size, self.num_patches + 1, self.dim, dtype=pixel_values.dtype)
        return SimpleNamespace(hidden_states=[None, penultimate, last])


def _make_clip_bundle(*, num_patches: int = 257, dim: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        vision_encoder=_FakeVisionEncoder(num_patches=num_patches, dim=dim),
        image_processor=_FakeImageProcessor(num_patches=num_patches, dim=dim),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _make_t2v_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        vision_encoder=None,
        image_processor=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_t2v_bundle_does_not_construct_clip_stage() -> None:
    bundle = _make_t2v_bundle()
    # Sanity: the property contract used by the pipeline gate
    # ``self.bundle.uses_clip_vision`` -- a fresh ``SimpleNamespace`` does
    # not synthesize one, so the pipeline-side check would simply skip
    # construction. Constructing the stage anyway against a T2V bundle is
    # a fail-fast.
    assert getattr(bundle, "vision_encoder", None) is None
    with pytest.raises(ValueError):
        WAN21CLIPVisionEncodeStage(bundle)


def test_clip_encode_emits_image_embed_condition_with_expected_shape() -> None:
    bundle = _make_clip_bundle(num_patches=257, dim=1024)
    stage = WAN21CLIPVisionEncodeStage(bundle)
    images = Images(pixels=torch.rand(2, 3, 32, 32))
    cond = stage.encode(images)
    assert isinstance(cond, ImageEmbedCondition)
    assert cond.embeds is not None
    assert cond.embeds.shape == (2, 257, 1024)
    # All-ones attention mask covering the full patch grid (CLIP ViT has
    # no padding inside the encoder).
    assert cond.attn_mask is not None
    assert cond.attn_mask.shape == (2, 257)
    assert cond.attn_mask.dtype == torch.long
    assert torch.all(cond.attn_mask == 1)


def test_clip_encode_uses_penultimate_hidden_state() -> None:
    bundle = _make_clip_bundle(num_patches=10, dim=8)
    stage = WAN21CLIPVisionEncodeStage(bundle)
    images = Images(pixels=torch.rand(1, 3, 16, 16))
    cond = stage.encode(images)
    # Fake encoder fills the penultimate state with 0.42; using
    # ``hidden_states[-1]`` would yield zeros (different shape too).
    assert cond.embeds is not None
    assert torch.allclose(cond.embeds, torch.full_like(cond.embeds, 0.42))


def test_clip_encode_rejects_non_images() -> None:
    bundle = _make_clip_bundle()
    stage = WAN21CLIPVisionEncodeStage(bundle)
    with pytest.raises(TypeError):
        stage.encode("not-an-images-instance")  # type: ignore[arg-type]
