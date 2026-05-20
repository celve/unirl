"""Tests for ``NewRolloutPipeline._build_images_primitive`` and its caller."""

from __future__ import annotations

import pytest
import torch

from diffusionrl.rollout.new_pipeline import _build_images_primitive
from diffusionrl.types.primitives import Image as PrimImage
from diffusionrl.types.primitives import Images


def _img(value: float = 0.5) -> PrimImage:
    return PrimImage(pixels=torch.full((3, 4, 4), value))


def test_empty_images_list_returns_none() -> None:
    """Pure T2V batch: empty list → no ``primitives['image']`` injection."""
    assert _build_images_primitive([], prompt_count=0, context="t") is None


def test_all_none_returns_none() -> None:
    """All prompts opt out → no ``primitives['image']`` key emitted."""
    assert _build_images_primitive([None, None, None], prompt_count=3, context="t") is None


def test_homogeneous_image_batch_emits_images_primitive() -> None:
    images = [_img(0.2), _img(0.3)]
    out = _build_images_primitive(images, prompt_count=2, context="t")
    assert isinstance(out, Images)
    assert out.pixels.shape == (2, 3, 4, 4)
    # Sanity: per-sample distinct values preserved.
    assert torch.allclose(out.pixels[0], torch.full((3, 4, 4), 0.2))
    assert torch.allclose(out.pixels[1], torch.full((3, 4, 4), 0.3))


def test_length_mismatch_raises() -> None:
    """Caller-supplied prompt_count must match images list length."""
    with pytest.raises(ValueError, match="length"):
        _build_images_primitive([_img()], prompt_count=2, context="ctx")


def test_heterogeneous_batch_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Mixing T2V- and I2V-shaped prompts in one request is unsupported."""
    images = [_img(0.4), None, _img(0.5)]
    with pytest.raises(ValueError, match="heterogeneous"):
        _build_images_primitive(images, prompt_count=3, context="plan_requests")


def test_heterogeneous_error_message_names_missing_index() -> None:
    images = [_img(), _img(), None]
    with pytest.raises(ValueError, match="prompt index 2"):
        _build_images_primitive(images, prompt_count=3, context="plan_requests")
