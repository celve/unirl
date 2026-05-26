"""Tests for image validation in the data source → RolloutInputs path."""

from __future__ import annotations

import pytest
import torch

from diffusionrl.data.data_source import _validate_homogeneous_images
from diffusionrl.types.primitives import Image as PrimImage
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.prompts import RolloutInputs


def _img(value: float = 0.5) -> PrimImage:
    return PrimImage(pixels=torch.full((3, 4, 4), value))


def test_text_only_rollout_inputs() -> None:
    """Pure T2V batch: no ``primitives['image']``."""
    inputs = RolloutInputs(
        primitives={"text": Texts(texts=["a", "b"])},
        sample_ids=["prompt:0:sample:0", "prompt:1:sample:0"],
        group_ids=["prompt:0", "prompt:1"],
    )
    assert "image" not in inputs.primitives


def test_rollout_inputs_with_images() -> None:
    images = [_img(0.2), _img(0.3)]
    inputs = RolloutInputs(
        primitives={
            "text": Texts(texts=["a", "b"]),
            "image": Images.from_list(images),
        },
        sample_ids=["prompt:0:sample:0", "prompt:1:sample:0"],
        group_ids=["prompt:0", "prompt:1"],
    )
    assert isinstance(inputs.primitives["image"], Images)
    assert inputs.primitives["image"].pixels.shape == (2, 3, 4, 4)
    assert torch.allclose(inputs.primitives["image"].pixels[0], torch.full((3, 4, 4), 0.2))
    assert torch.allclose(inputs.primitives["image"].pixels[1], torch.full((3, 4, 4), 0.3))


def test_validate_homogeneous_images_all_present() -> None:
    _validate_homogeneous_images([_img(), _img()])


def test_validate_homogeneous_images_all_none() -> None:
    _validate_homogeneous_images([None, None, None])


def test_validate_homogeneous_images_heterogeneous_raises() -> None:
    with pytest.raises(ValueError, match="[Hh]eterogeneous"):
        _validate_homogeneous_images([_img(0.4), None, _img(0.5)])


def test_validate_homogeneous_images_error_names_missing_index() -> None:
    with pytest.raises(ValueError, match="prompt index 2"):
        _validate_homogeneous_images([_img(), _img(), None])
