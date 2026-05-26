"""End-to-end test: images flow correctly through ``RolloutInputs`` primitives.

Exercises the path from direct ``RolloutInputs`` construction to the
``primitives["image"]`` slot, and verifies ``expand`` replicates images.
"""

from __future__ import annotations

import torch

from diffusionrl.types.primitives import Image as PrimImage
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.prompts import RolloutInputs


def _img(value: float = 0.5) -> PrimImage:
    return PrimImage(pixels=torch.full((3, 4, 4), value))


def test_rollout_inputs_carries_images() -> None:
    images = [_img(0.1), _img(0.2)]
    inputs = RolloutInputs(
        primitives={
            "text": Texts(texts=["a", "b"]),
            "image": Images.from_list(images),
        },
        sample_ids=["prompt:p0:sample:0", "prompt:p1:sample:0"],
        group_ids=["p0", "p1"],
    )
    assert "image" in inputs.primitives
    assert isinstance(inputs.primitives["image"], Images)
    assert inputs.primitives["image"].pixels.shape == (2, 3, 4, 4)


def test_expand_replicates_image_pixels() -> None:
    """``RolloutInputs.expand`` replicates image pixels via repeat_interleave."""
    images = [_img(0.7), _img(0.8)]
    inputs = RolloutInputs(
        primitives={
            "text": Texts(texts=["a", "b"]),
            "image": Images.from_list(images),
        },
        sample_ids=["prompt:p0:sample:0", "prompt:p1:sample:0"],
        group_ids=["p0", "p1"],
    )
    expanded = inputs.expand(3)
    assert isinstance(expanded.primitives["image"], Images)
    assert expanded.primitives["image"].pixels.shape == (6, 3, 4, 4)


def test_text_only_has_no_image_primitive() -> None:
    """T2V path: text-only primitives."""
    inputs = RolloutInputs(
        primitives={"text": Texts(texts=["a", "b"])},
        sample_ids=["prompt:p0:sample:0", "prompt:p1:sample:0"],
        group_ids=["p0", "p1"],
    )
    assert "image" not in inputs.primitives
    assert "text" in inputs.primitives
