"""End-to-end test: ``Prompts.images`` flows correctly through the rollout primitives.

Exercises the path from data-layer-emitted ``Prompts(images=...)`` to the
``RolloutReq.primitives["image"]`` slot, with heterogeneous batches
failing loudly at ``plan_requests`` time.
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.rollout.new_pipeline import _build_images_primitive
from diffusionrl.types.primitives import Image as PrimImage
from diffusionrl.types.primitives import Images
from diffusionrl.types.prompts import Prompts


def _img(value: float = 0.5) -> PrimImage:
    return PrimImage(pixels=torch.full((3, 4, 4), value))


def test_prompts_carries_images_through_from_unique_prompts() -> None:
    """``Prompts.from_unique_prompts`` propagates the per-prompt image list."""
    images = [_img(0.1), _img(0.2)]
    prompts = Prompts.from_unique_prompts(["a", "b"], prompt_ids=["p0", "p1"], images=images)
    assert len(prompts.images) == 2
    assert isinstance(prompts.images[0], PrimImage)
    assert isinstance(prompts.images[1], PrimImage)


def test_prompts_expand_replicates_image_refs() -> None:
    """``Prompts.expand`` propagates the same Image ref to each sample —
    no tensor copy, no None fill-in."""
    images = [_img(0.7), _img(0.8)]
    prompts = Prompts.from_unique_prompts(["a", "b"], prompt_ids=["p0", "p1"], images=images)
    expanded = prompts.expand(3)
    assert len(expanded.images) == 6
    # First 3 samples share the same Image instance (cheap reference copy).
    assert expanded.images[0] is images[0]
    assert expanded.images[1] is images[0]
    assert expanded.images[2] is images[0]
    assert expanded.images[3] is images[1]
    assert expanded.images[4] is images[1]
    assert expanded.images[5] is images[1]


def test_prompts_from_unique_prompts_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="images length"):
        Prompts.from_unique_prompts(
            ["a", "b", "c"],
            images=[_img()],
        )


def test_homogeneous_image_prompts_yield_images_primitive() -> None:
    images = [_img(0.1), _img(0.2)]
    prompts = Prompts.from_unique_prompts(["a", "b"], prompt_ids=["p0", "p1"], images=images)
    out = _build_images_primitive(prompts.images, prompt_count=len(prompts.prompts), context="plan_requests")
    assert isinstance(out, Images)
    assert out.pixels.shape == (2, 3, 4, 4)


def test_heterogeneous_batch_raises_at_primitive_build() -> None:
    """A batch where some prompts have images and others don't fails fast."""
    images = [_img(), None, _img()]
    prompts = Prompts.from_unique_prompts(["a", "b", "c"], prompt_ids=["p0", "p1", "p2"], images=images)
    with pytest.raises(ValueError, match="heterogeneous"):
        _build_images_primitive(prompts.images, prompt_count=len(prompts.prompts), context="plan_requests")


def test_all_none_images_yields_no_primitive() -> None:
    """A batch where every prompt opts out → no ``image`` primitive injection."""
    prompts = Prompts.from_unique_prompts(["a", "b"], prompt_ids=["p0", "p1"], images=[None, None])
    out = _build_images_primitive(prompts.images, prompt_count=len(prompts.prompts), context="plan_requests")
    assert out is None


def test_prompts_without_images_yields_no_primitive() -> None:
    """T2V path: prompts with no images field still works (empty list default)."""
    prompts = Prompts.from_unique_prompts(["a", "b"], prompt_ids=["p0", "p1"])
    out = _build_images_primitive(prompts.images, prompt_count=len(prompts.prompts), context="plan_requests")
    assert out is None
