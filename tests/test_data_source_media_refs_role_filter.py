"""Tests for the data layer's ``(image, condition)`` loader.

``MultimodalRLDataSource._load_condition_images`` (used by both the
collator and the eval batch builder) loads ``MediaRef(modality="image",
role="condition")`` URIs into per-prompt ``Image`` tensors. Other
``(modality, role)`` combinations are left to the rollout-side guard
(``_reject_unsupported_media_refs``) — the loader itself is concerned
only with the supported pair.
"""

from __future__ import annotations

from pathlib import Path

import PIL.Image
import pytest

from diffusionrl.data.data_source import _load_condition_images
from diffusionrl.types.media import MediaRef
from diffusionrl.types.primitives import Image as PrimImage


def _write_red_png(tmpdir: Path, *, name: str, size: int = 4) -> str:
    """Write a tiny red PNG and return its path string."""
    img = PIL.Image.new("RGB", (size, size), color=(255, 0, 0))
    out_path = tmpdir / name
    img.save(out_path, format="PNG")
    return str(out_path)


def test_returns_none_when_no_media_refs() -> None:
    """An entirely empty list signals the caller can skip the ``images`` key."""
    assert _load_condition_images([]) is None


def test_returns_none_when_all_per_prompt_lists_empty() -> None:
    assert _load_condition_images([[], [], []]) is None


def test_returns_none_when_no_image_condition_refs_present() -> None:
    """Refs exist but none match (image, condition) — caller should omit the
    ``images`` key entirely, and the rollout guard handles the rest."""
    refs = [
        [MediaRef(modality="audio", role="condition", uri="/data/x.wav")],
        [MediaRef(modality="image", role="reference", uri="/data/y.png")],
    ]
    assert _load_condition_images(refs) is None


def test_loads_image_condition_into_primimage(tmp_path: Path) -> None:
    p = _write_red_png(tmp_path, name="cond.png", size=8)
    refs = [[MediaRef(modality="image", role="condition", uri=p)]]
    out = _load_condition_images(refs)
    assert out is not None
    assert len(out) == 1
    img = out[0]
    assert isinstance(img, PrimImage)
    assert img.pixels is not None
    # to_tensor → [3, H, W] in [0, 1].
    assert img.pixels.shape == (3, 8, 8)
    assert float(img.pixels.max().item()) <= 1.0
    assert float(img.pixels.min().item()) >= 0.0


def test_mixed_batch_emits_none_for_prompts_without_condition_image(
    tmp_path: Path,
) -> None:
    p = _write_red_png(tmp_path, name="cond.png")
    refs = [
        [MediaRef(modality="image", role="condition", uri=p)],
        [],
        [MediaRef(modality="image", role="condition", uri=p)],
    ]
    out = _load_condition_images(refs)
    assert out is not None
    assert len(out) == 3
    assert isinstance(out[0], PrimImage)
    assert out[1] is None
    assert isinstance(out[2], PrimImage)


def test_extra_irrelevant_refs_are_ignored(tmp_path: Path) -> None:
    """A prompt may legitimately carry extra refs the data layer doesn't
    consume (e.g. metadata images for a future modality). As long as
    exactly one (image, condition) is present, the loader picks it."""
    p = _write_red_png(tmp_path, name="cond.png")
    refs = [
        [
            MediaRef(modality="image", role="reference", uri="/data/skipped.png"),
            MediaRef(modality="image", role="condition", uri=p),
        ],
    ]
    out = _load_condition_images(refs)
    assert out is not None
    assert isinstance(out[0], PrimImage)


def test_multiple_condition_images_per_prompt_raises(tmp_path: Path) -> None:
    """WAN I2V is single-frame-conditioned; >1 (image, condition) per
    prompt is ambiguous and must fail loudly."""
    p1 = _write_red_png(tmp_path, name="a.png")
    p2 = _write_red_png(tmp_path, name="b.png")
    refs = [
        [
            MediaRef(modality="image", role="condition", uri=p1),
            MediaRef(modality="image", role="condition", uri=p2),
        ],
    ]
    with pytest.raises(ValueError, match="<=1"):
        _load_condition_images(refs)
