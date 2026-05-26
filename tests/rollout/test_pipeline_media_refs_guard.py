"""Unit tests for the driver's media_refs filter.

The driver consumes ``(modality="image", role="condition")`` refs
through ``RolloutInputs.primitives['image']: Images``. The
:func:`_reject_unsupported_media_refs` helper raises
``NotImplementedError`` on any **unsupported** ``(modality, role)``
combination so misconfigured datasets fail loudly.
"""

from __future__ import annotations

import pytest

from diffusionrl.rollout.pipeline import _reject_unsupported_media_refs
from diffusionrl.types.media import MediaRef


def test_no_media_refs_key_is_silently_ok() -> None:
    """Most NEW-path datasets never produce media_refs — no-op."""
    _reject_unsupported_media_refs({"prompts": ["a", "b"]}, context="t")


def test_none_media_refs_is_silently_ok() -> None:
    _reject_unsupported_media_refs({"media_refs": None}, context="t")


def test_empty_top_level_list_is_silently_ok() -> None:
    _reject_unsupported_media_refs({"media_refs": []}, context="t")


def test_all_empty_per_prompt_lists_silently_ok() -> None:
    """Pure T2I/T2V dataset where the loader still emits the key but every
    prompt has [] — no actual media to drop, so no fail-fast needed."""
    _reject_unsupported_media_refs({"media_refs": [[], [], []]}, context="t")


def test_supported_image_condition_refs_are_silently_ok() -> None:
    """``(image, condition)`` refs flow through to
    ``RolloutInputs.primitives['image']`` and must NOT raise."""
    refs = [
        [MediaRef(modality="image", role="condition", uri="/data/a.png")],
        [],
        [MediaRef(modality="image", role="condition", uri="/data/c.png")],
    ]
    _reject_unsupported_media_refs({"media_refs": refs}, context="t")


def test_unsupported_role_raises_not_implemented() -> None:
    """A modality we recognize (image) under an unsupported role must fail loud."""
    refs = [
        [MediaRef(modality="image", role="reference", uri="/data/x.png")],
    ]
    with pytest.raises(NotImplementedError, match="unsupported"):
        _reject_unsupported_media_refs({"media_refs": refs}, context="MyDriver.load")


def test_unsupported_modality_raises_not_implemented() -> None:
    """Any non-image modality is unsupported on the path."""
    refs = [
        [MediaRef(modality="video", role="condition", uri="/data/x.mp4")],
    ]
    with pytest.raises(NotImplementedError, match="unsupported"):
        _reject_unsupported_media_refs({"media_refs": refs}, context="MyDriver.load")


def test_context_label_is_included_in_error() -> None:
    refs = [[MediaRef(modality="image", role="reference", uri="/data/x.png")]]
    with pytest.raises(NotImplementedError, match="MyContext"):
        _reject_unsupported_media_refs({"media_refs": refs}, context="MyContext")


def test_non_list_media_refs_raises_type_error() -> None:
    with pytest.raises(TypeError, match="must be a list"):
        _reject_unsupported_media_refs({"media_refs": "oops-string"}, context="t")
