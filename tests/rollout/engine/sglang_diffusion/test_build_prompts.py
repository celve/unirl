"""Unit tests for ``build_prompts`` after de-expand removal (LIN-513).

Covers the base ``ImageAdapter`` (T2I) and the ``QwenImageEditPlusAdapter``
override (text+image edit): every sample's prompt — and, for Edit-Plus, its
source image — is emitted as its own request, no group collapse, never
``num_outputs_per_prompt``. Importing an adapter pulls the engine stack, so these
are skipped where torch is unavailable. ``build_prompts`` ignores ``self`` on the
paths under test, so each is exercised as an unbound call over a light request
stand-in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _build_prompts(texts):
    pytest.importorskip("torch")
    from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter

    req = SimpleNamespace(primitives={"text": SimpleNamespace(texts=texts)})
    return ImageAdapter.build_prompts(None, req)


def test_emits_all_prompts_without_grouping():
    out = _build_prompts(["p", "p", "q", "q"])
    assert out == {"prompt": ["p", "p", "q", "q"]}
    assert "num_outputs_per_prompt" not in out


def test_single_prompt_is_scalar():
    out = _build_prompts(["only"])
    assert out == {"prompt": "only"}


def _build_prompts_edit_plus(texts, pils):
    pytest.importorskip("torch")
    from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image_edit_plus import (
        QwenImageEditPlusAdapter,
    )

    req = SimpleNamespace(
        primitives={
            "text": SimpleNamespace(texts=texts),
            "image": SimpleNamespace(to_pils=lambda: pils),
        }
    )
    return QwenImageEditPlusAdapter.build_prompts(None, req)


def test_edit_plus_emits_all_prompts_and_images_without_grouping():
    # Two groups of K=2 (prompts repeat; source image repeats within a group).
    out = _build_prompts_edit_plus(["p", "p", "q", "q"], ["ia", "ia", "ib", "ib"])
    assert out["prompt"] == ["p", "p", "q", "q"]
    assert out["condition_image"] == ["ia", "ia", "ib", "ib"]
    assert "num_outputs_per_prompt" not in out


def test_edit_plus_single_sample_is_scalar():
    out = _build_prompts_edit_plus(["only"], ["img"])
    assert out == {"prompt": "only", "condition_image": "img"}


def test_edit_plus_image_prompt_count_mismatch_raises():
    with pytest.raises(ValueError, match="image batch"):
        _build_prompts_edit_plus(["p", "q"], ["only-one"])
