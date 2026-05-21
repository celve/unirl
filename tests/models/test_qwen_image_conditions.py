"""Tests for ``QwenImageConditions`` round-trip with the generic ``Conditions`` dict.

Mirrors ``tests/test_sd3_conditions.py`` — the schema shape (text +
optional negative_text, both as :class:`TextEmbedCondition`) is the
same; what differs at runtime is that Qwen-Image's
``TextEmbedCondition`` carries ``embeds`` + ``attn_mask`` without a
``pooled`` vector (the Qwen-VL encoder doesn't emit one).
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.models.qwen_image.conditions import QwenImageConditions
from diffusionrl.types.conditions import ImageLatentCondition, TextEmbedCondition


def _make_text_cond(b: int = 2, seq: int = 4, hidden: int = 8) -> TextEmbedCondition:
    """Qwen-Image-flavored text condition: embeds + attn_mask, no pooled."""
    return TextEmbedCondition(
        embeds=torch.randn(b, seq, hidden),
        attn_mask=torch.ones(b, seq, dtype=torch.long),
        pooled=None,
    )


def test_qwen_image_conditions_to_dict_roundtrip_cfg_off():
    """When negative_text is None, to_dict emits only the 'text' key."""
    cond = _make_text_cond()
    qwen = QwenImageConditions(text=cond)
    d = qwen.to_dict()
    assert set(d.keys()) == {"text"}
    assert d["text"] is cond
    qwen_round = QwenImageConditions.from_dict(d)
    assert qwen_round.text is cond
    assert qwen_round.negative_text is None


def test_qwen_image_conditions_to_dict_roundtrip_cfg_on():
    """With negative_text set, to_dict emits both keys; from_dict rebuilds them."""
    pos = _make_text_cond()
    neg = _make_text_cond()
    qwen = QwenImageConditions(text=pos, negative_text=neg)
    d = qwen.to_dict()
    assert set(d.keys()) == {"text", "negative_text"}
    assert d["text"] is pos
    assert d["negative_text"] is neg
    qwen_round = QwenImageConditions.from_dict(d)
    assert qwen_round.text is pos
    assert qwen_round.negative_text is neg


def test_qwen_image_conditions_from_dict_rejects_missing_text():
    with pytest.raises(TypeError, match="text"):
        QwenImageConditions.from_dict({})


def test_qwen_image_conditions_from_dict_rejects_wrong_text_type():
    bad = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="TextEmbedCondition"):
        QwenImageConditions.from_dict({"text": bad})


def test_qwen_image_conditions_from_dict_rejects_wrong_negative_text_type():
    bad_neg = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="negative_text"):
        QwenImageConditions.from_dict({"text": _make_text_cond(), "negative_text": bad_neg})


def test_qwen_image_conditions_to_dict_rejects_unset_text():
    qwen = QwenImageConditions()
    with pytest.raises(ValueError, match="text"):
        qwen.to_dict()
