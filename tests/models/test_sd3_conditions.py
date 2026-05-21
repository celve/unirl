"""Tests for ``SD3Conditions`` round-trip with the generic ``Conditions`` dict."""

from __future__ import annotations

import pytest
import torch

from diffusionrl.models.sd3.conditions import SD3Conditions
from diffusionrl.types.conditions import ImageLatentCondition, TextEmbedCondition


def _make_text_cond(b: int = 2, seq: int = 4, hidden: int = 8) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=torch.randn(b, seq, hidden),
        pooled=torch.randn(b, hidden * 2),
    )


def test_sd3_conditions_to_dict_roundtrip_cfg_off():
    """When negative_text is None, to_dict emits only the 'text' key."""
    cond = _make_text_cond()
    sd3 = SD3Conditions(text=cond)
    d = sd3.to_dict()
    assert set(d.keys()) == {"text"}
    assert d["text"] is cond
    # Round trip back through from_dict.
    sd3_round = SD3Conditions.from_dict(d)
    assert sd3_round.text is cond
    assert sd3_round.negative_text is None


def test_sd3_conditions_to_dict_roundtrip_cfg_on():
    """With negative_text set, to_dict emits both keys; from_dict rebuilds them."""
    pos = _make_text_cond()
    neg = _make_text_cond()
    sd3 = SD3Conditions(text=pos, negative_text=neg)
    d = sd3.to_dict()
    assert set(d.keys()) == {"text", "negative_text"}
    assert d["text"] is pos
    assert d["negative_text"] is neg
    sd3_round = SD3Conditions.from_dict(d)
    assert sd3_round.text is pos
    assert sd3_round.negative_text is neg


def test_sd3_conditions_from_dict_rejects_missing_text():
    with pytest.raises(TypeError, match="text"):
        SD3Conditions.from_dict({})


def test_sd3_conditions_from_dict_rejects_wrong_text_type():
    bad = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="TextEmbedCondition"):
        SD3Conditions.from_dict({"text": bad})


def test_sd3_conditions_from_dict_rejects_wrong_negative_text_type():
    bad_neg = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="negative_text"):
        SD3Conditions.from_dict({"text": _make_text_cond(), "negative_text": bad_neg})


def test_sd3_conditions_to_dict_rejects_unset_text():
    sd3 = SD3Conditions()
    with pytest.raises(ValueError, match="text"):
        sd3.to_dict()
