"""Tests for ``HunyuanVideo15Conditions`` round-trip.

Diverges from ``test_sd3_conditions.py`` / ``test_qwen_image_conditions.py``
in that HunyuanVideo-1.5 carries **two** parallel text streams
(``text_mllm`` for the Qwen2.5-VL MLLM and ``text_glyph`` for the ByT5
glyph encoder); both are required by ``from_dict`` and ``to_dict``
because the transformer's cross-attention is dual-stream by contract.
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.models_new.hunyuan_video15.conditions import HunyuanVideo15Conditions
from diffusionrl.types.conditions import (
    ImageLatentCondition,
    TextEmbedCondition,
)


def _make_text_cond(b: int = 2, seq: int = 4, hidden: int = 8) -> TextEmbedCondition:
    """HunyuanVideo-1.5-flavored text condition: embeds + attn_mask, no pooled."""
    return TextEmbedCondition(
        embeds=torch.randn(b, seq, hidden),
        attn_mask=torch.ones(b, seq, dtype=torch.long),
        pooled=None,
    )


def test_conditions_to_dict_roundtrip_cfg_off():
    """When the negative branches are None, to_dict emits only the two
    required text keys."""
    mllm = _make_text_cond(seq=16, hidden=32)
    glyph = _make_text_cond(seq=8, hidden=64)
    cond = HunyuanVideo15Conditions(text_mllm=mllm, text_glyph=glyph)
    d = cond.to_dict()
    assert set(d.keys()) == {"text_mllm", "text_glyph"}
    assert d["text_mllm"] is mllm
    assert d["text_glyph"] is glyph
    cond_round = HunyuanVideo15Conditions.from_dict(d)
    assert cond_round.text_mllm is mllm
    assert cond_round.text_glyph is glyph
    assert cond_round.negative_text_mllm is None
    assert cond_round.negative_text_glyph is None
    assert cond_round.vision is None


def test_conditions_to_dict_roundtrip_cfg_on():
    """With negative branches set, to_dict emits all four text keys."""
    pos_mllm = _make_text_cond(seq=16, hidden=32)
    pos_glyph = _make_text_cond(seq=8, hidden=64)
    neg_mllm = _make_text_cond(seq=16, hidden=32)
    neg_glyph = _make_text_cond(seq=8, hidden=64)
    cond = HunyuanVideo15Conditions(
        text_mllm=pos_mllm,
        text_glyph=pos_glyph,
        negative_text_mllm=neg_mllm,
        negative_text_glyph=neg_glyph,
    )
    d = cond.to_dict()
    assert set(d.keys()) == {
        "text_mllm",
        "text_glyph",
        "negative_text_mllm",
        "negative_text_glyph",
    }
    cond_round = HunyuanVideo15Conditions.from_dict(d)
    assert cond_round.text_mllm is pos_mllm
    assert cond_round.text_glyph is pos_glyph
    assert cond_round.negative_text_mllm is neg_mllm
    assert cond_round.negative_text_glyph is neg_glyph


def test_conditions_from_dict_rejects_missing_text_mllm():
    """text_mllm is required — transformer is dual-stream by contract."""
    with pytest.raises(TypeError, match="text_mllm"):
        HunyuanVideo15Conditions.from_dict({"text_glyph": _make_text_cond()})


def test_conditions_from_dict_rejects_missing_text_glyph():
    """text_glyph is required — transformer is dual-stream by contract."""
    with pytest.raises(TypeError, match="text_glyph"):
        HunyuanVideo15Conditions.from_dict({"text_mllm": _make_text_cond()})


def test_conditions_from_dict_rejects_wrong_text_mllm_type():
    bad = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="TextEmbedCondition"):
        HunyuanVideo15Conditions.from_dict({"text_mllm": bad, "text_glyph": _make_text_cond()})


def test_conditions_from_dict_rejects_wrong_negative_text_type():
    bad_neg = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="negative_text_mllm"):
        HunyuanVideo15Conditions.from_dict(
            {
                "text_mllm": _make_text_cond(),
                "text_glyph": _make_text_cond(),
                "negative_text_mllm": bad_neg,
            }
        )


def test_conditions_to_dict_rejects_unset_text():
    """to_dict must raise when either stream is None."""
    cond = HunyuanVideo15Conditions()
    with pytest.raises(ValueError, match="text_mllm and text_glyph"):
        cond.to_dict()
    # Only one set is also invalid.
    cond2 = HunyuanVideo15Conditions(text_mllm=_make_text_cond())
    with pytest.raises(ValueError, match="text_mllm and text_glyph"):
        cond2.to_dict()
