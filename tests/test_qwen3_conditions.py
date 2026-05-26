"""Tests for ``Qwen3ARConditions`` round-trip with the generic ``Conditions`` dict.

Mirrors ``tests/test_qwen_image_conditions.py``. The Qwen3 AR conditions
wrap a single :class:`TextTokenCondition` carrying the chat-template
``input_ids`` + ``attention_mask`` that the AR stage consumes.
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.models.qwen3.conditions import Qwen3ARConditions
from diffusionrl.types.conditions import ImageLatentCondition, TextTokenCondition


def _make_prompt_cond(b: int = 2, seq: int = 4) -> TextTokenCondition:
    return TextTokenCondition(
        input_ids=torch.zeros(b, seq, dtype=torch.long),
        attention_mask=torch.ones(b, seq, dtype=torch.long),
    )


def test_qwen3_conditions_to_dict_roundtrip():
    cond = _make_prompt_cond()
    qwen = Qwen3ARConditions(prompt=cond)
    d = qwen.to_dict()
    assert set(d.keys()) == {"prompt"}
    assert d["prompt"] is cond
    qwen_round = Qwen3ARConditions.from_dict(d)
    assert qwen_round.prompt is cond


def test_qwen3_conditions_from_dict_rejects_missing_prompt():
    with pytest.raises(TypeError, match="prompt"):
        Qwen3ARConditions.from_dict({})


def test_qwen3_conditions_from_dict_rejects_wrong_prompt_type():
    bad = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="TextTokenCondition"):
        Qwen3ARConditions.from_dict({"prompt": bad})


def test_qwen3_conditions_to_dict_rejects_unset_prompt():
    qwen = Qwen3ARConditions()
    with pytest.raises(ValueError, match="prompt"):
        qwen.to_dict()
