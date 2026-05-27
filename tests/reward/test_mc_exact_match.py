"""Tests for MCExactMatchRewardScorer and _extract_answer_letter.

Uses ``sys.modules`` patching to load the mc_exact_match module without
triggering the ``diffusionrl.reward.scorers`` package ``__init__`` (which
eagerly imports OCR/video scorers with heavy native deps like tqdm/paddleocr).
"""

from __future__ import annotations

import os
import sys
import types

import pytest

# Stub the scorers package so Python doesn't execute __init__.py when
# importing the mc_exact_match submodule. The __init__ eagerly imports
# OCR/video scorers with heavy native deps (tqdm, paddleocr).
_SCORERS_PKG = "diffusionrl.reward.scorers"
if _SCORERS_PKG not in sys.modules:
    import diffusionrl.reward  # ensure parent is loaded

    _stub = types.ModuleType(_SCORERS_PKG)
    _scorers_dir = os.path.join(os.path.dirname(diffusionrl.reward.__file__), "scorers")
    _stub.__path__ = [_scorers_dir]
    _stub.__package__ = _SCORERS_PKG
    sys.modules[_SCORERS_PKG] = _stub

from diffusionrl.reward.scorers.mc_exact_match import (  # noqa: E402
    MCExactMatchRewardScorer,
    MCExactMatchSpec,
    _extract_answer_letter,
)
from diffusionrl.types.reward import RewardRequest  # noqa: E402


class TestExtractAnswerLetter:
    def test_single_letter(self):
        assert _extract_answer_letter("B") == "B"

    def test_single_letter_lowercase(self):
        assert _extract_answer_letter("c") == "C"

    def test_answer_is_pattern(self):
        assert _extract_answer_letter("The answer is B") == "B"

    def test_answer_colon_pattern(self):
        assert _extract_answer_letter("Answer: D") == "D"

    def test_option_is_pattern(self):
        assert _extract_answer_letter("option is A") == "A"

    def test_parenthesized(self):
        assert _extract_answer_letter("The answer is (C)") == "C"

    def test_verbose_cot_with_answer(self):
        text = "First I note the triangle is isosceles. The base angles are equal. The answer is B."
        assert _extract_answer_letter(text) == "B"

    def test_last_standalone_letter_fallback(self):
        assert _extract_answer_letter("Looking at this, I'd go with D") == "D"

    def test_no_valid_letter(self):
        assert _extract_answer_letter("I don't know the solution") == ""

    def test_empty_string(self):
        assert _extract_answer_letter("") == ""

    def test_whitespace_only(self):
        assert _extract_answer_letter("   ") == ""

    def test_letter_with_whitespace(self):
        assert _extract_answer_letter("  A  ") == "A"

    def test_non_abcd_letter(self):
        assert _extract_answer_letter("E") == ""


def _t(texts):
    """Helper: build Texts primitive."""
    from diffusionrl.types.primitives import Texts

    return Texts(texts=list(texts))


class TestMCExactMatchScorer:
    @pytest.fixture
    def scorer(self):
        spec = MCExactMatchSpec(weight=1.0)
        return MCExactMatchRewardScorer(config=spec, base_device="cpu")

    def test_correct_answer(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["What is the angle?"])},
            generated={"text": _t(["C"])},
            metadata=[{"answer": "C", "choices": ["30", "45", "60", "90"]}],
        )
        response = scorer.compute_rewards(request)
        assert response.rewards == [1.0]

    def test_wrong_answer(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["What is the angle?"])},
            generated={"text": _t(["A"])},
            metadata=[{"answer": "C"}],
        )
        response = scorer.compute_rewards(request)
        assert response.rewards == [0.0]

    def test_missing_metadata(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["What is the angle?"])},
            generated={"text": _t(["B"])},
            metadata=[None],
        )
        response = scorer.compute_rewards(request)
        assert response.rewards == [0.0]

    def test_no_metadata_field(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["What is the angle?"])},
            generated={"text": _t(["B"])},
        )
        response = scorer.compute_rewards(request)
        assert response.rewards == [0.0]

    def test_batch(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["q1", "q2", "q3", "q4"])},
            generated={"text": _t(["A", "B", "C", "D"])},
            metadata=[
                {"answer": "A"},
                {"answer": "C"},
                {"answer": "C"},
                {"answer": "D"},
            ],
        )
        response = scorer.compute_rewards(request)
        assert response.rewards == [1.0, 0.0, 1.0, 1.0]

    def test_verbose_answer_extraction(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["What is x?"])},
            generated={"text": _t(["The answer is B"])},
            metadata=[{"answer": "B"}],
        )
        response = scorer.compute_rewards(request)
        assert response.rewards == [1.0]

    def test_is_available(self, scorer):
        assert scorer.is_available()

    def test_input_kind(self, scorer):
        assert scorer.preferred_input_kind == "text"
