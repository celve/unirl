"""Tests for GSM8KExactMatchRewardScorer and _extract_answer.

Mirrors test_mc_exact_match.py: stubs the ``diffusionrl.reward.local`` package
``__init__`` (which eagerly imports OCR/video scorers with heavy native deps)
so the submodule can be imported in isolation.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

_SCORERS_PKG = "diffusionrl.reward.local"
if _SCORERS_PKG not in sys.modules:
    import diffusionrl.reward  # ensure parent is loaded

    _stub = types.ModuleType(_SCORERS_PKG)
    _scorers_dir = os.path.join(os.path.dirname(diffusionrl.reward.__file__), "local")
    _stub.__path__ = [_scorers_dir]
    _stub.__package__ = _SCORERS_PKG
    sys.modules[_SCORERS_PKG] = _stub

from diffusionrl.reward.local.gsm8k_exact_match import (  # noqa: E402
    GSM8KExactMatchRewardScorer,
    GSM8KExactMatchSpec,
    _extract_answer,
)
from diffusionrl.types.reward import RewardRequest  # noqa: E402


class TestExtractAnswer:
    def test_hash_marker(self):
        assert _extract_answer("reasoning ... #### 1234") == "1234"

    def test_hash_marker_strips_commas(self):
        assert _extract_answer("#### 1,234") == "1234"

    def test_hash_marker_negative(self):
        assert _extract_answer("#### -5") == "-5"

    def test_hash_marker_decimal(self):
        assert _extract_answer("#### 3.5") == "3.5"

    def test_hash_marker_wins_over_trailing_number(self):
        assert _extract_answer("#### 7 (since 1 + 6 = 7, but 99 is noise)") == "7"

    def test_last_number_fallback(self):
        assert _extract_answer("he worked 356 hours") == "356"

    def test_last_of_many_numbers(self):
        assert _extract_answer("a=2, b=3, total 42") == "42"

    def test_no_number(self):
        assert _extract_answer("no digits here") == ""

    def test_empty(self):
        assert _extract_answer("") == ""


def _t(texts):
    from diffusionrl.types.primitives import Texts

    return Texts(texts=list(texts))


class TestGSM8KScorer:
    @pytest.fixture
    def scorer(self):
        return GSM8KExactMatchRewardScorer(config=GSM8KExactMatchSpec(), base_device="cpu")

    def test_correct_answer(self, scorer):
        request = RewardRequest(
            primitives={"text": _t(["What is 6 * 7?"])},
            generated={"text": _t(["#### 42"])},
            metadata=[{"answer": "42"}],
        )
        assert scorer.compute_rewards(request).rewards == [1.0]

    def test_wrong_answer(self, scorer):
        request = RewardRequest(
            generated={"text": _t(["#### 41"])},
            metadata=[{"answer": "42"}],
        )
        assert scorer.compute_rewards(request).rewards == [0.0]

    def test_ground_truth_comma_normalized(self, scorer):
        request = RewardRequest(
            generated={"text": _t(["the total is #### 1234"])},
            metadata=[{"answer": "1,234"}],
        )
        assert scorer.compute_rewards(request).rewards == [1.0]

    def test_missing_metadata(self, scorer):
        request = RewardRequest(
            generated={"text": _t(["#### 42"])},
            metadata=[None],
        )
        assert scorer.compute_rewards(request).rewards == [0.0]

    def test_no_answer_field(self, scorer):
        request = RewardRequest(
            generated={"text": _t(["#### 42"])},
            metadata=[{"foo": "bar"}],
        )
        assert scorer.compute_rewards(request).rewards == [0.0]

    def test_batch(self, scorer):
        request = RewardRequest(
            generated={"text": _t(["#### 42", "the answer is 100", "#### 7"])},
            metadata=[{"answer": "42"}, {"answer": "99"}, {"answer": "7"}],
        )
        assert scorer.compute_rewards(request).rewards == [1.0, 0.0, 1.0]

    def test_input_kind(self, scorer):
        assert scorer.preferred_input_kind == "text"
