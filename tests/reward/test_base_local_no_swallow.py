"""Regression test: ``BaseLocalRewardScorer.compute_rewards`` no longer swallows.

Before the fail-fast fix, an exception inside ``_compute_model_rewards``
was caught and silently turned into ``rewards=[0.0] * batch_size`` with
the real error parked in ``RewardResponse.errors`` (which nothing reads).
This test locks in that the exception now propagates with the original
type and message intact.
"""

from __future__ import annotations

from typing import List

import pytest

from diffusionrl.reward.scorers.base_local import BaseLocalRewardScorer
from diffusionrl.types.reward import RewardRequest


class _DummyLocalScorer(BaseLocalRewardScorer):
    canonical_model_name = "dummy"

    def _load_model(self) -> None:
        return None

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        raise ValueError("boom")


def test_compute_rewards_propagates_errors() -> None:
    scorer = _DummyLocalScorer(model_name="dummy", device="cpu")
    request = RewardRequest(prompts=["a", "b"])
    with pytest.raises(ValueError, match="boom"):
        scorer.compute_rewards(request)
