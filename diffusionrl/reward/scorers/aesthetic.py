"""Placeholder aesthetic reward scorer."""

from __future__ import annotations

from typing import List

from diffusionrl.types.reward import RewardRequest

from .base_local import BaseLocalRewardScorer


class AestheticRewardScorer(BaseLocalRewardScorer):
    """Placeholder scorer for future aesthetic reward support."""

    canonical_model_name = "aesthetic"

    def _load_model(self) -> None:
        raise NotImplementedError("Aesthetic model loading not yet implemented")

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        raise NotImplementedError("Aesthetic reward computation not yet implemented")
