"""Placeholder aesthetic reward scorer."""

from __future__ import annotations

from typing import List

from diffusionrl.config.registration import register_config
from diffusionrl.reward.base import BaseRewardComponentSpec
from diffusionrl.types.reward import RewardRequest

from .base import LocalRewardBackend


class AestheticRewardScorer(LocalRewardBackend):
    """Placeholder scorer for future aesthetic reward support."""

    canonical_model_name = "aesthetic"

    def __init__(self, *, config: "AestheticSpec", base_device: str) -> None:
        # Unused: aesthetic is a placeholder; both args ignored.
        del config, base_device
        super().__init__()

    def _load_model(self) -> None:
        raise NotImplementedError("Aesthetic model loading not yet implemented")

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        raise NotImplementedError("Aesthetic reward computation not yet implemented")


@register_config(
    group="reward/component",
    name="aesthetic",
    target="diffusionrl.reward.local.aesthetic.AestheticRewardScorer",
)
class AestheticSpec(BaseRewardComponentSpec):
    """Placeholder Spec for the aesthetic reward component (unimplemented)."""
