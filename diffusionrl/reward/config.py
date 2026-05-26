"""Typed reward config: flat ``RewardConfig`` carrying recipe + runtime knob.

Components are a polymorphic dict dispatched by key against the
``reward/component`` ConfigStore group; each scorer registers its own typed
Spec inheriting from :class:`BaseRewardComponentSpec`.
"""

from __future__ import annotations

from typing import Dict

from diffusionrl.config.polymorphic import polymorphic_field
from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.reward.base import BaseRewardComponentSpec


@register_config(group="reward", name="default")
class RewardConfig:
    """Reward recipe + cluster-level runtime knob.

    ``base_device`` is the default device for local scorers whose Spec sets
    ``device="auto"``. Per-Spec ``device`` overrides (``cpu``/``cuda``) win.
    """

    aggregation_method: str = "weighted_sum"
    base_device: str = "cpu"
    components: Dict[str, BaseRewardComponentSpec] = polymorphic_field(
        group="reward/component",
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        require(
            self.aggregation_method in {"weighted_sum", "mean", "min", "max", "concat"},
            f"RewardConfig.aggregation_method must be one of "
            f"weighted_sum/mean/min/max/concat; got {self.aggregation_method!r}",
        )
        require(
            str(self.base_device or "").strip().lower() in {"cpu", "cuda", "auto"},
            f"RewardConfig.base_device must be cpu/cuda/auto; got {self.base_device!r}",
        )
        require(
            len(self.components) > 0,
            "RewardConfig.components must be non-empty",
        )


__all__ = [
    "RewardConfig",
]
