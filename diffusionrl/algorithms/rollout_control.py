"""Algorithm config for the singular ``cfg.algorithm`` section.

The singular ``cfg.algorithm`` carries driver-side knobs (scheduler,
sampling multipliers, advantage normalization scope) and nests the
plural ``cfg.algorithm.algorithms.<slot>`` trainer-side configs.

Previously this module also held ``GRPORolloutControl`` /
``NFTRolloutControl`` runtime classes. Those were eliminated: SDE
scheduling is now called directly via ``create_indices_scheduler``,
advantage normalization is inlined at call sites using functions from
``normalizers.py``, and sampling fields live on typed params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from diffusionrl.config.polymorphic import polymorphic_field
from diffusionrl.config.registration import register_config
from diffusionrl.utils.scheduler_utils import SchedulerConfig

from .base import BaseAlgorithmConfig


@register_config(
    group="algorithm",
    name="grpo",
    mutable=True,
)
@dataclass
class GRPORolloutControlConfig(BaseAlgorithmConfig):
    """Driver-side GRPO algorithm config.

    Fields read by the driver (``train.py``, ``rollout/pipeline.py``):

    - ``prompts_per_rollout``
    - ``pe_rewrites_per_prompt``
    - ``scheduler`` (built into a ``TimestepScheduler`` by the driver)

    Fields read by the actor-side advantage normalization
    (``ray/mixins/rollout_pipeline.py``):

    - ``adv_normalization_scope``
    - ``use_global_std``

    ``kl_coef`` is kept only as a fail-fast guard (``__post_init__`` rejects
    ``kl_coef > 0``; the KL term is unimplemented). Per-algorithm
    hyperparameters (``clip_range``, ``clip_schedule``, EMA) live solely on
    the plural ``cfg.algorithm.algorithms.<slot>`` configs.
    """

    prompts_per_rollout: int = 1
    pe_rewrites_per_prompt: int = 1

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    adv_normalization_scope: str = "group"
    use_global_std: bool = False
    kl_coef: float = 0.0

    algorithms: Dict[str, BaseAlgorithmConfig] = polymorphic_field(
        group="algorithm",
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if float(self.kl_coef) > 0:
            raise ValueError(
                f"GRPORolloutControlConfig.kl_coef={self.kl_coef!r}: the KL "
                f"penalty against the reference policy is not implemented. "
                f"Set kl_coef=0 (the recipe default) or implement the term."
            )


@register_config(
    group="algorithm",
    name="nft",
    mutable=True,
)
@dataclass
class NFTRolloutControlConfig(GRPORolloutControlConfig):
    """NFT algorithm config. Schema identical to GRPO; the behavioral
    difference (no SDE steps) is encoded by the driver checking
    ``scheduler.num_sde_steps == 0`` and setting ``sde_scheduler = None``.
    """


__all__ = [
    "GRPORolloutControlConfig",
    "NFTRolloutControlConfig",
]
