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

from omegaconf import SI

from diffusionrl.config.polymorphic import polymorphic_field
from diffusionrl.config.registration import register_config
from diffusionrl.types.sampling import DiffusionSamplingParams
from diffusionrl.utils.scheduler_utils import SchedulerConfig

from .base import BaseAlgorithmConfig


@dataclass
class GRPOEMAConfig:
    enable_eval_ema: bool = True
    eval_ema_decay: float = 0.9
    eval_ema_update_interval: int = 1
    reference_mode: str = "none"
    reference_decay: float = 0.0
    reference_decay_type: str = "constant"
    reference_flat_steps: int = 0
    reference_uprate: float = 0.001
    reference_uphold: float = 0.5
    reference_update_timing: str = "optimizer_step"
    old_adapter_name: str = "old"
    new_adapter_name: str = "default"


@register_config(
    group="algorithm",
    name="grpo",
    mutable=True,
)
@dataclass
class GRPORolloutControlConfig(BaseAlgorithmConfig):
    """Driver-side GRPO algorithm config.

    Fields read by the driver (``train.py``, ``rollout/pipeline.py``):

    - ``samples_per_prompt`` (also on ``DiffusionSamplingParams``)
    - ``prompts_per_rollout``
    - ``pe_rewrites_per_prompt``
    - ``scheduler`` (built into a ``TimestepScheduler`` by the driver)

    Fields read by the actor-side advantage normalization
    (``ray/mixins/rollout_pipeline.py``):

    - ``adv_normalization_scope``
    - ``use_global_std``

    Fields carried for recipe-shape compat (consumed by the plural
    ``cfg.algorithms.<slot>`` trainer-side configs, not here):

    - ``clip_range``, ``clip_schedule``
    - ``use_kl_penalty``, ``kl_coef``
    - ``ema``
    """

    sampling: DiffusionSamplingParams = field(default_factory=lambda: SI("${sampling}"))

    prompts_per_rollout: int = 1
    samples_per_prompt: int = 1
    pe_rewrites_per_prompt: int = 1

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    adv_normalization_scope: str = "group"
    use_global_std: bool = False
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    use_kl_penalty: bool = True
    kl_coef: float = 0.0
    ema: GRPOEMAConfig = field(default_factory=GRPOEMAConfig)

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
    "GRPOEMAConfig",
    "GRPORolloutControlConfig",
    "NFTRolloutControlConfig",
]
