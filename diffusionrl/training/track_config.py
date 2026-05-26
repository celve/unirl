"""Per-track training schema for the multi-track ``StageTrainStack``.

A *track* is one parallel training context: one model bundle, one set of
wrap-function configs (LoRA, EMA, FSDP), one optimizer, one scheduler,
one algorithm.  Track keys in the cfg match ``RolloutResp.tracks`` keys
and ``cfg.algorithms.<track_name>`` keys.

Shape (Hydra-composed at runtime)::

    training:
      tracks:
        image:
          model: {_target_: ...sd3.SD3Pipeline, ...}
          source_stage_attr: diffusion
          lora: {rank: 32, alpha: 64}
          fsdp: {param_dtype: bf16, activation_checkpointing: true}
          optimizer: {_target_: ..., learning_rate: 3.0e-4, ...}
          lr_scheduler: {_target_: ..., type: constant, ...}
          plan_overrides: {micro_batch_size: 4}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from diffusionrl.config.registration import register_config
from diffusionrl.training.backends.base import LrSchedulerConfig, OptimizerConfig
from diffusionrl.training.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)


@register_config(group="training/track_plan_overrides", name="default")
@dataclass
class TrackPlanOverrides:
    """Per-track overrides for the shared ``cfg.training.plan``.

    Only ``micro_batch_size`` is exposed today — global batch sizing,
    optimizer-step count, and topology stay shared across tracks.
    """

    micro_batch_size: Optional[int] = None


@register_config(group="training/track", name="default", mutable=True)
@dataclass
class TrainingTrackConfig:
    """One training track: model + wrap-function configs + optimizer.

    ``lora`` and ``ema_lora`` are mutually exclusive (both inject LoRA
    adapters).  ``ema`` can combine with ``lora`` (mirrors the
    post-LoRA trainable parameter set).  ``fsdp`` is always present.
    """

    # Any: OmegaConf requires it for polymorphic Hydra config nodes.
    model: Any
    source_stage_attr: str
    optimizer: OptimizerConfig
    lr_scheduler: LrSchedulerConfig

    lora: Optional[LoraConfig] = None
    ema_lora: Optional[EmaLoraConfig] = None
    ema: Optional[EmaFullConfig] = None
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    plan_overrides: Optional[TrackPlanOverrides] = None


__all__ = [
    "TrackPlanOverrides",
    "TrainingTrackConfig",
]
