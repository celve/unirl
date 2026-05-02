"""FSDP direct-sampling engine configuration.

Registered under ``rollout/engine: fsdp`` with ``_target_`` pointing at
``FSDPSamplingEngine``. ``TrainActor`` builds the engine via
``build(cfg.rollout.engine)`` — the same unified path ``RolloutActor``
uses — instead of the legacy ad-hoc construction from ``cfg.sampling``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import SI

from diffusionrl.config.registration import register_config
from diffusionrl.types.sampling import SamplingParams


@register_config(
    group="rollout/engine",
    name="fsdp",
    target="diffusionrl.samplers.fsdp.engine.FSDPSamplingEngine",
)
@dataclass
class FSDPEngineConfig:
    """Configuration for the in-process FSDP direct-sampling engine."""

    # Default to a live interpolation back to top-level cfg.sampling so the
    # engine's sampling block tracks the canonical recipe spec without each
    # recipe having to override every field. Mirrors the pattern on
    # BaseAlgorithmConfig.sampling (algorithms/base.py). Without this, the
    # engine falls through to a fresh SamplingParams() with SDEConfig defaults
    # (eta=1.0, shift=3.0), so the recipe's sampling.sde_config.eta=0.7 never
    # propagates into rollout-side log_prob math.
    sampling: SamplingParams = field(default_factory=lambda: SI("${sampling}"))


__all__ = ["FSDPEngineConfig"]
