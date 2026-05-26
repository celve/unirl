"""Composed rollout-engine configuration.

Registered under ``rollout/engine: composed_pe`` with ``_target_`` pointing at
:class:`ComposedRolloutEngine`. The composed engine holds two child engines
(``ar`` and ``diffusion``) and orchestrates the prompt-enhancement (PE)
serial flow internally — it is a peer of the existing ``sglang``,
``sglang_llm``, ``vllm_omni``, and ``trainside`` engines as far as the
actor / pipeline layer is concerned.

Children are single polymorphic fields resolved against the
``rollout/engine`` ConfigStore group. Each child carries its own typed
config after ``expand_polymorphic_fields`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffusionrl.config.polymorphic import polymorphic_field
from diffusionrl.config.registration import register_config
from diffusionrl.rollout.engine.base import BaseEngineConfig


@register_config(
    group="rollout/engine",
    name="composed_pe",
    target="diffusionrl.rollout.engine.composed.engine.ComposedRolloutEngine",
)
@dataclass
class ComposedRolloutEngineConfig(BaseEngineConfig):
    """Two-stage prompt-enhancement (PE) composed rollout engine."""

    ar: BaseEngineConfig = polymorphic_field(group="rollout/engine")
    diffusion: BaseEngineConfig = polymorphic_field(group="rollout/engine")

    sleep_diffusion_on_start: bool = True


__all__ = ["ComposedRolloutEngineConfig"]
