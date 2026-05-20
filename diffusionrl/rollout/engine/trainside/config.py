"""Hydra registration for the trainside (in-process) rollout engine.

Empty dataclass — the engine's only runtime deps (``pipeline``,
``policy``) are Python handles owned by the train actor, not Hydra
leaves. ``NewTrainActor`` injects them via ``build(cfg.rollout.engine,
pipeline=..., policy=...)`` the same way ``NewRolloutActor`` injects
``device`` / ``strategy`` / ``rank`` / ``model_config`` into
``VLLMOmniRolloutEngine`` at build time.

The dataclass exists so the Hydra ConfigStore has a leaf under
``rollout/engine: trainside`` whose ``_target_`` points at
:class:`TrainsideRolloutEngine` — which is also what
:func:`diffusionrl.config.validation.is_direct_sampling` keys off.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffusionrl.config.registration import register_config


@register_config(
    group="rollout/engine",
    name="trainside",
    target="diffusionrl.rollout.engine.trainside.engine.TrainsideRolloutEngine",
    expand=True,
)
@dataclass
class TrainsideEngineConfig:
    """No static fields — pipeline/policy are runtime handles, not cfg leaves.

    ``expand=True`` so :func:`build` unpacks the (empty) field set as
    kwargs rather than wrapping into a ``config=`` argument. This lets
    :class:`TrainsideRolloutEngine.__init__` keep its keyword-only
    ``(pipeline, policy)`` signature without accepting a vestigial
    ``config`` parameter.
    """

    pass


__all__ = ["TrainsideEngineConfig"]
