"""Hydra registration for the trainside (in-process) rollout engine.

Empty dataclass — the engine's only runtime deps (``pipeline``,
``policy``) are Python handles owned by the train actor, not Hydra
leaves. ``TrainActor`` injects them via ``build(cfg.rollout.engine,
pipeline=..., policy=...)`` the same way ``RolloutActor`` injects
``device`` / ``strategy`` / ``rank`` / ``model_config`` into
``VLLMOmniRolloutEngine`` at build time.

The dataclass exists so the Hydra ConfigStore has a leaf under
``rollout/engine: trainside`` whose ``_target_`` points at
:class:`TrainsideRolloutEngine` — which is also what
:func:`unirl.config.validation.is_direct_sampling` keys off.
"""

from __future__ import annotations

from dataclasses import dataclass

from unirl.config.registration import register_config
from unirl.rollout.engine.base import BaseEngineConfig


@register_config(
    group="rollout/engine",
    name="trainside",
    target="unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine",
    expand=True,
)
@dataclass
class TrainsideEngineConfig(BaseEngineConfig):
    """No static fields — pipeline/policy are runtime handles, not cfg leaves.

    ``expand=True`` so :func:`build` unpacks the (empty) field set as
    kwargs rather than wrapping into a ``config=`` argument. This lets
    :class:`TrainsideRolloutEngine.__init__` keep its keyword-only
    ``(pipeline, policy)`` signature without accepting a vestigial
    ``config`` parameter.
    """

    pass


__all__ = ["TrainsideEngineConfig"]
