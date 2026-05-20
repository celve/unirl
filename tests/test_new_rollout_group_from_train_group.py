"""Tests for :meth:`NewRolloutActorGroup.from_train_group`.

CPU-only. Builds a fake train group with two dummy actor handles and
verifies the alt constructor:

- adopts the handles (no spawn loop, no health-check fan-out),
- materializes ``rollout_plan`` from ``cfg.rollout.plan``,
- exposes the standard rollout-group surface (``num_actors``,
  ``get_actors``, ``rollout_plan``),
- raises when ``cfg.rollout.engine`` does not target a direct-sampling
  engine.

No Ray, no real actors.
"""

from __future__ import annotations

from typing import Any, List

import pytest
from omegaconf import OmegaConf

# Ensure the trainside engine is importable so its registration runs
# (the alt-ctor's predicate keys off the registered target suffix).
from diffusionrl.rollout.engine.trainside import (  # noqa: F401
    TrainsideRolloutEngine,
)


class _FakeTrainGroup:
    """Stand-in for :class:`NewTrainActorGroup`.

    Implements only the surface used by ``from_train_group``: ``get_actors``.
    """

    def __init__(self, handles: List[Any]) -> None:
        self._handles = list(handles)

    def get_actors(self) -> List[Any]:
        return list(self._handles)


def _make_cfg(engine_target: str, forward_batch_size: int = 4) -> Any:
    """Build a cfg with a *structured* rollout.plan so ``materialize`` returns
    a real ``RolloutPlan`` instance rather than a plain dict."""
    from diffusionrl.rollout.plan import RolloutPlan

    plan_cfg = OmegaConf.structured(RolloutPlan(forward_batch_size=int(forward_batch_size)))
    cfg = OmegaConf.create(
        {
            "rollout": {
                "engine": {"_target_": engine_target},
            }
        }
    )
    OmegaConf.update(cfg, "rollout.plan", plan_cfg, force_add=True)
    return cfg


def test_adopts_train_actor_handles_without_spawning() -> None:
    from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup

    handles = [object(), object()]
    train_group = _FakeTrainGroup(handles)
    cfg = _make_cfg(
        engine_target="diffusionrl.rollout.engine.trainside.engine.TrainsideRolloutEngine",
        forward_batch_size=8,
    )

    rollout_group = NewRolloutActorGroup.from_train_group(train_group, cfg=cfg)

    assert rollout_group.num_actors == 2
    assert rollout_group.get_actors() == handles
    assert rollout_group.rollout_plan.forward_batch_size == 8
    assert rollout_group._sampler_engine_type == "trainside"


def test_rejects_non_direct_sampling_engine() -> None:
    from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup

    train_group = _FakeTrainGroup([object()])
    cfg = _make_cfg(
        engine_target="diffusionrl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine",
    )

    with pytest.raises(ValueError, match="direct-sampling engine"):
        NewRolloutActorGroup.from_train_group(train_group, cfg=cfg)


def test_rejects_empty_train_group() -> None:
    from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup

    train_group = _FakeTrainGroup([])
    cfg = _make_cfg(
        engine_target="diffusionrl.rollout.engine.trainside.engine.TrainsideRolloutEngine",
    )

    with pytest.raises(ValueError, match="no actor handles"):
        NewRolloutActorGroup.from_train_group(train_group, cfg=cfg)
