"""Tests for :class:`TrainsideRolloutEngine`.

CPU-only. Uses fake ``Pipeline`` / ``Stage`` stand-ins to verify:

- ``generate(req)`` delegates to the wrapped pipeline's ``generate`` and
  returns the response verbatim.
- The trainable module(s) are flipped to ``eval()`` for the call duration
  and restored to their prior training mode afterward — for a single stage
  (``stage_attr``) and for multiple stages (``stage_attrs``, e.g. PE).
- ``torch.is_grad_enabled()`` is ``False`` inside the call.
- Lifecycle methods are safe no-ops; weight-sync methods raise
  ``NotImplementedError`` from the base class.

No real model, no FSDP, no Ray.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from diffusionrl.rollout.engine.trainside import (
    TrainsideEngineConfig,
    TrainsideRolloutEngine,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStage:
    """Minimal stage stand-in exposing the trainable ``nn.Module``."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def trainable_module(self) -> nn.Module:
        return self._model


class _FakePipeline:
    """Pipeline stand-in exposing named stages + recording eval/grad state.

    ``bundle.pretrained_path=None`` + ``shift`` satisfy the engine's
    schedule-policy fallback (``FlowMatchSchedulePolicy.from_pretrained(None,
    shift=...)`` is static / no I/O). Each stage is exposed as an attribute
    (e.g. ``self.diffusion``, ``self.ar``) so ``stage_attr`` / ``stage_attrs``
    resolve via ``getattr``.
    """

    bundle = SimpleNamespace(pretrained_path=None)
    shift = 1.0

    def __init__(self, **stages: _FakeStage) -> None:
        for name, st in stages.items():
            setattr(self, name, st)
        self._stages = stages
        self.observations: list = []

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.observations.append(
            {
                "modes": {name: st.trainable_module().training for name, st in self._stages.items()},
                "grad_enabled": torch.is_grad_enabled(),
                "sample_ids": list(req.sample_ids),
            }
        )
        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions={},
                    segment=None,
                    decoded=None,
                ),
            }
        )


def _single_stage_pipeline(*, training: bool = True) -> tuple[_FakePipeline, nn.Module]:
    model = nn.Linear(2, 2)
    model.train(bool(training))
    return _FakePipeline(diffusion=_FakeStage(model)), model


def _make_req(n: int = 2) -> RolloutReq:
    # Pre-pin σ schedule so generate() does not derive one via the
    # schedule_policy (which would require a real checkpoint). These tests
    # cover engine wrapping semantics, not schedule-policy logic.
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=[f"g{i}" for i in range(n)],
        primitives={"text": Texts(texts=[f"p{i}" for i in range(n)])},
        sigmas=torch.linspace(1.0, 0.0, steps=4),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_registration_target_resolves_to_engine() -> None:
    """Hydra ``_target_`` on the registered dataclass points at the engine class."""
    inst = TrainsideEngineConfig()
    assert inst._target_.endswith("TrainsideRolloutEngine"), inst._target_


def test_generate_returns_pipeline_response_verbatim() -> None:
    pipeline, _ = _single_stage_pipeline()
    engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=["diffusion"])

    req = _make_req(n=3)
    resp = engine.generate(req)

    assert isinstance(resp, RolloutResp)
    track = resp.tracks["image"]
    assert list(track.sample_ids) == list(req.sample_ids)
    assert list(track.group_ids) == list(req.group_ids)
    assert len(pipeline.observations) == 1


def test_generate_scopes_eval_no_grad_and_restores_train_mode() -> None:
    pipeline, model = _single_stage_pipeline(training=True)
    engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=["diffusion"])

    assert model.training is True
    engine.generate(_make_req())

    # Inside the call: eval + no_grad.
    obs = pipeline.observations[0]
    assert obs["modes"]["diffusion"] is False
    assert obs["grad_enabled"] is False
    # After the call: training mode restored.
    assert model.training is True


def test_generate_restores_eval_when_stage_started_in_eval_mode() -> None:
    pipeline, model = _single_stage_pipeline(training=False)
    engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=["diffusion"])

    assert model.training is False
    engine.generate(_make_req())
    # Final state should still be eval.
    assert model.training is False


def test_generate_restores_train_mode_even_when_pipeline_raises() -> None:
    model = nn.Linear(2, 2)
    model.train(True)

    class _RaisingPipeline:
        bundle = SimpleNamespace(pretrained_path=None)
        shift = 1.0

        def __init__(self) -> None:
            self.diffusion = _FakeStage(model)

        def generate(self, req: RolloutReq) -> RolloutResp:
            raise RuntimeError("synthetic failure")

    engine = TrainsideRolloutEngine(pipeline=_RaisingPipeline(), stage_attrs=["diffusion"])

    with pytest.raises(RuntimeError, match="synthetic failure"):
        engine.generate(_make_req())

    # finally-block must have restored training mode.
    assert model.training is True


def test_generate_eval_scopes_all_stages_for_multi_stage_pipeline() -> None:
    """``stage_attrs`` eval-scopes every listed trainable module (PE: diffusion + ar)."""
    m_diff = nn.Linear(2, 2)
    m_ar = nn.Linear(2, 2)
    m_diff.train(True)
    m_ar.train(True)
    pipeline = _FakePipeline(diffusion=_FakeStage(m_diff), ar=_FakeStage(m_ar))

    engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=["diffusion", "ar"])
    assert len(engine._models) == 2

    engine.generate(_make_req())

    # Both stages were in eval (and grad disabled) during generate.
    obs = pipeline.observations[0]
    assert obs["modes"] == {"diffusion": False, "ar": False}
    assert obs["grad_enabled"] is False
    # Both restored to train afterward.
    assert m_diff.training is True
    assert m_ar.training is True


def test_lifecycle_methods_are_safe_noops() -> None:
    pipeline, _ = _single_stage_pipeline()
    engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=["diffusion"])

    # None of these should raise.
    engine.sleep()
    engine.wake_up()
    engine.shutdown()
    assert engine.health_check() is True


def test_weight_sync_methods_raise_not_implemented() -> None:
    """Direct sampling = sampler==trainer; weight sync is meaningless."""
    pipeline, _ = _single_stage_pipeline()
    engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=["diffusion"])

    with pytest.raises(NotImplementedError):
        engine.update_weights_from_ipc()
    with pytest.raises(NotImplementedError):
        engine.init_weights_update_group(master_address="x", master_port=0, rank_offset=0, world_size=1, group_name="g")
    with pytest.raises(NotImplementedError):
        engine.update_weights_from_distributed(names=[], dtypes=[], shapes=[], group_name="g")
    with pytest.raises(NotImplementedError):
        engine.destroy_weights_update_group(group_name="g")
