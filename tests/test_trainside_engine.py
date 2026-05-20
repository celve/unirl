"""Tests for :class:`TrainsideRolloutEngine`.

CPU-only. Uses fake ``Pipeline`` and ``Policy`` stand-ins to verify:

- ``generate(req)`` delegates to the wrapped pipeline's ``generate`` and
  returns the response verbatim.
- The policy is flipped to ``eval()`` for the call duration and restored
  to its prior training mode afterward.
- ``torch.is_grad_enabled()`` is ``False`` inside the call.
- Lifecycle methods are safe no-ops; weight-sync methods raise
  ``NotImplementedError`` from the base class.

No real model, no FSDP, no Ray.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from diffusionrl.rollout.engine.trainside import (
    TrainsideEngineConfig,
    TrainsideRolloutEngine,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePolicy:
    """Minimal Policy stand-in. Holds an ``nn.Module`` so ``.training`` is real."""

    def __init__(self, *, start_training: bool = True) -> None:
        self.model = nn.Linear(2, 2)
        self.model.train(bool(start_training))
        self.train_calls: list = []

    def train(self, mode: bool = True) -> None:
        self.train_calls.append(("train", bool(mode)))
        self.model.train(bool(mode))

    def eval(self) -> None:
        self.train_calls.append(("eval", False))
        self.model.eval()


class _FakePipeline:
    """Minimal Pipeline stand-in.

    Records whether the policy was in eval mode and whether grad was
    disabled at the moment ``generate`` was invoked, so the engine's
    scoping behavior can be asserted.
    """

    def __init__(self, *, policy: _FakePolicy) -> None:
        self._policy = policy
        self.observations: list = []

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.observations.append(
            {
                "model_training": self._policy.model.training,
                "grad_enabled": torch.is_grad_enabled(),
                "sample_ids": list(req.sample_ids),
            }
        )
        return RolloutResp(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            conditions={},
            rollout_traces={},
            decoded={},
        )


def _make_req(n: int = 2) -> RolloutReq:
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=[f"g{i}" for i in range(n)],
        primitives={"text": Texts(texts=[f"p{i}" for i in range(n)])},
        stage_params={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_registration_target_resolves_to_engine() -> None:
    """Hydra ``_target_`` on the registered dataclass points at the engine class.

    ``register_config(..., target=...)`` synthesizes a dataclass subclass
    with ``_target_: str = <target>`` as a kw-only field (see
    ``diffusionrl/config/registration.py:148``).
    """
    inst = TrainsideEngineConfig()
    assert inst._target_.endswith("TrainsideRolloutEngine"), inst._target_


def test_generate_returns_pipeline_response_verbatim() -> None:
    policy = _FakePolicy()
    pipeline = _FakePipeline(policy=policy)
    engine = TrainsideRolloutEngine(pipeline=pipeline, policy=policy)

    req = _make_req(n=3)
    resp = engine.generate(req)

    assert isinstance(resp, RolloutResp)
    assert list(resp.sample_ids) == list(req.sample_ids)
    assert list(resp.group_ids) == list(req.group_ids)
    assert len(pipeline.observations) == 1


def test_generate_scopes_eval_no_grad_and_restores_train_mode() -> None:
    policy = _FakePolicy(start_training=True)
    pipeline = _FakePipeline(policy=policy)
    engine = TrainsideRolloutEngine(pipeline=pipeline, policy=policy)

    assert policy.model.training is True

    engine.generate(_make_req())

    # Inside the call: eval + no_grad.
    obs = pipeline.observations[0]
    assert obs["model_training"] is False
    assert obs["grad_enabled"] is False

    # After the call: training mode restored.
    assert policy.model.training is True
    # The engine should have called .eval() and then .train(True).
    assert ("eval", False) in policy.train_calls
    assert ("train", True) in policy.train_calls


def test_generate_restores_eval_when_policy_started_in_eval_mode() -> None:
    policy = _FakePolicy(start_training=False)
    pipeline = _FakePipeline(policy=policy)
    engine = TrainsideRolloutEngine(pipeline=pipeline, policy=policy)

    assert policy.model.training is False

    engine.generate(_make_req())

    # Final state should still be eval.
    assert policy.model.training is False
    # The engine restored to .train(False).
    assert ("train", False) in policy.train_calls


def test_generate_restores_train_mode_even_when_pipeline_raises() -> None:
    policy = _FakePolicy(start_training=True)

    class _RaisingPipeline:
        def generate(self, req: RolloutReq) -> RolloutResp:
            raise RuntimeError("synthetic failure")

    engine = TrainsideRolloutEngine(pipeline=_RaisingPipeline(), policy=policy)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        engine.generate(_make_req())

    # finally-block must have restored training mode.
    assert policy.model.training is True


def test_lifecycle_methods_are_safe_noops() -> None:
    policy = _FakePolicy()
    pipeline = _FakePipeline(policy=policy)
    engine = TrainsideRolloutEngine(pipeline=pipeline, policy=policy)

    # None of these should raise.
    engine.sleep()
    engine.wake_up()
    engine.shutdown()
    assert engine.health_check() is True


def test_weight_sync_methods_raise_not_implemented() -> None:
    """Direct sampling = sampler==trainer; weight sync is meaningless."""
    policy = _FakePolicy()
    pipeline = _FakePipeline(policy=policy)
    engine = TrainsideRolloutEngine(pipeline=pipeline, policy=policy)

    with pytest.raises(NotImplementedError):
        engine.update_weights_from_ipc()
    with pytest.raises(NotImplementedError):
        engine.init_weights_update_group(master_address="x", master_port=0, rank_offset=0, world_size=1, group_name="g")
    with pytest.raises(NotImplementedError):
        engine.update_weights_from_distributed(names=[], dtypes=[], shapes=[], group_name="g")
    with pytest.raises(NotImplementedError):
        engine.destroy_weights_update_group(group_name="g")
