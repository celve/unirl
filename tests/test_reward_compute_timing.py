"""Tests for _stamp_actor_reward_total (per-actor reward time aggregation).

The helper rewrites every per-handle ``samples.reward_compute_s`` to the
per-actor sum, so that ``RolloutSamples.reward_compute_s`` (a ``max_field``)
reduces to ``max(per-actor totals)`` on the driver after Batched.concat —
the wall-clock contribution to ``rollout_phase_s``.

We stub ``diffusionrl.utils.transfer_queue_utils`` (Ray-dependent) before
importing the mixin so the test runs on macOS without Ray installed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Ray-free import of the helper
# ---------------------------------------------------------------------------


def _stub_transfer_queue_utils() -> None:
    """Pre-stub the Ray-dependent helper module before importing the mixin."""
    name = "diffusionrl.utils.transfer_queue_utils"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.create_transferqueue_client = lambda *_a, **_kw: None
    stub.reset_zero_copy_buffer_free = lambda *_a, **_kw: None

    def _identity_decorator(*_dargs, **_dkwargs):
        def _wrap(fn):
            return fn

        return _wrap

    stub.tqbridge = _identity_decorator
    sys.modules[name] = stub


_stub_transfer_queue_utils()

from diffusionrl.ray.mixins.rollout_pipeline import _stamp_actor_reward_total  # noqa: E402

# ---------------------------------------------------------------------------
# Duck-typed stand-ins
# ---------------------------------------------------------------------------


@dataclass
class _SamplesStub:
    reward_compute_s: float = 0.0


@dataclass
class _ResponseStub:
    samples: _SamplesStub


def _make_responses(per_handle_seconds):
    return [_ResponseStub(samples=_SamplesStub(reward_compute_s=s)) for s in per_handle_seconds]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStampActorRewardTotal:
    def test_empty_responses_is_a_noop(self):
        # No responses -> nothing to stamp; just must not raise.
        responses = []
        _stamp_actor_reward_total(responses)
        assert responses == []

    def test_single_handle_keeps_value(self):
        responses = _make_responses([1.5])
        _stamp_actor_reward_total(responses)
        assert responses[0].samples.reward_compute_s == pytest.approx(1.5)

    def test_three_handles_get_per_actor_sum(self):
        responses = _make_responses([1.0, 0.8, 0.2])
        _stamp_actor_reward_total(responses)
        for r in responses:
            assert r.samples.reward_compute_s == pytest.approx(2.0)

    def test_zero_per_handle_yields_zero_total(self):
        responses = _make_responses([0.0, 0.0])
        _stamp_actor_reward_total(responses)
        for r in responses:
            assert r.samples.reward_compute_s == 0.0


# ---------------------------------------------------------------------------
# Integration with RolloutSamples max_field reduction
# ---------------------------------------------------------------------------


class TestRolloutSamplesMaxFieldReduction:
    """Verify the new max_field on RolloutSamples reduces to MAX across actors."""

    def _build_minimal_samples(self, *, reward_compute_s):
        # Construct a RolloutSamples with the bare minimum so concat works.
        # latents is the only non-default required field; we use a 1-sample tensor.
        import torch

        from diffusionrl.types.sample import RolloutSamples

        return RolloutSamples(
            latents=torch.zeros(1, 1),
            timesteps=torch.zeros(1),
            reward_compute_s=reward_compute_s,
        )

    def test_concat_reduces_by_max(self):
        from diffusionrl.types.sample import RolloutSamples

        a = self._build_minimal_samples(reward_compute_s=1.8)
        b = self._build_minimal_samples(reward_compute_s=2.8)
        c = self._build_minimal_samples(reward_compute_s=0.5)
        merged = RolloutSamples.concat([a, b, c])
        assert merged.reward_compute_s == pytest.approx(2.8)

    def test_default_is_zero(self):
        s = self._build_minimal_samples(reward_compute_s=0.0)
        assert s.reward_compute_s == 0.0
