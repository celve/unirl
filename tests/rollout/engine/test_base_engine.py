"""Tests for :class:`BaseRolloutEngine` ABC defaults.

Specifically pins the ``stage_ids`` kwarg on ``sleep`` / ``wake_up``
(added in LIN-279 to support :class:`ComposedRolloutEngine`'s child-routing
semantics). Subclasses that don't need stage routing accept the kwarg and
ignore it (mirroring the weight-sync ABC pattern).
"""

from __future__ import annotations

from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp


class _Stub(BaseRolloutEngine):
    def shutdown(self) -> None:
        pass

    def generate(self, req: RolloutReq) -> RolloutResp:
        return RolloutResp(tracks={})


def test_base_sleep_signature_accepts_stage_ids():
    """Default no-op accepts the kwarg; both ``None`` and a list work."""
    s = _Stub()
    s.sleep(stage_ids=[0])
    s.sleep(stage_ids=None)
    s.sleep()  # positional default still works


def test_base_wake_up_signature_accepts_stage_ids():
    s = _Stub()
    s.wake_up(stage_ids=[1])
    s.wake_up(stage_ids=None)
    s.wake_up()


def test_base_lifecycle_returns_none():
    """Defaults are no-ops returning None."""
    s = _Stub()
    assert s.sleep(stage_ids=[0]) is None
    assert s.wake_up(stage_ids=[0]) is None
    assert s.is_offloaded is False
    assert s.health_check() is True
