"""State machine for async rollout/train overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class InflightRollout:
    """A rollout launched but not yet consumed."""

    rollout_id: int
    future: Any


@dataclass(frozen=True)
class ResolvedRollout:
    """A rollout payload resolved from an inflight future."""

    rollout_id: int
    payload: Any


class AsyncPipelineRuntime:
    """
    Minimal producer-consumer state for async pipeline.

    The runtime tracks:
    - inflight rollout futures with bounded queue size
    - rollout id ordering
    """

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        initial_rollout_id: int = 0,
    ) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")

        self.max_inflight = int(max_inflight)
        self._inflight: Dict[int, InflightRollout] = {}

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def can_launch(self) -> bool:
        return self.inflight_count < self.max_inflight

    def launch_rollout(
        self,
        rollout_id: int,
        future: Any,
    ) -> InflightRollout:
        rid = int(rollout_id)
        if not self.can_launch():
            raise RuntimeError(
                f"Async inflight queue full: inflight={self.inflight_count}, max_inflight={self.max_inflight}"
            )
        if rid in self._inflight:
            raise RuntimeError(f"Rollout {rid} is already inflight")

        inflight = InflightRollout(rollout_id=rid, future=future)
        self._inflight[rid] = inflight
        return inflight

    def resolve_next_rollout(self, resolver: Callable[[Any], Any]) -> ResolvedRollout:
        if not self._inflight:
            raise RuntimeError("No inflight rollout to resolve")

        rid = min(self._inflight.keys())
        inflight = self._inflight.pop(rid)
        payload = resolver(inflight.future)
        return ResolvedRollout(
            rollout_id=inflight.rollout_id,
            payload=payload,
        )

    def assert_no_inflight_for_weight_sync(self) -> None:
        if self._inflight:
            pending = sorted(self._inflight.keys())
            raise RuntimeError(
                "Weight sync requires empty inflight queue, but found pending rollouts: "
                f"{pending}"
            )
