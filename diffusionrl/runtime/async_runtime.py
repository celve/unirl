"""State machine for async rollout/train overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class InflightRollout:
    """A rollout launched but not yet consumed."""

    rollout_id: int
    weight_version: int
    future: Any


@dataclass(frozen=True)
class ResolvedRollout:
    """A rollout payload resolved from an inflight future."""

    rollout_id: int
    weight_version: int
    payload: Any


class AsyncPipelineRuntime:
    """
    Minimal producer-consumer state for async pipeline.

    The runtime tracks:
    - inflight rollout futures with bounded queue size
    - rollout id ordering
    - expected weight version for consistency checks
    """

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        initial_rollout_id: int = 0,
        initial_weight_version: int = 0,
    ) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")

        self.max_inflight = int(max_inflight)
        self.next_rollout_id = int(initial_rollout_id)
        self.expected_weight_version = int(initial_weight_version)
        self._inflight: Dict[int, InflightRollout] = {}

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def can_launch(self) -> bool:
        return self.inflight_count < self.max_inflight

    def has_rollout(self, rollout_id: int) -> bool:
        return int(rollout_id) in self._inflight

    def launch_rollout(
        self,
        rollout_id: int,
        future: Any,
        *,
        weight_version: Optional[int] = None,
    ) -> InflightRollout:
        rid = int(rollout_id)
        if not self.can_launch():
            raise RuntimeError(
                f"Async inflight queue full: inflight={self.inflight_count}, max_inflight={self.max_inflight}"
            )
        if rid in self._inflight:
            raise RuntimeError(f"Rollout {rid} is already inflight")

        version = self.expected_weight_version if weight_version is None else int(weight_version)
        inflight = InflightRollout(rollout_id=rid, weight_version=version, future=future)
        self._inflight[rid] = inflight
        self.next_rollout_id = max(self.next_rollout_id, rid + 1)
        return inflight

    def resolve_next_rollout(self, resolver: Callable[[Any], Any]) -> ResolvedRollout:
        if not self._inflight:
            raise RuntimeError("No inflight rollout to resolve")

        rid = min(self._inflight.keys())
        inflight = self._inflight.pop(rid)
        payload = resolver(inflight.future)
        return ResolvedRollout(
            rollout_id=inflight.rollout_id,
            weight_version=inflight.weight_version,
            payload=payload,
        )

    def ensure_rollout_version(
        self,
        rollout: ResolvedRollout,
        *,
        allow_stale: bool = False,
    ) -> bool:
        """Return False when stale rollout is allowed and detected; otherwise True."""
        if rollout.weight_version > self.expected_weight_version:
            raise RuntimeError(
                "Resolved rollout uses newer weight version than trainer state: "
                f"rollout_version={rollout.weight_version}, expected={self.expected_weight_version}"
            )
        if rollout.weight_version < self.expected_weight_version:
            if allow_stale:
                return False
            raise RuntimeError(
                "Resolved rollout is stale relative to trainer state: "
                f"rollout_version={rollout.weight_version}, expected={self.expected_weight_version}"
            )
        return True

    def assert_no_inflight_for_weight_sync(self) -> None:
        if self._inflight:
            pending = sorted(self._inflight.keys())
            raise RuntimeError(
                "Weight sync requires empty inflight queue, but found pending rollouts: "
                f"{pending}"
            )

    def advance_weight_version(self) -> int:
        self.expected_weight_version += 1
        return self.expected_weight_version
