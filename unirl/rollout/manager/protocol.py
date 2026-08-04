"""The manager surface the async trainers program against.

Mechanism only. Every knob a trainer sets — batch size, over-sample width, tail
policy, sync cadence — stays in the trainer; a manager answers *what is ready* and
*what is in flight*, never *how much to train on*.

Two implementations, differing only where they genuinely must:
:class:`~unirl.rollout.manager.batch.BatchManager` (one batch per call, slab-wide
dispatch, versions stamped at launch) and
:class:`~unirl.rollout.manager.agentic.AgenticManager` (one trajectory per call,
point-to-point dispatch, versions stamped at completion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Protocol

if TYPE_CHECKING:
    from unirl.types.sample import Sample


class RolloutManager(Protocol):
    """Consumer verbs shared by the batch and agentic managers."""

    @property
    def weight_version(self) -> int: ...

    def sync_weights(self, weight_sync: Any) -> int: ...

    def poll(self) -> int: ...

    def drain_freshest(self, n: int, *, max_staleness: Optional[int]) -> Optional[List[Any]]: ...

    def pop_evicted(self) -> List[Any]: ...

    def quiesce(self) -> List["Sample"]: ...


__all__ = ["RolloutManager"]
