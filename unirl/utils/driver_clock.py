"""Monotonic driver clock and full-iteration phase accounting — see README.md."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

# The stages of one async iteration, in execution order. `batch` covers prompt
# admission plus the buffer wait; `train` covers advantage and optimizer work.
ITERATION_PHASES = ("batch", "train", "sync", "eval", "checkpoint")


class DriverClock:
    """Run-scoped monotonic clock: per-iteration phase spans on a single origin."""

    def __init__(self) -> None:
        self._origin = time.monotonic()
        self._iteration_t0: Optional[float] = None
        self._phases: Dict[str, float] = {}
        self._open: List[str] = []

    @property
    def origin(self) -> float:
        """The run origin stamp, so separately archived series can be aligned to it."""
        return self._origin

    def elapsed(self) -> float:
        """Monotonic seconds since the clock was constructed."""
        return time.monotonic() - self._origin

    def begin_iteration(self) -> None:
        """Open a new iteration, discarding any phases from the previous one."""
        self._iteration_t0 = time.monotonic()
        self._phases = {}
        self._open = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Accumulate the enclosed block's wall clock under ``name``."""
        if self._iteration_t0 is None:
            raise RuntimeError("DriverClock.phase requires begin_iteration() first")
        if name in self._open:
            raise RuntimeError(f"DriverClock.phase({name!r}) is already open — phases must not self-nest")
        self._open.append(name)
        t = time.monotonic()
        try:
            yield
        finally:
            self._open.pop()
            self._phases[name] = self._phases.get(name, 0.0) + (time.monotonic() - t)

    def iteration_elapsed(self) -> float:
        """Wall clock of the current iteration, tail included."""
        if self._iteration_t0 is None:
            raise RuntimeError("DriverClock.iteration_elapsed requires begin_iteration() first")
        return time.monotonic() - self._iteration_t0

    def iteration_metrics(self) -> Dict[str, float]:
        """Full-iteration series: total, per-phase spans, and the unattributed remainder."""
        total = self.iteration_elapsed()
        metrics: Dict[str, float] = {
            "driver_iteration_time_s": total,
            "driver_elapsed_s": self.elapsed(),
        }
        for name in ITERATION_PHASES:
            metrics[f"driver_{name}_time_s"] = float(self._phases.get(name, 0.0))
        for name, value in self._phases.items():
            if name not in ITERATION_PHASES:
                metrics[f"driver_{name}_time_s"] = float(value)
        attributed = sum(self._phases.values())
        metrics["driver_unattributed_time_s"] = max(total - attributed, 0.0)
        return metrics


def buffer_accounting_metrics(
    *,
    inflight_groups: int,
    ready_groups: int,
    batch_size: int,
    staleness_budget: int,
    published_version: int,
    train_version: int,
    quiesce_carried: Optional[int] = None,
    barrier_wait_s: Optional[float] = None,
) -> Dict[str, float]:
    """Per-step buffer / lag / barrier series for the E6 staleness sweep."""
    if batch_size <= 0:
        raise ValueError(f"buffer_accounting_metrics: batch_size must be positive, got {batch_size}")
    metrics: Dict[str, float] = {
        "buffer/inflight_groups": float(inflight_groups),
        "buffer/ready_groups": float(ready_groups),
        "buffer/outstanding_groups": float(inflight_groups + ready_groups),
        "buffer/ready_batches": ready_groups / batch_size,
        "buffer/staleness_budget_updates": float(staleness_budget),
        "buffer/publish_lag_updates": float(train_version - published_version),
    }
    if staleness_budget > 0:
        metrics["buffer/publish_lag_fraction_of_budget"] = (train_version - published_version) / staleness_budget
    if quiesce_carried is not None:
        metrics["buffer/quiesce_carried_groups"] = float(quiesce_carried)
    if barrier_wait_s is not None:
        metrics["buffer/barrier_wait_s"] = float(barrier_wait_s)
    return metrics


__all__ = ["DriverClock", "ITERATION_PHASES", "buffer_accounting_metrics"]
