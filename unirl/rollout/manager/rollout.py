"""Driver-side rollout manager over per-slot launchers; see :class:`RolloutManager`."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Literal, Optional, Sequence

from unirl.rollout.manager.buffers import CompleteGroups, PendingGroups, roots_of
from unirl.rollout.manager.dispatch import RolloutPool
from unirl.rollout.manager.filters import RolloutFilter, identity, validate_filter_output

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

_PROGRESS_PROBE_S = 0.01


@dataclass(frozen=True)
class AdmissionPolicy:
    """Bounded-staleness root admission, in root units, mirroring AReaL's staleness manager."""

    max_concurrent_roots: int
    max_staleness: int
    consumer_batch: int

    def __post_init__(self) -> None:
        if self.max_concurrent_roots <= 0:
            raise ValueError(f"max_concurrent_roots must be positive; got {self.max_concurrent_roots}")
        if self.max_staleness < 0:
            raise ValueError(f"max_staleness must be non-negative; got {self.max_staleness}")
        if self.consumer_batch <= 0:
            raise ValueError(f"consumer_batch must be positive; got {self.consumer_batch}")
        if self.max_concurrent_roots < self.consumer_batch:
            raise ValueError(
                f"max_concurrent_roots must be >= consumer_batch; "
                f"got {self.max_concurrent_roots} < {self.consumer_batch}"
            )

    @property
    def window(self) -> int:
        """Root-units the producer may run ahead of the consumer."""
        return (self.max_staleness + 1) * self.consumer_batch


@dataclass(frozen=True)
class RolloutStats:
    pending_roots: int
    active_roots: int
    ready_groups: int
    accepted: int
    rejected: int
    lead: int
    ready_age_seconds: float
    oldest_active_age_seconds: float


class RolloutManager:
    """Driver-side rollout scheduling: bounded dispatch, sibling-group assembly, root-atomic filtering, published version.

    With an :class:`AdmissionPolicy` the manager also owns a pending-root queue and a
    background thread that admits roots as running slots free, so production continues
    while the caller trains; without one it dispatches on submit and is driven entirely
    by ``collect``.

    A failure while resolving or routing completed work poisons the manager: samples may
    already be lost, so every later call (including ``empty``/``counts``) re-raises the
    original error instead of reporting clean state; only ``close()`` remains safe.
    """

    def __init__(
        self,
        rollout: "Handle",
        *,
        launchers: Sequence[Callable[["Sample"], Any]],
        capacities: Sequence[int],
        group_size: int,
        filter_fn: RolloutFilter = identity,
        policy: Optional[AdmissionPolicy] = None,
    ) -> None:
        self.rollout = rollout
        self.pool = RolloutPool(launchers, capacities)
        self.pending = PendingGroups(group_size)
        self.complete = CompleteGroups()
        self.filter = filter_fn
        self.policy = policy

        self.pending_roots: Deque[List["Sample"]] = deque()
        self.active_roots: Dict[str, float] = {}
        self.lead = 0

        self.state: Literal["running", "paused", "closed"] = "running"
        self.failure: Optional[BaseException] = None
        self.condition = threading.Condition()

        self.published_version = 0
        self.accepted = 0
        self.rejected = 0

        self.progress_thread: Optional[threading.Thread] = None
        if self.policy is not None:
            self.progress_thread = threading.Thread(target=self._progress, name="rollout-admission", daemon=True)
            self.progress_thread.start()

    @property
    def group_size(self) -> int:
        return self.pending.group_size

    def submit(self, tasks: List["Sample"]) -> None:
        self._ensure_open()
        if self.policy is None:
            self.pool.add(list(tasks))
            return
        roots = self._bucket_by_root(list(tasks))
        with self.condition:
            self._raise_if_failed()
            if len(self.pending_roots) + len(roots) > self.policy.window:
                raise RuntimeError(
                    f"submit exceeds the pending window: {len(self.pending_roots)} queued + {len(roots)} "
                    f"> {self.policy.window}; size the pull with pending_capacity"
                )
            self.pending_roots.extend(roots)
            self.condition.notify_all()

    def collect(self, n: int, *, current_version: int) -> List[List["Sample"]]:
        self._ensure_open()
        n = int(n)
        if n <= 0:
            raise ValueError(f"collect count must be positive; got {n}")
        current_version = int(current_version)
        if current_version < 0:
            raise ValueError(f"current_version must be non-negative; got {current_version}")
        try:
            if self.policy is not None:
                return self._collect_admitted(n, current_version)
            while True:
                self._route(self._resolve(self.pool.take_completed(block=False)), allow_suspended=False)
                self._filter_complete(current_version)
                selected = self.complete.take(n)
                if selected is not None:
                    return selected
                if not self.pool.live:
                    raise RuntimeError(f"needed {n} rollout groups, collected {self.complete.group_count}")
                self._route(self._resolve(self.pool.take_completed(block=True)), allow_suspended=False)
        except BaseException as exc:
            self._poison(exc)
            raise

    def quiesce(self, *, current_version: int) -> List["Sample"]:
        self._ensure_open()
        current_version = int(current_version)
        if current_version < 0:
            raise ValueError(f"current_version must be non-negative; got {current_version}")
        try:
            with self.condition:
                if self.state == "running":
                    self.state = "paused"
                self.condition.notify_all()
            undispatched = self.pool.pause()
            self.rollout.set_stopping(True)
            completed = self._resolve(self.pool.drain())
            self.rollout.set_stopping(False)

            with self.condition:
                suspended = self._route(completed, allow_suspended=True)
                self._filter_complete_locked(current_version)
                candidates = [*undispatched, *suspended, *self._drain_pending_roots()]
                tails_by_root: Dict[str, List["Sample"]] = defaultdict(list)
                carried = []
                for sample in candidates:
                    roots = roots_of(sample)
                    if len(roots) == 1:
                        tails_by_root[roots[0]].append(sample)
                    elif self._keep_root([sample], current_version=current_version):
                        carried.append(sample)

                for root, tails in tails_by_root.items():
                    known = [*self.pending.get(root), *tails]
                    if self._keep_root(known, current_version=current_version):
                        carried.extend(tails)
                    else:
                        discarded = self.pending.discard(root)
                        logger.info(
                            "rollout filter discarded incomplete root=%s tails=%d completed=%d",
                            root,
                            len(tails),
                            discarded,
                        )
                return carried
        except BaseException as exc:
            self._poison(exc)
            raise

    def sync_weights(self, weight_sync: object, *, output_version: int) -> int:
        self._ensure_open()
        next_version = int(output_version)
        if next_version < self.published_version:
            raise ValueError(f"output_version must be monotonic; current={self.published_version}, next={next_version}")
        try:
            if self.policy is None:
                self._route(self._resolve(self.pool.take_completed(block=False)), allow_suspended=False)
                if self.pool.live:
                    raise RuntimeError("sync_weights requires no queued or in-flight rollout work")
                weight_sync.sync()
                self.rollout.set_version(next_version)
                self.published_version = next_version
                return self.published_version
            return self._publish_paused(weight_sync, next_version)
        except BaseException as exc:
            self._poison(exc)
            raise

    @property
    def pending_capacity(self) -> int:
        """Roots the caller may still submit; the depth of the staleness window over the queue."""
        if self.policy is None:
            raise RuntimeError("pending_capacity requires an AdmissionPolicy")
        with self.condition:
            self._raise_if_failed()
            return max(0, self.policy.window - len(self.pending_roots))

    @property
    def stats(self) -> RolloutStats:
        with self.condition:
            self._raise_if_failed()
            now = time.monotonic()
            oldest = min(self.active_roots.values(), default=now)
            return RolloutStats(
                pending_roots=len(self.pending_roots),
                active_roots=len(self.active_roots),
                ready_groups=self.complete.group_count,
                accepted=self.accepted,
                rejected=self.rejected,
                lead=self.lead,
                ready_age_seconds=self.complete.oldest_age_seconds,
                oldest_active_age_seconds=max(0.0, now - oldest),
            )

    @property
    def counts(self) -> tuple[int, int]:
        inflight_count, completed_count = self.pool.counts
        with self.condition:
            self._raise_if_failed()
            return inflight_count + len(self.pending_roots) * self.group_size, completed_count + len(self.complete)

    @property
    def empty(self) -> bool:
        live = self.pool.live
        with self.condition:
            self._raise_if_failed()
            return not live and not self.complete and not self.pending and not self.pending_roots

    def close(self) -> None:
        with self.condition:
            if self.state == "closed":
                return
            was_failed = self.failure is not None
            self.state = "closed"
            self.condition.notify_all()
        thread = self.progress_thread
        if thread is not None:
            thread.join(timeout=30.0)
        try:
            if not was_failed and self.pool.live:
                self._quiesce_at_close()
        except BaseException:
            # Teardown path: log instead of re-raising so close() cannot mask the primary error.
            logger.exception("RolloutManager.close: discarding in-flight rollout work after a failure")
        finally:
            self.pool.close()

    def _quiesce_at_close(self) -> None:
        self.pool.pause()
        self.rollout.set_stopping(True)
        self._resolve(self.pool.drain())
        self.rollout.set_stopping(False)

    # ── admission ─────────────────────────────────────────────────────────────

    def _bucket_by_root(self, tasks: List["Sample"]) -> List[List["Sample"]]:
        """Group a flat sibling list into whole roots, preserving first-appearance order."""
        buckets: Dict[str, List["Sample"]] = {}
        for task in tasks:
            roots = roots_of(task)
            if len(roots) != 1:
                raise RuntimeError(f"admission-gated submit requires one root per task; got {roots}")
            buckets.setdefault(roots[0], []).append(task)
        malformed = {root: len(group) for root, group in buckets.items() if len(group) != self.group_size}
        if malformed:
            raise RuntimeError(
                f"admission-gated submit requires whole roots of {self.group_size} siblings: {malformed}"
            )
        return list(buckets.values())

    def _capacity(self) -> int:
        """Admittable root count; caller holds the lock."""
        assert self.policy is not None
        concurrency_capacity = self.policy.max_concurrent_roots - len(self.active_roots)
        staleness_capacity = self.policy.window - self.lead
        return max(0, min(concurrency_capacity, staleness_capacity))

    def _progress(self) -> None:
        while True:
            with self.condition:
                if self.state == "closed" or self.failure is not None:
                    return
            try:
                results = self._resolve(self.pool.take_completed(block=False))
                launched = self._advance(results)
            except BaseException as exc:
                self._poison(exc)
                return
            if not results and not launched:
                with self.condition:
                    if self.state == "closed" or self.failure is not None:
                        return
                    self.condition.wait(timeout=_PROGRESS_PROBE_S)

    def _advance(self, results: List[tuple[int, "Sample"]]) -> int:
        """Route completions, retire finished roots, admit replacements; returns roots admitted."""
        with self.condition:
            self._route(results, allow_suspended=False)
            if self.state != "running":
                return 0
            admit = min(len(self.pending_roots), self._capacity())
            if admit <= 0:
                return 0
            tasks: List["Sample"] = []
            now = time.monotonic()
            for _ in range(admit):
                group = self.pending_roots.popleft()
                self.active_roots[roots_of(group[0])[0]] = now
                self.lead += 1
                tasks.extend(group)
            # Held under the manager lock so a concurrent publish cannot be undone by
            # RolloutPool.add, which clears the pool's own paused flag.
            self.pool.add(tasks)
            self.condition.notify_all()
            return admit

    def _collect_admitted(self, n: int, current_version: int) -> List[List["Sample"]]:
        with self.condition:
            while True:
                self._raise_if_failed()
                self._filter_complete_locked(current_version)
                selected = self.complete.take(n)
                if selected is not None:
                    self.condition.notify_all()
                    return selected
                if self.state == "closed":
                    raise RuntimeError(f"needed {n} rollout groups, collected {self.complete.group_count}")
                if not self.pending_roots and not self.active_roots and not self.pool.live:
                    raise RuntimeError(f"needed {n} rollout groups, collected {self.complete.group_count}")
                self.condition.wait(timeout=_PROGRESS_PROBE_S)

    def _publish_paused(self, weight_sync: object, next_version: int) -> int:
        with self.condition:
            self._raise_if_failed()
            self.state = "paused"
        try:
            self.rollout.pause()
            weight_sync.sync()
            self.rollout.set_version(next_version)
            with self.condition:
                published = self.published_version
                self.published_version = next_version
                assert self.policy is not None
                self.lead -= (next_version - published) * self.policy.consumer_batch
        finally:
            with self.condition:
                if self.state == "paused":
                    self.state = "running"
                self.condition.notify_all()
            self.rollout.resume()
        return self.published_version

    def _drain_pending_roots(self) -> List["Sample"]:
        """Empty the pending queue back to the caller; caller holds the lock."""
        drained = [sample for group in self.pending_roots for sample in group]
        self.pending_roots.clear()
        return drained

    # ── routing ───────────────────────────────────────────────────────────────

    def _resolve(self, units: List[Any]) -> List[tuple[int, "Sample"]]:
        return [(unit.sequence, unit.pending.result()) for unit in units]

    def _route(self, results: List[tuple[int, "Sample"]], *, allow_suspended: bool) -> List["Sample"]:
        terminal_trajectories = []
        suspended = []
        for _, sample in results:
            status = sample.parts[-1].harness_status if sample.parts else None
            if status == "suspended":
                if not allow_suspended:
                    raise RuntimeError("trajectory suspended outside quiesce")
                suspended.append(sample)
            elif status is None:
                self.complete.add(self._batch_group_count(sample), [sample])
            else:
                self._require_stamped_generated_parts(sample)
                terminal_trajectories.append(sample)
        for group in self.pending.add(terminal_trajectories):
            self._retire(group)
        return suspended

    def _retire(self, group: List["Sample"]) -> None:
        """Free the root's slot, then accept or reject the assembled group."""
        self.active_roots.pop(roots_of(group[0])[0], None)
        if self.policy is not None and not self._keep_root(group, current_version=self.published_version):
            self.lead -= 1
            self.rejected += 1
            return
        self.complete.add(1, group)
        self.accepted += 1

    def _batch_group_count(self, sample: "Sample") -> int:
        roots = roots_of(sample)
        if not sample.gen_parts():
            raise RuntimeError("completed batch rollout has no generated Parts")
        self._require_stamped_generated_parts(sample)
        descendants = Counter(sample.root_group_ids(-1))
        malformed = {root: descendants.get(root, 0) for root in roots if descendants.get(root, 0) != self.group_size}
        extra = set(descendants) - set(roots)
        if malformed or extra:
            raise RuntimeError(
                f"batch rollout fan-out does not match group_size={self.group_size}: "
                f"malformed={malformed}, extra_roots={sorted(extra)}"
            )
        return len(roots)

    def _filter_complete(self, current_version: int) -> None:
        with self.condition:
            self._filter_complete_locked(current_version)

    def _filter_complete_locked(self, current_version: int) -> None:
        before = self.complete.group_count
        self.complete.filter(lambda samples: self._apply_filter(samples, current_version=current_version))
        dropped = before - self.complete.group_count
        if dropped:
            if self.policy is not None:
                self.lead -= dropped
                self.rejected += dropped
            logger.warning(
                "rollout filter dropped %d/%d buffered rollout group(s) at version %d",
                dropped,
                before,
                current_version,
            )

    def _apply_filter(self, samples: List["Sample"], *, current_version: int) -> List["Sample"]:
        candidates = list(samples)
        kept = list(self.filter(list(candidates), current_version))
        validate_filter_output(candidates, kept)
        return kept

    def _keep_root(self, samples: List["Sample"], *, current_version: int) -> bool:
        candidates = list(samples)
        kept = self._apply_filter(candidates, current_version=current_version)
        if kept and Counter(map(id, kept)) != Counter(map(id, candidates)):
            raise RuntimeError("rollout filter must retain or discard an entire incomplete root")
        return bool(kept)

    @staticmethod
    def _require_stamped_generated_parts(sample: "Sample") -> None:
        unstamped = [index for index, part in enumerate(sample.gen_parts()) if part.output_version is None]
        if unstamped:
            raise RuntimeError(f"completed rollout has unstamped generated Parts at indices {unstamped}")

    def _poison(self, exc: BaseException) -> None:
        # Completed work may have been lost mid-route; the manager must not report clean state after.
        with self.condition:
            if self.failure is None:
                self.failure = exc
            self.condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self.failure is not None:
            raise RuntimeError("RolloutManager is unusable after an earlier failure") from self.failure

    def _ensure_open(self) -> None:
        self._raise_if_failed()
        if self.state == "closed":
            raise RuntimeError("RolloutManager is closed")


__all__ = ["AdmissionPolicy", "RolloutManager", "RolloutStats"]
