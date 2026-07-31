"""CPU-only tests for the driver-side async rollout engines (stubbed handles)."""

import pytest

from unirl.rollout.engine.asynchronous import AsyncAgenticRolloutEngine, AsyncBatchRolloutEngine, root_of

# ── batch engine stubs ──


class StubPending:
    def __init__(self, value):
        self.value = value

    def ready(self):
        return True

    def wait(self):
        pass

    def result(self):
        return self.value


class StubBatchHandle:
    """launch_nowait echoes the submitted sample back as the completed value."""

    def __init__(self):
        self.launched = []

    def launch_nowait(self, method, sample):
        self.launched.append((method, sample))
        return StubPending(f"done:{sample}")


# ── agentic stubs ──


class _Part:
    def __init__(self, root):
        self.sample_ids = [root]


class Traj:
    def __init__(self, root):
        self.parts = [_Part(root)]

    def __repr__(self):
        return f"Traj({self.parts[0].sample_ids[0]})"


class StubAgenticHandle:
    """Coordinator handle: every value returns BROADCAST+RANK_ZERO-shaped [value]."""

    def __init__(self):
        self.submitted = []
        self.poll_batches = []
        self.finalize_value = None
        self.carried = []
        self.abort_calls = 0

    def submit(self, tasks):
        self.submitted.append(list(tasks))

    def poll(self):
        batch = self.poll_batches.pop(0) if self.poll_batches else []
        return [batch]

    def finalize_if_drained(self):
        return [self.finalize_value]

    def abort(self):
        self.abort_calls += 1
        return [list(self.carried)]


# ── AsyncBatchRolloutEngine ──


def test_batch_stamps_weight_version_at_launch():
    handle = StubBatchHandle()
    engine = AsyncBatchRolloutEngine(handle, complete=lambda gen_id, done: [done])
    engine.submit("s0")           # launched under version 0
    engine.bump_weight_version()  # sync happens while s0 is in flight
    assert engine.poll() == 1

    # Launch-time stamp: at staleness 0 the group is one sync stale → evicted.
    assert engine.drain_freshest(1, max_staleness=0) is None
    assert engine.pop_evicted() == ["done:s0"]


def test_batch_gen_id_seed_and_next_gen_id():
    handle = StubBatchHandle()
    engine = AsyncBatchRolloutEngine(handle, complete=lambda g, d: [d], start_gen_id=5)
    assert engine.next_gen_id == 5
    assert engine.submit("s") == 5
    assert engine.next_gen_id == 6
    assert engine.inflight == 1


def test_batch_quiesce_drains_everything_and_returns_empty():
    handle = StubBatchHandle()
    seen = []
    engine = AsyncBatchRolloutEngine(handle, complete=lambda g, d: seen.append(g) or [d])
    engine.submit("s0")
    engine.submit("s1")
    assert engine.quiesce() == []
    assert engine.inflight == 0
    assert seen == [0, 1]
    assert engine.drain_freshest(2, max_staleness=0) == ["done:s1", "done:s0"]


def test_batch_complete_failure_causes_no_double_put_on_retry():
    handle = StubBatchHandle()
    fails = [True]

    def complete(gen_id, done):
        if fails and fails.pop():
            raise RuntimeError("score boom")
        return [done]

    engine = AsyncBatchRolloutEngine(handle, complete=complete)
    engine.submit("s0")
    with pytest.raises(RuntimeError, match="score boom"):
        engine.poll()
    assert engine.inflight == 1  # retained for retry, nothing buffered

    assert engine.poll() == 1
    assert engine.drain_freshest(1, max_staleness=0) == ["done:s0"]
    assert engine.drain_freshest(1, max_staleness=0) is None  # exactly once


def test_batch_complete_may_split_into_multiple_groups():
    handle = StubBatchHandle()
    engine = AsyncBatchRolloutEngine(handle, complete=lambda g, d: [f"{d}#0", f"{d}#1"])
    engine.submit("s0")
    engine.poll()
    assert engine.drain_freshest(2, max_staleness=0) == ["done:s0#0", "done:s0#1"]


# ── AsyncAgenticRolloutEngine ──


def test_agentic_poll_unwraps_and_assembles_n_sibling_groups():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=2)
    a0, a1, b0 = Traj("a"), Traj("a"), Traj("b")

    handle.poll_batches.append([a0, b0])
    assert engine.poll() == 2
    assert engine.pending_groups() == 2  # both roots incomplete
    assert engine.buffered_groups() == 0

    handle.poll_batches.append([a1])
    assert engine.poll() == 1
    assert engine.buffered_groups() == 1
    assert engine.pending_groups() == 1  # root b still waiting

    assert engine.drain_freshest(1, max_staleness=0) == [[a0, a1]]


def test_agentic_stamps_weight_version_at_completion():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=2)
    handle.poll_batches.append([Traj("a")])
    engine.poll()                 # first sibling under version 0
    engine.bump_weight_version()
    handle.poll_batches.append([Traj("a")])
    engine.poll()                 # group completes under version 1

    # Completion-time stamp: fresh at current version, staleness 0.
    assert engine.drain_freshest(1, max_staleness=0) is not None


def test_agentic_quiesce_polls_after_abort_and_before_bump():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=1)
    handle.carried = [Traj("tail")]
    handle.poll_batches.append([Traj("done-during-quiesce")])

    carried = engine.quiesce()
    engine.bump_weight_version()  # trainer syncs after the quiesce

    assert handle.abort_calls == 1
    assert [root_of(t) for t in carried] == ["tail"]
    # The group that completed during the quiesce was stamped pre-bump …
    assert engine.buffered_groups() == 1
    # … so at staleness 0 it is one sync stale and must evict, proving the
    # stamp happened before bump_weight_version.
    assert engine.drain_freshest(1, max_staleness=0) is None
    assert len(engine.pop_evicted()) == 1


def test_agentic_finalize_if_drained_passthrough_and_ingest():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=1)

    handle.finalize_value = None
    assert engine.finalize_if_drained() is None

    handle.finalize_value = [Traj("a")]
    assert engine.finalize_if_drained() == 1
    assert engine.buffered_groups() == 1


def test_agentic_discard_roots_drops_partial_buckets():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=2)
    handle.poll_batches.append([Traj("a"), Traj("b")])
    engine.poll()

    assert engine.discard_roots(["a"]) == 1
    assert engine.pending_groups() == 1


def test_agentic_submit_passthrough():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=1)
    tasks = [Traj("a")]
    engine.submit(tasks)
    assert handle.submitted == [tasks]


def test_agentic_gen_id_orders_groups_by_completion():
    handle = StubAgenticHandle()
    engine = AsyncAgenticRolloutEngine(handle, group_size=1)
    first, second = Traj("a"), Traj("b")
    handle.poll_batches.append([first])
    engine.poll()
    handle.poll_batches.append([second])
    engine.poll()

    # Freshest-first: the later-completed group drains first.
    assert engine.drain_freshest(1, max_staleness=0) == [[second]]
    assert engine.drain_freshest(1, max_staleness=0) == [[first]]
