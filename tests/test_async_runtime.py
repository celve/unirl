"""CPU-only tests for the driver-side async mechanisms (VersionedBuffer, InflightPool)."""

import pytest

from unirl.rollout.engine.asynchronous import InflightPool, VersionedBuffer


class StubPending:
    """Stands in for handle.PendingHandleCall."""

    def __init__(self, value, *, ready=True, fail_results=0):
        self.value = value
        self._ready = ready
        self._fail_results = fail_results
        self.result_calls = 0
        self.wait_calls = 0

    def ready(self):
        return self._ready

    def wait(self):
        self.wait_calls += 1

    def result(self):
        self.result_calls += 1
        if self._fail_results > 0:
            self._fail_results -= 1
            raise RuntimeError("result boom")
        return self.value


class StubRollout:
    """Hands out pre-built StubPending objects in launch order."""

    def __init__(self, pendings):
        self._pendings = list(pendings)
        self.launched = []

    def launch_nowait(self, method, sample):
        self.launched.append((method, sample))
        return self._pendings.pop(0)


class Recorder:
    def __init__(self, fail_times=0, raise_cls=RuntimeError):
        self.calls = []
        self._fail_times = fail_times
        self._raise_cls = raise_cls

    def __call__(self, gen_id, weight_version, payload):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._raise_cls("complete boom")
        self.calls.append((gen_id, weight_version, payload))


# ── VersionedBuffer ──


def test_drain_freshest_orders_by_gen_id_desc():
    buf = VersionedBuffer()
    for gen_id in (1, 3, 2):
        buf.put(f"g{gen_id}", weight_version=0, gen_id=gen_id)
    assert buf.drain_freshest(2) == ["g3", "g2"]
    assert buf.size() == 1
    assert buf.drain_freshest(1) == ["g1"]


def test_drain_freshest_stable_on_gen_id_ties():
    buf = VersionedBuffer()
    buf.put("first", weight_version=0, gen_id=7)
    buf.put("second", weight_version=0, gen_id=7)
    assert buf.drain_freshest(2) == ["first", "second"]


def test_drain_freshest_returns_none_without_consuming():
    buf = VersionedBuffer()
    buf.put("only", weight_version=0, gen_id=0)
    assert buf.drain_freshest(2) is None
    assert buf.size() == 1


def test_staleness_eviction_and_pop_evicted():
    buf = VersionedBuffer()
    buf.put("stale", weight_version=0, gen_id=0)
    buf.put("fresh", weight_version=2, gen_id=1)
    assert buf.drain_freshest(2, current_version=2, max_staleness=1) is None
    assert buf.pop_evicted() == ["stale"]
    assert buf.pop_evicted() == []
    assert buf.drain_freshest(1, current_version=2, max_staleness=1) == ["fresh"]


# ── InflightPool ──


def test_launch_stamps_and_reap_completes():
    rollout = StubRollout([StubPending("a"), StubPending("b")])
    pool = InflightPool(rollout, start_gen_id=10)
    complete = Recorder()

    assert pool.launch("s0", weight_version=0) == 10
    assert pool.launch("s1", weight_version=1) == 11
    assert pool.next_gen_id == 12
    assert len(pool) == 2
    assert rollout.launched == [("generate", "s0"), ("generate", "s1")]

    assert pool.reap_ready(complete) == 2
    assert complete.calls == [(10, 0, "a"), (11, 1, "b")]
    assert len(pool) == 0


def test_reap_skips_unready_jobs():
    rollout = StubRollout([StubPending("a", ready=False), StubPending("b")])
    pool = InflightPool(rollout)
    complete = Recorder()
    pool.launch("s0", weight_version=0)
    pool.launch("s1", weight_version=0)

    assert pool.reap_ready(complete) == 1
    assert complete.calls == [(1, 0, "b")]
    assert len(pool) == 1


def test_complete_failure_keeps_job_in_flight_for_retry():
    rollout = StubRollout([StubPending("a")])
    pool = InflightPool(rollout)
    complete = Recorder(fail_times=1)
    pool.launch("s0", weight_version=0)

    with pytest.raises(RuntimeError, match="complete boom"):
        pool.reap_ready(complete)
    assert len(pool) == 1

    assert pool.reap_ready(complete) == 1
    assert complete.calls == [(0, 0, "a")]
    assert len(pool) == 0


def test_result_failure_keeps_job_in_flight_for_retry():
    pending = StubPending("a", fail_results=1)
    rollout = StubRollout([pending])
    pool = InflightPool(rollout)
    complete = Recorder()
    pool.launch("s0", weight_version=0)

    with pytest.raises(RuntimeError, match="result boom"):
        pool.reap_ready(complete)
    assert len(pool) == 1 and complete.calls == []

    assert pool.reap_ready(complete) == 1
    assert complete.calls == [(0, 0, "a")]


def test_first_error_reraised_after_full_sweep():
    rollout = StubRollout([StubPending("a"), StubPending("b")])
    pool = InflightPool(rollout)
    complete = Recorder(fail_times=2)
    pool.launch("s0", weight_version=0)
    pool.launch("s1", weight_version=0)

    with pytest.raises(RuntimeError):
        pool.reap_ready(complete)
    assert len(pool) == 2  # both swept, both retained


def test_keyboard_interrupt_propagates_immediately():
    rollout = StubRollout([StubPending("a"), StubPending("b")])
    pool = InflightPool(rollout)
    complete = Recorder(fail_times=1, raise_cls=KeyboardInterrupt)
    pool.launch("s0", weight_version=0)
    pool.launch("s1", weight_version=0)

    with pytest.raises(KeyboardInterrupt):
        pool.reap_ready(complete)
    assert complete.calls == []  # second job never attempted
    assert len(pool) == 2  # sweep aborted before reassignment


def test_drain_all_completes_unready_jobs_too():
    rollout = StubRollout([StubPending("a", ready=False), StubPending("b")])
    pool = InflightPool(rollout)
    complete = Recorder()
    pool.launch("s0", weight_version=0)
    pool.launch("s1", weight_version=0)

    assert pool.drain_all(complete) == 2
    assert len(pool) == 0


def test_wait_oldest_blocks_on_first_job_only():
    p0, p1 = StubPending("a", ready=False), StubPending("b", ready=False)
    rollout = StubRollout([p0, p1])
    pool = InflightPool(rollout)
    pool.launch("s0", weight_version=0)
    pool.launch("s1", weight_version=0)

    pool.wait_oldest()
    assert (p0.wait_calls, p1.wait_calls) == (1, 0)
