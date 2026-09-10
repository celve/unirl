"""E0 gate: the driver clock covers the whole async iteration, not just the batch.

`perf/step_time_s` stops at the optimizer update, which is why RQ2/RQ4 timing is
not measurable at the audited commit. These assert the replacement actually spans
the sync/eval/checkpoint tail.
"""

from __future__ import annotations

import time

import pytest

from unirl.utils.driver_clock import ITERATION_PHASES, DriverClock, buffer_accounting_metrics


def test_iteration_total_includes_the_publication_tail():
    clock = DriverClock()
    clock.begin_iteration()
    with clock.phase("batch"):
        time.sleep(0.01)
    with clock.phase("train"):
        time.sleep(0.01)
    with clock.phase("sync"):
        time.sleep(0.02)
    metrics = clock.iteration_metrics()
    assert metrics["driver_sync_time_s"] >= 0.015
    assert metrics["driver_iteration_time_s"] >= sum(
        metrics[f"driver_{p}_time_s"] for p in ("batch", "train", "sync")
    )


def test_every_declared_phase_is_emitted_even_when_it_did_not_run():
    clock = DriverClock()
    clock.begin_iteration()
    with clock.phase("batch"):
        pass
    metrics = clock.iteration_metrics()
    for phase in ITERATION_PHASES:
        assert f"driver_{phase}_time_s" in metrics
    assert metrics["driver_eval_time_s"] == 0.0


def test_unattributed_time_is_reported_not_hidden():
    clock = DriverClock()
    clock.begin_iteration()
    time.sleep(0.02)
    with clock.phase("train"):
        pass
    assert clock.iteration_metrics()["driver_unattributed_time_s"] >= 0.015


def test_repeated_phase_accumulates():
    clock = DriverClock()
    clock.begin_iteration()
    for _ in range(3):
        with clock.phase("sync"):
            time.sleep(0.005)
    assert clock.iteration_metrics()["driver_sync_time_s"] >= 0.012


def test_begin_iteration_clears_the_previous_iteration():
    clock = DriverClock()
    clock.begin_iteration()
    with clock.phase("sync"):
        time.sleep(0.02)
    clock.begin_iteration()
    assert clock.iteration_metrics()["driver_sync_time_s"] == 0.0


def test_clock_is_monotonic_across_iterations():
    clock = DriverClock()
    clock.begin_iteration()
    first = clock.elapsed()
    time.sleep(0.01)
    clock.begin_iteration()
    assert clock.elapsed() > first


def test_phase_requires_an_open_iteration():
    clock = DriverClock()
    with pytest.raises(RuntimeError, match="begin_iteration"):
        with clock.phase("batch"):
            pass


def test_self_nesting_a_phase_is_rejected():
    clock = DriverClock()
    clock.begin_iteration()
    with pytest.raises(RuntimeError, match="self-nest"):
        with clock.phase("sync"):
            with clock.phase("sync"):
                pass


def test_phase_time_survives_an_exception():
    clock = DriverClock()
    clock.begin_iteration()
    with pytest.raises(ValueError):
        with clock.phase("train"):
            raise ValueError("boom")
    assert clock.iteration_metrics()["driver_train_time_s"] >= 0.0


def test_buffer_metrics_express_lag_against_the_budget():
    metrics = buffer_accounting_metrics(
        inflight_groups=4,
        ready_groups=8,
        batch_size=4,
        staleness_budget=4,
        published_version=10,
        train_version=12,
    )
    assert metrics["buffer/outstanding_groups"] == 12
    assert metrics["buffer/ready_batches"] == 2.0
    assert metrics["buffer/publish_lag_updates"] == 2
    assert metrics["buffer/publish_lag_fraction_of_budget"] == 0.5


def test_buffer_metrics_omit_optional_series_when_absent():
    metrics = buffer_accounting_metrics(
        inflight_groups=0,
        ready_groups=0,
        batch_size=8,
        staleness_budget=0,
        published_version=1,
        train_version=1,
    )
    assert "buffer/barrier_wait_s" not in metrics
    assert "buffer/quiesce_carried_groups" not in metrics
    assert "buffer/publish_lag_fraction_of_budget" not in metrics


def test_buffer_metrics_reject_a_nonpositive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        buffer_accounting_metrics(
            inflight_groups=0,
            ready_groups=0,
            batch_size=0,
            staleness_budget=1,
            published_version=1,
            train_version=1,
        )
