"""E0 gate 6: raw per-step metrics reach an archived file, wandb off or on.

Gate 6 requires unaggregated per-step metrics as archived evidence. At the audited
commit wandb was the only sink, so `report_to_wandb=false` dropped every series and
left a 1-second-resolution stdout line as the sole record.
"""

from __future__ import annotations

import json

from unirl.utils.wandb_logger import UniRLWandBLogger


def _logger(tmp_path, monkeypatch, *, rank: int = 0):
    sink = tmp_path / "metrics" / "steps.jsonl"
    monkeypatch.setenv("UNIRL_METRICS_JSONL", str(sink))
    return UniRLWandBLogger(project=None, enabled=False, rank=rank), sink


def _rows(sink):
    return [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]


def test_metrics_are_archived_even_with_wandb_disabled(tmp_path, monkeypatch):
    logger, sink = _logger(tmp_path, monkeypatch)
    logger.log_perf(7, {"driver_iteration_time_s": 1.5})
    rows = _rows(sink)
    assert len(rows) == 1
    assert rows[0]["step"] == 7
    assert rows[0]["perf/driver_iteration_time_s"] == 1.5
    assert rows[0]["step_key"] == "rollout/step"


def test_every_namespace_lands_in_one_file(tmp_path, monkeypatch):
    logger, sink = _logger(tmp_path, monkeypatch)
    logger.log_perf(1, {"driver_sync_time_s": 0.25})
    logger.log_with_step(step_key="rollout/step", step=1, metrics={"buffer/ready_groups": 4}, prefix="")
    logger.log_eval(1, {"reward": 0.5})
    rows = _rows(sink)
    assert len(rows) == 3
    assert {r["step_key"] for r in rows} == {"rollout/step", "eval/step"}


def test_rows_carry_a_wall_time_for_external_telemetry_alignment(tmp_path, monkeypatch):
    logger, sink = _logger(tmp_path, monkeypatch)
    logger.log_perf(1, {"driver_elapsed_s": 12.0})
    assert _rows(sink)[0]["wall_time"] > 0


def test_non_scalar_values_are_dropped_not_fatal(tmp_path, monkeypatch):
    logger, sink = _logger(tmp_path, monkeypatch)
    logger.log_perf(1, {"good": 1.0, "bad": {"nested": "dict"}})
    row = _rows(sink)[0]
    assert row["perf/good"] == 1.0
    assert "perf/bad" not in row


def test_appends_across_steps_rather_than_truncating(tmp_path, monkeypatch):
    logger, sink = _logger(tmp_path, monkeypatch)
    for step in range(4):
        logger.log_perf(step, {"driver_iteration_time_s": float(step)})
    assert [r["step"] for r in _rows(sink)] == [0, 1, 2, 3]


def test_non_zero_ranks_do_not_write(tmp_path, monkeypatch):
    logger, sink = _logger(tmp_path, monkeypatch, rank=3)
    logger.log_perf(1, {"driver_iteration_time_s": 1.0})
    assert not sink.exists()


def test_no_sink_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("UNIRL_METRICS_JSONL", raising=False)
    logger = UniRLWandBLogger(project=None, enabled=False, rank=0)
    logger.log_perf(1, {"driver_iteration_time_s": 1.0})
    assert logger._metrics_sink is None


def test_finish_closes_the_sink(tmp_path, monkeypatch):
    logger, _ = _logger(tmp_path, monkeypatch)
    logger.finish()
    assert logger._metrics_sink is None
