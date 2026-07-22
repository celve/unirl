"""CPU contracts for infrastructure-aware agentic rollout handling."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine  # noqa: E402
from unirl.train.stack import TrainStepResult  # noqa: E402
from unirl.trainer.agentic import (  # noqa: E402
    AgenticTrainer,
    _infrastructure_group_exclusion,
)
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample, _part_with_field  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402
from unirl.utils.trajectory_dump import _render_trajectory  # noqa: E402


def _trajectory(
    root_id: str,
    *,
    diagnostic: Dict[str, Any] | None = None,
    tokens: int = 2,
) -> Sample:
    root = Part.input(
        [root_id],
        primitive=Texts(texts=[f"question-{root_id}"]),
        metadata=[{"answer": "reference"}],
    )
    segment = TextSegment.pack(
        tokens=[torch.arange(1, tokens + 1)],
        log_probs=[torch.zeros(tokens)],
    )
    traj = (
        Sample.request(root)
        .fork(1, sampling_params=ARSamplingParams())
        .with_filled_frontier(
            segment=segment,
            primitive=Texts(texts=["<answer>answer</answer>"]),
        )
    )
    if diagnostic is not None:
        frontier = _part_with_field(
            traj.parts[-1],
            "metadata",
            [{"tool_diagnostics": diagnostic}],
        )
        traj = traj.with_parts([*traj.parts[:-1], frontier])
    return traj


class _CapturingStack:
    dp_size = 1

    def __init__(self) -> None:
        self.parts: List[Part] = []

    def train_track(self, part: Part, *, training_progress: float) -> TrainStepResult:
        del training_progress
        self.parts.append(part)
        return TrainStepResult(1.0, 2.0, 3.0, True, [], {})


class _CapturingLogger:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_rollout_step(
        self,
        rollout_id: int,
        result: TrainStepResult,
        sample: Sample,
        *,
        step_time_s: float,
        extra_metrics: Dict[str, Any],
    ) -> None:
        self.calls.append(
            {
                "rollout_id": rollout_id,
                "result": result,
                "sample": sample,
                "step_time_s": step_time_s,
                "metrics": extra_metrics,
            }
        )


def _trainer() -> AgenticTrainer:
    trainer = object.__new__(AgenticTrainer)
    trainer.adv_normalization_scope = "group"
    trainer.normalize_adv_by_std = True
    trainer._mask_answer_rescue_trigger_task_credit = False
    trainer._answer_rescue_trigger_penalty = 0.0
    trainer.sampling_params = {"ar": ARSamplingParams()}
    trainer.num_devices = 1
    trainer.stack = _CapturingStack()
    trainer.wandb_logger = _CapturingLogger()
    return trainer


def test_engine_copies_row_aligned_tool_diagnostics_to_generated_frontier():
    traj = _trajectory("p0")
    assert (
        AgenticRolloutEngine._attach_tool_diagnostics(
            traj, {"tool_diagnostics": [None]}
        )
        is traj
    )
    diagnostic = {
        "request_count": 2,
        "retry_count": 1,
        "transient_exhausted_count": 1,
    }

    attached = AgenticRolloutEngine._attach_tool_diagnostics(traj, {"tool_diagnostics": [diagnostic]})
    diagnostic["request_count"] = 999

    assert attached.parts[-1].metadata == [
        {
            "tool_diagnostics": {
                "request_count": 2,
                "retry_count": 1,
                "transient_exhausted_count": 1,
            }
        }
    ]
    assert traj.parts[-1].metadata == []


def test_infrastructure_failure_invalidates_every_sibling_in_root_group():
    trajs = [
        _trajectory("g0", diagnostic={"transient_exhausted_count": 1}),
        _trajectory("g0", diagnostic={"auth_error": 1}),
        _trajectory("g1", diagnostic={"permanent_error_count": 3}),
        _trajectory("g1"),
    ]

    mask, reasons, totals, invalid_groups = _infrastructure_group_exclusion(trajs, ["g0", "g0", "g1", "g1"])

    assert mask.tolist() == [True, True, False, False]
    assert reasons[:2] == [
        "infrastructure_group:auth_error+transient_exhausted",
        "infrastructure_group:auth_error+transient_exhausted",
    ]
    assert reasons[2:] == [None, None]
    assert invalid_groups == {"g0": ("auth_error", "transient_exhausted")}
    assert totals[2]["permanent_error_count"] == 3


def test_mixed_rollout_trains_only_valid_group_and_logs_raw_vs_effective(monkeypatch):
    monkeypatch.delenv("TRAJ_DUMP_DIR", raising=False)
    trainer = _trainer()
    trajs = [
        _trajectory(
            "bad",
            diagnostic={
                "tool": "search",
                "provider": "polaris",
                "request_count": 3,
                "retry_count": 2,
                "recovered_transient_count": 1,
                "transient_exhausted_count": 1,
                "status_counts": {"http_429": 1},
            },
        ),
        _trajectory("bad"),
        _trajectory(
            "good",
            diagnostic={
                "tool": "visit",
                "provider": "polaris",
                "request_count": 2,
                "success_count": 2,
                "cache_hit_count": 1,
            },
        ),
        _trajectory("good"),
    ]

    result, mean_reward = trainer._advantage_train_and_log(
        trajs,
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
        ["bad", "bad", "good", "good"],
        rollout_id=7,
        training_progress=0.25,
        t0=time.perf_counter(),
    )

    assert result.has_backward is True
    assert mean_reward == pytest.approx(0.5)
    assert len(trainer.stack.parts) == 1
    trained = trainer.stack.parts[0]
    assert trained.batch_size == 2
    assert torch.allclose(trained.advantages, torch.tensor([1.0, -1.0]))

    metrics = trainer.wandb_logger.calls[0]["metrics"]
    assert metrics["reward_mean"] == pytest.approx(0.5)
    assert metrics["agent/raw_all_trajectory_mean_reward"] == pytest.approx(0.75)
    assert metrics["agent/valid_effective_mean_reward"] == pytest.approx(0.5)
    assert metrics["agent/infra_invalid_groups"] == 1
    assert metrics["agent/infra_invalid_trajectories"] == 2
    assert metrics["agent/train_rows_before_padding"] == 2
    assert metrics["agent/tool_request_count"] == 5
    assert metrics["agent/tool_retry_count"] == 2
    assert metrics["agent/tool_cache_hit_count"] == 1
    assert metrics["agent/tool_recovered_transient_count"] == 1
    assert metrics["agent/tool_transient_exhausted_count"] == 1
    assert metrics["agent/tool_transient_recovery_rate"] == pytest.approx(0.5)
    assert metrics["agent/tool_logical_call_count"] == 3
    assert metrics["agent/tool_cache_hit_rate"] == pytest.approx(1 / 3)
    assert metrics["agent/tool_tool_search_request_count"] == 3
    assert metrics["agent/tool_tool_visit_request_count"] == 2
    assert metrics["agent/tool_provider_polaris_request_count"] == 5
    assert metrics["agent/tool_status_http_429_count"] == 1
    assert metrics["agent/tool_tool_search_status_http_429_count"] == 1
    assert metrics["agent/tool_rate_limited_count"] == 1
    assert metrics["agent/raw_finite_trajectory_count"] == 4
    assert metrics["agent/effective_valid_trajectory_count"] == 2


def test_all_invalid_rollout_skips_optimizer_but_still_logs(monkeypatch):
    monkeypatch.delenv("TRAJ_DUMP_DIR", raising=False)
    trainer = _trainer()
    trajs = [
        _trajectory("bad", diagnostic={"request_count": 1, "auth_error_count": 1}),
        _trajectory("bad"),
    ]

    result, mean_reward = trainer._advantage_train_and_log(
        trajs,
        torch.tensor([1.0, 0.0]),
        ["bad", "bad"],
        rollout_id=8,
        training_progress=0.5,
        t0=time.perf_counter(),
    )

    assert result.has_backward is False
    assert mean_reward == 0.0
    assert trainer.stack.parts == []
    assert len(trainer.wandb_logger.calls) == 1
    call = trainer.wandb_logger.calls[0]
    assert call["metrics"]["agent/train_rows"] == 0
    assert call["metrics"]["reward_mean"] == 0.0
    assert call["metrics"]["agent/infra_invalid_group_rate"] == 1.0
    assert torch.isnan(call["sample"].gen_parts()[-1].rewards).all()


def test_mixed_invalid_with_only_genless_valid_rows_still_logs(monkeypatch):
    monkeypatch.delenv("TRAJ_DUMP_DIR", raising=False)
    trainer = _trainer()
    trajs = [
        _trajectory("bad", diagnostic={"transient_exhausted_count": 1}),
        _trajectory("valid-but-empty", tokens=0),
    ]

    result, mean_reward = trainer._advantage_train_and_log(
        trajs,
        torch.tensor([1.0, 0.0]),
        ["bad", "valid-but-empty"],
        rollout_id=10,
        training_progress=0.5,
        t0=time.perf_counter(),
    )

    assert result.has_backward is False
    assert mean_reward == 0.0
    assert trainer.stack.parts == []
    assert len(trainer.wandb_logger.calls) == 1
    metrics = trainer.wandb_logger.calls[0]["metrics"]
    assert metrics["agent/infra_invalid_trajectories"] == 1
    assert metrics["agent/effective_valid_trajectory_count"] == 1
    assert metrics["agent/train_rows"] == 0


def test_dump_distinguishes_infrastructure_exclusion_from_crash_and_is_safe():
    traj = _trajectory(
        "bad",
        diagnostic={
            "tool": "search",
            "provider": "polaris",
            "request_count": 2,
            "transient_exhausted_count": 1,
            "status_counts": {
                "http_503": 2,
                "TOP_SECRET_QUERY": 7,
                "bad": "not-a-count",
            },
            "api_key": "must-not-leak",
            "exception": "must-not-leak",
        },
    )

    record = _render_trajectory(
        traj,
        rollout_id=9,
        traj_index=0,
        reward=float("nan"),
        raw_reward=1.0,
        advantage=0.0,
        group_id="bad",
        max_chars=0,
        excluded_from_training=True,
        exclusion_reason="infrastructure_group:transient_exhausted",
    )

    assert record["reward"] is None
    assert record["reward_schema_version"] == 2
    assert record["raw_reward"] == 1.0
    assert record["effective_reward"] is None
    assert record["excluded_from_training"] is True
    assert record["exclusion_reason"] == "infrastructure_group:transient_exhausted"
    assert record["crashed"] is False
    diagnostic = record["turns"][-1]["tool_diagnostics"]
    assert diagnostic == {
        "tool": "search",
        "provider": "polaris",
        "request_count": 2,
        "transient_exhausted_count": 1,
        "status_counts": {"http_503": 2},
    }
    assert "must-not-leak" not in str(record)
