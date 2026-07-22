"""Async consumer parity for infrastructure-contaminated GRPO groups."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from unirl.train.stack import TrainStepResult  # noqa: E402
from unirl.trainer.agentic_async import AsyncAgenticTrainer, _GroupBuffer  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample, _part_with_field  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402


def _trajectory(
    root_id: str,
    *,
    diagnostic: Dict[str, Any] | None = None,
    weight_version: int | None = None,
) -> Sample:
    root = Part.input(
        [root_id],
        primitive=Texts(texts=[f"question-{root_id}"]),
        metadata=[{"answer": "reference"}],
    )
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2])],
        log_probs=[torch.zeros(2)],
    )
    traj = (
        Sample.request(root)
        .fork(1, sampling_params=ARSamplingParams())
        .with_filled_frontier(
            segment=segment,
            primitive=Texts(texts=["<answer>answer</answer>"]),
        )
    )
    frontier = traj.parts[-1]
    if diagnostic is not None:
        frontier = _part_with_field(
            frontier,
            "metadata",
            [{"tool_diagnostics": diagnostic}],
        )
    if weight_version is not None:
        frontier = _part_with_field(frontier, "weight_version", weight_version)
    return traj.with_parts([*traj.parts[:-1], frontier])


class _CapturingStack:
    dp_size = 1

    def __init__(self) -> None:
        self.parts: List[Part] = []

    def train_track(self, part: Part, *, training_progress: float) -> TrainStepResult:
        assert training_progress == pytest.approx(0.25)
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


def test_async_consumer_excludes_failed_group_before_normalization_and_training(monkeypatch):
    monkeypatch.delenv("TRAJ_DUMP_DIR", raising=False)
    trainer = object.__new__(AsyncAgenticTrainer)
    trainer.adv_normalization_scope = "global"
    trainer.normalize_adv_by_std = True
    trainer._mask_answer_rescue_trigger_task_credit = False
    trainer._answer_rescue_trigger_penalty = 0.0
    trainer.sampling_params = {"ar": ARSamplingParams()}
    trainer.num_devices = 1
    trainer.stack = _CapturingStack()
    trainer.wandb_logger = _CapturingLogger()
    trainer._gt_by_root = {"bad": "reference", "good": "reference"}
    trainer._buffer = _GroupBuffer()
    trainer._weight_version = 7

    reset_calls: List[bool] = []
    trainer._reset_transport_buffers = lambda: reset_calls.append(True)
    trainer._rewards_and_groups = lambda request, trajs, rollout_id: (
        torch.tensor([10.0, -10.0, 1.0, 0.0]),
        [traj.parts[0].sample_ids[0] for traj in trajs],
    )

    groups = [
        [
            _trajectory(
                "bad",
                diagnostic={
                    "request_count": 3,
                    "retry_count": 2,
                    "transient_exhausted_count": 1,
                },
                weight_version=2,
            ),
            _trajectory("bad", weight_version=3),
        ],
        [
            _trajectory("good", weight_version=4),
            _trajectory("good", weight_version=5),
        ],
    ]

    result, mean_reward = trainer._train_on_groups(
        groups,
        training_progress=0.25,
        rollout_id=11,
        t0=time.perf_counter(),
    )

    assert result.has_backward is True
    assert mean_reward == pytest.approx(0.5)
    assert reset_calls == [True]

    # Both siblings of the contaminated group are absent, not zero-advantage rows
    # that would dilute the global-token loss denominator. The remaining rewards
    # normalize against each other to +1/-1 under population-std global GRPO.
    assert len(trainer.stack.parts) == 1
    trained = trainer.stack.parts[0]
    assert trained.batch_size == 2
    assert torch.allclose(trained.advantages, torch.tensor([1.0, -1.0]))

    assert len(trainer.wandb_logger.calls) == 1
    call = trainer.wandb_logger.calls[0]
    assert call["rollout_id"] == 11
    assert torch.isnan(call["sample"].gen_parts()[-1].rewards[:2]).all()
    assert torch.equal(
        call["sample"].gen_parts()[-1].rewards[2:],
        torch.tensor([1.0, 0.0]),
    )
    metrics = call["metrics"]
    assert metrics["agent/infra_invalid_groups"] == 1
    assert metrics["agent/infra_invalid_trajectories"] == 2
    assert metrics["agent/tool_transient_exhausted_count"] == 1
    assert metrics["async/buffer_groups"] == 0
    assert metrics["async/weight_version"] == 7
    assert metrics["async/version_span"] == 3
