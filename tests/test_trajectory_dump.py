"""CPU contracts for the trajectory-dump answer-format diagnostics."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample, _part_with_field  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402
from unirl.utils.trajectory_dump import _render_trajectory  # noqa: E402


def _trajectory(answer: str) -> Sample:
    root = Part.input(
        ["prompt-0"],
        primitive=Texts(texts=["question"]),
        metadata=[{"answer": "reference"}],
    )
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2, 3])],
        log_probs=[torch.zeros(3)],
    )
    return (
        Sample.request(root)
        .fork(1, sampling_params=ARSamplingParams())
        .with_filled_frontier(segment=segment, primitive=Texts(texts=[answer]))
    )


@pytest.mark.parametrize(
    ("text", "has_answer_tag", "final_answer"),
    [
        ("<think>done</think><answer> 42 </answer>", True, "42"),
        ("the unwrapped answer is 42", False, "the unwrapped answer is 42"),
        ("", False, ""),
    ],
)
def test_render_trajectory_records_answer_tag_explicitly(text, has_answer_tag, final_answer):
    record = _render_trajectory(
        _trajectory(text),
        rollout_id=3,
        traj_index=4,
        reward=1.0,
        advantage=0.5,
        group_id="prompt-0",
        max_chars=0,
    )

    assert record["has_answer_tag"] is has_answer_tag
    assert record["final_answer"] == final_answer
    assert f"has_answer_tag={has_answer_tag}" in record["transcript"]


def test_render_trajectory_distinguishes_physical_repair_decode_from_logical_turn():
    traj = _trajectory("plain deliberation")
    repair = traj.fork(1, sampling_params=ARSamplingParams()).with_filled_frontier(
        segment=TextSegment.pack(
            tokens=[torch.tensor([4, 5])],
            log_probs=[torch.zeros(2)],
        ),
        primitive=Texts(texts=["<answer>42</answer>"]),
    )
    flagged = _part_with_field(repair.parts[-1], "metadata", [{"answer_injected": True}])
    repair = repair.with_parts([*repair.parts[:-1], flagged])
    record = _render_trajectory(
        repair,
        rollout_id=0,
        traj_index=0,
        reward=1.0,
        advantage=1.0,
        group_id="prompt-0",
        max_chars=0,
    )
    assert record["num_turns"] == 2
    assert record["num_logical_turns"] == 1
    assert record["answer_injected"] is True
    assert record["turns"][-1]["answer_injected"] is True
