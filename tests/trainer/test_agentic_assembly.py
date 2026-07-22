"""CPU contract tests for AgenticTrainer's assembly logic (LIN-519).

Exercises the pure, error-prone pieces of ``train_step`` in isolation — ``<answer>``
extraction, group-relative GRPO advantage over a flat trajectory list (completion
order), and the DP-divisibility padding — without a GPU / Ray / the full trainer.
The end-to-end multi-turn on-policy run is the M1 GPU recipe.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

import torch  # noqa: E402

from unirl.distributed.tensor import TensorRef, TensorSpan  # noqa: E402
from unirl.trainer.agentic import (  # noqa: E402
    AgenticTrainer,
    _extract_answer,
    _intervention_aware_advantage,
    _is_answer_repair,
    _is_answer_rescue,
    _is_answer_rescue_trigger,
    _prepare_agentic_train_part,
    _trajectory_token_counts,
)
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.prompts import RolloutInputs  # noqa: E402
from unirl.types.sample import Part  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402

# --------------------------------------------------------------------------- #
# <answer> extraction
# --------------------------------------------------------------------------- #


def test_extract_answer_last_wins_and_fallback():
    assert _extract_answer("blah <answer> 42 </answer> tail") == "42"
    assert _extract_answer("<answer>a</answer> ... <answer> b </answer>") == "b"  # last wins
    # no tag -> whole text (the verifier is tolerant of an unwrapped / \boxed{} answer)
    assert _extract_answer("the answer is \\boxed{7}") == "the answer is \\boxed{7}"
    assert _extract_answer("") == ""
    assert _extract_answer(None) == ""
    assert _extract_answer("<answer>\nline1\nline2\n</answer>") == "line1\nline2"


# --------------------------------------------------------------------------- #
# group-relative GRPO advantage over the flat trajectory list
# --------------------------------------------------------------------------- #


def _adv(scope, by_std, rewards, group_ids, token_counts=None):
    stub = SimpleNamespace(adv_normalization_scope=scope, normalize_adv_by_std=by_std)
    weights = None if token_counts is None else torch.tensor(token_counts)
    return AgenticTrainer._group_advantages(stub, torch.tensor(rewards), group_ids, token_counts=weights)


def test_group_advantages_group_zscore():
    # p0=[1,0] -> mean .5 popstd .5 -> [1,-1]; p1=[1,1] -> std 0 -> [0,0]
    adv = _adv("group", True, [1.0, 0.0, 1.0, 1.0], ["p0", "p0", "p1", "p1"])
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-4)


def test_group_advantages_completion_order_non_contiguous():
    # siblings interleaved (the engine returns trajectories in completion order):
    # grouping must be by root id, not position.
    adv = _adv("group", True, [1.0, 1.0, 0.0, 1.0], ["p0", "p1", "p0", "p1"])
    # p0=[idx0:1, idx2:0] -> [1,-1]; p1=[idx1:1, idx3:1] -> [0,0]
    assert torch.allclose(adv, torch.tensor([1.0, 0.0, -1.0, 0.0]), atol=1e-4)


def test_group_advantages_mean_center_only():
    adv = _adv("group", False, [1.0, 0.0, 1.0, 1.0], ["p0", "p0", "p1", "p1"])
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.0, 0.0]), atol=1e-4)


def test_group_advantages_global_is_mean_zero():
    adv = _adv("global", True, [1.0, 0.0, 1.0, 1.0], ["p0", "p0", "p1", "p1"])
    assert adv.shape == (4,)
    assert abs(float(adv.mean())) < 1e-5


def test_token_global_advantages_match_areal_unbiased_masked_normalization():
    # Conceptual token-expanded rewards are [1, 1, 1, 0]: mean=.75 and
    # unbiased std=sqrt((3*.25^2 + .75^2)/(4-1))=.5. AReaL adds eps=1e-5
    # after reward_scaling=10, equivalent to eps=1e-6 on the raw 0/1 scale.
    adv = _adv("token-global", True, [1.0, 0.0], ["p0", "p1"], token_counts=[3, 1])
    expected = torch.tensor([0.25 / 0.500001, -0.75 / 0.500001])
    assert torch.allclose(adv, expected, atol=1e-6)
    assert abs(float((adv * torch.tensor([3.0, 1.0])).sum())) < 1e-5
    assert float(adv.mean()) < 0  # sequence-weighted baseline sees a negative stop bias


def test_token_global_mean_center_and_inactive_rows():
    adv = _adv(
        "token-global",
        False,
        [1.0, 0.0, float("nan"), 1.0],
        ["p0", "p1", "p2", "p3"],
        token_counts=[3, 1, 100, 0],
    )
    assert torch.allclose(adv, torch.tensor([0.25, -0.75, 0.0, 0.0]))


def test_trajectory_token_counts_sum_generated_turns_only():
    turn_a = SimpleNamespace(segment=SimpleNamespace(lengths=torch.tensor([2])))
    turn_b = SimpleNamespace(
        segment=SimpleNamespace(
            lengths=torch.tensor([5]),
            loss_mask=torch.tensor([True, False, True, False, True]),
        )
    )
    genless = SimpleNamespace(gen_parts=lambda: [])
    two_turn = SimpleNamespace(gen_parts=lambda: [turn_a, turn_b])
    assert torch.equal(_trajectory_token_counts([two_turn, genless]), torch.tensor([5, 0]))


def test_train_assembly_strips_heterogeneous_repair_metadata_without_losing_source_marker():
    """Regression for U5 rollout-0: ordinary generated Parts use ``metadata=[]``
    while decoder repair suffixes carry one diagnostic dict. Train copies must
    concat cleanly without mutating the source trajectory marker used by dumps and
    logical-turn metrics."""
    segment_a = TextSegment.pack(tokens=[torch.tensor([1, 2])], log_probs=[torch.zeros(2)])
    segment_b = TextSegment.pack(tokens=[torch.tensor([3])], log_probs=[torch.zeros(1)])
    ordinary = Part(sample_ids=["a"], segment=segment_a, metadata=[])
    repair = Part(
        sample_ids=["b"],
        segment=segment_b,
        primitive=Texts(texts=["<answer>42</answer>"]),
        metadata=[{"answer_injected": True, "format_repair": "neither_answer_prefix"}],
    )

    assembled = Part.concat(
        [
            _prepare_agentic_train_part(ordinary, -0.5),
            _prepare_agentic_train_part(repair, 0.5),
        ]
    )

    assert assembled.batch_size == 2
    assert assembled.metadata == []
    assert torch.allclose(assembled.advantages, torch.tensor([-0.5, 0.5]))
    assert assembled.primitive is None
    assert _is_answer_repair(repair) is True
    assert repair.metadata[0]["answer_injected"] is True
    assert repair.primitive.texts == ["<answer>42</answer>"]


def test_user_rescue_credit_boundary_and_task_token_counts():
    research = Part(
        sample_ids=["r"],
        segment=TextSegment.pack(tokens=[torch.tensor([1, 2])], log_probs=[torch.zeros(2)]),
        metadata=[],
    )
    trigger = Part(
        sample_ids=["t"],
        segment=TextSegment.pack(tokens=[torch.tensor([3, 4, 5])], log_probs=[torch.zeros(3)]),
        metadata=[{"answer_rescue_trigger": True}],
    )
    rescued = Part(
        sample_ids=["a"],
        segment=TextSegment.pack(tokens=[torch.tensor([6, 7])], log_probs=[torch.zeros(2)]),
        metadata=[{"answer_rescued": True}],
    )
    traj = SimpleNamespace(gen_parts=lambda: [research, trigger, rescued])

    assert _is_answer_rescue_trigger(trigger) is True
    assert _is_answer_rescue(rescued) is True
    assert _trajectory_token_counts([traj]).item() == 7
    assert _trajectory_token_counts([traj], exclude_answer_rescue_triggers=True).item() == 4

    for trajectory_advantage in (2.0, -2.0):
        assert _intervention_aware_advantage(
            research,
            trajectory_advantage,
            mask_trigger_task_credit=True,
            trigger_penalty=0.05,
        ) == trajectory_advantage
        assert _intervention_aware_advantage(
            rescued,
            trajectory_advantage,
            mask_trigger_task_credit=True,
            trigger_penalty=0.05,
        ) == trajectory_advantage
        assert _intervention_aware_advantage(
            trigger,
            trajectory_advantage,
            mask_trigger_task_credit=True,
            trigger_penalty=0.05,
        ) == pytest.approx(-0.05)

    assert _intervention_aware_advantage(
        trigger,
        2.0,
        mask_trigger_task_credit=False,
        trigger_penalty=0.0,
    ) == 2.0


# --------------------------------------------------------------------------- #
# per-request tool-boundary control
# --------------------------------------------------------------------------- #


def _request_with_boundary_control(*, no_stop_trim: bool):
    trainer = object.__new__(AgenticTrainer)
    trainer._stop = ["</tool_call>"]
    trainer._no_stop_trim = no_stop_trim
    inputs = RolloutInputs(
        primitives={"text": Texts(texts=["question"])},
        sample_ids=["prompt-0"],
        metadata=[{"answer": "reference"}],
    )
    return AgenticTrainer._build_request_sample(trainer, inputs, rollout_id=7)


def test_closed_tool_boundary_is_carried_in_request_control():
    request = _request_with_boundary_control(no_stop_trim=True)

    assert request.parts[0].control == {
        "ar": {"stop": ["</tool_call>"], "no_stop_trim": True}
    }


def test_default_tool_boundary_omits_no_stop_trim_for_reproducibility():
    request = _request_with_boundary_control(no_stop_trim=False)

    assert request.parts[0].control == {"ar": {"stop": ["</tool_call>"]}}


# --------------------------------------------------------------------------- #
# DP-divisibility padding
# --------------------------------------------------------------------------- #


def _pad(part, dp):
    stub = SimpleNamespace(stack=SimpleNamespace(dp_size=dp), num_devices=dp)
    return AgenticTrainer._pad_to_dp_multiple(stub, part)


def test_pad_to_dp_multiple_pads_with_zero_advantage():
    part = Part(sample_ids=["a", "b", "c"], advantages=torch.tensor([1.0, 2.0, 3.0]))
    out = _pad(part, 2)
    assert out.batch_size == 4  # 3 -> 4 (a multiple of dp=2)
    assert float(out.advantages[-1]) == 0.0  # pad row -> zero gradient
    assert torch.allclose(out.advantages[:3], torch.tensor([1.0, 2.0, 3.0]))  # real rows preserved


def test_pad_to_dp_multiple_noop_when_divisible():
    part = Part(sample_ids=["a", "b"], advantages=torch.tensor([1.0, 2.0]))
    out = _pad(part, 2)
    assert out.batch_size == 2
    assert out is part  # exact no-op when already divisible


def test_pad_to_dp_multiple_marks_synthetic_tokens_inactive():
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4, 5, 6])],
        log_probs=[torch.zeros(2), torch.zeros(1), torch.zeros(3)],
    )
    part = Part(sample_ids=["a", "b", "c"], segment=segment, advantages=torch.ones(3))
    out = _pad(part, 2)
    assert out.batch_size == 4
    assert out.segment.loss_mask is not None
    cu = out.segment.cu_seqlens
    assert bool(out.segment.loss_mask[: int(cu[3])].all())
    assert not bool(out.segment.loss_mask[int(cu[3]) : int(cu[4])].any())


def test_pad_to_dp_multiple_builds_mask_for_tensorref_tokens():
    class LocalHandle:
        def __init__(self, value):
            self.value = value
            self.shape = tuple(value.shape)
            self.dtype = value.dtype
            self.device = str(value.device)

        def local(self):
            return self.value

    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4, 5, 6])],
        log_probs=[torch.zeros(2), torch.zeros(1), torch.zeros(3)],
    )
    handle = LocalHandle(segment.tokens)
    segment.tokens = TensorRef(
        spans=[TensorSpan(handle, 0, 6)],
        shape=(6,),
        dtype=torch.long,
        device="cpu",
    )
    part = Part(sample_ids=["a", "b", "c"], segment=segment, advantages=torch.ones(3))

    out = _pad(part, 2)

    assert isinstance(out.segment.tokens, TensorRef)
    assert out.segment.loss_mask.tolist() == [True, True, True, True, True, True, False]


def test_pad_to_dp_multiple_never_uses_zero_token_source():
    segment = TextSegment.pack(
        tokens=[torch.tensor([], dtype=torch.long), torch.tensor([2, 3]), torch.tensor([4, 5, 6])],
        log_probs=[torch.tensor([]), torch.zeros(2), torch.zeros(3)],
    )
    part = Part(sample_ids=["empty", "short", "long"], segment=segment, advantages=torch.ones(3))
    out = _pad(part, 2)
    assert out.segment.lengths.tolist() == [0, 2, 3, 2]
