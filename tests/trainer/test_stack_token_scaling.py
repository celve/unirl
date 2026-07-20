"""CPU contracts for exact DP-global token-mean micro scaling."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from unirl.train.stack.base import _global_token_loss_scales, _micro_token_counts  # noqa: E402
from unirl.algorithms.grpo import GRPO  # noqa: E402
from unirl.types.sample import Part  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402


def test_single_rank_token_share_differs_from_sample_share():
    scales = _global_token_loss_scales([1, 3], global_token_count=4, dp_world_size=1)
    assert scales == pytest.approx([0.25, 0.75])
    assert scales != pytest.approx([0.5, 0.5])


def test_two_rank_scales_reconstruct_direct_global_token_mean():
    rank0 = _global_token_loss_scales([1, 3], global_token_count=10, dp_world_size=2)
    rank1 = _global_token_loss_scales([2, 4], global_token_count=10, dp_world_size=2)
    assert rank0 == pytest.approx([0.2, 0.6])
    assert rank1 == pytest.approx([0.4, 0.8])

    # FSDP averages the two rank-local accumulated gradients.
    fsdp_gradient = ((0.2 * 10 + 0.6 * 20) + (0.4 * 30 + 0.8 * 40)) / 2
    direct_token_mean = (1 * 10 + 3 * 20 + 2 * 30 + 4 * 40) / 10
    assert fsdp_gradient == pytest.approx(direct_token_mean)


def test_micro_token_counts_use_active_loss_mask():
    segment = TextSegment.pack(
        tokens=[torch.tensor([1]), torch.tensor([2, 3, 4])],
        log_probs=[torch.zeros(1), torch.zeros(3)],
        loss_mask=[torch.tensor([True]), torch.tensor([True, False, True])],
    )
    part = Part(sample_ids=["a", "b"], segment=segment)
    assert _micro_token_counts(part, [(0, 1), (1, 2)]) == [1, 2]


def test_global_token_scaling_rejects_missing_or_empty_denominator():
    with pytest.raises(ValueError, match="at least one active token"):
        _global_token_loss_scales([0, 0], global_token_count=0, dp_world_size=1)

    unpacked = Part(
        sample_ids=["a"],
        segment=TextSegment(tokens=torch.tensor([1]), log_probs=torch.zeros(1)),
    )
    with pytest.raises(ValueError, match="segment.lengths"):
        _micro_token_counts(unpacked, [(0, 1)])


def test_grpo_global_token_mean_masks_inactive_token_gradient():
    class FakeStage:
        def __init__(self):
            self.logp = torch.tensor([0.0, 0.0], requires_grad=True)

        def replay(self, conditions, *, segment, temperature):
            return self.logp

    stage = FakeStage()
    algorithm = GRPO(stage=stage, loss_agg_mode="global-token-mean")
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2])],
        log_probs=[torch.zeros(2)],
        loss_mask=[torch.tensor([True, False])],
    )
    result = algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.tensor([1.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert result.has_backward
    assert stage.logp.grad is not None
    assert float(stage.logp.grad[0]) != 0.0
    assert float(stage.logp.grad[1]) == 0.0
