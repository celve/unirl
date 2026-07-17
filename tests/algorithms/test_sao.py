from __future__ import annotations

import math

import pytest
import torch

from unirl.algorithms.sao import SAO, sao_policy_loss
from unirl.types.segments.text import TextSegment


def test_text_segment_sao_fields_follow_packed_batch_operations() -> None:
    segment = TextSegment.pack(
        tokens=[torch.tensor([10, 11]), torch.tensor([20])],
        log_probs=[torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])],
        loss_mask=[torch.tensor([1.0, 0.0]), torch.tensor([1.0])],
        token_advantages=[torch.tensor([0.1, 0.2]), torch.tensor([0.3])],
        value_targets=[torch.tensor([1.1, 1.2]), torch.tensor([1.3])],
        value_mask=[torch.tensor([1.0, 1.0]), torch.tensor([0.0])],
    )

    assert segment.lengths is not None
    assert segment.lengths.tolist() == [2, 1]
    assert torch.equal(segment.token_advantages, torch.tensor([0.1, 0.2, 0.3]))

    selected = segment.select(torch.tensor([1, 0]))
    assert selected.lengths is not None
    assert selected.lengths.tolist() == [1, 2]
    assert torch.equal(selected.tokens, torch.tensor([20, 10, 11]))
    assert torch.allclose(selected.value_targets, torch.tensor([1.3, 1.1, 1.2]))
    assert torch.equal(selected.value_mask, torch.tensor([0.0, 1.0, 1.0]))

    sliced = segment.slice(1, 2)
    assert sliced.lengths is not None
    assert sliced.lengths.tolist() == [1]
    assert torch.equal(sliced.token_advantages, torch.tensor([0.3]))

    joined = TextSegment.concat([sliced, segment.slice(0, 1)])
    assert joined.lengths is not None
    assert joined.lengths.tolist() == [1, 2]
    assert torch.equal(joined.value_targets, torch.tensor([1.3, 1.1, 1.2]))


def test_text_segment_rejects_misaligned_packed_sao_fields() -> None:
    with pytest.raises(ValueError, match="per-sample sizes"):
        TextSegment.pack(
            tokens=[torch.tensor([1, 2])],
            token_advantages=[torch.tensor([1.0])],
        )


def test_sao_dis_has_strict_sign_independent_bounds() -> None:
    ratios = torch.tensor([0.8, 0.8001, 1.0, 1.4999, 1.5])
    current = ratios.log().requires_grad_()
    rollout = torch.zeros_like(current)
    positive = sao_policy_loss(
        current_logp=current,
        rollout_logp=rollout,
        advantages=torch.ones_like(current),
        eps_low=0.2,
        eps_high=0.5,
    )
    negative = sao_policy_loss(
        current_logp=current,
        rollout_logp=rollout,
        advantages=-torch.ones_like(current),
        eps_low=0.2,
        eps_high=0.5,
    )

    expected = torch.tensor([False, True, True, True, False])
    assert torch.equal(positive.valid_mask, expected)
    assert torch.equal(negative.valid_mask, expected)
    assert positive.denominator.item() == 5
    assert positive.metrics["dis_accept_fraction"].item() == pytest.approx(3 / 5)


def test_sao_uses_detached_score_coefficient_and_structural_denominator() -> None:
    ratios = torch.tensor([1.2, 2.0, 1.1])
    current = ratios.log().requires_grad_()
    advantages = torch.tensor([2.0, -3.0, 7.0])
    output = sao_policy_loss(
        current_logp=current,
        rollout_logp=torch.zeros_like(current),
        advantages=advantages,
        eps_low=0.2,
        eps_high=0.5,
        structural_mask=torch.tensor([1.0, 1.0, 0.0]),
    )

    # Token 1 is DIS-rejected but stays in the denominator; token 2 is padding.
    assert output.denominator.item() == 2
    assert torch.equal(output.valid_mask, torch.tensor([True, False, False]))
    output.loss.backward()
    assert torch.allclose(current.grad, torch.tensor([-1.2, 0.0, 0.0]), atol=1e-6)


def test_sao_nonfinite_inputs_are_zeroed_but_counted() -> None:
    current = torch.tensor([float("nan"), float("inf"), 0.0], requires_grad=True)
    output = sao_policy_loss(
        current_logp=current,
        rollout_logp=torch.tensor([0.0, float("-inf"), 0.0]),
        advantages=torch.tensor([1.0, float("nan"), 2.0]),
        eps_low=0.3,
        eps_high=5.0,
    )

    assert torch.isfinite(output.loss)
    assert output.denominator.item() == 3
    assert torch.equal(output.valid_mask, torch.tensor([False, False, True]))
    assert output.metrics["dis_nonfinite_fraction"].item() == pytest.approx(2 / 3)
    assert all(torch.isfinite(metric) for metric in output.metrics.values())
    output.loss.backward()
    assert torch.equal(current.grad, torch.tensor([0.0, 0.0, -2.0 / 3.0]))


def test_sao_empty_and_extreme_diagnostics_remain_finite() -> None:
    empty = sao_policy_loss(
        current_logp=torch.empty(0, requires_grad=True),
        rollout_logp=torch.empty(0),
        advantages=torch.empty(0),
        eps_low=0.3,
        eps_high=5.0,
    )
    assert empty.denominator.item() == 0
    assert empty.loss.item() == 0
    assert all(torch.isfinite(metric) for metric in empty.metrics.values())

    extreme = sao_policy_loss(
        current_logp=torch.tensor([1.0e38], requires_grad=True),
        rollout_logp=torch.tensor([-1.0e38]),
        advantages=torch.ones(1),
        eps_low=0.3,
        eps_high=5.0,
    )
    assert torch.isfinite(extreme.loss)
    assert all(torch.isfinite(metric) for metric in extreme.metrics.values())


@pytest.mark.parametrize(
    ("eps_low", "eps_high"),
    [(-0.1, 1.0), (1.0, 1.0), (0.1, -1.0), (0.1, float("inf"))],
)
def test_sao_rejects_invalid_bounds(eps_low: float, eps_high: float) -> None:
    with pytest.raises(ValueError):
        sao_policy_loss(
            current_logp=torch.zeros(1),
            rollout_logp=torch.zeros(1),
            advantages=torch.ones(1),
            eps_low=eps_low,
            eps_high=eps_high,
        )


class _FakeARStage:
    def __init__(self, current_logp: torch.Tensor) -> None:
        self.current_logp = current_logp

    def replay(self, conditions: object, *, segment: TextSegment, temperature: float) -> torch.Tensor:
        assert conditions == {}
        assert temperature == 1.0
        assert segment.tokens is not None
        return self.current_logp


def test_sao_stage_algorithm_reads_packed_advantages_and_honors_loss_scale() -> None:
    current = torch.tensor([math.log(1.1), math.log(0.9)], requires_grad=True)
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2])],
        log_probs=[torch.zeros(2)],
        token_advantages=[torch.tensor([2.0, -4.0])],
        loss_mask=[torch.ones(2)],
    )
    algorithm = SAO(
        stage=_FakeARStage(current),
        eps_low=0.3,
        eps_high=5.0,
        sampling_temperature=1.0,
    )

    result = algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.tensor([999.0]),  # segment.token_advantages wins
        training_progress=0.5,
        loss_scale=0.25,
    )

    assert result.has_backward
    assert result.num_steps_or_tokens == 2
    expected = -torch.tensor([1.1 * 2.0, 0.9 * -4.0]) / 2.0 * 0.25
    assert torch.allclose(current.grad, expected, atol=1e-6)
