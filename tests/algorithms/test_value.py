from __future__ import annotations

import pytest
import torch

from unirl.algorithms.value import TokenValueAlgorithm, masked_value_loss
from unirl.types.segments.text import TextSegment


def test_masked_value_loss_exposes_sum_count_and_exact_gradient() -> None:
    predictions = torch.tensor([1.0, 3.0, 100.0], requires_grad=True)
    output = masked_value_loss(
        predictions=predictions,
        targets=torch.tensor([0.0, 1.0, 0.0]),
        mask=torch.tensor([1.0, 1.0, 0.0]),
    )

    assert output.loss_sum.item() == pytest.approx(5.0)
    assert output.denominator.item() == 2
    assert output.loss.item() == pytest.approx(2.5)
    output.loss.backward()
    assert torch.equal(predictions.grad, torch.tensor([1.0, 2.0, 0.0]))


def test_value_explained_variance_is_one_for_perfect_predictions() -> None:
    output = masked_value_loss(
        predictions=torch.tensor([-1.0, 0.0, 3.0]),
        targets=torch.tensor([-1.0, 0.0, 3.0]),
    )
    assert output.loss.item() == 0.0
    assert output.metrics["value_explained_variance"].item() == pytest.approx(1.0)


def test_value_loss_handles_constant_targets_and_nonfinite_rows() -> None:
    predictions = torch.tensor([float("nan"), 1.0, 2.0], requires_grad=True)
    output = masked_value_loss(
        predictions=predictions,
        targets=torch.tensor([0.0, 1.0, 1.0]),
        mask=torch.ones(3),
    )

    assert torch.isfinite(output.loss)
    assert output.denominator.item() == 3
    assert output.loss.item() == pytest.approx(1 / 3)
    assert output.metrics["value_nonfinite_fraction"].item() == pytest.approx(1 / 3)
    output.loss.backward()
    assert torch.allclose(predictions.grad, torch.tensor([0.0, 0.0, 2 / 3]))


class _FakeValueStage:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def predict_values(self, conditions: object, *, segment: TextSegment) -> torch.Tensor:
        assert conditions == {}
        assert segment.tokens is not None
        return self.values


def test_token_value_algorithm_predicts_and_honors_loss_scale() -> None:
    values = torch.tensor([0.0, 1.0], requires_grad=True)
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2])],
        value_targets=[torch.tensor([1.0, 3.0])],
        value_mask=[torch.ones(2)],
    )
    algorithm = TokenValueAlgorithm(stage=_FakeValueStage(values))

    assert algorithm.predict_values({}, segment) is values
    result = algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.empty(0),
        training_progress=0.0,
        loss_scale=0.5,
    )

    assert result.has_backward
    assert result.num_steps_or_tokens == 2
    assert result.loss == pytest.approx(2.5)
    assert torch.equal(values.grad, torch.tensor([-0.5, -1.0]))
