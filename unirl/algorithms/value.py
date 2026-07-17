"""Token-value regression helpers and stage-driven critic algorithm for SAO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import AlgorithmStepResult, BaseAlgorithmConfig, StageAlgorithm, typed_conditions


@dataclass(frozen=True)
class ValueLossOutput:
    """Masked token-value loss with a distributed-normalization seam."""

    loss: torch.Tensor
    loss_sum: torch.Tensor
    denominator: torch.Tensor
    structural_mask: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


def _validate_value_shapes(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
    expected = tuple(predictions.shape)
    if tuple(targets.shape) != expected:
        raise ValueError(f"token value loss: targets shape={tuple(targets.shape)} != predictions shape={expected}")
    if mask is not None and tuple(mask.shape) != expected:
        raise ValueError(f"token value loss: mask shape={tuple(mask.shape)} != predictions shape={expected}")


def _masked_moments(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=values.dtype)
    count = weights.sum()
    mean = torch.where(mask, values, torch.zeros_like(values)).sum() / count.clamp_min(1.0)
    centered = torch.where(mask, values - mean, torch.zeros_like(values))
    variance = centered.square().sum() / count.clamp_min(1.0)
    zero = values.sum() * 0.0
    return torch.where(count > 0, mean, zero), torch.where(count > 0, variance, zero)


def masked_value_loss(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> ValueLossOutput:
    """Return action-token MSE and explained-variance diagnostics.

    ``mask != 0`` defines the denominator.  Structurally selected non-finite
    predictions or targets contribute zero rather than poisoning a whole
    distributed step, remain in the denominator, and are surfaced through
    ``value_nonfinite_fraction``.
    """

    if not torch.is_floating_point(predictions):
        raise TypeError(f"token value loss: predictions must be floating point; got {predictions.dtype}")
    _validate_value_shapes(predictions, targets, mask)

    prediction = predictions.float()
    target = targets.to(device=prediction.device, dtype=torch.float32).detach()
    if mask is None:
        structural = torch.ones_like(prediction, dtype=torch.bool)
    else:
        raw_mask = mask.to(device=prediction.device)
        structural = raw_mask != 0

    finite = torch.isfinite(prediction) & torch.isfinite(target)
    valid = structural & finite
    safe_prediction = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))
    safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    squared_error = torch.where(valid, (safe_prediction - safe_target).square(), torch.zeros_like(prediction))
    loss_sum = squared_error.sum()
    denominator = structural.sum().to(dtype=loss_sum.dtype)
    loss = loss_sum / denominator.clamp_min(1.0)

    prediction_mean, _ = _masked_moments(safe_prediction, valid)
    target_mean, target_variance = _masked_moments(safe_target, valid)
    residual = safe_target - safe_prediction
    _, residual_variance = _masked_moments(residual, valid)
    explained_variance = torch.where(
        target_variance > 0,
        1.0 - residual_variance / target_variance.clamp_min(torch.finfo(target_variance.dtype).eps),
        target_variance * 0.0,
    )
    metric_denom = denominator.detach().clamp_min(1.0)
    metrics: Dict[str, torch.Tensor] = {
        "value_structural_tokens": denominator.detach(),
        "value_finite_tokens": valid.sum().to(dtype=loss_sum.dtype).detach(),
        "value_nonfinite_fraction": ((structural & ~finite).sum().to(dtype=loss_sum.dtype) / metric_denom).detach(),
        "value_mse": loss.detach(),
        "value_explained_variance": explained_variance.detach(),
        "value_prediction_mean": prediction_mean.detach(),
        "value_target_mean": target_mean.detach(),
        "value_target_std": target_variance.sqrt().detach(),
    }
    return ValueLossOutput(
        loss=loss,
        loss_sum=loss_sum,
        denominator=denominator,
        structural_mask=structural,
        metrics=metrics,
    )


@dataclass
class TokenValueConfig(BaseAlgorithmConfig):
    stage_attr: str = "value"
    conditions_cls: str = ""


class TokenValueAlgorithm(StageAlgorithm):
    """Stage-driven token-value critic using ``TextSegment.value_targets``."""

    supports_multi_update = False
    loss_agg_mode = "token-mean"

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "value",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("TokenValueAlgorithm: either `stage` or `pipeline` must be provided")
        self.stage = stage if stage is not None else getattr(pipeline, stage_attr)
        self.conditions_cls = conditions_cls

    def predict_values(
        self,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
    ) -> torch.Tensor:
        """Predict one packed value per action token through the critic stage."""

        predictor = getattr(self.stage, "predict_values", None)
        if predictor is None:
            raise TypeError(f"{type(self.stage).__name__} must implement predict_values(conditions, *, segment)")
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        values = predictor(typed_conds, segment=segment)
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"critic stage predict_values returned {type(values).__name__}, expected torch.Tensor")
        if segment.tokens is not None and tuple(values.shape) != tuple(segment.tokens.shape):
            raise ValueError(
                f"critic predicted shape={tuple(values.shape)} but segment tokens shape={tuple(segment.tokens.shape)}"
            )
        return values

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress
        if segment.tokens is None or int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.value_targets is None:
            raise ValueError("TokenValueAlgorithm requires packed segment.value_targets")

        predictions = self.predict_values(conditions, segment)
        output = masked_value_loss(
            predictions=predictions,
            targets=segment.value_targets,
            mask=segment.value_mask if segment.value_mask is not None else segment.loss_mask,
        )
        token_count = int(output.denominator.detach().item())
        if token_count == 0:
            return AlgorithmStepResult(
                loss=0.0,
                metrics={k: float(v.item()) for k, v in output.metrics.items()},
                num_steps_or_tokens=0,
                has_backward=False,
            )
        (output.loss * float(loss_scale)).backward()
        metrics: Dict[str, Any] = {
            "critic_loss": float(output.loss.detach().item()),
            **{k: float(v.item()) for k, v in output.metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(output.loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=token_count,
            has_backward=True,
        )


# Concise aliases for recipe readability and external callers that do not need
# to distinguish the token-specific implementation from future value critics.
ValueAlgorithm = TokenValueAlgorithm
ValueConfig = TokenValueConfig


__all__ = [
    "TokenValueAlgorithm",
    "TokenValueConfig",
    "ValueAlgorithm",
    "ValueConfig",
    "ValueLossOutput",
    "masked_value_loss",
]
