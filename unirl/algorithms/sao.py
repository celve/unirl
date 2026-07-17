"""Single-rollout asynchronous optimization (SAO) actor objective.

SAO's Direct Importance Sampling (DIS) is a *hard*, sign-independent token
filter.  Given the rollout behavior log-probability ``log_mu`` and current
policy log-probability ``log_pi``, a token contributes only when::

    1 - eps_low < exp(log_pi - log_mu) < 1 + eps_high

Rejected tokens remain in the structural action-token denominator.  This is
intentionally different from PPO clipping and from renormalizing over accepted
tokens: rejecting more tokens reduces the strength of the update.

The paper writes an objective containing both ``rho`` and ``log pi``.  We make
the score-function interpretation explicit by detaching ``rho * advantage``;
otherwise autograd introduces an unintended extra ``1 + log pi`` factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import AlgorithmStepResult, BaseAlgorithmConfig, StageAlgorithm, typed_conditions


@dataclass(frozen=True)
class SAOLossOutput:
    """Pure DIS loss result.

    ``loss_sum`` is differentiable and ``denominator`` is the number of
    structurally trainable action tokens.  A distributed caller can therefore
    scale ``loss_sum`` by an all-reduced denominator without reconstructing the
    DIS mask.  ``loss`` is the local structural-token mean used by
    :class:`SAO`'s regular :class:`StageAlgorithm` wrapper.
    """

    loss: torch.Tensor
    loss_sum: torch.Tensor
    denominator: torch.Tensor
    valid_mask: torch.Tensor
    structural_mask: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


def _same_shape(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    expected = tuple(reference.shape)
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected:
            raise ValueError(f"SAO: {name} shape={tuple(tensor.shape)} != expected {expected}")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum().to(dtype=values.dtype)
    total = torch.where(mask, values, torch.zeros_like(values)).sum()
    return torch.where(count > 0, total / count.clamp_min(1), total * 0.0)


def sao_policy_loss(
    *,
    current_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    advantages: torch.Tensor,
    eps_low: float,
    eps_high: float,
    structural_mask: Optional[torch.Tensor] = None,
) -> SAOLossOutput:
    """Compute SAO's strict, double-sided DIS score-function loss.

    All inputs are packed action-token tensors with identical shape.  The
    optional ``structural_mask`` identifies real policy-loss tokens (normally
    ``TextSegment.loss_mask != 0``); when omitted every token is structural.
    DIS-rejected and non-finite tokens have zero contribution but *remain* in
    the structural denominator.  Padding/non-loss tokens do not.

    Comparisons happen in log space, and exponentiation is performed only for
    accepted finite ratios.  This keeps extreme stale-policy gaps and NaNs from
    contaminating either the loss or its gradients.
    """

    low = float(eps_low)
    high = float(eps_high)
    if not (0.0 <= low < 1.0):
        raise ValueError(f"SAO: eps_low must be in [0, 1); got {eps_low!r}")
    if not (high >= 0.0 and math.isfinite(high)):
        raise ValueError(f"SAO: eps_high must be finite and >= 0; got {eps_high!r}")
    if not torch.is_floating_point(current_logp):
        raise TypeError(f"SAO: current_logp must be floating point; got {current_logp.dtype}")

    _same_shape(current_logp, rollout_logp=rollout_logp, advantages=advantages)
    if structural_mask is not None:
        _same_shape(current_logp, structural_mask=structural_mask)

    # Work in fp32 even when replay runs in bf16.  The cast remains
    # differentiable with respect to current_logp.
    current = current_logp.float()
    rollout = rollout_logp.to(device=current.device, dtype=torch.float32)
    advantage = advantages.to(device=current.device, dtype=torch.float32)
    if structural_mask is None:
        structural = torch.ones_like(current, dtype=torch.bool)
    else:
        raw_mask = structural_mask.to(device=current.device)
        structural = raw_mask != 0

    finite_logp = torch.isfinite(current) & torch.isfinite(rollout)
    finite_inputs = finite_logp & torch.isfinite(advantage)
    safe_current = torch.where(torch.isfinite(current), current, torch.zeros_like(current))
    safe_rollout = torch.where(torch.isfinite(rollout), rollout, torch.zeros_like(rollout))
    safe_advantage = torch.where(torch.isfinite(advantage), advantage, torch.zeros_like(advantage))
    log_ratio = safe_current - safe_rollout

    # Build thresholds in the same dtype/device and through the same logarithm
    # as normal tensor log-prob inputs.  Besides avoiding host/device churn,
    # this makes exactly representable boundary tests strict despite fp32's
    # rounding of ``log(1 - eps_low)``.
    log_lower = torch.log(current.new_tensor(1.0 - low))
    log_upper = torch.log(current.new_tensor(1.0 + high))
    valid = structural & finite_inputs & (log_ratio > log_lower) & (log_ratio < log_upper)

    # Only accepted log-ratios are exponentiated.  Invalid positions use zero
    # before exp (ratio=1) and are then zeroed by where, avoiding 0 * inf / NaN.
    accepted_log_ratio = torch.where(valid, log_ratio, torch.zeros_like(log_ratio))
    ratio_for_score = torch.exp(accepted_log_ratio)
    coefficient = torch.where(valid, ratio_for_score * safe_advantage, torch.zeros_like(current)).detach()
    loss_per_token = -coefficient * safe_current
    loss_sum = torch.where(structural, loss_per_token, torch.zeros_like(loss_per_token)).sum()
    denominator = structural.sum().to(dtype=loss_sum.dtype)
    loss = loss_sum / denominator.clamp_min(1.0)

    # Diagnostics remain finite even for arbitrarily stale policies.  Ratio
    # moments are over finite structural log-prob pairs and use a conventional
    # exp([-20, 20]) diagnostic clamp; DIS acceptance itself is never clamped.
    finite_ratio_mask = structural & finite_logp
    diagnostic_log_ratio = log_ratio.clamp(min=-20.0, max=20.0)
    diagnostic_ratio = torch.exp(diagnostic_log_ratio)
    ratio_mean = _masked_mean(diagnostic_ratio, finite_ratio_mask)
    ratio_second = _masked_mean(diagnostic_ratio.square(), finite_ratio_mask)
    ratio_std = (ratio_second - ratio_mean.square()).clamp_min(0.0).sqrt()
    finite_count = finite_ratio_mask.sum()
    if diagnostic_ratio.numel() == 0:
        ratio_min = ratio_mean * 0.0
        ratio_max = ratio_mean * 0.0
        absdiff_max = ratio_mean * 0.0
    else:
        ratio_min_raw = torch.where(
            finite_ratio_mask, diagnostic_ratio, torch.full_like(diagnostic_ratio, float("inf"))
        ).amin()
        ratio_max_raw = torch.where(
            finite_ratio_mask, diagnostic_ratio, torch.full_like(diagnostic_ratio, float("-inf"))
        ).amax()
        ratio_min = torch.where(finite_count > 0, ratio_min_raw, ratio_mean * 0.0)
        ratio_max = torch.where(finite_count > 0, ratio_max_raw, ratio_mean * 0.0)
        absdiff_max = torch.where(
            finite_count > 0,
            torch.where(finite_ratio_mask, diagnostic_log_ratio.abs(), torch.zeros_like(log_ratio)).amax(),
            ratio_mean * 0.0,
        )

    metric_denom = denominator.detach().clamp_min(1.0)
    lower_rejected = structural & finite_inputs & (log_ratio <= log_lower)
    upper_rejected = structural & finite_inputs & (log_ratio >= log_upper)
    nonfinite = structural & ~finite_inputs
    metrics: Dict[str, torch.Tensor] = {
        "dis_structural_tokens": denominator.detach(),
        "dis_accepted_tokens": valid.sum().to(dtype=loss_sum.dtype).detach(),
        "dis_accept_fraction": (valid.sum().to(dtype=loss_sum.dtype) / metric_denom).detach(),
        "dis_reject_lower_fraction": (lower_rejected.sum().to(dtype=loss_sum.dtype) / metric_denom).detach(),
        "dis_reject_upper_fraction": (upper_rejected.sum().to(dtype=loss_sum.dtype) / metric_denom).detach(),
        "dis_nonfinite_fraction": (nonfinite.sum().to(dtype=loss_sum.dtype) / metric_denom).detach(),
        "ratio_mean": ratio_mean.detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio_min.detach(),
        "ratio_max": ratio_max.detach(),
        "approx_kl": (0.5 * _masked_mean(diagnostic_log_ratio.square(), finite_ratio_mask)).detach(),
        "rollout_replay_logp_absdiff_mean": _masked_mean(diagnostic_log_ratio.abs(), finite_ratio_mask).detach(),
        "rollout_replay_logp_absdiff_max": absdiff_max.detach(),
    }
    return SAOLossOutput(
        loss=loss,
        loss_sum=loss_sum,
        denominator=denominator,
        valid_mask=valid,
        structural_mask=structural,
        metrics=metrics,
    )


@dataclass
class SAOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    eps_low: float = 0.3
    eps_high: float = 5.0
    sampling_temperature: Optional[float] = None


class SAO(StageAlgorithm):
    """Stage-driven SAO actor algorithm over packed action tokens."""

    supports_multi_update = False
    loss_agg_mode = "token-mean"

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        eps_low: float = 0.3,
        eps_high: float = 5.0,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("SAO: either `stage` or `pipeline` must be provided")
        self.stage = stage if stage is not None else getattr(pipeline, stage_attr)
        # Validate bounds once at construction; the pure helper validates again
        # because it is a public standalone API.
        if not (0.0 <= float(eps_low) < 1.0):
            raise ValueError(f"SAO: eps_low must be in [0, 1); got {eps_low!r}")
        if not (float(eps_high) >= 0.0 and math.isfinite(float(eps_high))):
            raise ValueError(f"SAO: eps_high must be finite and >= 0; got {eps_high!r}")
        self.eps_low = float(eps_low)
        self.eps_high = float(eps_high)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        if float(sampling_temperature) <= 0.0:
            raise ValueError(f"SAO: sampling_temperature must be > 0; got {sampling_temperature!r}")
        self.sampling_temperature = float(sampling_temperature)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del training_progress
        if segment.tokens is None or segment.log_probs is None or int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        current_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        if segment.token_advantages is not None:
            token_advantages = segment.token_advantages
        elif advantages.numel() == current_logp.numel():
            # Explicit compatibility path for a custom stack that already passes
            # packed token advantages through the abstract StageAlgorithm slot.
            token_advantages = advantages
        else:
            raise ValueError(
                "SAO requires packed per-token advantages in segment.token_advantages "
                f"(got {advantages.numel()} fallback values for {current_logp.numel()} tokens)"
            )

        output = sao_policy_loss(
            current_logp=current_logp,
            rollout_logp=segment.log_probs,
            advantages=token_advantages,
            eps_low=self.eps_low,
            eps_high=self.eps_high,
            structural_mask=segment.loss_mask,
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
            "policy_loss": float(output.loss.detach().item()),
            "eps_low": self.eps_low,
            "eps_high": self.eps_high,
            **{k: float(v.item()) for k, v in output.metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(output.loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=token_count,
            has_backward=True,
        )


__all__ = ["SAO", "SAOConfig", "SAOLossOutput", "sao_policy_loss"]
