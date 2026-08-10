"""AReaL's actor-only, decoupled PPO objective for text trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import AlgorithmStepResult, BaseAlgorithmConfig, StageAlgorithm, _grpo_clip_loss, typed_conditions


@dataclass
class ARealPPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 0.4
    behavior_ratio_upper: float = 5.0
    sampling_temperature: float = 1.0


class ARealPPO(StageAlgorithm):
    """AReaL actor loss over behavior and proximal log-probabilities."""

    supports_multi_update = False
    loss_weighting = "token"
    anchor_fields = ("log_probs", "rollout_log_probs")

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 0.4,
        behavior_ratio_upper: float = 5.0,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("ARealPPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        if not 0.0 < float(clip_range) < 1.0:
            raise ValueError(f"ARealPPO: clip_range must be in (0, 1); got {clip_range}")
        if float(behavior_ratio_upper) <= 1.0:
            raise ValueError(f"ARealPPO: behavior_ratio_upper must be greater than 1; got {behavior_ratio_upper}")
        if float(sampling_temperature) <= 0.0:
            raise ValueError(f"ARealPPO: sampling_temperature must be positive; got {sampling_temperature}")
        self.stage = stage
        self.clip_range = float(clip_range)
        self.behavior_ratio_upper = float(behavior_ratio_upper)
        self.conditions_cls = conditions_cls
        self.sampling_temperature = float(sampling_temperature)

    def recomputes_anchor(self) -> bool:
        return True

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
    ) -> None:
        """Save rollout log-probabilities and freeze the pre-update policy."""
        if segment.tokens is None or segment.log_probs is None or int(segment.tokens.numel()) == 0:
            return
        if segment.log_probs.shape != segment.tokens.shape:
            raise ValueError(
                "ARealPPO.prepare_segment: behavior log-probabilities must align with tokens; "
                f"got {tuple(segment.log_probs.shape)} and {tuple(segment.tokens.shape)}"
            )
        if segment.rollout_log_probs is None:
            segment.rollout_log_probs = segment.log_probs.detach().cpu().clone()
        elif segment.rollout_log_probs.shape != segment.tokens.shape:
            raise ValueError("ARealPPO.prepare_segment: rollout log-probabilities must align with tokens")

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            proximal = self.stage.replay(
                typed_conds,
                segment=segment,
                temperature=self.sampling_temperature,
            )
        if proximal.shape != segment.tokens.shape:
            raise ValueError(
                "ARealPPO.prepare_segment: proximal replay must align with tokens; "
                f"got {tuple(proximal.shape)} and {tuple(segment.tokens.shape)}"
            )
        segment.log_probs = proximal.detach().cpu()

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
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.rollout_log_probs is None:
            raise ValueError("ARealPPO: prepare_segment must populate rollout_log_probs before training")
        if segment.loss_mask is None:
            raise ValueError("ARealPPO requires the AReaL trajectory loss mask")
        if int(segment.tokens.numel()) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        current = self.stage.replay(
            typed_conds,
            segment=segment,
            temperature=self.sampling_temperature,
        )
        expected = segment.tokens.shape
        proximal = segment.log_probs.to(device=current.device, dtype=current.dtype)
        behavior = segment.rollout_log_probs.to(device=current.device, dtype=current.dtype)
        active = segment.loss_mask.to(device=current.device, dtype=torch.bool)
        named = {"current": current, "proximal": proximal, "behavior": behavior, "loss_mask": active}
        mismatched = {name: tuple(value.shape) for name, value in named.items() if value.shape != expected}
        if mismatched:
            raise ValueError(f"ARealPPO: packed tensor shape mismatch; expected {tuple(expected)}, got {mismatched}")

        token_advantages = self._expand_advantages(
            advantages,
            segment.lengths,
            dtype=current.dtype,
            device=current.device,
        )
        active_count = int(active.count_nonzero().item())
        if active_count == 0:
            zero_loss = current.sum() * 0.0
            (zero_loss * float(loss_scale)).backward()
            return AlgorithmStepResult(
                loss=0.0,
                metrics={"policy_loss": 0.0, "active_tokens": 0.0, "kept_tokens": 0.0},
                num_steps_or_tokens=0,
                has_backward=True,
            )

        current_active = current[active]
        proximal_active = proximal[active]
        behavior_active = behavior[active]
        advantages_active = token_advantages[active]

        behavior_log_ratio = proximal_active.detach().float() - behavior_active.detach().float()
        behavior_log_ratio = torch.where(
            torch.isfinite(behavior_log_ratio),
            behavior_log_ratio,
            torch.zeros_like(behavior_log_ratio),
        )
        behavior_weight = torch.exp(behavior_log_ratio)
        keep = behavior_weight <= self.behavior_ratio_upper
        corrected_weight = torch.where(keep, behavior_weight, torch.zeros_like(behavior_weight)).detach()

        policy_per_token, ratio_metrics = _grpo_clip_loss(
            new_logp=current_active,
            old_logp=proximal_active,
            advantages=advantages_active,
            clip_range=self.clip_range,
        )
        loss = (policy_per_token * corrected_weight).sum() / active_count
        (loss * float(loss_scale)).backward()

        kept_count = int(keep.count_nonzero().item())
        behavior_absdiff = behavior_log_ratio.abs()
        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach()),
            "clip_range": self.clip_range,
            "behavior_ratio_upper": self.behavior_ratio_upper,
            "behavior_ratio_mean": float(behavior_weight.mean()),
            "behavior_ratio_max": float(behavior_weight.max()),
            "behavior_prox_logp_absdiff_mean": float(behavior_absdiff.mean()),
            "behavior_prox_logp_absdiff_max": float(behavior_absdiff.max()),
            "rejected_tokens": float(active_count - kept_count),
            "rejection_fraction": float((active_count - kept_count) / active_count),
            "active_tokens": float(active_count),
            "kept_tokens": float(kept_count),
            "original_denominator": float(active_count),
            **{name: float(value) for name, value in ratio_metrics.items()},
        }
        if behavior_weight.numel() > 1:
            finite_weights = behavior_weight[torch.isfinite(behavior_weight)]
            if finite_weights.numel():
                metrics["behavior_ratio_p95"] = float(torch.quantile(finite_weights, 0.95))
        return AlgorithmStepResult(
            loss=float(loss.detach()),
            metrics=metrics,
            num_steps_or_tokens=active_count,
            has_backward=True,
        )

    @staticmethod
    def _expand_advantages(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if advantages.ndim != 1 or lengths.ndim != 1 or advantages.shape != lengths.shape:
            raise ValueError(
                "ARealPPO: advantages and packed lengths must be aligned 1D tensors; "
                f"got {tuple(advantages.shape)} and {tuple(lengths.shape)}"
            )
        return torch.repeat_interleave(
            advantages.detach().to(device=device, dtype=dtype),
            lengths.to(device=device, dtype=torch.long),
        )


__all__ = ["ARealPPO", "ARealPPOConfig"]
