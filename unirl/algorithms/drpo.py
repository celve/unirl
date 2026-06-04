"""DRPO (AR): Divergence Regularized Policy Optimization for token-level RL.

Implements **DRPO**, the proposed method of the AdaSPO paper "Rethinking the
Divergence Regularization in LLM Reinforcement Learning", for autoregressive
(token-level, discrete) policies. DRPO is introduced in the paper's
**§3 (Methodology)** and is the headline method evaluated in **§4 / Figure 3**.

Derivation (paper §3): DRPO takes DPPO's Binary-TV trust region
(§2.4, Eq 6-7: ``|π(a|s) − μ(a|s)| ≤ δ``), rewrites it as a **token-adaptive
ratio bound** ``|r_t − 1| ≤ δ / μ(a_t|s_t)``, and applies the SPO construction
(§2.3, Eq 5: the advantage-weighted χ² / ℓ²₂ regularizer). Substituting the
token-adaptive ``ε_t = ε / μ`` into SPO yields the per-token loss::

    L_t = −A_t · r_t  +  |A_t| · μ(a_t|s_t) · (r_t − 1)² / (2 · ε)

with ``r_t = π(a_t|s_t) / μ(a_t|s_t)`` the importance ratio and ``μ`` the
rollout-policy token probability. The induced **gradient weight** (paper Table 1;
gradient in Eq 9) is the *smooth*, advantage-aware
``1 − sign(Â_t(r_t−1)) · |π−μ| / δ`` — a continuous trust region that attenuates
diverging updates and provides a corrective signal beyond the boundary, unlike
DPPO's hard 0/1 mask (§2.4) or SPO's fixed ratio bound (§2.3).

This is exactly verl's ``spo_adaptive_eps`` policy loss
(``actor.policy_loss.loss_mode``) — the loss the reference DRPO recipe selects.
``ε`` is the "regularization threshold" set to 12.5 in the paper (§4).

(AR-only by design; a diffusion DRPO sibling can be added in its own PR.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .ar_grpo import ARGRPO
from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    rollout_replay_logp_absdiff,
    typed_conditions,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ARDRPOConfig(BaseAlgorithmConfig):
    """Config for :class:`ARDRPO` (the paper's DRPO method, §3).

    Attributes:
        stage_attr: Which stage slot to bind to (``"ar"``).
        conditions_cls: Dotted path to the stage-typed conditions class.
        drpo_epsilon: Regularization threshold ``ε`` of the token-adaptive SPO
            quadratic. Paper §4: "For SPO and DRPO, we set the regularization
            threshold to 12.5." Larger ``ε`` ⇒ weaker regularization; the
            per-token trust region is ``ε_t = ε / μ`` (§3).
        loss_agg_mode: ``"token-mean"`` or ``"seq-mean-token-sum-norm"``
            (per-seq token-SUM / horizon, then mean over sequences).
        horizon: Fixed length normalizer for ``seq-mean-token-sum-norm``
            (= max response length).
        sampling_temperature: Rollout sampling temperature; replay rescales
            logits by it so ``log_softmax(logits / T)`` matches the sampling
            distribution. MUST equal ``sampling.temperature``. Falls back to the
            :class:`ARSamplingParams` default when None.
    """

    stage_attr: str = "ar"
    conditions_cls: str = ""
    # Paper §4: "For SPO and DRPO, we set the regularization threshold to 12.5."
    drpo_epsilon: float = 12.5
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# Loss helper — AR (token-level)
# ---------------------------------------------------------------------------


def _ar_drpo_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DRPO per-token loss (paper §3; gradient = Eq 9, weight = Table 1).

    Operates on packed-varlen ``[total_tokens]`` tensors (the natural setting for
    DRPO, which targets discrete token-level policies). Per token::

        L_t = −A_t · r_t  +  |A_t| · μ · (r_t − 1)² / (2 · ε)

    where ``r_t = π/μ`` is the importance ratio, ``μ = exp(old_logp)`` is the
    rollout-policy token probability, and ``ε`` is the regularization threshold
    (12.5, §4). The first term is the importance-weighted policy gradient; the
    second is SPO's advantage-weighted quadratic regularizer (§2.3, Eq 5) carrying
    the Binary-TV's token-adaptive ``ε_t = ε / μ`` (§3). ``r_t`` is kept
    differentiable (no ``.detach()``, no TIS truncation), so the smooth Table-1
    gradient weight ``1 − sign(Â_t(r_t−1)) · |π−μ| / δ`` arises via autograd.

    Mirrors verl ``compute_policy_loss_spo_adaptive_eps`` exactly.

    Args:
        new_logp: New-policy log-probs at current weights. ``[total_tokens]``.
        old_logp: Rollout-policy (μ) log-probs, frozen. ``[total_tokens]``.
        advantages: Per-token advantages (expanded from per-sample).
        epsilon: Regularization threshold ``ε`` (12.5).

    Returns:
        ``(loss_per_element, metrics_dict)``. Reduction is the caller's job.
    """
    log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)  # r_t = π/μ (differentiable through new_logp)
    adv = advantages.detach()
    old_prob = torch.exp(old_logp).detach()  # μ = rollout-policy token probability

    # SPO advantage-weighted quadratic (§2.3 Eq 5) with the Binary-TV
    # token-adaptive trust region ε_t = ε / μ (§3). r_t stays differentiable.
    ratio_delta = ratio - 1.0
    quadratic_penalty = adv.abs() * old_prob * ratio_delta.square() / (2.0 * epsilon)
    pg_losses = -adv * ratio + quadratic_penalty

    # Token-adaptive trust-region boundary r* = 1 ± ε_t (§3) — diagnostics only.
    adaptive_eps = torch.where(old_prob > 0.0, epsilon / old_prob, torch.full_like(old_prob, float("inf")))
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": ((ratio - 1.0) - log_diff).mean().detach(),  # k3 estimator
        "drpo_penalty_mean": quadratic_penalty.mean().detach(),
        "clipfrac_upper": (ratio > (1.0 + adaptive_eps)).float().mean().detach(),
        "clipfrac_lower": (ratio < (1.0 - adaptive_eps)).float().mean().detach(),
    }
    return pg_losses, metrics


# ---------------------------------------------------------------------------
# Algorithm class — AR (token-level)
# ---------------------------------------------------------------------------


class ARDRPO(StageAlgorithm):
    """DRPO for AR token-level policies — the paper's proposed method (§3).

    DRPO (Divergence Regularized Policy Optimization) replaces DPPO's hard
    Binary-TV mask (§2.4) with a **smooth, advantage-weighted, token-adaptive
    quadratic regularizer** — SPO's construction (§2.3) applied to the Binary-TV
    token-adaptive ratio bound ``|r−1| ≤ δ/μ``. Per-token loss (§3; gradient
    Eq 9; gradient weight Table 1)::

        L_t = −A_t · r_t  +  |A_t| · μ · (r_t − 1)² / (2 · ε)

    Args:
        pipeline: The trainer-injected pipeline; the stage is resolved from it via
            ``getattr(pipeline, stage_attr)``. v2-only — there is no v1 ``stage=``
            path.
        stage_attr: Which pipeline attribute holds the AR stage (``"ar"``).
        drpo_epsilon: Regularization threshold ``ε`` (12.5; paper §4).
        loss_agg_mode: ``"token-mean"`` or ``"seq-mean-token-sum-norm"``.
        horizon: Fixed length normalizer for ``seq-mean-token-sum-norm``.
        sampling_temperature: Rollout sampling temperature, passed to
            ``stage.replay`` so its log-softmax matches the sampling distribution.
            MUST equal ``sampling.temperature``.
        conditions_cls: Stage-typed conditions container.
    """

    def __init__(
        self,
        *,
        pipeline: Any = None,
        stage_attr: str = "ar",
        drpo_epsilon: float = 12.5,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        sampling_temperature: Optional[float] = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        # v2-only: the trainer injects the shared ``pipeline``
        # (remote_hydra(algorithm_cfg, pipeline=...)) and we resolve the stage
        # from it. There is no v1 ``stage=`` path — ARDRPO is v2-only.
        if pipeline is None:
            raise ValueError("ARDRPO: `pipeline` must be provided (the v2 trainer injects it)")
        self.stage = getattr(pipeline, stage_attr)
        self.drpo_epsilon = float(drpo_epsilon)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        # replay rescales logits by this temperature so its log-softmax matches
        # the rollout sampling distribution (log_softmax(logits / T)); MUST equal
        # sampling.temperature. Mirrors ARGRPO. Falls back to the ARSamplingParams
        # default when unset.
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.conditions_cls = conditions_cls

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens]
        # NOTE(multi-epoch): old_logp is the rollout log-prob — correct only for a
        # single on-policy update. Before enabling num_updates_per_batch>1 for AR,
        # snapshot a frozen train-side old_logp here (mirror
        # DiffusionGRPO.prepare_segment); reusing rollout logp across PPO epochs
        # conflates the rollout-vs-train engine gap with real policy drift.
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)

        # Expand per-sample advantages to per-token
        adv_per_token = ARGRPO._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        loss_per_elem, ratio_metrics = _ar_drpo_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            epsilon=self.drpo_epsilon,
        )

        # Apply loss_mask if present (token-level masking for padding/eos)
        if segment.loss_mask is not None:
            mask = segment.loss_mask.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
            loss_per_elem = loss_per_elem * mask

        if self.loss_agg_mode == "seq-mean-token-sum-norm" and segment.lengths is not None:
            # Reference seq-mean-token-sum-norm: per-sequence token-SUM / horizon, then
            # mean over the micro-batch's sequences. The stack's loss_scale=1/num_micros
            # then averages across micro-batches, giving the overall mean-over-sequences.
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "drpo_epsilon": self.drpo_epsilon,
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["ARDRPO"]
