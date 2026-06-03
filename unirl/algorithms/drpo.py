"""DRPO (AR): divergence-based trust-region masking for token-level RL.

Implements the DPPO-Binary-TV and DPPO-Binary-KL policy losses from
"Rethinking the Trust Region in LLM Reinforcement Learning"
(https://arxiv.org/pdf/2602.04879) for autoregressive (token-level,
discrete) policies. ``ARDRPO`` applies trust-region masking per-token
over packed-varlen ``TextSegment`` data.

Two divergence variants:
  - **TV** (``variant="tv"``): Total Variation |π_θ(a|s) - π_old(a|s)| —
    the absolute change in the chosen token's probability.
  - **KL** (``variant="kl"``): Binary KL between the old and new Bernoulli
    token probabilities, computed from per-token log-probs only (no
    full-vocabulary logits required).

(AR-only by design; a diffusion DRPO sibling can be added in its own PR.)

Both variants share:
  1. Truncated Importance Sampling (TIS): ``truncated_ratio = clamp(ratio, max=C).detach()``
  2. Advantage-aware asymmetric mask: separate thresholds for adv>0 and adv<0
  3. Loss form: ``-advantages * truncated_ratio * new_logp * valid_mask``
     (REINFORCE + TIS correction + trust-region mask)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

import torch

from unirl.config.registration import register_config
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


@register_config(
    group="algorithm",
    name="ar_drpo",
    target="unirl.algorithms.drpo.ARDRPO",
)
@dataclass
class ARDRPOConfig(BaseAlgorithmConfig):
    """Config for :class:`ARDRPO`.

    Attributes:
        stage_attr: Which stage slot to bind to (``"ar"``).
        conditions_cls: Dotted path to the stage-typed conditions class.
        variant: Which divergence measure to use — ``"tv"`` or ``"kl"``.
            TV uses |π_θ - π_old| (true Total Variation) as the divergence.
            KL uses Binary KL between old and new Bernoulli token
            probabilities (computed from per-token log-probs; no logits
            required).
        clip_divergence: Divergence threshold for the trust region mask.
            For TV: threshold on |π_θ - π_old| (absolute probability diff).
            For KL: threshold on Binary KL (nats).
        clip_divergence_low: Asymmetric lower threshold (adv < 0 direction).
            Defaults to ``clip_divergence`` if None.
        clip_divergence_high: Asymmetric upper threshold (adv > 0 direction).
            Defaults to ``clip_divergence`` if None.
        clip_ratio_c: TIS truncation bound for the importance ratio.
        sampling_temperature: Rollout sampling temperature; replay rescales
            logits by it so ``log_softmax(logits / T)`` matches the sampling
            distribution. MUST equal ``sampling.temperature``. Falls back to
            the :class:`ARSamplingParams` default when None.
    """

    stage_attr: str = "ar"
    conditions_cls: str = ""
    variant: str = "tv"  # "tv" | "kl" | "pg_tv_penalty"
    clip_divergence: float = 0.2
    clip_divergence_low: Optional[float] = None
    clip_divergence_high: Optional[float] = None
    clip_ratio_c: float = 20.0
    # pg_tv_penalty (reference run_qwen3_4b.sh default): L = -A*logp + penalty_coef*|ratio-1|.
    # penalty_coef is the reference's actor.clip_ratio used as the TV-penalty coefficient (0.15).
    penalty_coef: float = 0.15
    # "token-mean" (current) or "seq-mean-token-sum-norm" (reference: per-seq token-SUM / horizon,
    # then mean over sequences). horizon = max response length (the fixed normalizer).
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# Loss helpers — AR (token-level)
# ---------------------------------------------------------------------------


def _ar_drpo_tv_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_divergence_low: float,
    clip_divergence_high: float,
    clip_ratio_c: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DRPO TV-variant loss for AR token-level policies.

    Operates on packed-varlen ``[total_tokens]`` tensors (the natural
    setting for the DRPO paper, which was designed for discrete
    token-level policies).

    Uses |π_θ(a|s) - π_old(a|s)| as the divergence measure — the true
    Total Variation distance from the DRPO paper. For discrete tokens
    the chosen-token probability is a well-defined scalar in [0, 1], so the
    absolute difference is the correct TV (not a ratio-based proxy).

    Trust-region mask (asymmetric, advantage-aware):
      - adv > 0: keep if (prob - old_prob) <= clip_divergence_high
      - adv < 0: keep if (prob - old_prob) >= -clip_divergence_low

    Loss: ``-advantages * truncated_ratio.detach() * new_logp * valid_mask``

    Args:
        new_logp: New policy log-probs at current weights. ``[total_tokens]``.
        old_logp: Old policy log-probs frozen at pre-update weights.
            ``[total_tokens]``.
        advantages: Per-token advantages (already expanded from per-sample).
            ``[total_tokens]``.
        clip_divergence_low: Divergence threshold for adv < 0 direction.
        clip_divergence_high: Divergence threshold for adv > 0 direction.
        clip_ratio_c: TIS truncation bound for the importance ratio.

    Returns:
        ``(loss_per_element, metrics_dict)``. Reduction is the caller's job.
    """
    log_diff = new_logp - old_logp
    # Clamp for numerical stability
    log_diff = torch.clamp(log_diff, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)
    adv = advantages.detach()

    # Truncated Importance Sampling (TIS) — large threshold to minimise bias
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c).detach()

    # True TV divergence: |π_θ(a|s) - π_old(a|s)|
    # This is the original DRPO paper formulation, not |ratio-1|.
    prob = torch.exp(new_logp)
    old_prob = torch.exp(old_logp)
    prob_delta = prob - old_prob
    valid_positive_mask = prob_delta <= clip_divergence_high  # adv > 0: small prob increase ok
    valid_negative_mask = prob_delta >= -clip_divergence_low  # adv < 0: small prob decrease ok
    valid_mask = torch.where(adv > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    # REINFORCE + TIS + trust-region mask
    pg_losses = -adv * truncated_ratio * new_logp * valid_mask

    # Metrics
    approx_kl = ((ratio - 1.0) - log_diff).mean()  # k3 estimator
    clip_fraction = (1.0 - valid_mask).float().mean()
    clipfrac_lower = ((ratio > clip_ratio_c).float() * valid_mask).mean()

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)

    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "clipfrac_lower": clipfrac_lower.detach(),
        "valid_fraction": valid_mask.mean().detach(),
        "pos_masked_fraction": ((~valid_positive_mask) & (adv > 0)).float().mean().detach(),
        "neg_masked_fraction": ((~valid_negative_mask) & (adv <= 0)).float().mean().detach(),
    }
    return pg_losses, metrics


def _ar_drpo_kl_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_divergence_low: float,
    clip_divergence_high: float,
    clip_ratio_c: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DRPO KL-variant loss for AR token-level policies.

    Operates on packed-varlen ``[total_tokens]`` tensors. Uses Binary KL
    as the divergence measure — the natural discrete analogue from the
    DRPO paper for token-level policies.

    Binary KL treats each token position as a Bernoulli distribution
    parameterised by the probability of the chosen token under the old
    and new policies:

        binary_kl = old_prob * (old_logp - new_logp)
                  + (1 - old_prob) * log((1 - old_prob + eps) / (1 - new_prob + eps))

    This requires only per-token log-probs of the *chosen* token — no
    full-vocabulary logits needed.

    Trust-region mask (asymmetric, advantage-aware, mirrors DRPO
    Binary-KL logic):
      - adv > 0: keep if binary_kl <= clip_divergence_high OR ratio <= 1
        (prob decreasing = conservative update, always safe)
      - adv < 0: keep if binary_kl <= clip_divergence_low OR ratio >= 1
        (prob increasing = conservative update, always safe)

    Loss: ``-advantages * truncated_ratio.detach() * new_logp * valid_mask``

    Args:
        new_logp: New policy log-probs at current weights. ``[total_tokens]``.
        old_logp: Old policy log-probs frozen at pre-update weights.
            ``[total_tokens]``.
        advantages: Per-token advantages (already expanded from per-sample).
            ``[total_tokens]``.
        clip_divergence_low: Binary KL threshold for adv < 0 direction.
        clip_divergence_high: Binary KL threshold for adv > 0 direction.
        clip_ratio_c: TIS truncation bound for the importance ratio.

    Returns:
        ``(loss_per_element, metrics_dict)``. Reduction is the caller's job.
    """
    log_diff = new_logp - old_logp
    # Clamp for numerical stability
    log_diff = torch.clamp(log_diff, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)
    adv = advantages.detach()

    # Truncated Importance Sampling (TIS) — large threshold to minimise bias
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c).detach()

    # Binary KL divergence per token
    #   old_prob = exp(old_logp) — probability of chosen token under old policy
    #   new_prob = exp(new_logp) — probability of chosen token under new policy
    #   KL(Bernoulli(old_prob) || Bernoulli(new_prob))
    #     = old_prob * log(old_prob / new_prob) + (1 - old_prob) * log((1 - old_prob) / (1 - new_prob))
    old_prob = torch.exp(old_logp)
    new_prob = torch.exp(new_logp)
    eps = 1e-8
    binary_kl = old_prob * (old_logp - new_logp) + (1.0 - old_prob) * torch.log(
        (1.0 - old_prob + eps) / (1.0 - new_prob + eps)
    )
    # Clamp KL to avoid inf from numerical edge cases
    binary_kl = torch.clamp(binary_kl, min=0.0, max=1e6)

    # Advantage-aware asymmetric mask (mirrors DRPO Binary-KL logic)
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (ratio <= 1.0)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (ratio >= 1.0)
    valid_mask = torch.where(adv > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    # REINFORCE + TIS + trust-region mask
    pg_losses = -adv * truncated_ratio * new_logp * valid_mask

    # Metrics
    approx_kl = ((ratio - 1.0) - log_diff).mean()  # k3 estimator
    clip_fraction = (1.0 - valid_mask).float().mean()
    clipfrac_lower = ((ratio > clip_ratio_c).float() * valid_mask).mean()

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)

    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": approx_kl.detach(),
        "binary_kl_mean": binary_kl.mean().detach(),
        "binary_kl_max": binary_kl.max().detach(),
        "clip_fraction": clip_fraction.detach(),
        "clipfrac_lower": clipfrac_lower.detach(),
        "valid_fraction": valid_mask.mean().detach(),
        "pos_masked_fraction": ((~valid_positive_mask) & (adv > 0)).float().mean().detach(),
        "neg_masked_fraction": ((~valid_negative_mask) & (adv <= 0)).float().mean().detach(),
    }
    return pg_losses, metrics


def _ar_pg_tv_penalty_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    penalty_coef: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reference ``pg_tv_penalty`` loss (MaxwellJryao/DRPO core_algos.py).

    Verbatim form::

        ratio   = exp(new_logp - old_logp)
        L_t     = -A_t * new_logp_t  +  penalty_coef * |ratio_t - 1|

    Plain REINFORCE (no importance-sampling weight, no trust-region mask) plus a
    soft TV penalty that continuously pulls the ratio toward 1 — the reference's
    only regularizer. ``penalty_coef`` is the reference ``actor.clip_ratio`` (0.15).
    """
    log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)
    adv = advantages.detach()

    tv_penalty = penalty_coef * torch.abs(ratio - 1.0)
    pg_losses = -adv * new_logp + tv_penalty

    approx_kl = (-log_diff).mean()  # reference ppo_kl = masked_mean(old-new)
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": approx_kl.detach(),
        "tv_penalty_mean": tv_penalty.mean().detach(),
    }
    return pg_losses, metrics


# ---------------------------------------------------------------------------
# Algorithm class — AR (token-level)
# ---------------------------------------------------------------------------


class ARDRPO(StageAlgorithm):
    """DRPO trust-region masking for AR token-level policies.

    The original DRPO paper (https://arxiv.org/pdf/2602.04879) targets
    discrete token-level policies — this is the natural setting.

    Two variants:

    **TV variant** (``variant="tv"``):
    - Uses |π_θ(a|s) - π_old(a|s)| as the divergence (true Total Variation)
    - Advantage-aware asymmetric mask with separate thresholds for adv>0
      and adv<0

    **KL variant** (``variant="kl"``):
    - Uses Binary KL between old and new Bernoulli token probabilities
    - Computed from per-token log-probs only (no logits required)
    - Advantage-aware asymmetric mask with "always allow conservative
      update" logic: even when KL is high, updates that move the policy
      *opposite* to the advantage direction are allowed

    Both variants use:
    1. Truncated Importance Sampling (TIS) with large default bound
    2. REINFORCE + TIS form: ``-adv * truncated_ratio.detach() * logp * mask``
    3. Asymmetric advantage-aware divergence masking

    Args:
        pipeline: The trainer-injected pipeline; the stage is resolved from
            it via ``getattr(pipeline, stage_attr)``. v2-only — there is no
            v1 ``stage=`` path.
        stage_attr: Which pipeline attribute holds the AR stage (``"ar"``).
        variant: ``"tv"`` or ``"kl"`` — which divergence measure to use.
        clip_divergence: Divergence threshold for the trust region mask.
            For TV: threshold on |π_θ - π_old| (absolute probability diff).
            For KL: threshold on Binary KL (nats).
        clip_divergence_low: Asymmetric lower threshold. Defaults to
            ``clip_divergence`` if None.
        clip_divergence_high: Asymmetric upper threshold. Defaults to
            ``clip_divergence`` if None.
        clip_ratio_c: TIS truncation bound for the importance ratio.
        sampling_temperature: Rollout sampling temperature, passed to
            ``stage.replay`` so its log-softmax matches the sampling
            distribution. MUST equal ``sampling.temperature``.
        conditions_cls: Stage-typed conditions container.
    """

    def __init__(
        self,
        *,
        pipeline: Any = None,
        stage_attr: str = "ar",
        variant: str = "tv",
        clip_divergence: float = 0.2,
        clip_divergence_low: Optional[float] = None,
        clip_divergence_high: Optional[float] = None,
        clip_ratio_c: float = 20.0,
        penalty_coef: float = 0.15,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        sampling_temperature: Optional[float] = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if variant not in ("tv", "kl", "pg_tv_penalty"):
            raise ValueError(f"ARDRPO variant must be 'tv', 'kl', or 'pg_tv_penalty', got '{variant}'")
        # v2-only: the trainer injects the shared ``pipeline``
        # (remote_hydra(algorithm_cfg, pipeline=...)) and we resolve the stage
        # from it. There is no v1 ``stage=`` path — ARDRPO is v2-only.
        if pipeline is None:
            raise ValueError("ARDRPO: `pipeline` must be provided (the v2 trainer injects it)")
        self.stage = getattr(pipeline, stage_attr)
        self.variant = variant
        self.clip_divergence = float(clip_divergence)
        self.clip_divergence_low = (
            float(clip_divergence_low) if clip_divergence_low is not None else self.clip_divergence
        )
        self.clip_divergence_high = (
            float(clip_divergence_high) if clip_divergence_high is not None else self.clip_divergence
        )
        self.clip_ratio_c = float(clip_ratio_c)
        self.penalty_coef = float(penalty_coef)
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

        if self.variant == "tv":
            loss_per_elem, ratio_metrics = _ar_drpo_tv_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=adv_per_token,
                clip_divergence_low=self.clip_divergence_low,
                clip_divergence_high=self.clip_divergence_high,
                clip_ratio_c=self.clip_ratio_c,
            )
        elif self.variant == "pg_tv_penalty":
            loss_per_elem, ratio_metrics = _ar_pg_tv_penalty_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=adv_per_token,
                penalty_coef=self.penalty_coef,
            )
        else:  # kl
            loss_per_elem, ratio_metrics = _ar_drpo_kl_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=adv_per_token,
                clip_divergence_low=self.clip_divergence_low,
                clip_divergence_high=self.clip_divergence_high,
                clip_ratio_c=self.clip_ratio_c,
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
            "clip_divergence_low": self.clip_divergence_low,
            "clip_divergence_high": self.clip_divergence_high,
            "clip_ratio_c": self.clip_ratio_c,
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
