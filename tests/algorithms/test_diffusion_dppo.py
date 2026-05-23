"""Tests for the Flow-DPPO (DiffusionDPPO) stage-driven algorithm.

Validates :class:`DiffusionDPPO` against a fake diffusion stage that returns
both ``log_probs`` and ``prev_sample_means`` from ``replay``. Tests cover:

1. prepare_segment populates sde_means and sde_logp
2. compute_loss_and_backward fires gradients and returns expected metrics
3. KL masking behavior: low-KL passes, high-KL conflicting direction masked
4. No SDE indices → no-op step
5. Integration with StageTrainStack dispatch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import pytest
import torch
import torch.nn as nn

from diffusionrl.algorithms import AlgorithmStepResult, DiffusionDPPO
from diffusionrl.algorithms.dppo import _dppo_kl_adv_loss, _gaussian_kl_div
from diffusionrl.models.types.replay_result import ReplayResult
from diffusionrl.types.conditions import Condition, TextEmbedCondition
from diffusionrl.types.segments import LatentSegment

# ---------------------------------------------------------------------------
# Fake stage for DPPO testing
# ---------------------------------------------------------------------------


@dataclass
class _FakeParams:
    eta: float = 0.7


class _FakeDPPOStage:
    """Minimal stage that returns both log_probs and prev_sample_means.

    ``replay`` returns:
    - ``log_probs``: ``param * sum(text.embeds)`` broadcast to ``[B, S']``
    - ``prev_sample_means``: ``param * latents[:, :S]`` with spatial dims

    The ``param`` is an ``nn.Parameter`` so we can assert gradients.
    """

    def __init__(self, *, init_value: float = 0.5, means_offset: float = 0.0) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))
        self._means_offset = means_offset

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: LatentSegment,
        params: Any = None,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        text = conditions["text"]
        assert text.embeds is not None
        B = int(text.embeds.shape[0])
        if step_indices is None:
            S = 0 if segment.sde_indices is None else int(segment.sde_indices.shape[0])
        else:
            S = len(step_indices)
        feat = text.embeds.float().reshape(B, -1).mean(dim=1)  # [B]
        out = self.param * feat
        log_probs = out.unsqueeze(1).expand(B, max(S, 1))[:, :S].contiguous()

        # Produce means: [B, S, C, H, W] from latents shape
        means = self.param * segment.latents[:, :S].float() + self._means_offset
        return ReplayResult(log_probs=log_probs, prev_sample_means=means)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_latent_segment_for_dppo(
    *, batch_size: int = 2, num_steps: int = 3, latent_channels: int = 4, spatial: int = 8
) -> LatentSegment:
    """Synthetic LatentSegment for DPPO testing (no sde_logp or sde_means pre-filled)."""
    sde_indices_list = list(range(num_steps))
    return LatentSegment(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        latents=torch.randn(batch_size, num_steps + 1, latent_channels, spatial, spatial),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
        sde_logp=None,  # Will be populated by prepare_segment
        sde_means=None,  # Will be populated by prepare_segment
        sde_indices=torch.tensor(sde_indices_list, dtype=torch.long),
    )


def _make_latent_segment_with_logp(*, batch_size: int = 2, num_steps: int = 3) -> LatentSegment:
    """Synthetic LatentSegment with pre-filled sde_logp (native mode)."""
    sde_indices_list = list(range(num_steps))
    sde_logp = torch.full((batch_size, num_steps), -1.0, dtype=torch.float32)
    return LatentSegment(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        latents=torch.randn(batch_size, num_steps + 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
        sde_logp=sde_logp,
        sde_means=None,
        sde_indices=torch.tensor(sde_indices_list, dtype=torch.long),
    )


def _conditions_with_text(*, batch_size: int) -> Mapping[str, Condition]:
    return {"text": TextEmbedCondition(embeds=torch.randn(batch_size, 4, 8))}


# ---------------------------------------------------------------------------
# _gaussian_kl_div unit test
# ---------------------------------------------------------------------------


def test_gaussian_kl_div_zero_diff() -> None:
    """KL between identical means is zero."""
    p = torch.randn(2, 3, 4, 8, 8)
    sigma = torch.ones(1, 3, 1, 1, 1)
    kl = _gaussian_kl_div(p, p, sigma)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7)


def test_gaussian_kl_div_known_value() -> None:
    """KL = (p-q)^2 / (2*sigma^2) for known inputs."""
    p = torch.tensor([2.0])
    q = torch.tensor([1.0])
    sigma = torch.tensor([1.0])
    kl = _gaussian_kl_div(p, q, sigma)
    assert torch.allclose(kl, torch.tensor([0.5]))


def test_gaussian_kl_div_sigma_scaling() -> None:
    """Larger sigma → smaller KL (less divergence relative to noise)."""
    p = torch.tensor([2.0])
    q = torch.tensor([1.0])
    kl_small_sigma = _gaussian_kl_div(p, q, torch.tensor([0.5]))
    kl_large_sigma = _gaussian_kl_div(p, q, torch.tensor([2.0]))
    assert kl_small_sigma > kl_large_sigma


# ---------------------------------------------------------------------------
# _dppo_kl_adv_loss unit test
# ---------------------------------------------------------------------------


def test_dppo_kl_adv_loss_all_below_threshold() -> None:
    """When all KL < threshold, keep_adv_mask is all 1 (no masking)."""
    B, S = 4, 2
    new_logp = torch.zeros(B, S)
    old_logp = torch.zeros(B, S)
    # Identical means → KL = 0 < any threshold
    means = torch.randn(B, S, 4, 8, 8)
    sigma_t = torch.ones(1, S, 1, 1, 1)

    loss, metrics = _dppo_kl_adv_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        new_means=means,
        old_means=means.clone(),
        advantages=torch.ones(B, S),
        sigma_t=sigma_t,
        kl_mask_threshold=1e-5,
    )
    # All kept → unmasked_fraction = 1.0
    assert float(metrics["unmasked_fraction"].item()) == 1.0
    assert float(metrics["masked_fraction"].item()) == 0.0


def test_dppo_kl_adv_loss_masking_conflicting_direction() -> None:
    """High-KL + conflicting direction → masked out."""
    B, S = 2, 1
    # ratio > 1 (log_diff > 0) AND adv > 0 AND high KL → should mask
    new_logp = torch.tensor([[1.0]])  # log_diff = 1 → ratio = e^1 > 1
    old_logp = torch.tensor([[0.0]])
    # Make means very different to get high KL
    new_means = torch.ones(B, S, 4, 8, 8) * 10.0
    old_means = torch.zeros(B, S, 4, 8, 8)
    sigma_t = torch.ones(1, S, 1, 1, 1)  # sigma=1 → KL = 50 >> threshold
    adv = torch.tensor([[1.0]])  # positive advantage

    # Expand to batch
    new_logp = new_logp.expand(B, S)
    old_logp = old_logp.expand(B, S)
    adv = adv.expand(B, S)

    loss, metrics = _dppo_kl_adv_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        new_means=new_means,
        old_means=old_means,
        advantages=adv,
        sigma_t=sigma_t,
        kl_mask_threshold=1e-5,
    )
    # Should be fully masked (ratio > 1 AND adv > 0 AND KL high)
    assert float(metrics["masked_fraction"].item()) == 1.0
    # Loss should be zero (all masked out)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_dppo_kl_adv_loss_non_conflicting_direction_passes() -> None:
    """High-KL but non-conflicting direction → NOT masked."""
    B, S = 2, 1
    # ratio < 1 (log_diff < 0) AND adv > 0 → non-conflicting
    new_logp = torch.tensor([[-1.0]])  # ratio = e^-1 < 1
    old_logp = torch.tensor([[0.0]])
    # High KL
    new_means = torch.ones(B, S, 4, 8, 8) * 10.0
    old_means = torch.zeros(B, S, 4, 8, 8)
    sigma_t = torch.ones(1, S, 1, 1, 1)
    adv = torch.tensor([[1.0]])  # positive advantage

    new_logp = new_logp.expand(B, S)
    old_logp = old_logp.expand(B, S)
    adv = adv.expand(B, S)

    loss, metrics = _dppo_kl_adv_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        new_means=new_means,
        old_means=old_means,
        advantages=adv,
        sigma_t=sigma_t,
        kl_mask_threshold=1e-5,
    )
    # High KL + ratio < 1 + adv > 0 → not conflicting → keep
    assert float(metrics["unmasked_fraction"].item()) == 1.0


# ---------------------------------------------------------------------------
# DiffusionDPPO.prepare_segment
# ---------------------------------------------------------------------------


def test_dppo_prepare_segment_populates_means_and_logp() -> None:
    """prepare_segment fills both sde_logp and sde_means when starting empty."""
    stage = _FakeDPPOStage(init_value=0.5)
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), kl_mask_threshold=1e-5, conditions_cls=None)
    seg = _make_latent_segment_for_dppo(batch_size=2, num_steps=3)
    conds = _conditions_with_text(batch_size=2)

    assert seg.sde_logp is None
    assert seg.sde_means is None

    alg.prepare_segment(conditions=conds, segment=seg)

    assert seg.sde_logp is not None
    assert seg.sde_means is not None
    assert seg.sde_logp.shape == (2, 3)
    assert seg.sde_means.shape == (2, 3, 4, 8, 8)


def test_dppo_prepare_segment_preserves_existing_logp() -> None:
    """When sde_logp is already set (native mode), prepare_segment does NOT overwrite it."""
    stage = _FakeDPPOStage(init_value=0.5)
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), kl_mask_threshold=1e-5, conditions_cls=None)
    seg = _make_latent_segment_with_logp(batch_size=2, num_steps=3)
    conds = _conditions_with_text(batch_size=2)

    original_logp = seg.sde_logp.clone()
    alg.prepare_segment(conditions=conds, segment=seg)

    # sde_logp should NOT be overwritten
    assert torch.allclose(seg.sde_logp, original_logp)
    # But sde_means SHOULD be populated
    assert seg.sde_means is not None
    assert seg.sde_means.shape == (2, 3, 4, 8, 8)


def test_dppo_prepare_segment_no_sde_indices_is_noop() -> None:
    """A segment without sde_indices triggers no replay."""
    stage = _FakeDPPOStage()
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), conditions_cls=None)
    seg = LatentSegment(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        latents=torch.zeros(2, 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, 2),
        indices=torch.arange(2, dtype=torch.long),
        sde_logp=None,
        sde_means=None,
        sde_indices=None,
    )
    alg.prepare_segment(conditions=_conditions_with_text(batch_size=2), segment=seg)
    assert seg.sde_logp is None
    assert seg.sde_means is None


# ---------------------------------------------------------------------------
# DiffusionDPPO.compute_loss_and_backward
# ---------------------------------------------------------------------------


def test_dppo_compute_loss_backward_fires() -> None:
    """compute_loss_and_backward produces loss, fires backward, and logs metrics."""
    stage = _FakeDPPOStage(init_value=0.5)
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), kl_mask_threshold=1e-5, conditions_cls=None)
    seg = _make_latent_segment_for_dppo(batch_size=2, num_steps=3)
    conds = _conditions_with_text(batch_size=2)
    advantages = torch.tensor([0.5, -0.3])

    # Prepare segment first (populates sde_logp + sde_means)
    alg.prepare_segment(conditions=conds, segment=seg)

    # Now compute loss
    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )

    assert isinstance(result, AlgorithmStepResult)
    assert result.has_backward
    assert result.num_steps_or_tokens == 3
    assert torch.isfinite(torch.tensor(result.loss)).item()
    # Param grad fired
    assert stage.param.grad is not None
    assert stage.param.grad.abs().item() > 0.0
    # Expected metrics keys
    expected_keys = {
        "policy_loss",
        "kl_mask_threshold",
        "ratio_mean",
        "ratio_std",
        "ratio_min",
        "ratio_max",
        "approx_kl",
        "kl_new_old_mean",
        "kl_new_old_max",
        "kl_mask_fraction",
        "pos_rm_fraction",
        "neg_rm_fraction",
        "masked_fraction",
        "unmasked_fraction",
    }
    assert expected_keys.issubset(result.metrics.keys())


def test_dppo_no_sde_indices_returns_no_backward() -> None:
    """A segment without sde_indices yields a no-op step."""
    stage = _FakeDPPOStage()
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), conditions_cls=None)
    seg = LatentSegment(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        latents=torch.zeros(2, 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, 2),
        indices=torch.arange(2, dtype=torch.long),
        sde_logp=None,
        sde_means=None,
        sde_indices=None,
    )
    result = alg.compute_loss_and_backward(
        conditions=_conditions_with_text(batch_size=2),
        segment=seg,
        advantages=torch.tensor([0.0, 0.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert not result.has_backward
    assert result.num_steps_or_tokens == 0
    assert stage.param.grad is None


def test_dppo_on_policy_ratio_near_one() -> None:
    """On first step (same weights), ratio should be ~1 and KL ~0."""
    stage = _FakeDPPOStage(init_value=0.5)
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), kl_mask_threshold=1e-5, conditions_cls=None)
    seg = _make_latent_segment_for_dppo(batch_size=2, num_steps=3)
    conds = _conditions_with_text(batch_size=2)

    # prepare_segment uses same weights as compute_loss_and_backward
    # (no optimizer step in between) → ratio should be exactly 1
    alg.prepare_segment(conditions=conds, segment=seg)
    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=torch.tensor([1.0, -1.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )

    # On-policy: ratio = exp(new_logp - old_logp) = 1 since weights are same
    assert abs(result.metrics["ratio_mean"] - 1.0) < 1e-4
    # KL should be ~0 since means are same (within float precision)
    assert result.metrics["kl_new_old_mean"] < 1e-6
    # No masking (KL < threshold)
    assert result.metrics["unmasked_fraction"] == 1.0


def test_dppo_sigma_t_computation_add_kl_coefficient_true() -> None:
    """Verify sigma_t computation with add_kl_coefficient=True."""
    stage = _FakeDPPOStage()
    alg = DiffusionDPPO(
        stage=stage,
        params=_FakeParams(eta=0.7),
        kl_mask_threshold=1e-5,
        add_kl_coefficient=True,
        conditions_cls=None,
    )
    seg = _make_latent_segment_for_dppo(batch_size=2, num_steps=3)
    target_steps = [0, 1, 2]
    sigma_t = alg._compute_sigma_t(seg, target_steps, device=torch.device("cpu"))

    # Shape should be [1, 3, 1, 1, 1]
    assert sigma_t.shape == (1, 3, 1, 1, 1)
    # All values should be positive
    assert (sigma_t > 0).all()


def test_dppo_sigma_t_computation_add_kl_coefficient_false() -> None:
    """Verify sigma_t = 1 when add_kl_coefficient=False."""
    stage = _FakeDPPOStage()
    alg = DiffusionDPPO(
        stage=stage,
        params=_FakeParams(eta=0.7),
        kl_mask_threshold=1e-5,
        add_kl_coefficient=False,
        conditions_cls=None,
    )
    seg = _make_latent_segment_for_dppo(batch_size=2, num_steps=3)
    target_steps = [0, 1, 2]
    sigma_t = alg._compute_sigma_t(seg, target_steps, device=torch.device("cpu"))

    # Should be all ones when add_kl_coefficient=False
    assert sigma_t.shape == (1, 3, 1, 1, 1)
    assert torch.allclose(sigma_t, torch.ones_like(sigma_t))


def test_dppo_missing_sde_means_raises() -> None:
    """compute_loss_and_backward without prepare_segment raises ValueError."""
    stage = _FakeDPPOStage()
    alg = DiffusionDPPO(stage=stage, params=_FakeParams(), conditions_cls=None)
    seg = _make_latent_segment_for_dppo(batch_size=2, num_steps=3)
    conds = _conditions_with_text(batch_size=2)

    # Manually set sde_logp but NOT sde_means to simulate partial prep
    seg.sde_logp = torch.full((2, 3), -1.0)

    with pytest.raises(ValueError, match="sde_means"):
        alg.compute_loss_and_backward(
            conditions=conds,
            segment=seg,
            advantages=torch.tensor([0.5, -0.3]),
            training_progress=0.0,
            loss_scale=1.0,
        )
