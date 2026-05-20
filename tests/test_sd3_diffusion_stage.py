"""Tests for ``SD3DiffusionStage.diffuse`` and ``replay`` with a fake transformer.

Parity tests against legacy SD3 (``SD3Sampler``) need a real checkpoint and
GPU; those are run manually as a smoke step. These tests verify the loop
plumbing, segment shape contract, and replay API on CPU with a stub
transformer that returns deterministic zero-noise predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusionrl.models_new.sd3.bundle import SD3Bundle
from diffusionrl.models_new.sd3.conditions import SD3Conditions
from diffusionrl.models_new.sd3.diffusion import (
    SD3DiffusionParams,
    SD3DiffusionStage,
    SD3DiffusionStep,
)
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.conditions import TextEmbedCondition
from diffusionrl.types.segments import LatentSegment


class _FakeTransformer(nn.Module):
    """Returns zero noise prediction matching ``hidden_states`` shape."""

    def forward(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        timestep,
        pooled_projections,
        return_dict,
    ):
        return (torch.zeros_like(hidden_states),)


def _make_bundle(device: str = "cpu") -> SD3Bundle:
    return SD3Bundle(
        transformer=_FakeTransformer(),
        vae=None,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        scheduler=None,
        dtype=torch.float32,
        device=torch.device(device),
        pretrained_path="fake",
    )


def _make_conditions(b: int = 2, with_negative: bool = True) -> SD3Conditions:
    seq, hidden = 4, 8
    pos = TextEmbedCondition(
        embeds=torch.randn(b, seq, hidden),
        pooled=torch.randn(b, hidden * 2),
    )
    neg = (
        TextEmbedCondition(
            embeds=torch.randn(b, seq, hidden),
            pooled=torch.randn(b, hidden * 2),
        )
        if with_negative
        else None
    )
    return SD3Conditions(text=pos, negative_text=neg)


def _make_stage(latent_channels: int = 4) -> SD3DiffusionStage:
    return SD3DiffusionStage(
        model=_make_bundle(),
        step=SD3DiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="fp32",
        trajectory_precision="fp32",
        logprob_precision="fp32",
        latent_channels=latent_channels,
    )


def test_diffuse_full_sde_cfg_returns_latent_segment():
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=True)
    params = SD3DiffusionParams(
        num_inference_steps=T,
        guidance_scale=2.0,
        height=8,
        width=8,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    assert isinstance(seg, LatentSegment)
    # Latents [N=2, K=T+1, C=4, H=1, W=1] (8 / vae_scale_factor=8).
    assert seg.latents.shape == (2, T + 1, 4, 1, 1)
    # All positions stored when full SDE.
    assert seg.indices.tolist() == list(range(T + 1))
    # sde_logp [N=2, S=T] dense — no NaN sentinels.
    assert seg.sde_logp is not None
    assert seg.sde_logp.shape == (2, T)
    assert torch.isfinite(seg.sde_logp).all()
    assert seg.sde_indices.tolist() == list(range(T))


def test_diffuse_no_cfg_path():
    stage = _make_stage(latent_channels=4)
    T = 2
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    assert seg.latents.shape == (1, T + 1, 4, 1, 1)
    assert seg.sde_logp.shape == (1, T)


def test_diffuse_partial_sde_always_stores_clean_position():
    """Position T (the clean latent) must always be in the trajectory so the
    VAE decode stage can read ``s.latents[:, -1]`` regardless of which
    timesteps were SDE."""
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        seed=0,
        sde_indices=[0, 1],  # only the first two steps are SDE
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    # SDE pairs for {0, 1} -> needed positions {0, 1, 2}; plus T=4 for decode.
    assert seg.indices.tolist() == [0, 1, 2, T]
    assert int(seg.indices[-1].item()) == T
    # sde_logp tracks only the SDE-indexed transitions.
    assert seg.sde_logp.shape == (1, 2)
    assert seg.sde_indices.tolist() == [0, 1]


def test_replay_returns_logp_aligned_with_segment_sde_logp():
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=True)
    params = SD3DiffusionParams(
        num_inference_steps=T,
        guidance_scale=2.0,
        height=8,
        width=8,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )
    seg = stage.diffuse(cond, schedule=schedule, params=params)

    result = stage.replay(cond, segment=seg, params=params)
    log_prob = result.log_probs

    assert log_prob.shape == seg.sde_logp.shape  # (B=2, S=T=4)
    assert log_prob.dtype == torch.float32
    assert torch.isfinite(log_prob).all()
    # Numerical equivalence: with the same FakeTransformer (deterministic
    # zero-noise predictions), rollout-time and replay log-probs should match.
    assert torch.allclose(log_prob, seg.sde_logp, atol=1e-5, rtol=1e-4)
    assert result.prev_sample_means is not None
    assert result.prev_sample_means.shape[:2] == log_prob.shape


def test_replay_step_indices_subset():
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )
    seg = stage.diffuse(cond, schedule=schedule, params=params)

    result = stage.replay(cond, segment=seg, params=params, step_indices=[1, 2])
    log_prob = result.log_probs

    assert log_prob.shape == (1, 2)
    assert torch.isfinite(log_prob).all()
    assert result.prev_sample_means is not None
    assert result.prev_sample_means.shape[:2] == log_prob.shape


# ---------------------------------------------------------------------------
# initial_latents wiring — driver-shipped x_T overrides internal generate_latents
# (test the Stage-level contract; the Pipeline-level extraction from
# RolloutReq.request_conditions["initial_latents"] is covered separately).
# ---------------------------------------------------------------------------


def test_diffuse_initial_latents_used_verbatim_as_x_T():
    """Driver-shipped initial_latents lands at segment.latents[:, 0, ...]
    unchanged (after device/dtype cast), proving the internal RNG path
    is bypassed when the request pre-ships noise."""
    stage = _make_stage(latent_channels=4)
    T = 2
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        seed=12345,
        sde_indices=list(range(T)),
        eta=0.5,
    )
    # Deterministic non-Gaussian tensor that internal RNG would never produce.
    fixed_x_T = torch.full((2, 4, 1, 1), 0.42)

    seg = stage.diffuse(cond, schedule=schedule, params=params, initial_latents=fixed_x_T)

    assert seg.latents.shape == (2, T + 1, 4, 1, 1)
    assert torch.allclose(seg.latents[:, 0].float(), fixed_x_T.float())


def test_diffuse_initial_latents_batch_mismatch_raises():
    stage = _make_stage(latent_channels=4)
    T = 2
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T, guidance_scale=1.0, height=8, width=8, seed=0, sde_indices=[0, 1], eta=0.0
    )
    wrong_batch = torch.randn(3, 4, 1, 1)  # batch=3 but cond is b=2
    import pytest

    with pytest.raises(ValueError, match=r"initial_latents.shape\[0\]=3 != batch_size=2"):
        stage.diffuse(cond, schedule=schedule, params=params, initial_latents=wrong_batch)


def test_diffuse_initial_latents_shape_mismatch_raises():
    stage = _make_stage(latent_channels=4)
    T = 2
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T, guidance_scale=1.0, height=8, width=8, seed=0, sde_indices=[0, 1], eta=0.0
    )
    wrong_tail = torch.randn(2, 4, 2, 2)  # spatial 2x2 instead of 1x1 for height=8/vae=8
    import pytest

    with pytest.raises(ValueError, match=r"initial_latents.shape\[1:\]="):
        stage.diffuse(cond, schedule=schedule, params=params, initial_latents=wrong_tail)


def test_diffuse_initial_latents_none_uses_internal_rng():
    """Backwards-compat: when initial_latents is None, internal generate_latents
    fires — same behavior as pre-Batch-5. We can't assert determinism (the
    default ``noise_group_ids=None`` path uses plain ``torch.randn``, not
    seed-keyed), so we just verify the internal path produced finite noise
    of the right shape AND that it is NOT the all-0.42 sentinel the
    pre-shipped test uses (distinguishing the two branches)."""
    stage = _make_stage(latent_channels=4)
    T = 2
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=False)
    params = SD3DiffusionParams(
        num_inference_steps=T, guidance_scale=1.0, height=8, width=8, seed=7, sde_indices=[0, 1], eta=0.0
    )
    seg = stage.diffuse(cond, schedule=schedule, params=params, initial_latents=None)
    assert seg.latents[:, 0].shape == (2, 4, 1, 1)
    assert torch.isfinite(seg.latents[:, 0]).all()
    # Internal RNG must NOT match the sentinel value used in the pre-shipped
    # test (0.42 everywhere) — distinguishes "internal path fired" from
    # "we accidentally reused stale data".
    assert not torch.allclose(seg.latents[:, 0].float(), torch.full((2, 4, 1, 1), 0.42))
