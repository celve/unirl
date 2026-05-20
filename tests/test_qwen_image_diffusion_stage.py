"""Tests for ``QwenImageDiffusionStage.diffuse`` and ``replay`` with a fake transformer.

Parity tests against legacy Qwen-Image (``QwenImageSampler``, PR #104)
need a real checkpoint and GPU; those are run manually as a smoke step.
These tests verify the loop plumbing, segment shape contract, pack/
unpack round-trip math, and replay API on CPU with a stub transformer
that returns deterministic zero-noise predictions.

The fake transformer accepts the Qwen-Image kwarg signature
(``hidden_states``, ``timestep``, ``guidance``,
``encoder_hidden_states_mask``, ``encoder_hidden_states``, ``img_shapes``,
``return_dict``) and returns zeros matching ``hidden_states`` shape.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from diffusionrl.models_new.qwen_image.bundle import QwenImageBundle
from diffusionrl.models_new.qwen_image.conditions import QwenImageConditions
from diffusionrl.models_new.qwen_image.diffusion import (
    QwenImageDiffusionParams,
    QwenImageDiffusionStage,
    QwenImageDiffusionStep,
    _pack_latents,
    _unpack_latents,
)
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.conditions import TextEmbedCondition
from diffusionrl.types.segments import LatentSegment


class _FakeTransformer(nn.Module):
    """Returns zero noise prediction matching ``hidden_states`` shape.

    Carries a ``.config`` namespace with the two attributes
    ``QwenImageDiffusionStage`` reads off the transformer:
    ``in_channels`` (used to derive ``latent_channels = in_channels // 4``)
    and ``guidance_embeds`` (set to False so the stage's guidance scalar
    path is skipped — the fake forward doesn't model it).
    """

    def __init__(self, *, in_channels: int = 16, guidance_embeds: bool = False):
        super().__init__()
        self.config = SimpleNamespace(in_channels=in_channels, guidance_embeds=guidance_embeds)

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        guidance,
        encoder_hidden_states_mask,
        encoder_hidden_states,
        img_shapes,
        return_dict,
    ):
        # Non-zero deterministic output: Qwen-Image's combined-CFG with norm
        # correction (``comb * cond_norm / comb_norm``) divides 0/0 → NaN
        # when both branches return all-zero predictions; a tiny non-zero
        # function of ``hidden_states`` keeps the norm rescaling finite
        # while remaining deterministic (cond/uncond agree → CFG is a
        # no-op, so replay==rollout still holds bit-exactly).
        return (0.001 * hidden_states,)


def _make_bundle(device: str = "cpu") -> QwenImageBundle:
    return QwenImageBundle(
        transformer=_FakeTransformer(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
        dtype=torch.float32,
        device=torch.device(device),
        pretrained_path="fake",
    )


def _make_conditions(b: int = 2, with_negative: bool = True, seq: int = 4, hidden: int = 8) -> QwenImageConditions:
    pos = TextEmbedCondition(
        embeds=torch.randn(b, seq, hidden),
        attn_mask=torch.ones(b, seq, dtype=torch.long),
        pooled=None,
    )
    neg = (
        TextEmbedCondition(
            embeds=torch.randn(b, seq, hidden),
            attn_mask=torch.ones(b, seq, dtype=torch.long),
            pooled=None,
        )
        if with_negative
        else None
    )
    return QwenImageConditions(text=pos, negative_text=neg)


def _make_stage(latent_channels: int = 4) -> QwenImageDiffusionStage:
    return QwenImageDiffusionStage(
        model=_make_bundle(),
        step=QwenImageDiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="fp32",
        trajectory_precision="fp32",
        logprob_precision="fp32",
        latent_channels=latent_channels,
    )


# --------------------------------------------------------------------------
# Pack / unpack round-trip
# --------------------------------------------------------------------------


def test_pack_unpack_round_trip_exact():
    """``_pack_latents`` and ``_unpack_latents`` are inverses (reshape +
    permute only, no data loss)."""
    x = torch.randn(2, 16, 64, 96)
    packed = _pack_latents(x)
    assert packed.shape == (2, (64 // 2) * (96 // 2), 16 * 4)
    unpacked = _unpack_latents(packed, latent_h=64, latent_w=96)
    assert unpacked.shape == x.shape
    assert torch.equal(x, unpacked)


def test_pack_unpack_round_trip_small():
    """Smallest valid case (latent_h=2, latent_w=2 → packed seq=1)."""
    x = torch.randn(1, 4, 2, 2)
    packed = _pack_latents(x)
    assert packed.shape == (1, 1, 16)
    unpacked = _unpack_latents(packed, latent_h=2, latent_w=2)
    assert torch.equal(x, unpacked)


# --------------------------------------------------------------------------
# Diffuse — segment shape contract
# --------------------------------------------------------------------------


def test_diffuse_full_sde_cfg_returns_latent_segment():
    """Full-SDE run with CFG-on: trajectory is dense (T+1 positions),
    sde_logp is dense across all steps, segment latents are
    ``[N, T+1, C, latent_h, latent_w]`` (unpacked spatial shape)."""
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=True)
    # latent_h = 2 * (32 // (8 * 2)) = 4
    params = QwenImageDiffusionParams(
        num_inference_steps=T,
        guidance_scale=2.0,
        height=32,
        width=32,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    assert isinstance(seg, LatentSegment)
    # Latents [N=2, K=T+1, C=4, H=4, W=4] (latent_h = 2*(32//16) = 4).
    assert seg.latents.shape == (2, T + 1, 4, 4, 4)
    assert seg.indices.tolist() == list(range(T + 1))
    assert seg.sde_logp is not None
    assert seg.sde_logp.shape == (2, T)
    assert torch.isfinite(seg.sde_logp).all()
    assert seg.sde_indices.tolist() == list(range(T))


def test_diffuse_no_cfg_path():
    """guidance_scale=1.0 skips the negative-branch forward; segment
    shape is unchanged."""
    stage = _make_stage(latent_channels=4)
    T = 2
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = QwenImageDiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=32,
        width=32,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    assert seg.latents.shape == (1, T + 1, 4, 4, 4)
    assert seg.sde_logp.shape == (1, T)


def test_diffuse_partial_sde_always_stores_clean_position():
    """Position T (the clean latent) must always be in the trajectory so
    :class:`QwenImageVAEDecodeStage` can read ``s.latents[:, -1]``
    regardless of which timesteps were SDE."""
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = QwenImageDiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=32,
        width=32,
        seed=0,
        sde_indices=[0, 1],  # only the first two steps are SDE
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    # SDE pairs for {0, 1} -> needed positions {0, 1, 2}; plus T=4 for decode.
    assert seg.indices.tolist() == [0, 1, 2, T]
    assert int(seg.indices[-1].item()) == T
    assert seg.sde_logp.shape == (1, 2)
    assert seg.sde_indices.tolist() == [0, 1]


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def test_replay_returns_logp_aligned_with_segment_sde_logp():
    """With the same FakeTransformer (deterministic zero-noise
    predictions), rollout-time and replay log-probs match exactly. This
    validates that the pack/unpack round-trip inside ``predict_noise``
    does not introduce drift."""
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=2, with_negative=True)
    params = QwenImageDiffusionParams(
        num_inference_steps=T,
        guidance_scale=2.0,
        height=32,
        width=32,
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
    assert torch.allclose(log_prob, seg.sde_logp, atol=1e-5, rtol=1e-4)
    assert result.prev_sample_means is not None
    assert result.prev_sample_means.shape[:2] == log_prob.shape


def test_replay_step_indices_subset():
    stage = _make_stage(latent_channels=4)
    T = 4
    schedule = get_sigma_schedule(T, shift=3.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = QwenImageDiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=32,
        width=32,
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
