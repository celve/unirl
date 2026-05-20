"""Tests for ``HunyuanVideo15DiffusionStage.diffuse`` and ``replay`` with a fake transformer.

Verifies the dual-stream CFG plumbing, the channel-dim latent packing
(``[B, 2C+1, T, H, W]``-shaped transformer input → ``[B, C, T, H, W]``
noise prediction), 5D video latent shape contract, and replay log-prob
parity against rollout — all on CPU with a stub transformer that emits
a deterministic non-zero noise prediction.

The fake transformer accepts the HunyuanVideo-1.5 kwarg signature
(``hidden_states``, ``timestep``, ``encoder_hidden_states``,
``encoder_attention_mask``, ``encoder_hidden_states_2``,
``encoder_attention_mask_2``, ``image_embeds``, ``return_dict``) and
returns the **first ``latent_channels`` channels** of ``hidden_states``
scaled by 1e-3 — that yields a stable, non-zero, deterministic
prediction whose shape matches the latent stream (which is what the
real transformer also returns).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from diffusionrl.models_new.hunyuan_video15.bundle import HunyuanVideo15Bundle
from diffusionrl.models_new.hunyuan_video15.conditions import HunyuanVideo15Conditions
from diffusionrl.models_new.hunyuan_video15.diffusion import (
    HunyuanVideo15DiffusionParams,
    HunyuanVideo15DiffusionStage,
    HunyuanVideo15DiffusionStep,
)
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.conditions import TextEmbedCondition
from diffusionrl.types.segments import LatentSegment

# Fixed test geometry so the fake can carve the latent stream cleanly.
_LATENT_CHANNELS = 4
_VISION_TOKENS = 6
_VISION_DIM = 8


class _FakeTransformer(nn.Module):
    """Returns the first ``latent_channels`` channels of ``hidden_states``
    scaled by 1e-3 — matches the real transformer's output shape contract
    (only the latent stream, not the packed cond_latents / cond_mask)."""

    def __init__(self, latent_channels: int = _LATENT_CHANNELS) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        # Stage reads ``out_channels`` from the transformer config when no
        # explicit ``latent_channels`` is passed; we DO pass it explicitly
        # below, so the fallback path is unused. Still expose a config to
        # avoid surprising any future audit code.
        self.config = SimpleNamespace(out_channels=self.latent_channels, use_meanflow=False)

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
        return_dict,
    ):
        # Real transformer emits noise prediction for the LATENT stream only
        # (first ``latent_channels`` channels); cond_latents + cond_mask
        # channels are read-only conditioning input. Mirror that contract.
        return (0.001 * hidden_states[:, : self.latent_channels, ...],)


def _make_bundle(device: str = "cpu") -> HunyuanVideo15Bundle:
    return HunyuanVideo15Bundle(
        transformer=_FakeTransformer(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        vision_encoder=None,
        image_processor=None,
        scheduler=None,
        dtype=torch.float32,
        device=torch.device(device),
        pretrained_path="fake",
    )


def _make_text(b: int, seq: int, hidden: int) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=torch.randn(b, seq, hidden),
        attn_mask=torch.ones(b, seq, dtype=torch.long),
        pooled=None,
    )


def _make_conditions(b: int, with_negative: bool) -> HunyuanVideo15Conditions:
    text_mllm = _make_text(b, seq=8, hidden=16)
    text_glyph = _make_text(b, seq=4, hidden=32)
    if with_negative:
        neg_mllm = _make_text(b, seq=8, hidden=16)
        neg_glyph = _make_text(b, seq=4, hidden=32)
    else:
        neg_mllm = None
        neg_glyph = None
    return HunyuanVideo15Conditions(
        text_mllm=text_mllm,
        text_glyph=text_glyph,
        negative_text_mllm=neg_mllm,
        negative_text_glyph=neg_glyph,
    )


def _make_stage() -> HunyuanVideo15DiffusionStage:
    return HunyuanVideo15DiffusionStage(
        model=_make_bundle(),
        step=HunyuanVideo15DiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="fp32",
        trajectory_precision="fp32",
        logprob_precision="fp32",
        # Small geometry: spatial=4, temporal=2 → manageable test latent shape.
        spatial_compression_ratio=4,
        temporal_compression_ratio=2,
        latent_channels=_LATENT_CHANNELS,
        vision_num_semantic_tokens=_VISION_TOKENS,
        vision_states_dim=_VISION_DIM,
    )


# --------------------------------------------------------------------------
# Diffuse — segment shape contract (6D video latents)
# --------------------------------------------------------------------------


def test_diffuse_full_sde_cfg_returns_video_latent_segment():
    """Full-SDE run with CFG-on: 6D segment ``[N, K, C, T_lat, H_lat, W_lat]``
    and modality=VIDEO."""
    stage = _make_stage()
    T = 3
    schedule = get_sigma_schedule(T, shift=5.0)
    cond = _make_conditions(b=2, with_negative=True)
    # height=8, spatial=4 → latent_h=2; num_frames=5, temporal=2 → latent_t=3
    params = HunyuanVideo15DiffusionParams(
        num_inference_steps=T,
        guidance_scale=2.0,
        height=8,
        width=8,
        num_frames=5,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    assert isinstance(seg, LatentSegment)
    # [N=2, K=T+1=4, C=4, T_lat=3, H_lat=2, W_lat=2]
    assert seg.latents.shape == (2, T + 1, _LATENT_CHANNELS, 3, 2, 2)
    # All trajectory positions stored when full SDE.
    assert seg.indices.tolist() == list(range(T + 1))
    assert seg.sde_logp is not None
    assert seg.sde_logp.shape == (2, T)
    assert torch.isfinite(seg.sde_logp).all()
    assert seg.sde_indices.tolist() == list(range(T))
    # Stamped as VIDEO modality via make_video_segment.
    from diffusionrl.types.conditions.base import Modality

    assert seg.modality == Modality.VIDEO


def test_diffuse_no_cfg_path():
    """guidance_scale=1.0 skips the dual-stream CFG branch; shape unchanged."""
    stage = _make_stage()
    T = 2
    schedule = get_sigma_schedule(T, shift=5.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = HunyuanVideo15DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        num_frames=3,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    # [B=1, K=T+1=3, C=4, T_lat=2, H_lat=2, W_lat=2]
    assert seg.latents.shape == (1, T + 1, _LATENT_CHANNELS, 2, 2, 2)
    assert seg.sde_logp.shape == (1, T)


def test_diffuse_partial_sde_always_stores_clean_position():
    """Position T (the clean latent) must always be in the trajectory so
    the VAE decode stage can read ``s.latents[:, -1]`` regardless of which
    timesteps were SDE."""
    stage = _make_stage()
    T = 4
    schedule = get_sigma_schedule(T, shift=5.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = HunyuanVideo15DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        num_frames=3,
        seed=0,
        sde_indices=[0, 1],
        eta=0.5,
    )

    seg = stage.diffuse(cond, schedule=schedule, params=params)

    assert seg.indices.tolist() == [0, 1, 2, T]
    assert int(seg.indices[-1].item()) == T
    assert seg.sde_logp.shape == (1, 2)
    assert seg.sde_indices.tolist() == [0, 1]


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def test_replay_returns_logp_aligned_with_segment_sde_logp():
    """Deterministic transformer + same conditions → rollout and replay
    log-probs match exactly. Validates that the channel-dim packing inside
    ``predict_noise`` does not introduce drift."""
    stage = _make_stage()
    T = 3
    schedule = get_sigma_schedule(T, shift=5.0)
    cond = _make_conditions(b=2, with_negative=True)
    params = HunyuanVideo15DiffusionParams(
        num_inference_steps=T,
        guidance_scale=2.0,
        height=8,
        width=8,
        num_frames=5,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )
    seg = stage.diffuse(cond, schedule=schedule, params=params)

    result = stage.replay(cond, segment=seg, params=params)
    log_prob = result.log_probs

    assert log_prob.shape == seg.sde_logp.shape  # (B=2, S=T=3)
    assert log_prob.dtype == torch.float32
    assert torch.isfinite(log_prob).all()
    assert torch.allclose(log_prob, seg.sde_logp, atol=1e-5, rtol=1e-4)
    assert result.prev_sample_means is not None
    assert result.prev_sample_means.shape[:2] == log_prob.shape


def test_replay_step_indices_subset():
    stage = _make_stage()
    T = 3
    schedule = get_sigma_schedule(T, shift=5.0)
    cond = _make_conditions(b=1, with_negative=False)
    params = HunyuanVideo15DiffusionParams(
        num_inference_steps=T,
        guidance_scale=1.0,
        height=8,
        width=8,
        num_frames=3,
        seed=0,
        sde_indices=list(range(T)),
        eta=0.5,
    )
    seg = stage.diffuse(cond, schedule=schedule, params=params)

    result = stage.replay(cond, segment=seg, params=params, step_indices=[1])
    log_prob = result.log_probs

    assert log_prob.shape == (1, 1)
    assert torch.isfinite(log_prob).all()


def test_predict_noise_packs_channel_dim_correctly():
    """The transformer must see hidden_states of shape [B, 2C+1, T, H, W]
    (latents + cond_latents + cond_mask channel concat)."""
    bundle = _make_bundle()
    step = HunyuanVideo15DiffusionStep()

    cond = _make_conditions(b=1, with_negative=False)
    sample = torch.randn(1, _LATENT_CHANNELS, 2, 4, 4)
    sigma = torch.tensor(0.5)

    # Spy on the fake's forward to capture the hidden_states shape.
    seen_shapes = []
    real_forward = bundle.transformer.forward

    def spy(*, hidden_states, **kw):
        seen_shapes.append(tuple(hidden_states.shape))
        return real_forward(hidden_states=hidden_states, **kw)

    bundle.transformer.forward = spy
    noise_pred = step.predict_noise(
        bundle,
        sample,
        sigma,
        cond,
        guidance_scale=1.0,
        vision_num_semantic_tokens=_VISION_TOKENS,
        vision_states_dim=_VISION_DIM,
    )

    # Hidden states packed: [B, 2*C+1, T, H, W] = [1, 9, 2, 4, 4]
    assert seen_shapes == [(1, 2 * _LATENT_CHANNELS + 1, 2, 4, 4)]
    # Output unpacked: [B, C, T, H, W]
    assert noise_pred.shape == (1, _LATENT_CHANNELS, 2, 4, 4)
