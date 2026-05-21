"""CPU tests for ``encoder_hidden_states_image`` plumbing in WAN21/22 diffusion steps."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from diffusionrl.models.wan21.conditions import WAN21Conditions
from diffusionrl.models.wan21.diffusion import WAN21DiffusionStep
from diffusionrl.models.wan22.diffusion import WAN22DiffusionStep
from diffusionrl.types.conditions import ImageEmbedCondition, TextEmbedCondition


class _CaptureTransformer:
    """Records the last call's kwargs and returns 16-channel zeros."""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        hidden = kwargs["hidden_states"]
        out = torch.zeros(
            hidden.shape[0],
            16,
            hidden.shape[2],
            hidden.shape[3],
            hidden.shape[4],
            dtype=hidden.dtype,
        )
        return (out,)


def _make_wan21_bundle() -> SimpleNamespace:
    return SimpleNamespace(transformer=_CaptureTransformer())


def _make_wan22_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        transformer=_CaptureTransformer(),
        boundary_ratio=0.5,
        guidance_scale_2=None,
    )


def _make_text_cond(batch_size: int) -> TextEmbedCondition:
    return TextEmbedCondition(embeds=torch.zeros(batch_size, 8, 32, dtype=torch.float32))


def _make_sample(batch_size: int = 1, t_lat: int = 2, h: int = 4, w: int = 4) -> torch.Tensor:
    return torch.zeros(batch_size, 16, t_lat, h, w, dtype=torch.float32)


def test_t2v_omits_encoder_hidden_states_image_kwarg() -> None:
    bundle = _make_wan21_bundle()
    step = WAN21DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.5)
    conds = WAN21Conditions(text=_make_text_cond(2))
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=1.0)
    # T2V must not leak the I2V-only kwarg to a T2V transformer.
    assert "encoder_hidden_states_image" not in bundle.transformer.last_kwargs


def test_wan21_predict_noise_clip_kwarg_single_branch() -> None:
    bundle = _make_wan21_bundle()
    step = WAN21DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.5)
    image_embeds = torch.full((2, 257, 1024), 0.7)
    conds = WAN21Conditions(
        text=_make_text_cond(2),
        image_embed=ImageEmbedCondition(embeds=image_embeds),
    )
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=1.0)
    kwargs = bundle.transformer.last_kwargs
    assert "encoder_hidden_states_image" in kwargs
    assert kwargs["encoder_hidden_states_image"].shape == (2, 257, 1024)


def test_wan21_predict_noise_clip_kwarg_cfg_doubles_batch() -> None:
    bundle = _make_wan21_bundle()
    step = WAN21DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.5)
    image_embeds = torch.full((2, 257, 1024), 0.7)
    conds = WAN21Conditions(
        text=_make_text_cond(2),
        negative_text=_make_text_cond(2),
        image_embed=ImageEmbedCondition(embeds=image_embeds),
    )
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=4.0)
    kwargs = bundle.transformer.last_kwargs
    assert kwargs["encoder_hidden_states_image"].shape == (4, 257, 1024)
    # Both halves of the batch must be identical (cond/uncond image branch).
    img_uncond, img_cond = kwargs["encoder_hidden_states_image"].chunk(2, dim=0)
    assert torch.equal(img_uncond, img_cond)


def test_wan22_predict_noise_clip_kwarg_routes_through_dual_transformer() -> None:
    bundle = _make_wan22_bundle()
    step = WAN22DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.7)  # >= boundary_ratio=0.5 → high noise
    image_embeds = torch.full((2, 257, 1024), 0.3)
    conds = WAN21Conditions(
        text=_make_text_cond(2),
        image_embed=ImageEmbedCondition(embeds=image_embeds),
    )
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=1.0)
    kwargs = bundle.transformer.last_kwargs
    assert "encoder_hidden_states_image" in kwargs
    assert kwargs["encoder_hidden_states_image"].shape == (2, 257, 1024)
    # WanDualTransformer routes via ``use_high_noise``; the bundle's
    # composite transformer still receives the kwarg and is responsible
    # for forwarding it to both sub-transformers.
    assert kwargs.get("use_high_noise") is True


def test_wan22_t2v_omits_encoder_hidden_states_image_kwarg() -> None:
    bundle = _make_wan22_bundle()
    step = WAN22DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.7)
    conds = WAN21Conditions(text=_make_text_cond(2))
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=1.0)
    assert "encoder_hidden_states_image" not in bundle.transformer.last_kwargs
