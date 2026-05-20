"""CPU tests for ``WAN21DiffusionStep.predict_noise`` kwarg shape under I2V."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from diffusionrl.models_new.wan21.conditions import WAN21Conditions
from diffusionrl.models_new.wan21.diffusion import WAN21DiffusionStep
from diffusionrl.types.conditions import ImageLatentCondition, TextEmbedCondition


class _CaptureTransformer:
    """Records the last call's kwargs and returns a noise tensor matching B and 16 ch."""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        hidden = kwargs["hidden_states"]
        # Transformer outputs 16-channel noise regardless of in_channels.
        out = torch.zeros(
            hidden.shape[0],
            16,
            hidden.shape[2],
            hidden.shape[3],
            hidden.shape[4],
            dtype=hidden.dtype,
        )
        return (out,)


def _make_bundle() -> SimpleNamespace:
    return SimpleNamespace(transformer=_CaptureTransformer())


def _make_text_cond(batch_size: int) -> TextEmbedCondition:
    return TextEmbedCondition(embeds=torch.zeros(batch_size, 8, 32, dtype=torch.float32))


def _make_sample(batch_size: int = 1, t_lat: int = 2, h: int = 4, w: int = 4) -> torch.Tensor:
    return torch.zeros(batch_size, 16, t_lat, h, w, dtype=torch.float32)


def test_predict_noise_t2v_hidden_states_16_channels() -> None:
    bundle = _make_bundle()
    step = WAN21DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.5)
    conds = WAN21Conditions(text=_make_text_cond(2))
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=1.0)
    hidden = bundle.transformer.last_kwargs["hidden_states"]
    assert hidden.shape == (2, 16, 2, 4, 4)


def test_predict_noise_i2v_hidden_states_36_channels() -> None:
    bundle = _make_bundle()
    step = WAN21DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.5)
    image_latent = ImageLatentCondition(latents=torch.zeros(2, 20, 2, 4, 4, dtype=torch.float32))
    conds = WAN21Conditions(text=_make_text_cond(2), image_latent=image_latent)
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=1.0)
    hidden = bundle.transformer.last_kwargs["hidden_states"]
    assert hidden.shape == (2, 36, 2, 4, 4)


def test_predict_noise_i2v_cfg_doubles_batch() -> None:
    bundle = _make_bundle()
    step = WAN21DiffusionStep()
    sample = _make_sample(batch_size=2)
    sigma = torch.tensor(0.5)
    image_latent = ImageLatentCondition(latents=torch.zeros(2, 20, 2, 4, 4, dtype=torch.float32))
    conds = WAN21Conditions(
        text=_make_text_cond(2),
        negative_text=_make_text_cond(2),
        image_latent=image_latent,
    )
    step.predict_noise(bundle, sample, sigma, conds, guidance_scale=4.0)
    hidden = bundle.transformer.last_kwargs["hidden_states"]
    assert hidden.shape == (4, 36, 2, 4, 4)
