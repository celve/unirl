"""CPU tests for the WAN 2.1 / 2.2 I2V image-condition path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from diffusionrl.models_new.wan21.conditions import WAN21Conditions
from diffusionrl.models_new.wan21.image_encode import WAN21ImageLatentEncodeStage
from diffusionrl.types.conditions import ImageLatentCondition, TextEmbedCondition
from diffusionrl.types.primitives import Images

# ---------------------------------------------------------------------------
# WAN21Conditions: image_latent slot round-trip + bad-type rejection
# ---------------------------------------------------------------------------


def test_conditions_roundtrip_with_image_latent() -> None:
    text = TextEmbedCondition(embeds=torch.zeros(2, 4, 8))
    img = ImageLatentCondition(latents=torch.zeros(2, 20, 1, 3, 3))
    conds = WAN21Conditions(text=text, image_latent=img)
    d = conds.to_dict()
    assert set(d.keys()) == {"text", "image_latent"}
    restored = WAN21Conditions.from_dict(d)
    assert restored.text is text
    assert restored.image_latent is img


def test_conditions_to_dict_omits_image_latent_when_none() -> None:
    text = TextEmbedCondition(embeds=torch.zeros(1, 2, 4))
    conds = WAN21Conditions(text=text)
    d = conds.to_dict()
    assert "image_latent" not in d


def test_conditions_from_dict_rejects_bad_image_latent_type() -> None:
    text = TextEmbedCondition(embeds=torch.zeros(1, 2, 4))
    with pytest.raises(TypeError):
        WAN21Conditions.from_dict({"text": text, "image_latent": "not-a-condition"})


# ---------------------------------------------------------------------------
# WAN21ImageLatentEncodeStage
# ---------------------------------------------------------------------------


class _FakeLatentDist:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent

    def sample(self) -> torch.Tensor:  # not used; here to make .sample() drift easy to detect
        raise AssertionError("stage must use .mode(), not .sample()")


class _FakeVAEEncoded:
    def __init__(self, latent: torch.Tensor) -> None:
        self.latent_dist = _FakeLatentDist(latent)


def _make_bundle(
    *,
    raw_latent_value: float = 0.5,
    z_dim: int = 16,
    latents_mean=None,
    latents_std=None,
    scaling_factor: float = 1.0,
    latent_shape=(1, 16, 1, 4, 4),
) -> SimpleNamespace:
    raw = torch.full(latent_shape, raw_latent_value)
    vae_dtype = torch.float32

    class _FakeVAE:
        config = SimpleNamespace(
            z_dim=z_dim,
            latents_mean=latents_mean,
            latents_std=latents_std,
            scaling_factor=scaling_factor,
        )
        dtype = vae_dtype

        def encode(self, x: torch.Tensor) -> _FakeVAEEncoded:  # noqa: ARG002
            return _FakeVAEEncoded(raw)

    return SimpleNamespace(vae=_FakeVAE(), device=torch.device("cpu"), dtype=torch.float32)


def test_encode_output_shape_and_mask_layout() -> None:
    # num_frames=5 → T_lat=2 (= (5-1)//4+1); height/width=32 → latent_h/w=4.
    bundle = _make_bundle(latent_shape=(2, 16, 2, 4, 4))
    stage = WAN21ImageLatentEncodeStage(bundle, num_frames=5, height=32, width=32)
    images = Images(pixels=torch.rand(2, 3, 16, 16))
    cond = stage.encode(images)
    assert cond.latents is not None
    assert cond.latents.shape == (2, 20, 2, 4, 4)

    # Mask channels are the first 4. After the temporal repeat+transpose,
    # the first latent temporal position carries mask=1 (the "given" frame)
    # and the remaining latent positions carry 0.
    mask = cond.latents[:, :4]
    assert torch.allclose(mask[:, :, 0], torch.ones(2, 4, 4, 4))
    assert torch.allclose(mask[:, :, 1:], torch.zeros(2, 4, 1, 4, 4))


def test_encode_deterministic_across_calls() -> None:
    bundle = _make_bundle(latent_shape=(1, 16, 1, 2, 2))
    stage = WAN21ImageLatentEncodeStage(bundle, num_frames=1, height=16, width=16)
    images = Images(pixels=torch.rand(1, 3, 8, 8))
    a = stage.encode(images).latents
    b = stage.encode(images).latents
    assert a is not None and b is not None
    assert torch.equal(a, b)


def test_encode_per_channel_norm_inverse_of_decode() -> None:
    # raw_latent=0.5, mean=0.1, std=2.0 → (0.5-0.1)/2.0 = 0.2.
    bundle = _make_bundle(
        raw_latent_value=0.5,
        z_dim=16,
        latents_mean=[0.1] * 16,
        latents_std=[2.0] * 16,
        latent_shape=(1, 16, 1, 2, 2),
    )
    stage = WAN21ImageLatentEncodeStage(bundle, num_frames=1, height=16, width=16)
    images = Images(pixels=torch.zeros(1, 3, 8, 8))
    payload = stage.encode(images).latents
    assert payload is not None
    image_part = payload[:, 4:]  # last 16 channels = normalized latent
    assert torch.allclose(image_part, torch.full_like(image_part, 0.2), atol=1e-6)


def test_encode_rejects_non_images() -> None:
    bundle = _make_bundle(latent_shape=(1, 16, 1, 2, 2))
    stage = WAN21ImageLatentEncodeStage(bundle, num_frames=1, height=16, width=16)
    with pytest.raises(TypeError):
        stage.encode("not-an-images-instance")  # type: ignore[arg-type]
