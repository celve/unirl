"""Unit tests for HunyuanVideo-1.5 model bundle.

These tests avoid loading real Qwen2.5-VL / ByT5 / SigLIP / VAE checkpoints by
constructing the bundle via ``object.__new__`` and patching the encoder /
transformer surface with light-weight fakes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch

from diffusionrl.models.hunyuan_veido1p5 import (
    HunyuanVeido1p5ModelBundle,
)
from diffusionrl.types.forward_context import (
    HunyuanVeido1p5ForwardContext,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTextEncoderWrapper:
    """Mimics :class:`HunyuanVeido1p5TextEncoderWrapper` for unit tests."""

    def __init__(self, mllm_dim: int = 8, byt5_dim: int = 4) -> None:
        self.mllm_dim = mllm_dim
        self.byt5_dim = byt5_dim
        self.calls: list[list[str]] = []

    def encode_prompt(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append(list(prompts))
        bsz = len(prompts)
        mllm_embeds = torch.zeros(bsz, 5, self.mllm_dim)
        mllm_mask = torch.ones(bsz, 5, dtype=torch.long)
        byt5_embeds = torch.zeros(bsz, 6, self.byt5_dim)
        byt5_mask = torch.zeros(bsz, 6, dtype=torch.long)
        # Tag the tensor with the prompt count so we can assert separation
        # between positive/negative encode calls.
        mllm_embeds[..., 0] = float(bsz)
        return mllm_embeds, mllm_mask, byt5_embeds, byt5_mask


def _make_bare_bundle() -> HunyuanVeido1p5ModelBundle:
    """Build a minimal ``HunyuanVeido1p5ModelBundle`` without running ``__init__``."""
    bundle = object.__new__(HunyuanVeido1p5ModelBundle)
    bundle.device = torch.device("cpu")
    bundle.dtype = torch.float32
    bundle.text_encoder_dtype = torch.float32
    bundle.vae_dtype = torch.float32
    bundle.training_only = False
    bundle.skip_device_move = False
    bundle.use_lora = False
    bundle.lora_rank = 16
    bundle.lora_alpha = 16
    bundle.lora_target_modules = None
    bundle.vision_num_semantic_tokens = 729
    bundle.vision_states_dim = 1152
    bundle.load_vision_encoder_flag = True
    bundle.mllm_max_length = 32
    bundle.byt5_max_length = 16
    bundle.mllm_skip_layers = 2
    bundle.mllm_crop_start = 4
    bundle.training_forward_autocast_dtype = None
    bundle._transformer = None
    bundle._vae = None
    bundle._text_encoder = None
    bundle._vision_encoder = None
    bundle._image_processor = None
    bundle._mllm_tokenizer = None
    bundle._mllm_encoder = None
    bundle._byt5_tokenizer = None
    bundle._byt5_encoder = None
    bundle._scheduler = None
    return bundle


# ---------------------------------------------------------------------------
# encode_prompt / encode_inputs
# ---------------------------------------------------------------------------


def test_encode_prompt_returns_4_tuple_with_correct_shapes() -> None:
    bundle = _make_bare_bundle()
    bundle._text_encoder = _FakeTextEncoderWrapper()

    embeds, mask, embeds_2, mask_2 = bundle.encode_prompt(["a", "b"])

    assert embeds.shape == (2, 5, 8)
    assert mask.shape == (2, 5)
    assert embeds_2.shape == (2, 6, 4)
    assert mask_2.shape == (2, 6)


def test_encode_prompt_to_input_dict_uses_named_fields_not_pooled_alias() -> None:
    bundle = _make_bare_bundle()
    bundle._text_encoder = _FakeTextEncoderWrapper()

    out = bundle._encode_prompt_to_input_dict(["alpha"])

    assert set(out.keys()) == {"prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2"}
    assert "pooled_prompt_embeds" not in out


def test_encode_inputs_t2v_emits_zero_image_conditioning() -> None:
    bundle = _make_bare_bundle()
    bundle._text_encoder = _FakeTextEncoderWrapper()

    encoded = bundle.encode_inputs(["a", "b"], height=64, width=128, num_frames=5)

    # Positive + negative streams (4 fields each) + 3 image-conditioning tensors.
    expected = {
        "prompt_embeds",
        "prompt_embeds_mask",
        "prompt_embeds_2",
        "prompt_embeds_mask_2",
        "negative_prompt_embeds",
        "negative_prompt_embeds_mask",
        "negative_prompt_embeds_2",
        "negative_prompt_embeds_mask_2",
        "image_embeds",
        "cond_latents",
        "cond_mask",
    }
    assert set(encoded.keys()) == expected
    # T2V: image / cond / mask are all zero placeholders.
    assert torch.all(encoded["image_embeds"] == 0)
    assert torch.all(encoded["cond_latents"] == 0)
    assert torch.all(encoded["cond_mask"] == 0)
    assert encoded["image_embeds"].shape == (2, 729, 1152)
    # Latent dims: 64/16=4, 128/16=8, (5-1)/4 + 1 = 2.
    assert encoded["cond_latents"].shape == (2, 32, 2, 4, 8)
    assert encoded["cond_mask"].shape == (2, 1, 2, 4, 8)


def test_encode_inputs_default_negative_prompt_is_empty_string() -> None:
    text_encoder = _FakeTextEncoderWrapper()
    bundle = _make_bare_bundle()
    bundle._text_encoder = text_encoder

    bundle.encode_inputs(["alpha", "beta"], height=64, width=128, num_frames=5)

    # First call: positive prompts; second call: empty-string negatives.
    assert text_encoder.calls == [
        ["alpha", "beta"],
        ["", ""],
    ]


def test_encode_inputs_scalar_negative_prompt_broadcasts() -> None:
    text_encoder = _FakeTextEncoderWrapper()
    bundle = _make_bare_bundle()
    bundle._text_encoder = text_encoder

    bundle.encode_inputs(
        ["alpha", "beta"],
        negative_prompt="ugly",
        height=64,
        width=128,
        num_frames=5,
    )

    assert text_encoder.calls[1] == ["ugly", "ugly"]


def test_encode_inputs_rejects_mismatched_negative_prompt_batch() -> None:
    bundle = _make_bare_bundle()
    bundle._text_encoder = _FakeTextEncoderWrapper()

    with pytest.raises(ValueError, match="negative_prompt batch size"):
        bundle.encode_inputs(
            ["a", "b"],
            negative_prompt=["only one"],
            height=64,
            width=128,
            num_frames=5,
        )


def test_encode_inputs_rejects_video_input() -> None:
    bundle = _make_bare_bundle()
    bundle._text_encoder = _FakeTextEncoderWrapper()

    with pytest.raises(NotImplementedError, match="video conditioning"):
        bundle.encode_inputs(
            ["a"],
            video=torch.zeros(1, 3, 4, 8, 8),
            height=64,
            width=128,
            num_frames=5,
        )


# ---------------------------------------------------------------------------
# forward_denoiser
# ---------------------------------------------------------------------------


def _patch_transformer(bundle: HunyuanVeido1p5ModelBundle, recorded: Dict[str, Any]) -> None:
    """Install a transformer stub that records its kwargs and returns zeros."""

    def fake_transformer(**kwargs: Any):
        recorded["kwargs"] = kwargs
        hidden_states = kwargs["hidden_states"]
        # Out channels = 32; the bundle returns ``out[0]`` which should match
        # the pure-latent shape (no cond/mask channels), i.e. 32 channels.
        out = torch.zeros(hidden_states.shape[0], 32, *hidden_states.shape[2:])
        return (out,)

    fake_transformer.config = SimpleNamespace(out_channels=32)
    bundle._transformer = fake_transformer  # type: ignore[assignment]


def test_forward_denoiser_assembles_concat_input_without_cfg() -> None:
    bundle = _make_bare_bundle()
    recorded: Dict[str, Any] = {}
    _patch_transformer(bundle, recorded)

    batch_size, latent_channels, latent_t, latent_h, latent_w = 2, 32, 2, 4, 8
    latents = torch.zeros(batch_size, latent_channels, latent_t, latent_h, latent_w)
    cond_latents = torch.zeros_like(latents)
    cond_mask = torch.zeros(batch_size, 1, latent_t, latent_h, latent_w)

    ctx = HunyuanVeido1p5ForwardContext(
        guidance_scale=1.0,  # disables CFG branch
        prompt_embeds=torch.zeros(batch_size, 5, 8),
        prompt_embeds_mask=torch.ones(batch_size, 5, dtype=torch.long),
        prompt_embeds_2=torch.zeros(batch_size, 6, 4),
        prompt_embeds_mask_2=torch.zeros(batch_size, 6, dtype=torch.long),
        image_embeds=torch.zeros(batch_size, 729, 1152),
        cond_latents=cond_latents,
        cond_mask=cond_mask,
    )

    sigma = torch.tensor(0.5)
    bundle.forward_denoiser(latents=latents, sigma=sigma, ctx=ctx)

    kwargs = recorded["kwargs"]
    assert kwargs["hidden_states"].shape == (
        batch_size,
        latent_channels * 2 + 1,
        latent_t,
        latent_h,
        latent_w,
    )
    assert kwargs["timestep"].shape == (batch_size,)
    # Timestep should have been multiplied by 1000 (flow-matching convention).
    assert torch.allclose(kwargs["timestep"], torch.full((batch_size,), 500.0))
    assert kwargs["encoder_hidden_states"].shape == (batch_size, 5, 8)
    assert kwargs["encoder_hidden_states_2"].shape == (batch_size, 6, 4)
    assert kwargs["image_embeds"].shape == (batch_size, 729, 1152)
    assert kwargs["return_dict"] is False
    assert "attention_kwargs" not in kwargs


def test_forward_denoiser_doubles_batch_for_cfg_branch() -> None:
    bundle = _make_bare_bundle()
    recorded: Dict[str, Any] = {}
    _patch_transformer(bundle, recorded)

    batch_size, latent_channels, latent_t, latent_h, latent_w = 2, 32, 2, 4, 8
    latents = torch.zeros(batch_size, latent_channels, latent_t, latent_h, latent_w)
    cond_latents = torch.zeros_like(latents)
    cond_mask = torch.zeros(batch_size, 1, latent_t, latent_h, latent_w)

    ctx = HunyuanVeido1p5ForwardContext(
        guidance_scale=6.0,
        prompt_embeds=torch.zeros(batch_size, 5, 8),
        prompt_embeds_mask=torch.ones(batch_size, 5, dtype=torch.long),
        prompt_embeds_2=torch.zeros(batch_size, 6, 4),
        prompt_embeds_mask_2=torch.zeros(batch_size, 6, dtype=torch.long),
        negative_prompt_embeds=torch.zeros(batch_size, 5, 8),
        negative_prompt_embeds_mask=torch.ones(batch_size, 5, dtype=torch.long),
        negative_prompt_embeds_2=torch.zeros(batch_size, 6, 4),
        negative_prompt_embeds_mask_2=torch.zeros(batch_size, 6, dtype=torch.long),
        image_embeds=torch.zeros(batch_size, 729, 1152),
        cond_latents=cond_latents,
        cond_mask=cond_mask,
    )

    sigma = torch.tensor(0.5)
    out = bundle.forward_denoiser(latents=latents, sigma=sigma, ctx=ctx)

    kwargs = recorded["kwargs"]
    assert kwargs["hidden_states"].shape[0] == batch_size * 2
    assert kwargs["timestep"].shape[0] == batch_size * 2
    assert kwargs["encoder_hidden_states"].shape[0] == batch_size * 2
    assert kwargs["encoder_hidden_states_2"].shape[0] == batch_size * 2
    assert kwargs["image_embeds"].shape[0] == batch_size * 2
    # CFG result keeps the original batch size.
    assert out.shape == (batch_size, latent_channels, latent_t, latent_h, latent_w)


def test_forward_denoiser_requires_full_text_stream_tuple() -> None:
    bundle = _make_bare_bundle()
    recorded: Dict[str, Any] = {}
    _patch_transformer(bundle, recorded)

    batch_size = 1
    ctx = HunyuanVeido1p5ForwardContext(
        guidance_scale=1.0,
        prompt_embeds=torch.zeros(batch_size, 5, 8),
        # missing prompt_embeds_mask / prompt_embeds_2 / prompt_embeds_mask_2
    )
    with pytest.raises(ValueError, match="full text-stream tuple"):
        bundle.forward_denoiser(
            latents=torch.zeros(batch_size, 32, 2, 4, 8),
            sigma=torch.tensor(0.5),
            ctx=ctx,
        )


# ---------------------------------------------------------------------------
# Aux-component error paths
# ---------------------------------------------------------------------------


def test_decode_latents_without_vae_raises_clear_error() -> None:
    bundle = _make_bare_bundle()
    with pytest.raises(RuntimeError, match="VAE"):
        bundle.decode_latents(torch.zeros(1, 32, 2, 4, 8))


def test_encode_inputs_without_vae_raises_when_image_provided() -> None:
    bundle = _make_bare_bundle()
    bundle._text_encoder = _FakeTextEncoderWrapper()
    bundle._vision_encoder = object()  # truthy stub
    bundle._image_processor = object()

    with pytest.raises(RuntimeError):
        bundle.encode_inputs(
            ["a"],
            image=torch.zeros(1, 3, 64, 64),
            height=64,
            width=128,
            num_frames=5,
        )
