from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from diffusionrl.cmdline.models import build_model_bundle_init_payload_from_args
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models import (
    ModelBundle,
    ModelBundleConfig,
    create_model_bundle_from_init_payload,
)
from diffusionrl.models.flux import FluxModelBundleConfig


class _FakeModelBundle(ModelBundle):
    model_type = "fake"

    def __init__(self, config: ModelBundleConfig, **_: Any) -> None:
        super().__init__(config)
        self.pretrained_path = config.pretrained_model_ckpt_path
        self.vae_ckpt_path = config.vae_ckpt_path
        self.text_encoder_ckpt_path = config.text_encoder_ckpt_path
        self.device = config.device
        self.training_only = config.training_only
        self.skip_device_move = config.skip_device_move
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = list(config.lora_target_modules or [])
        self.received_dtype = config.model_precision

    def load(self) -> None:
        return None

    def encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def encode_images(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        return None

    @property
    def media_type(self) -> str:
        return "image"


def test_build_model_bundle_init_payload_from_args_returns_typed_payload() -> None:
    args = SimpleNamespace(
        model=SimpleNamespace(
            pretrained_model_ckpt_path="/tmp/model",
            vae_ckpt_path="/tmp/vae",
            text_encoder_ckpt_path="/tmp/text",
        ),
        training=SimpleNamespace(
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules=["transformer.blocks.0"],
            use_gradient_checkpointing=True,
        ),
        precision=SimpleNamespace(model_precision="bf16"),
    )
    model_spec = SimpleNamespace(model_dotpath="diffusionrl.models.sd3.SD3ModelBundle")

    payload = build_model_bundle_init_payload_from_args(args, model_spec=model_spec)

    assert isinstance(payload, ComponentInitPayload)
    assert payload.component_dotpath.endswith("SD3ModelBundle")
    assert isinstance(payload.component_config, ModelBundleConfig)
    assert payload.component_config.pretrained_model_ckpt_path == "/tmp/model"
    assert payload.component_config.vae_ckpt_path == "/tmp/vae"
    assert payload.component_config.text_encoder_ckpt_path == "/tmp/text"
    assert payload.component_config.use_lora is True
    assert payload.component_config.lora_rank == 32
    assert payload.component_config.lora_alpha == 64
    assert payload.component_config.lora_target_modules == ["transformer.blocks.0"]


def test_build_model_bundle_init_payload_from_args_uses_none_for_default_lora_targets() -> None:
    args = SimpleNamespace(
        model=SimpleNamespace(
            pretrained_model_ckpt_path="/tmp/model",
            vae_ckpt_path=None,
            text_encoder_ckpt_path=None,
        ),
        training=SimpleNamespace(
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules=[],
            use_gradient_checkpointing=False,
        ),
        precision=SimpleNamespace(model_precision="bf16"),
    )
    model_spec = SimpleNamespace(model_dotpath="diffusionrl.models.sd3.SD3ModelBundle")

    payload = build_model_bundle_init_payload_from_args(args, model_spec=model_spec)

    assert payload.component_config.lora_target_modules is None


def test_model_bundle_config_rejects_empty_lora_target_modules() -> None:
    try:
        ModelBundleConfig(
            pretrained_model_ckpt_path="/tmp/model",
            lora_target_modules=[],
        )
    except ValueError as exc:
        assert "lora_target_modules" in str(exc)
    else:
        raise AssertionError("Expected ModelBundleConfig to reject empty lora_target_modules")


def test_create_model_bundle_from_init_payload_passes_typed_config() -> None:
    fake_payload = ComponentInitPayload(
        component_dotpath="tests.cmdline.test_model_construction._FakeModelBundle",
        component_config=ModelBundleConfig(
            pretrained_model_ckpt_path="/tmp/model",
            vae_ckpt_path="/tmp/vae",
            text_encoder_ckpt_path="/tmp/text",
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules=["transformer.blocks.0"],
            model_precision=torch.float16,
            device="cpu",
            training_only=True,
            skip_device_move=True,
        ),
    )

    model_bundle = create_model_bundle_from_init_payload(fake_payload)

    assert type(model_bundle).__name__ == "_FakeModelBundle"
    assert model_bundle.pretrained_path == "/tmp/model"
    assert model_bundle.vae_ckpt_path == "/tmp/vae"
    assert model_bundle.text_encoder_ckpt_path == "/tmp/text"
    assert model_bundle.device == "cpu"
    assert model_bundle.training_only is True
    assert model_bundle.skip_device_move is True
    assert model_bundle.use_lora is True
    assert model_bundle.lora_rank == 32
    assert model_bundle.lora_alpha == 64
    assert model_bundle.lora_target_modules == ["transformer.blocks.0"]
    assert model_bundle.received_dtype == torch.float16


def test_create_model_bundle_from_init_payload_omits_default_lora_targets() -> None:
    fake_payload = ComponentInitPayload(
        component_dotpath="tests.cmdline.test_model_construction._FakeModelBundle",
        component_config=ModelBundleConfig(
            pretrained_model_ckpt_path="/tmp/model",
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules=None,
            model_precision=torch.float16,
        ),
    )

    model_bundle = create_model_bundle_from_init_payload(fake_payload)

    assert model_bundle.use_lora is True
    assert model_bundle.lora_rank == 32
    assert model_bundle.lora_alpha == 64
    assert model_bundle.lora_target_modules == []


def test_model_bundle_base_dtype_fallbacks_follow_model_precision() -> None:
    config = ModelBundleConfig(
        pretrained_model_ckpt_path="/tmp/model",
        model_precision=torch.float16,
    )

    model_bundle = _FakeModelBundle(config)

    assert model_bundle.dtype == torch.float16
    assert model_bundle.vae_dtype == torch.float16
    assert model_bundle.text_encoder_dtype == torch.float16


def test_flux_model_bundle_config_preserves_second_text_encoder_path() -> None:
    config = FluxModelBundleConfig(
        pretrained_model_ckpt_path="/tmp/model",
        text_encoder_2_ckpt_path="/tmp/text_encoder_2",
    )

    assert config.text_encoder_2_ckpt_path == "/tmp/text_encoder_2"
