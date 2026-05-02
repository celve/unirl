"""Typed runtime config for model bundle construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.config.validation import validate_precision_type


@register_config(group="model", name="base", mutable=True)
@dataclass
class ModelBundleConfig:
    """Model-bundle construction args.

    Marked ``mutable=True``: ``device``, ``training_only``, and
    ``skip_device_move`` are runtime-injected onto ``cfg.model`` by
    ``TrainActor`` after compose (FSDP placement depends on the actor's
    local rank and the backend's ``cpu_offload`` flag). Every other field
    is set at compose time and read once during bundle construction.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: Optional[List[str]] = None
    use_gradient_checkpointing: bool = False
    model_precision: Any = "bf16"
    device: Any = None
    training_only: bool = False
    skip_device_move: bool = False

    def __post_init__(self) -> None:
        require(
            self.lora_target_modules is None
            or (isinstance(self.lora_target_modules, list) and bool(self.lora_target_modules)),
            "ModelBundleConfig.lora_target_modules must be None or a non-empty list.",
        )
        validate_precision_type(self.model_precision, field="ModelBundleConfig.model_precision")


__all__ = ["ModelBundleConfig"]
