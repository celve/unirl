"""Typed runtime config for model bundle construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class ModelBundleConfig:
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
        if self.lora_target_modules is None:
            return
        if (
            not isinstance(self.lora_target_modules, list)
            or not self.lora_target_modules
        ):
            raise ValueError(
                "ModelBundleConfig.lora_target_modules must be None or a non-empty list."
            )


__all__ = ["ModelBundleConfig"]
