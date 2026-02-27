"""
Wan2.1 Model Bundle — example plugin.

This file demonstrates how to add a new model to diffusionrl without
modifying any core framework files.

Usage:
    # Option A: explicit dotpath (always works)
    --model-path diffusionrl_plugins.models.wan21.Wan21ModelBundle

    # Option B: short name (requires PLUGIN_SPEC defaults, see bottom)
    --model-type wan21

Steps to create your own model plugin:
    1. Subclass ``diffusionrl.models.base.ModelBundle``
    2. Implement the required abstract methods
    3. Add ``PLUGIN_SPEC`` at the bottom of the file
    4. Point to it via ``--model-path`` or ``--model-type``
"""
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusionrl.models.base import ModelBundle

logger = logging.getLogger(__name__)


class Wan21ModelBundle(ModelBundle):
    """Wan2.1 Text-to-Video model bundle (example skeleton)."""

    def __init__(
        self,
        pretrained_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_target_modules=lora_target_modules,
            **kwargs,
        )
        # TODO: load Wan2.1 transformer, VAE, text encoder

    def load(self) -> None:
        raise NotImplementedError("Wan21ModelBundle.load() not yet implemented")

    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError

    def get_model_state_dict(self) -> dict:
        raise NotImplementedError

    def load_model_state_dict(self, state_dict: dict, strict: bool = False) -> None:
        raise NotImplementedError

    def get_no_split_modules(self) -> Tuple[type, ...]:
        return ()

    @property
    def model_type(self) -> str:
        return "wan21"

    @property
    def media_type(self) -> str:
        return "video"
