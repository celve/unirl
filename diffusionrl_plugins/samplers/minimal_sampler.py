"""Minimal sampler plugin template.

This class exists so custom model plugins can declare a valid default sampler
dotpath during config validation. Copy this file and replace `sample()`
with your model-specific implementation.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

import torch
import torch.nn as nn

from diffusionrl.samplers.base import BaseSampler
from diffusionrl.types import RolloutOutput


class MinimalSampler(BaseSampler):
    """Template sampler with constructor compatible with FSDP engine wiring."""

    def __init__(
        self,
        model: Optional[nn.Module],
        text_encoder: Optional[nn.Module],
        vae: Optional[nn.Module],
        eta: float = 1.0,
        sde_type: str = "sde",
        shift: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(eta=eta, sde_type=sde_type, shift=shift)
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.extra_kwargs = dict(kwargs)

    def sample(
        self,
        prompts: List[str],
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs: Any,
    ) -> RolloutOutput:
        del (
            prompts,
            prompt_embeds,
            pooled_prompt_embeds,
            num_inference_steps,
            guidance_scale,
            latents,
            generator,
            sde_indices,
            kwargs,
        )
        raise NotImplementedError(
            "MinimalSampler is a template. Copy "
            "`diffusionrl_plugins/samplers/minimal_sampler.py` and implement sample() "
            "to return a valid RolloutOutput."
        )
