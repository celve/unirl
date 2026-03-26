"""
Model Forward Plugins for GRPO Loss.

This module provides plugins that encapsulate model-specific forward pass logic,
allowing the loss function to be agnostic to the underlying model architecture.

Each plugin implements a consistent interface for:
- Preparing inputs (timesteps, guidance, position IDs)
- Executing the forward pass
- Handling model-specific quirks (CFG, timestep scaling, etc.)

Benefits:
- Loss class no longer needs model-specific if-elif chains
- New models can be added without modifying loss code
- Easier testing of model-specific logic in isolation
"""

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict, Optional, Protocol

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelForwardPlugin(Protocol):
    """Protocol defining the interface for model forward plugins.

    All plugins must implement the forward() method with this signature.
    Using Protocol instead of ABC allows duck-typing flexibility.
    """

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Execute forward pass through the model.

        Args:
            model: The diffusion model
            latents: Input latents [B, C, H, W] or [B, C, T, H, W]
            sigma: Current sigma/timestep value
            prompt_embeds: Text encoder hidden states [B, seq, hidden]
            pooled_prompt_embeds: Pooled text embeddings [B, hidden]
            guidance_scale: Classifier-free guidance scale
            text_ids: Position IDs for text tokens (FLUX)
            image_ids: Position IDs for image patches (FLUX)
            negative_prompt_embeds: Negative prompt embeddings (for CFG)
            negative_pooled_prompt_embeds: Pooled negative embeddings
            **kwargs: Additional model-specific arguments

        Returns:
            Model prediction (velocity/noise) [B, C, H, W] or [B, C, T, H, W]
        """
        ...

    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare model kwargs for forward pass without executing model call."""
        ...


class BaseForwardPlugin(ABC):
    """Base class for forward plugins with common utilities.

    Subclasses may set ``autocast_dtype`` (e.g. ``torch.bfloat16``) so that
    ``forward()`` wraps the model call in ``torch.autocast``, matching the
    precision used during sampling.  When ``None`` (default) no autocast is
    applied and the model runs in its native parameter dtype.
    """

    autocast_dtype: Optional[torch.dtype] = None

    def _build_autocast_ctx(self, device: torch.device):
        """Return an autocast context matching the sampling path."""
        if (
            self.autocast_dtype is not None
            and device.type == "cuda"
            and self.autocast_dtype in (torch.float16, torch.bfloat16)
        ):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    @abstractmethod
    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare kwargs for model forward call without executing it."""
        ...

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute forward pass through the model."""
        ...

    def _prepare_timestep(
        self,
        sigma: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Prepare timestep tensor from sigma."""
        if sigma.dim() == 0:
            sigma_expanded = sigma.unsqueeze(0)
        else:
            sigma_expanded = sigma
        return sigma_expanded.expand(batch_size).to(device, dtype=dtype)


class FluxForwardPlugin(BaseForwardPlugin):
    """Forward plugin for FLUX models.

    FLUX models require:
    - guidance tensor input
    - txt_ids and img_ids for position encoding
    - No timestep scaling (sigma is used directly)
    """

    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare FLUX model kwargs (timestep is not multiplied by 1000)."""
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype

        timestep = self._prepare_timestep(sigma, batch_size, device)

        # FLUX requires guidance tensor
        guidance_tensor = torch.tensor(
            [guidance_scale], device=device, dtype=dtype
        ).expand(batch_size)

        # Process text_ids: [B, seq, 3] -> [seq, 3]
        if text_ids is not None:
            if text_ids.dim() == 3:
                txt_ids = text_ids[0]
            elif text_ids.dim() == 2:
                txt_ids = text_ids
            else:
                txt_ids = text_ids
        else:
            txt_ids = None

        # Process image_ids: [B, seq, 3] -> [seq, 3] or keep [num_patches, 3]
        img_ids = image_ids
        if img_ids is not None:
            if img_ids.dim() == 3:
                img_ids = img_ids[0]
            elif img_ids.dim() > 3:
                img_ids = img_ids.squeeze(0)
                if img_ids.dim() > 2:
                    img_ids = img_ids[0]

        return {
            "hidden_states": latents.to(dtype),
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep,
            "guidance": guidance_tensor,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "pooled_projections": pooled_prompt_embeds,
            "return_dict": False,
        }

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute FLUX forward pass."""
        model_kwargs = self.prepare_model_kwargs(
            latents=latents,
            sigma=sigma,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            text_ids=text_ids,
            image_ids=image_ids,
            **kwargs,
        )
        with self._build_autocast_ctx(latents.device):
            pred = model(**model_kwargs)[0]

        return pred


class SD3ForwardPlugin(BaseForwardPlugin):
    """Forward plugin for SD3 models.

    SD3 models require:
    - timestep * 1000 scaling
    - CFG with batched forward (single forward with doubled batch)

    Memory optimization: Uses single forward with concatenated inputs
    instead of two separate forwards to reduce peak activation memory.
    """

    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare SD3 kwargs (timestep multiplied by 1000)."""
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        # Keep timestep in float32 to avoid bfloat16 precision loss
        timestep = self._prepare_timestep(sigma, batch_size, device) * 1000
        model_kwargs: Dict[str, Any] = {
            "hidden_states": latents,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep,
            "return_dict": False,
        }
        if pooled_prompt_embeds is not None:
            model_kwargs["pooled_projections"] = pooled_prompt_embeds
        encoder_attention_mask = kwargs.get("encoder_attention_mask")
        if encoder_attention_mask is not None:
            model_kwargs["encoder_attention_mask"] = encoder_attention_mask
        return model_kwargs

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute SD3 forward pass with memory-efficient CFG."""
        batch_size = latents.shape[0]
        device = latents.device

        sigma = self._prepare_timestep(sigma, batch_size, device)
        timestep = sigma * 1000
        encoder_attention_mask = kwargs.get("encoder_attention_mask")

        with self._build_autocast_ctx(device):
            if guidance_scale > 1.0:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                if negative_pooled_prompt_embeds is None:
                    negative_pooled_prompt_embeds = (
                        torch.zeros_like(pooled_prompt_embeds)
                        if pooled_prompt_embeds is not None
                        else None
                    )

                attn_kw: Dict[str, Any] = {}
                if encoder_attention_mask is not None:
                    attn_kw["encoder_attention_mask"] = torch.cat(
                        [encoder_attention_mask, encoder_attention_mask], dim=0
                    )

                noise_pred = model(
                    hidden_states=torch.cat([latents, latents], dim=0),
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                    timestep=torch.cat([timestep, timestep], dim=0),
                    pooled_projections=(
                        torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        if pooled_prompt_embeds is not None else None
                    ),
                    return_dict=False,
                    **attn_kw,
                )[0]

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                single_kw: Dict[str, Any] = {"return_dict": False}
                if encoder_attention_mask is not None:
                    single_kw["encoder_attention_mask"] = encoder_attention_mask
                pred = model(
                    **self.prepare_model_kwargs(
                        latents=latents,
                        sigma=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        **kwargs,
                    )
                )[0]

        return pred


class HunyuanForwardPlugin(BaseForwardPlugin):
    """Forward plugin for Hunyuan video models.

    Hunyuan models have similar interface to SD3 with some variations
    in how embeddings are handled.
    """
    GUIDANCE_VALUE = 6018.0

    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare Hunyuan kwargs (timestep multiplied by 1000)."""
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = torch.zeros(
                batch_size,
                768,
                device=device,
                dtype=dtype,
            )
        encoder_attention_mask = kwargs.get("encoder_attention_mask")
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                batch_size,
                prompt_embeds.shape[1],
                device=device,
                dtype=torch.long,
            )
        timestep_1000 = self._prepare_timestep(sigma, batch_size, device) * 1000
        guidance = torch.tensor(
            [self.GUIDANCE_VALUE], device=device, dtype=dtype
        )

        model_kwargs: Dict[str, Any] = {
            "hidden_states": latents.to(dtype),
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_attention_mask": encoder_attention_mask,
            "timestep": timestep_1000,
            "guidance": guidance,
            "return_dict": False,
        }
        return model_kwargs

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute Hunyuan forward pass."""
        model_kwargs = self.prepare_model_kwargs(
            latents=latents,
            sigma=sigma,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            text_ids=text_ids,
            image_ids=image_ids,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )
        with self._build_autocast_ctx(latents.device):
            pred = model(**model_kwargs)[0]
        return pred


class MochiForwardPlugin(BaseForwardPlugin):
    """Forward plugin for Mochi video models.

    Mochi transformer interface differs from Hunyuan:
    - requires encoder_attention_mask
    - does not use pooled_projections/guidance inputs
    - timestep follows SD3-style 0-1000 scaling
    """

    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare Mochi kwargs."""
        del pooled_prompt_embeds, guidance_scale, text_ids, image_ids
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        timestep_1000 = self._prepare_timestep(sigma, batch_size, device) * 1000
        model_kwargs: Dict[str, Any] = {
            "hidden_states": latents.to(dtype),
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep_1000,
            "return_dict": False,
        }
        encoder_attention_mask = kwargs.get("encoder_attention_mask")
        if encoder_attention_mask is not None:
            model_kwargs["encoder_attention_mask"] = encoder_attention_mask
        attention_kwargs = kwargs.get("attention_kwargs")
        if attention_kwargs is not None:
            model_kwargs["attention_kwargs"] = attention_kwargs
        return model_kwargs

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute Mochi forward pass with optional CFG."""
        del pooled_prompt_embeds, text_ids, image_ids, negative_pooled_prompt_embeds
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        timestep_1000 = self._prepare_timestep(sigma, batch_size, device) * 1000

        with self._build_autocast_ctx(device):
            if guidance_scale > 1.0:
                uncond_embeds = (
                    negative_prompt_embeds
                    if negative_prompt_embeds is not None
                    else torch.zeros_like(prompt_embeds)
                )
                latents_batched = torch.cat([latents, latents], dim=0)
                embeds_batched = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                timestep_batched = torch.cat([timestep_1000, timestep_1000], dim=0)

                model_kwargs: Dict[str, Any] = {
                    "hidden_states": latents_batched.to(dtype),
                    "encoder_hidden_states": embeds_batched,
                    "timestep": timestep_batched,
                    "return_dict": False,
                }
                if encoder_attention_mask is not None:
                    model_kwargs["encoder_attention_mask"] = torch.cat(
                        [encoder_attention_mask, encoder_attention_mask], dim=0
                    )
                attention_kwargs = kwargs.get("attention_kwargs")
                if attention_kwargs is not None:
                    model_kwargs["attention_kwargs"] = attention_kwargs

                noise_pred = model(**model_kwargs)[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                return noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            return model(
                **self.prepare_model_kwargs(
                    latents=latents,
                    sigma=sigma,
                    prompt_embeds=prompt_embeds,
                    encoder_attention_mask=encoder_attention_mask,
                    **kwargs,
                )
            )[0]


class DefaultForwardPlugin(BaseForwardPlugin):
    """Default forward plugin with fallback logic.

    Tries SD3-style interface first, falls back to generic UNet interface.
    This plugin is used when the model type is unknown.

    Memory optimization: Uses single forward with concatenated inputs for CFG.
    """

    def prepare_model_kwargs(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare default kwargs using SD3-style timestep scaling."""
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        # Keep timestep in float32 to avoid bfloat16 precision loss
        timestep = self._prepare_timestep(sigma, batch_size, device) * 1000
        model_kwargs: Dict[str, Any] = {
            "hidden_states": latents.to(dtype),
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep,
            "return_dict": False,
        }
        if pooled_prompt_embeds is not None:
            model_kwargs["pooled_projections"] = pooled_prompt_embeds
        encoder_attention_mask = kwargs.get("encoder_attention_mask")
        if encoder_attention_mask is not None:
            model_kwargs["encoder_attention_mask"] = encoder_attention_mask
        return model_kwargs

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute forward pass with fallback logic and memory-efficient CFG."""
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype

        timestep = self._prepare_timestep(sigma, batch_size, device)

        with self._build_autocast_ctx(device):
            # Try SD3-style interface (timestep * 1000)
            try:
                timestep_1000 = timestep * 1000
                if guidance_scale > 1.0:
                    # CFG mode with batched forward (memory efficient)
                    uncond_embeds = (
                        negative_prompt_embeds
                        if negative_prompt_embeds is not None
                        else torch.zeros_like(prompt_embeds)
                    )
                    uncond_pooled = (
                        negative_pooled_prompt_embeds
                        if negative_pooled_prompt_embeds is not None
                        else (
                            torch.zeros_like(pooled_prompt_embeds)
                            if pooled_prompt_embeds is not None
                            else None
                        )
                    )

                    # Concatenate for single batched forward
                    latents_batched = torch.cat([latents, latents], dim=0)
                    embeds_batched = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                    timestep_batched = torch.cat([timestep_1000, timestep_1000], dim=0)

                    if pooled_prompt_embeds is not None:
                        pooled_batched = torch.cat([uncond_pooled, pooled_prompt_embeds], dim=0)
                    else:
                        pooled_batched = None

                    # Single forward with doubled batch
                    noise_pred = model(
                        hidden_states=latents_batched.to(dtype),
                        encoder_hidden_states=embeds_batched,
                        timestep=timestep_batched,
                        pooled_projections=pooled_batched,
                        return_dict=False,
                    )[0]

                    # Split result
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                    pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    pred = model(
                        **self.prepare_model_kwargs(
                            latents=latents,
                            sigma=sigma,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            guidance_scale=guidance_scale,
                            **kwargs,
                        )
                    )[0]
            except TypeError:
                # Generic UNet interface
                pred = model(
                    latents.to(dtype),
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                )[0]

        return pred
