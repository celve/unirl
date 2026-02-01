"""
Model Forward Adapters for GRPO Loss.

This module provides adapters that encapsulate model-specific forward pass logic,
allowing the loss function to be agnostic to the underlying model architecture.

Each adapter implements a consistent interface for:
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
from typing import Optional, Protocol, Type, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelForwardAdapter(Protocol):
    """Protocol defining the interface for model forward adapters.

    All adapters must implement the forward() method with this signature.
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


class BaseForwardAdapter(ABC):
    """Base class for forward adapters with common utilities."""

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
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare timestep tensor from sigma."""
        if sigma.dim() == 0:
            sigma_expanded = sigma.unsqueeze(0)
        else:
            sigma_expanded = sigma
        return sigma_expanded.expand(batch_size).to(device, dtype=dtype)


class FluxForwardAdapter(BaseForwardAdapter):
    """Forward adapter for FLUX models.

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

        timestep = self._prepare_timestep(sigma, batch_size, device, dtype)

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
        pred = model(**model_kwargs)[0]

        return pred


class SD3ForwardAdapter(BaseForwardAdapter):
    """Forward adapter for SD3 models.

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
        timestep = self._prepare_timestep(sigma, batch_size, device, dtype) * 1000
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
        """Execute SD3 forward pass with memory-efficient CFG.

        Uses batched forward (single forward with 2x batch) instead of
        two separate forwards to reduce peak memory usage.
        """
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype

        timestep = self._prepare_timestep(sigma, batch_size, device, dtype)
        timestep_1000 = timestep * 1000

        if guidance_scale > 1.0:
            # CFG: single forward with batched inputs (memory efficient)
            # Following flow_grpo's approach: concat latents x2, concat [neg, pos] embeds
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
            # latents: [B, C, H, W] -> [2B, C, H, W]
            latents_batched = torch.cat([latents, latents], dim=0)
            # embeds: [B, seq, hidden] -> [2B, seq, hidden]
            embeds_batched = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            # timestep: [B] -> [2B]
            timestep_batched = torch.cat([timestep_1000, timestep_1000], dim=0)

            # pooled: [B, hidden] -> [2B, hidden]
            if pooled_prompt_embeds is not None:
                pooled_batched = torch.cat([uncond_pooled, pooled_prompt_embeds], dim=0)
            else:
                pooled_batched = None

            # Single forward pass with doubled batch
            noise_pred = model(
                hidden_states=latents_batched.to(dtype),
                encoder_hidden_states=embeds_batched,
                timestep=timestep_batched,
                pooled_projections=pooled_batched,
                return_dict=False,
            )[0]

            # Split result: [2B, ...] -> 2x [B, ...]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
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

        return pred


class HunyuanForwardAdapter(BaseForwardAdapter):
    """Forward adapter for Hunyuan video models.

    Hunyuan models have similar interface to SD3 with some variations
    in how embeddings are handled.
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
        """Prepare Hunyuan kwargs (timestep multiplied by 1000)."""
        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        timestep_1000 = self._prepare_timestep(sigma, batch_size, device, dtype) * 1000

        model_kwargs: Dict[str, Any] = {
            "hidden_states": latents.to(dtype),
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep_1000,
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
        pred = model(**model_kwargs)[0]
        return pred


class DefaultForwardAdapter(BaseForwardAdapter):
    """Default forward adapter with fallback logic.

    Tries SD3-style interface first, falls back to generic UNet interface.
    This adapter is used when the model type is unknown.

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
        timestep = self._prepare_timestep(sigma, batch_size, device, dtype) * 1000
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

        timestep = self._prepare_timestep(sigma, batch_size, device, dtype)

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
                pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
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


# Registry mapping model types to their adapters
ADAPTER_REGISTRY: Dict[str, Type[BaseForwardAdapter]] = {
    "flux": FluxForwardAdapter,
    "sd3": SD3ForwardAdapter,
    "hunyuan": HunyuanForwardAdapter,
    "mochi": HunyuanForwardAdapter,  # Mochi uses similar interface to Hunyuan
    "default": DefaultForwardAdapter,
}

# Cached adapter instances
_adapter_cache: Dict[str, BaseForwardAdapter] = {}


def get_forward_adapter(model_type: str) -> BaseForwardAdapter:
    """
    Get the forward adapter for a given model type.

    Args:
        model_type: Model type identifier (e.g., "flux", "sd3", "hunyuan")

    Returns:
        Configured forward adapter instance
    """
    if model_type not in _adapter_cache:
        adapter_cls = ADAPTER_REGISTRY.get(model_type.lower(), DefaultForwardAdapter)
        _adapter_cache[model_type] = adapter_cls()
        logger.debug(f"Created forward adapter for model_type={model_type}: {adapter_cls.__name__}")

    return _adapter_cache[model_type]


def detect_model_type(model: nn.Module) -> str:
    """
    Detect model type from the model's class name.

    Args:
        model: The model to detect type for

    Returns:
        Detected model type string
    """
    model_name = model.__class__.__name__.lower()
    base_model = model

    # Unwrap common wrappers
    if hasattr(model, 'module'):
        base_model = model.module
    if hasattr(base_model, 'get_base_model'):
        base_model = base_model.get_base_model()

    base_model_name = base_model.__class__.__name__.lower()

    # Check both wrapped and unwrapped names
    for name in [model_name, base_model_name]:
        if 'flux' in name:
            return 'flux'
        elif 'sd3' in name or 'stablevideo' in name:
            return 'sd3'
        elif 'hunyuan' in name:
            return 'hunyuan'
        elif 'mochi' in name:
            return 'mochi'

    return 'default'
