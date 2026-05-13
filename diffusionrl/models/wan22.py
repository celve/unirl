"""WAN 2.2 T2V dual-transformer model bundle.

Wan2.2-T2V uses two WanTransformer3DModel instances:
  - high_noise: handles timesteps t >= boundary_timestep (coarse structure)
  - low_noise: handles timesteps t < boundary_timestep (detail refinement)

Both are wrapped in a single WanDualTransformer nn.Module so that the existing
FSDPBackend (which wraps model_bundle.transformer) shards and trains both
transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from diffusionrl.config.registration import register_config

from .wan21 import WAN21ModelBundle, WAN21ModelBundleConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite module
# ---------------------------------------------------------------------------


class WanDualTransformer(nn.Module):
    """Thin nn.Module wrapper presenting both Wan2.2 transformers as one model.

    This exists so that FSDPBackend can call ``fully_shard(model_bundle.transformer)``
    and shard **both** sub-transformers in a single call. ``named_modules()`` recurses
    into both children, so ``_iter_target_modules`` discovers WanTransformerBlock
    instances in both.

    The ``forward()`` method routes to the correct sub-transformer based on a
    ``use_high_noise`` flag. This is essential for FSDP2: calling forward() on
    the root module triggers the FSDP pre-forward hook that all-gathers sharded
    parameters (patch_embedding, condition_embedder, etc.) before they are used.
    Bypassing root forward (calling sub-transformers directly) would leave those
    parameters as sharded DTensors → "mixed Tensor and DTensor" RuntimeError.
    """

    def __init__(self, high_noise: nn.Module, low_noise: nn.Module) -> None:
        super().__init__()
        self.high_noise = high_noise
        self.low_noise = low_noise

    @property
    def config(self):
        """Expose the high-noise transformer config for compatibility.

        Consumers like FSDPWanSampler._latent_channels() read self.model.config
        to discover out_channels / patch_size. Both sub-transformers share the
        same architecture config, so returning either one is correct.
        """
        return getattr(self.high_noise, "config", None)

    def forward(self, *, use_high_noise: bool, **kwargs) -> Any:
        """Route to the selected sub-transformer.

        Must be called instead of accessing sub-transformers directly so that
        FSDP2's pre-forward all-gather hook fires on this root module.

        Args:
            use_high_noise: If True, route to high_noise transformer; else low_noise.
            **kwargs: All keyword arguments forwarded to the sub-transformer.
        """
        if use_high_noise:
            return self.high_noise(**kwargs)
        else:
            return self.low_noise(**kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BOUNDARY_RATIO = 0.875


@register_config(
    group="model",
    name="wan22",
    target="diffusionrl.models.wan22.WAN22ModelBundle",
)
@dataclass
class WAN22ModelBundleConfig(WAN21ModelBundleConfig):
    """Wan2.2-T2V specific configuration.

    Extends the Wan2.1 config with dual-transformer parameters.
    """

    boundary_ratio: float = DEFAULT_BOUNDARY_RATIO
    guidance_scale_2: Optional[float] = None
    num_train_timesteps: int = 1000


# ---------------------------------------------------------------------------
# Model Bundle
# ---------------------------------------------------------------------------


class WAN22ModelBundle(WAN21ModelBundle):
    """WAN 2.2 T2V bundle with dual-transformer architecture.

    Loads two WanTransformer3DModel instances and presents them as a unified
    ``WanDualTransformer`` via the ``transformer`` property. During denoising,
    ``forward_denoiser`` routes to the correct sub-transformer based on the
    sigma-derived timestep and ``boundary_ratio``.
    """

    def __init__(self, config: WAN22ModelBundleConfig):
        # Store wan22-specific config before calling super().__init__()
        # which will trigger self.load() → self._load_transformer()
        self._boundary_ratio = config.boundary_ratio
        self._guidance_scale_2 = config.guidance_scale_2
        self._num_train_timesteps = config.num_train_timesteps

        self._high_noise_transformer: Optional[nn.Module] = None
        self._low_noise_transformer: Optional[nn.Module] = None

        super().__init__(config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return "wan22"

    @property
    def boundary_ratio(self) -> float:
        return self._boundary_ratio

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        return "diffusionrl.samplers.fsdp.wan22_sampler.FSDPWan22Sampler"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_transformer(self) -> None:
        """Load both Wan2.2 transformers and wrap in WanDualTransformer."""
        try:
            from diffusers import WanTransformer3DModel
        except ImportError:
            from diffusers import AutoModel

            WanTransformer3DModel = AutoModel

        self._high_noise_transformer = WanTransformer3DModel.from_pretrained(
            self.pretrained_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        self._low_noise_transformer = WanTransformer3DModel.from_pretrained(
            self.pretrained_path,
            subfolder="transformer_2",
            torch_dtype=self.dtype,
        )
        logger.info(
            "Loaded Wan2.2 transformers from %s (transformer + transformer_2)",
            self.pretrained_path,
        )

        if self.skip_device_move:
            for t in (self._high_noise_transformer, self._low_noise_transformer):
                first_param_device = next(t.parameters()).device
                if str(first_param_device) != "cpu":
                    t.to("cpu")
        else:
            # Dtype unification: diffusers leaves some buffers (RoPE freqs,
            # timestep embeddings) in fp32; FSDP2 asserts uniform param dtype.
            self._high_noise_transformer.to(self.device, dtype=self.dtype)
            self._low_noise_transformer.to(self.device, dtype=self.dtype)

        self._transformer = WanDualTransformer(
            high_noise=self._high_noise_transformer,
            low_noise=self._low_noise_transformer,
        )

        added_kv_proj_dim = getattr(getattr(self._high_noise_transformer, "config", None), "added_kv_proj_dim", None)
        self._is_i2v_model = added_kv_proj_dim is not None and added_kv_proj_dim > 0
        variant_str = "I2V" if self._is_i2v_model else "T2V"
        logger.info(
            "Wan2.2 dual-transformer ready (variant: %s, boundary_ratio: %.3f)",
            variant_str,
            self._boundary_ratio,
        )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_lora_to_module(self, module: nn.Module, lora_config) -> nn.Module:
        """Apply LoRA adapter to a single transformer module."""
        from peft import get_peft_model

        module = get_peft_model(module, lora_config)
        module.add_adapter("old", lora_config)
        module.set_adapter("default")
        return module

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to both sub-transformers."""
        if self._high_noise_transformer is None or self._low_noise_transformer is None:
            logger.warning("Cannot add LoRA: Wan2.2 transformers not loaded")
            return

        try:
            from peft import LoraConfig
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install with: pip install peft")
            return

        target_modules = self.lora_target_modules
        if target_modules is None:
            target_modules = type(self).default_lora_target_modules()
            if target_modules is None:
                raise RuntimeError("WAN22 LoRA requested but lora_target_modules is unresolved.")

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        self._high_noise_transformer = self._apply_lora_to_module(self._high_noise_transformer, lora_config)
        self._low_noise_transformer = self._apply_lora_to_module(self._low_noise_transformer, lora_config)

        for transformer in (self._high_noise_transformer, self._low_noise_transformer):
            for param in transformer.parameters():
                if param.requires_grad and param.dtype != self.dtype:
                    param.data = param.data.to(self.dtype)

        # Rebuild composite with LoRA-wrapped sub-transformers
        self._transformer = WanDualTransformer(
            high_noise=self._high_noise_transformer,
            low_noise=self._low_noise_transformer,
        )

        logger.info(
            "LoRA adapters added to both Wan2.2 transformers (rank=%s, alpha=%s)",
            self.lora_rank,
            self.lora_alpha,
        )

    # ------------------------------------------------------------------
    # Forward denoiser with boundary routing
    # ------------------------------------------------------------------

    def _select_guidance_for_sigma(
        self, sigma: torch.Tensor, guidance_scale: float, guidance_scale_2: Optional[float]
    ) -> Tuple[bool, float]:
        """Determine which sub-transformer and guidance to use based on sigma.

        Returns:
            (use_high_noise, effective_guidance_scale)
        """
        sigma_val = sigma.item() if sigma.dim() == 0 else sigma.mean().item()

        if sigma_val >= self._boundary_ratio:
            return True, guidance_scale
        else:
            gs = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
            return False, gs

    def forward_denoiser(self, *, latents: torch.Tensor, sigma: torch.Tensor, ctx) -> torch.Tensor:
        """Route to the correct sub-transformer based on sigma vs boundary.

        All calls go through self.transformer.forward() (WanDualTransformer)
        which triggers FSDP2's pre-forward all-gather hook, ensuring sharded
        parameters like patch_embedding are properly materialized.

        Args:
            latents: Noisy latent tensor [B, C, T, H, W]
            sigma: Current noise level (scalar or [B])
            ctx: WAN22ForwardContext (or WAN21ForwardContext for backward compat)

        Returns:
            Noise prediction tensor
        """
        prompt_embeds = ctx.prompt_embeds
        if prompt_embeds is None:
            raise ValueError("WAN22ModelBundle.forward_denoiser requires ctx.prompt_embeds.")

        guidance_scale = float(getattr(ctx, "guidance_scale", 5.0))
        guidance_scale_2 = getattr(ctx, "guidance_scale_2", self._guidance_scale_2)

        use_high_noise, active_guidance = self._select_guidance_for_sigma(sigma, guidance_scale, guidance_scale_2)

        negative_prompt_embeds = getattr(ctx, "negative_prompt_embeds", None)
        encoder_hidden_states_image = getattr(ctx, "encoder_hidden_states_image", None)
        negative_encoder_hidden_states_image = getattr(ctx, "negative_encoder_hidden_states_image", None)
        image_conditioning_latents = getattr(ctx, "image_conditioning_latents", None)
        first_frame_mask = getattr(ctx, "first_frame_mask", None)
        attention_kwargs = getattr(ctx, "attention_kwargs", None)

        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        timestep = self._prepare_training_timestep(sigma, batch_size, device, dtype) * self.TIMESTEP_SCALE

        with self._build_training_autocast_ctx(device):
            if active_guidance > 1.0:
                uncond_embeds = (
                    negative_prompt_embeds if negative_prompt_embeds is not None else torch.zeros_like(prompt_embeds)
                )
                latents_batched = torch.cat([latents, latents], dim=0)
                embeds_batched = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                timestep_batched = torch.cat([timestep, timestep], dim=0)
                hidden_states_batched = latents_batched.to(dtype)

                if first_frame_mask is not None and image_conditioning_latents is not None:
                    mask_batched = torch.cat([first_frame_mask, first_frame_mask], dim=0).to(device=device, dtype=dtype)
                    condition_batched = torch.cat(
                        [image_conditioning_latents, image_conditioning_latents],
                        dim=0,
                    ).to(device=device, dtype=dtype)
                    hidden_states_batched = (
                        1 - mask_batched
                    ) * condition_batched + mask_batched * hidden_states_batched
                    spatial_patch = self._spatial_patch_size()
                    timestep_template = (
                        first_frame_mask[0, 0, :, ::spatial_patch, ::spatial_patch].to(device=device, dtype=dtype)
                        * timestep[0]
                    ).flatten()
                    timestep_batched = timestep_template.unsqueeze(0).expand(latents_batched.shape[0], -1)
                elif image_conditioning_latents is not None:
                    hidden_states_batched = torch.cat(
                        [
                            hidden_states_batched,
                            torch.cat([image_conditioning_latents, image_conditioning_latents], dim=0).to(
                                device=device,
                                dtype=dtype,
                            ),
                        ],
                        dim=1,
                    )

                model_kwargs: Dict[str, Any] = {
                    "hidden_states": hidden_states_batched,
                    "encoder_hidden_states": embeds_batched,
                    "timestep": timestep_batched,
                    "return_dict": False,
                }
                if encoder_hidden_states_image is not None:
                    uncond_image_embeds = (
                        negative_encoder_hidden_states_image
                        if negative_encoder_hidden_states_image is not None
                        else encoder_hidden_states_image
                    )
                    model_kwargs["encoder_hidden_states_image"] = torch.cat(
                        [uncond_image_embeds, encoder_hidden_states_image],
                        dim=0,
                    )
                if attention_kwargs is not None:
                    model_kwargs["attention_kwargs"] = attention_kwargs

                # Call through self.transformer (WanDualTransformer) to trigger FSDP hooks
                noise_pred = self.transformer(use_high_noise=use_high_noise, **model_kwargs)[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                return noise_pred_uncond + active_guidance * (noise_pred_cond - noise_pred_uncond)

            # No CFG path — also routes through self.transformer for FSDP
            fwd_kwargs = self._prepare_forward_kwargs(latents, timestep, prompt_embeds, ctx)
            return self.transformer(use_high_noise=use_high_noise, **fwd_kwargs)[0]

    # ------------------------------------------------------------------
    # Spatial patch helper (needed for I2V timestep expansion)
    # ------------------------------------------------------------------

    def _spatial_patch_size(self) -> int:
        """Get spatial patch size from the high-noise transformer config.

        Overrides parent because self.transformer is WanDualTransformer (no .config.patch_size).
        Handles peft-wrapped models that hide config behind base_model.model.
        """
        config = getattr(self._high_noise_transformer, "config", None)
        if config is None:
            base = getattr(self._high_noise_transformer, "base_model", self._high_noise_transformer)
            base = getattr(base, "model", base)
            config = getattr(base, "config", None)
        patch_size = getattr(config, "patch_size", (1, 2, 2))
        if isinstance(patch_size, (list, tuple)):
            return int(patch_size[-1])
        return int(patch_size)

    # ------------------------------------------------------------------
    # No-split modules for FSDP
    # ------------------------------------------------------------------

    def get_no_split_modules(self) -> Tuple[Type[nn.Module], ...]:
        """Return block types for FSDP — same as Wan2.1.

        Only WanTransformerBlock is sharded individually. The parent
        WanTransformer3DModel modules do NOT need their own FSDP unit because
        all forward calls go through WanDualTransformer.forward() which is the
        FSDP root — its pre-forward hook all-gathers all parameters correctly.
        """
        try:
            from diffusers.models.transformers import transformer_wan

            block_types: list = []
            for name in ("WanTransformerBlock", "WanAttnProcessorBlock"):
                block = getattr(transformer_wan, name, None)
                if isinstance(block, type) and issubclass(block, nn.Module):
                    block_types.append(block)
            return tuple(block_types)
        except Exception:
            return tuple()


__all__ = [
    "WAN22ModelBundleConfig",
    "WAN22ModelBundle",
    "WanDualTransformer",
    "DEFAULT_BOUNDARY_RATIO",
]
