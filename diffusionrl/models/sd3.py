"""
SD3 (Stable Diffusion 3) Model Bundle for diffusionrl.

Supports SD3 and SD3.5 models for image generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import ModelBundle
from .config import ModelBundleConfig
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(component_cfg=ModelBundleConfig)
class SD3ModelBundle(ModelBundle):
    """
    Model bundle for Stable Diffusion 3 models.

    Supports:
    - stabilityai/stable-diffusion-3-medium
    - stabilityai/stable-diffusion-3.5-medium
    - stabilityai/stable-diffusion-3.5-large

    Example:
        bundle = SD3ModelBundle(
            config=ModelBundleConfig(
                pretrained_model_ckpt_path="stabilityai/stable-diffusion-3-medium",
            ),
        )
        prompt_embeds, pooled = bundle.encode_prompt(["a cat"])
    """

    def __init__(
        self,
        config: ModelBundleConfig,
    ):
        """
        Initialize SD3 model bundle.

        Args:
            config: Typed model-bundle config. If training_only is True, only
                load transformer (skip VAE, text encoders).
                          This saves significant memory for training actors.
        """
        super().__init__(config)

        self.pretrained_path = config.pretrained_model_ckpt_path
        self.device = torch.device(self.device)
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.training_only = config.training_only

        self._transformer = None
        self._text_encoder = None
        self._text_encoder_2 = None
        self._text_encoder_3 = None
        self._tokenizer = None
        self._tokenizer_2 = None
        self._tokenizer_3 = None
        self._vae = None
        self._scheduler = None

        self._load_models()

    def _load_models(self) -> None:
        """Load SD3 models from pretrained path."""
        if self.training_only:
            logger.info(f"Loading SD3 transformer only (training mode) from {self.pretrained_path}")
        else:
            logger.info(f"Loading SD3 model from {self.pretrained_path}")

        # Load transformer (DiT) - always needed
        self._load_transformer()

        # Skip VAE and text encoders in training_only mode
        if not self.training_only:
            self._load_vae()
            self._load_text_encoders()
            self._load_scheduler()

        # Add LoRA if requested
        if self.use_lora:
            self._add_lora_adapters()

        if self.training_only:
            logger.info("SD3 transformer loaded successfully (training_only mode)")
        else:
            logger.info("SD3 model loaded successfully")

    def _load_transformer(self) -> None:
        """Load SD3 transformer (DiT)."""
        try:
            from diffusers import SD3Transformer2DModel
        except ImportError as e:
            raise ImportError(
                "SD3 requires diffusers and transformers. Install with: pip install diffusers transformers"
            ) from e

        self._transformer = SD3Transformer2DModel.from_pretrained(
            self.pretrained_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        ).to(self.device)

    def _load_vae(self) -> None:
        """Load SD3 VAE."""
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:
            raise ImportError(
                "SD3 requires diffusers and transformers. Install with: pip install diffusers transformers"
            ) from e

        if self._vae is not None:
            return

        self._vae = AutoencoderKL.from_pretrained(
            self.pretrained_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
        ).to(self.device)
        self._vae.eval()
        self._vae.requires_grad_(False)

    def _load_text_encoders(self) -> None:
        """Load SD3 text encoders and tokenizers."""
        try:
            from transformers import (
                CLIPTextModelWithProjection,
                CLIPTokenizer,
                T5EncoderModel,
                T5TokenizerFast,
            )
        except ImportError as e:
            raise ImportError(
                "SD3 requires diffusers and transformers. Install with: pip install diffusers transformers"
            ) from e

        if self._text_encoder is None:
            self._text_encoder = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
            ).to(self.device)
        if self._text_encoder_2 is None:
            self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder_2",
                torch_dtype=self.text_encoder_dtype,
            ).to(self.device)
        if self._text_encoder_3 is None:
            self._text_encoder_3 = T5EncoderModel.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder_3",
                torch_dtype=self.text_encoder_dtype,
            ).to(self.device)

        if self._tokenizer is None:
            self._tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_path, subfolder="tokenizer")
        if self._tokenizer_2 is None:
            self._tokenizer_2 = CLIPTokenizer.from_pretrained(self.pretrained_path, subfolder="tokenizer_2")
        if self._tokenizer_3 is None:
            self._tokenizer_3 = T5TokenizerFast.from_pretrained(self.pretrained_path, subfolder="tokenizer_3")

        # Set eval mode for non-trainable components
        self._text_encoder.eval()
        self._text_encoder_2.eval()
        self._text_encoder_3.eval()

        # Disable gradients for non-trainable components
        self._text_encoder.requires_grad_(False)
        self._text_encoder_2.requires_grad_(False)
        self._text_encoder_3.requires_grad_(False)

    def _load_scheduler(self) -> None:
        """Load SD3 scheduler."""
        try:
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        except ImportError as e:
            raise ImportError(
                "SD3 requires diffusers and transformers. Install with: pip install diffusers transformers"
            ) from e

        if self._scheduler is not None:
            return

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.pretrained_path, subfolder="scheduler")

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to transformer for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install with: pip install peft")
            return

        # LoRA config for SD3 transformer
        target_modules = self.lora_target_modules
        if target_modules is None:
            # Fallback for non-CLI code paths (direct ModelBundle construction).
            # The CLI parser materialises this via ``default_lora_target_modules()``
            # at config-build time; this branch only triggers when someone
            # bypasses the CLI and constructs a ``ModelBundleConfig`` by hand.
            target_modules = type(self).default_lora_target_modules()
            if target_modules is None:
                raise RuntimeError(
                    "SD3 LoRA requested but lora_target_modules is unresolved. "
                    "Pass --training.lora-target-modules or ensure "
                    f"{type(self).__name__}.default_lora_target_modules() returns a list."
                )
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        self._transformer = get_peft_model(self._transformer, lora_config)

        # Add second adapter for NFT dual-adapter mechanism
        self._transformer.add_adapter("old", lora_config)
        self._transformer.set_adapter("default")

        # Ensure LoRA weights have the same dtype as base model (important for FSDP)
        for param in self._transformer.parameters():
            if param.requires_grad and param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)

        logger.info(f"LoRA adapters added (rank={self.lora_rank}, alpha={self.lora_alpha})")

    @property
    def model_type(self) -> str:
        return "sd3"

    @property
    def media_type(self) -> str:
        return "t2i"

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        return "diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler"

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        return "fsdp"

    @classmethod
    def default_lora_target_modules(cls) -> Optional[List[str]]:
        """Flow-GRPO canonical LoRA targets for SD3 joint-attention blocks.

        Matches the 8 keys injected by PEFT in ``_add_lora_adapters``.  This
        list is threaded all the way to SGLang ``ServerArgs.lora_target_modules``
        so the rollout engine only wraps layers the training side actually
        updates — otherwise SGLang defaults to wrapping every linear layer,
        producing a wall of ``LoRA adapter None does not contain the weights
        for layer '...'`` warnings and ``disable_lora=True`` for the unmatched
        layers.
        """
        return [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]

    @classmethod
    def supports_sglang_prompt_mode(cls) -> bool:
        return True

    def forward_denoiser(
        self,
        *,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        prompt_embeds = ctx.prompt_embeds
        if prompt_embeds is None:
            raise ValueError("SD3ModelBundle.forward_denoiser requires ctx.prompt_embeds.")
        pooled_prompt_embeds = getattr(ctx, "pooled_prompt_embeds", None)
        negative_prompt_embeds = getattr(ctx, "negative_prompt_embeds", None)
        negative_pooled_prompt_embeds = getattr(ctx, "negative_pooled_prompt_embeds", None)
        encoder_attention_mask = getattr(ctx, "encoder_attention_mask", None)
        guidance_scale = float(getattr(ctx, "guidance_scale", 3.5))

        batch_size = latents.shape[0]
        device = latents.device
        timestep = self._prepare_training_timestep(sigma, batch_size, device)
        timestep_1000 = timestep * 1000
        model = self.transformer

        with self._build_training_autocast_ctx(device):
            if guidance_scale > 1.0:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                if negative_pooled_prompt_embeds is None:
                    negative_pooled_prompt_embeds = (
                        torch.zeros_like(pooled_prompt_embeds) if pooled_prompt_embeds is not None else None
                    )
                attn_kw: Dict[str, Any] = {}
                if encoder_attention_mask is not None:
                    attn_kw["encoder_attention_mask"] = torch.cat(
                        [encoder_attention_mask, encoder_attention_mask], dim=0
                    )
                noise_pred = model(
                    hidden_states=torch.cat([latents, latents], dim=0),
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                    timestep=torch.cat([timestep_1000, timestep_1000], dim=0),
                    pooled_projections=(
                        torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        if pooled_prompt_embeds is not None
                        else None
                    ),
                    return_dict=False,
                    **attn_kw,
                )[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            model_kwargs: Dict[str, Any] = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "timestep": timestep_1000,
                "return_dict": False,
            }
            if pooled_prompt_embeds is not None:
                model_kwargs["pooled_projections"] = pooled_prompt_embeds
            if encoder_attention_mask is not None:
                model_kwargs["encoder_attention_mask"] = encoder_attention_mask
            return model(**model_kwargs)[0]

    @property
    def transformer(self) -> nn.Module:
        """Get the transformer (DiT) model."""
        return self._transformer

    @property
    def text_encoder(self) -> nn.Module:
        """Get the primary text encoder (CLIP)."""
        return self._text_encoder

    @property
    def vae(self) -> nn.Module:
        """Get the VAE."""
        return self._vae

    @property
    def scheduler(self):
        """Get the noise scheduler."""
        return self._scheduler

    def get_sampler_extra_kwargs(self) -> Dict[str, Any]:
        """Inject SD3 scheduler into sampler when available."""
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is None:
            return {}
        return {"scheduler": scheduler}

    def _require_text_encoding_components(self) -> None:
        missing = [
            name
            for name in (
                "_text_encoder",
                "_text_encoder_2",
                "_text_encoder_3",
                "_tokenizer",
                "_tokenizer_2",
                "_tokenizer_3",
            )
            if getattr(self, name, None) is None
        ]
        if missing:
            self._raise_aux_component_not_loaded("text encoders/tokenizers", missing=missing)

    def _require_vae(self) -> None:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")

    def encode_prompt(
        self,
        prompts: List[str],
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to embeddings.

        Args:
            prompts: List of text prompts
            max_sequence_length: Maximum sequence length for T5

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
            - prompt_embeds: [B, seq_len, hidden_dim] concatenated embeddings
            - pooled_prompt_embeds: [B, pooled_dim] pooled CLIP embeddings
        """
        self._require_text_encoding_components()

        # Encode with CLIP text encoders
        # CLIP 1
        text_inputs = self._tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            clip_output_1 = self._text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
            pooled_1 = clip_output_1.text_embeds

        # CLIP 2
        text_inputs_2 = self._tokenizer_2(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)

        with torch.no_grad():
            clip_output_2 = self._text_encoder_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            pooled_2 = clip_output_2.text_embeds

        # Concatenate CLIP embeddings (SD3 transformer uses T5 only; CLIP contributes pooled_embeds)
        pooled_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

        # T5 encoding
        text_inputs_3 = self._tokenizer_3(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_3 = text_inputs_3.input_ids.to(self.device)

        with torch.no_grad():
            t5_output = self._text_encoder_3(text_input_ids_3)
            t5_embeds = t5_output.last_hidden_state

        # SD3 transformer expects:
        # - encoder_hidden_states: T5 embeddings only [B, seq_len, 4096]
        # - pooled_projections: CLIP pooled embeddings [B, 2048]
        # The context_embedder in SD3 maps 4096 -> 1536
        prompt_embeds = t5_embeds

        return prompt_embeds, pooled_embeds

    def encode_inputs(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode SD3 prompts for inference, including CFG negative branch.
        """
        if image is not None:
            raise NotImplementedError(f"{type(self).__name__} does not support image conditioning")
        if video is not None:
            raise NotImplementedError(f"{type(self).__name__} does not support video conditioning")

        prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
        output = self._encode_prompt_to_input_dict(prompts_list, **kwargs)

        if negative_prompt is None:
            negative_prompts = [""] * len(prompts_list)
        elif isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt] * len(prompts_list)
        else:
            negative_prompts = list(negative_prompt)
            if len(negative_prompts) != len(prompts_list):
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompts)} does not match prompt batch size {len(prompts_list)}"
                )

        negative_output = self._encode_prompt_to_input_dict(negative_prompts, **kwargs)
        output["negative_prompt_embeds"] = negative_output.get("prompt_embeds")
        output["negative_pooled_prompt_embeds"] = negative_output.get("pooled_prompt_embeds")
        return output

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: [B, C, H, W] images in [-1, 1] range

        Returns:
            [B, C, H//8, W//8] latents
        """
        self._require_vae()

        with torch.no_grad():
            latents = self._vae.encode(images).latent_dist.sample()
            latents = latents * self._vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: [B, C, H//8, W//8] latents

        Returns:
            [B, C, H, W] images in [-1, 1] range
        """
        self._require_vae()

        # VAE decode doesn't support bfloat16 ("Got unsupported ScalarType BFloat16")
        # Convert VAE and latents to float32 for decoding
        latents = latents.to(dtype=torch.float32) / self._vae.config.scaling_factor
        with torch.no_grad():
            images = self._vae.to(torch.float32).decode(latents).sample
        return images

    def load(self) -> None:
        """Load all model components (already done in __init__)."""
        # Models are loaded in __init__ via _load_models()
        pass

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: [B, C, H, W] images in [-1, 1] range

        Returns:
            [B, C, H//8, W//8] latents
        """
        return self.encode_image(images)

    def get_no_split_modules(self) -> Tuple[type, ...]:
        """
        Get module types that should not be split in FSDP.

        Returns:
            Tuple of module types
        """
        results: list[type] = []
        # SD3.5 uses JointTransformerBlock
        try:
            from diffusers.models.transformers.transformer_sd3 import (
                JointTransformerBlock,
            )

            results.append(JointTransformerBlock)
        except ImportError:
            pass
        # Older diffusers may use SD3TransformerBlock
        try:
            from diffusers.models.transformers.transformer_sd3 import (
                SD3TransformerBlock,
            )

            results.append(SD3TransformerBlock)
        except ImportError:
            pass
        return tuple(results)
