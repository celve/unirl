"""
Mochi Video Model Bundle.

Mochi is a video generation model from Genmo with:
- Single-stream Transformer architecture
- T5 text encoder
- 3D VAE for video

Reference: https://github.com/genmoai/mochi
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .base import ModelBundle
from .config import ModelBundleConfig
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(component_name="mochi", component_cfg=ModelBundleConfig)
class MochiModelBundle(ModelBundle):
    """
    Mochi video model bundle.

    Components:
    - transformer: Mochi Transformer (single-stream)
    - vae: 3D VAE for video encoding/decoding
    - text_encoder: T5 text encoder
    """

    def __init__(
        self,
        config: ModelBundleConfig,
    ):
        """
        Initialize Mochi model bundle.

        Args:
            config: Typed model-bundle config.
        """
        super().__init__(config)

        self.vae_ckpt_path = config.vae_ckpt_path or config.pretrained_model_ckpt_path
        self.text_encoder_ckpt_path = config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path

        # Text encoder components
        self._t5_encoder = None
        self._t5_tokenizer = None

        self.load()

    @property
    def model_type(self) -> str:
        return "mochi"

    @property
    def media_type(self) -> str:
        return "video"

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        # Mochi rollout is served by the dedicated SGLang engine in the current runtime.
        return "diffusionrl.samplers.sglang.engine.SGLangRolloutEngine"

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        # Default rollout engine is SGLang.
        return "sglang"

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
            raise ValueError("MochiModelBundle.forward_denoiser requires ctx.prompt_embeds.")
        negative_prompt_embeds = getattr(ctx, "negative_prompt_embeds", None)
        encoder_attention_mask = getattr(ctx, "encoder_attention_mask", None)
        attention_kwargs = getattr(ctx, "attention_kwargs", None)
        guidance_scale = float(getattr(ctx, "guidance_scale", 3.5))

        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        timestep_1000 = self._prepare_training_timestep(sigma, batch_size, device) * 1000
        model = self.transformer

        with self._build_training_autocast_ctx(device):
            if guidance_scale > 1.0:
                uncond_embeds = (
                    negative_prompt_embeds if negative_prompt_embeds is not None else torch.zeros_like(prompt_embeds)
                )
                model_kwargs: Dict[str, Any] = {
                    "hidden_states": torch.cat([latents, latents], dim=0).to(dtype),
                    "encoder_hidden_states": torch.cat([uncond_embeds, prompt_embeds], dim=0),
                    "timestep": torch.cat([timestep_1000, timestep_1000], dim=0),
                    "return_dict": False,
                }
                if encoder_attention_mask is not None:
                    model_kwargs["encoder_attention_mask"] = torch.cat(
                        [encoder_attention_mask, encoder_attention_mask], dim=0
                    )
                if attention_kwargs is not None:
                    model_kwargs["attention_kwargs"] = attention_kwargs
                noise_pred = model(**model_kwargs)[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            model_kwargs = {
                "hidden_states": latents.to(dtype),
                "encoder_hidden_states": prompt_embeds,
                "timestep": timestep_1000,
                "return_dict": False,
            }
            if encoder_attention_mask is not None:
                model_kwargs["encoder_attention_mask"] = encoder_attention_mask
            if attention_kwargs is not None:
                model_kwargs["attention_kwargs"] = attention_kwargs
            return model(**model_kwargs)[0]

    def load(self) -> None:
        """Load all model components."""
        logger.info("Loading Mochi model bundle...")

        # Load transformer
        self._load_transformer()

        # Load VAE
        self._load_vae()

        # Load text encoder
        self._load_text_encoder()

        logger.info("Mochi model bundle loaded")

    def _load_transformer(self) -> None:
        """Load the Mochi transformer model."""
        try:
            # Try diffusers implementation first
            from diffusers import MochiTransformer3DModel

            self._transformer = MochiTransformer3DModel.from_pretrained(
                self.pretrained_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            self._transformer.to(self.device)
            logger.info(f"Loaded Mochi transformer from {self.pretrained_path}")

        except ImportError:
            logger.warning(
                "Could not import MochiTransformer3DModel from diffusers. "
                "Please install diffusers>=0.31.0 with Mochi support."
            )
            self._transformer = None

        except Exception as e:
            logger.warning(f"Could not load Mochi transformer: {e}")
            self._transformer = None

    def _load_vae(self) -> None:
        """Load the VAE model."""
        try:
            from diffusers import AutoencoderKLMochi

            self._vae = AutoencoderKLMochi.from_pretrained(
                self.vae_ckpt_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()  # VAE is always in eval mode
            logger.info(f"Loaded Mochi VAE from {self.vae_ckpt_path}")

        except ImportError:
            logger.warning("Could not import AutoencoderKLMochi from diffusers.")
            self._vae = None

        except Exception as e:
            logger.warning(f"Could not load Mochi VAE: {e}")
            self._vae = None

    def _load_text_encoder(self) -> None:
        """Load the T5 text encoder."""
        try:
            from transformers import T5EncoderModel, T5Tokenizer

            # Load T5 encoder
            self._t5_encoder = T5EncoderModel.from_pretrained(
                self.text_encoder_ckpt_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
            )
            self._t5_encoder.to(self.device)
            self._t5_encoder.eval()

            # Load tokenizer
            self._t5_tokenizer = T5Tokenizer.from_pretrained(
                self.text_encoder_ckpt_path,
                subfolder="tokenizer",
            )

            # Wrap for unified interface
            self._text_encoder = MochiTextEncoderWrapper(
                encoder=self._t5_encoder,
                tokenizer=self._t5_tokenizer,
                device=self.device,
                dtype=self.text_encoder_dtype,
            )

            logger.info(f"Loaded T5 text encoder from {self.text_encoder_ckpt_path}")

        except Exception as e:
            logger.warning(f"Could not load T5 text encoder: {e}")
            self._text_encoder = None

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using T5 text encoder.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Optional negative prompt(s)

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        if self._text_encoder is None:
            raise RuntimeError("Text encoder not loaded")

        # Handle string input
        if isinstance(prompt, str):
            prompt = [prompt]

        return self._text_encoder.encode_prompt(prompt, negative_prompt)

    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents to video frames.

        Args:
            latents: Latent tensor [B, C, T, H, W]

        Returns:
            Video tensor [B, C, T, H', W']
        """
        if self._vae is None:
            raise RuntimeError("VAE not loaded")

        with torch.no_grad():
            # Mochi VAE expects specific latent format
            # Scale latents
            latents = latents / self._vae.config.scaling_factor

            # Decode
            video = self._vae.decode(latents).sample

        return video

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images/video to latents.

        Args:
            images: Video tensor [B, C, T, H, W]

        Returns:
            Latent tensor [B, C', T', H', W']
        """
        if self._vae is None:
            raise RuntimeError("VAE not loaded")

        with torch.no_grad():
            # Encode
            latents = self._vae.encode(images).latent_dist.sample()

            # Scale
            latents = latents * self._vae.config.scaling_factor

        return latents

    def get_no_split_modules(self) -> Tuple[Type[nn.Module], ...]:
        """
        Get module types that should not be split in FSDP.

        For Mochi, we don't split the transformer blocks.
        """
        try:
            from diffusers.models.transformers.transformer_mochi import (
                MochiTransformerBlock,
            )

            return (MochiTransformerBlock,)
        except ImportError:
            return ()

    def get_sigma_schedule(
        self,
        num_steps: int,
        shift: float = 4.0,  # Mochi default shift
    ) -> torch.Tensor:
        """
        Get sigma schedule for Mochi.

        Mochi uses a slightly different shift value by default.
        """
        from diffusionrl.sde.runtime import get_sigma_schedule

        return get_sigma_schedule(num_steps, shift, self.device)


class MochiTextEncoderWrapper:
    """
    Wrapper for Mochi's T5 text encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
        max_length: int = 256,
    ):
        """
        Initialize text encoder wrapper.

        Args:
            encoder: T5 encoder model
            tokenizer: T5 tokenizer
            device: Device
            dtype: Data type
            max_length: Maximum sequence length
        """
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts to embeddings.

        Args:
            prompt: List of text prompts
            negative_prompt: Optional negative prompts

        Returns:
            Tuple of (prompt_embeds, attention_mask)
        """
        if self.encoder is None:
            raise RuntimeError("MochiTextEncoderWrapper is not initialized: T5 encoder is unavailable.")

        with torch.no_grad():
            # Tokenize
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)

            # Encode
            prompt_embeds = self.encoder(
                input_ids=text_input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

            prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        # Handle negative prompts
        if negative_prompt is not None:
            neg_embeds, neg_mask = self.encode_prompt(negative_prompt, None)
            # Concatenate for classifier-free guidance
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            attention_mask = torch.cat([neg_mask, attention_mask], dim=0)

        return prompt_embeds, attention_mask

    def to(self, device: Union[str, torch.device]) -> "MochiTextEncoderWrapper":
        """Move encoder to device."""
        self.device = device
        if self.encoder is not None:
            self.encoder.to(device)
        return self
