"""
FLUX Image Model Bundle.

FLUX is an image generation model from Black Forest Labs with:
- Single-stream Transformer architecture (MMDiT)
- T5 + CLIP text encoders
- VAE for image encoding/decoding

Reference: https://github.com/black-forest-labs/flux
"""
import logging
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .base import ModelBundle

logger = logging.getLogger(__name__)


class FluxModelBundle(ModelBundle):
    """
    FLUX image model bundle.

    Components:
    - transformer: FLUX Transformer (single-stream MMDiT)
    - vae: VAE for image encoding/decoding
    - text_encoder: T5 + CLIP dual encoder
    """

    def __init__(
        self,
        pretrained_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        vae_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        text_encoder_2_path: Optional[str] = None,
        variant: str = "dev",  # "dev" or "schnell"
        load_on_init: bool = True,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        training_only: bool = False,
        skip_device_move: bool = False,
        **kwargs,
    ):
        """
        Initialize FLUX model bundle.

        Args:
            pretrained_path: Path to pretrained transformer weights
            device: Device to load models on
            dtype: Data type for transformer weights
            vae_path: Optional separate path for VAE
            text_encoder_path: Optional separate path for CLIP encoder
            text_encoder_2_path: Optional separate path for T5 encoder
            variant: FLUX variant ("dev" or "schnell")
            load_on_init: Whether to load models immediately
            use_lora: Whether to add LoRA adapters for training
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_target_modules: Target modules for LoRA (None uses defaults)
            training_only: If True, only load transformer (skip VAE, text encoders)
                          This saves significant memory for training actors.
            skip_device_move: If True, don't move model to device after loading.
                             Required for FSDP CPU offload mode.
            **kwargs: Additional arguments
        """
        super().__init__(pretrained_path, device, dtype, **kwargs)

        self.vae_path = vae_path or pretrained_path
        self.text_encoder_path = text_encoder_path or pretrained_path
        self.text_encoder_2_path = text_encoder_2_path or pretrained_path
        self.variant = variant
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.training_only = training_only
        self.skip_device_move = skip_device_move

        # Text encoder components
        self._clip_encoder = None
        self._clip_tokenizer = None
        self._t5_encoder = None
        self._t5_tokenizer = None

        if load_on_init:
            self.load()

    @property
    def model_type(self) -> str:
        return "flux"

    def load(self) -> None:
        """Load all model components."""
        if self.training_only:
            logger.info(f"Loading FLUX transformer only (training mode, variant={self.variant})...")
        else:
            logger.info(f"Loading FLUX model bundle (variant={self.variant})...")

        # Load transformer - always needed
        self._load_transformer()

        # Skip VAE and text encoders in training_only mode
        if not self.training_only:
            # Load VAE
            self._load_vae()

            # Load text encoders
            self._load_text_encoders()

        # Add LoRA adapters if requested
        if self.use_lora:
            self._add_lora_adapters()

        if self.training_only:
            logger.info("FLUX transformer loaded (training_only mode)")
        else:
            logger.info("FLUX model bundle loaded")

    def _load_transformer(self) -> None:
        """Load the FLUX transformer model."""
        try:
            from diffusers import FluxTransformer2DModel

            self._transformer = FluxTransformer2DModel.from_pretrained(
                self.pretrained_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            # Check device after loading
            first_param_device = next(self._transformer.parameters()).device
            logger.info(f"FLUX transformer loaded on device: {first_param_device}")

            # Only move to device if not using FSDP CPU offload
            if not self.skip_device_move:
                self._transformer.to(self.device)
                logger.info(f"Moved FLUX transformer to {self.device}")
            else:
                # Ensure model stays on CPU for FSDP CPU offload
                if str(first_param_device) != 'cpu':
                    self._transformer.to('cpu')
                    logger.info(f"Moved FLUX transformer to CPU for FSDP CPU offload")
            logger.info(f"Loaded FLUX transformer from {self.pretrained_path} (skip_device_move={self.skip_device_move})")

        except ImportError:
            logger.warning(
                "Could not import FluxTransformer2DModel from diffusers. "
                "Please install diffusers>=0.30.0 with FLUX support."
            )
            self._transformer = None

        except Exception as e:
            logger.warning(f"Could not load FLUX transformer: {e}")
            self._transformer = None

    def _load_vae(self) -> None:
        """Load the VAE model."""
        try:
            from diffusers import AutoencoderKL

            self._vae = AutoencoderKL.from_pretrained(
                self.vae_path,
                subfolder="vae",
                torch_dtype=self.dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()
            logger.info(f"Loaded FLUX VAE from {self.vae_path}")

        except ImportError:
            logger.warning("Could not import AutoencoderKL from diffusers.")
            self._vae = None

        except Exception as e:
            logger.warning(f"Could not load FLUX VAE: {e}")
            self._vae = None

    def _load_text_encoders(self) -> None:
        """Load the CLIP and T5 text encoders."""
        try:
            from transformers import (
                CLIPTextModel,
                CLIPTokenizer,
                T5EncoderModel,
                T5TokenizerFast,
            )

            # Load CLIP encoder
            self._clip_encoder = CLIPTextModel.from_pretrained(
                self.text_encoder_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            )
            self._clip_encoder.to(self.device)
            self._clip_encoder.eval()

            self._clip_tokenizer = CLIPTokenizer.from_pretrained(
                self.text_encoder_path,
                subfolder="tokenizer",
            )

            # Load T5 encoder
            self._t5_encoder = T5EncoderModel.from_pretrained(
                self.text_encoder_2_path,
                subfolder="text_encoder_2",
                torch_dtype=self.dtype,
            )
            self._t5_encoder.to(self.device)
            self._t5_encoder.eval()

            self._t5_tokenizer = T5TokenizerFast.from_pretrained(
                self.text_encoder_2_path,
                subfolder="tokenizer_2",
            )

            # Create wrapper
            self._text_encoder = FluxTextEncoderWrapper(
                clip_encoder=self._clip_encoder,
                clip_tokenizer=self._clip_tokenizer,
                t5_encoder=self._t5_encoder,
                t5_tokenizer=self._t5_tokenizer,
                device=self.device,
                dtype=self.dtype,
            )

            logger.info("Loaded FLUX text encoders")

        except Exception as e:
            logger.warning(f"Could not load text encoders: {e}")
            self._text_encoder = None

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to transformer for efficient fine-tuning."""
        if self._transformer is None:
            logger.warning("Cannot add LoRA: transformer not loaded")
            return

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install with: pip install peft")
            return

        # Default target modules for FLUX transformer
        target_modules = self.lora_target_modules
        if target_modules is None:
            target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",  # Attention
                "proj_in", "proj_out",  # Projections
            ]

        # LoRA config for FLUX transformer
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

        logger.info(f"LoRA adapters added to FLUX (rank={self.lora_rank}, alpha={self.lora_alpha})")

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using CLIP + T5 encoders.

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
        Decode latents to images.

        Args:
            latents: Latent tensor [B, C, H, W]

        Returns:
            Image tensor [B, C, H', W']
        """
        if self._vae is None:
            raise RuntimeError("VAE not loaded")

        with torch.no_grad():
            # FLUX VAE scaling factor
            # Ensure latents match VAE dtype to avoid "Input type and bias type should be the same" errors
            latents = latents.to(dtype=self._dtype) / self._vae.config.scaling_factor

            # Decode
            images = self._vae.decode(latents, return_dict=False)[0]

        return images

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images to latents.

        Args:
            images: Image tensor [B, C, H, W]

        Returns:
            Latent tensor [B, C', H', W']
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

        For FLUX, we don't split the transformer blocks.
        """
        try:
            from diffusers.models.transformers.transformer_flux import (
                FluxSingleTransformerBlock,
                FluxTransformerBlock,
            )
            return (FluxSingleTransformerBlock, FluxTransformerBlock)
        except ImportError:
            return ()

    def get_sigma_schedule(
        self,
        num_steps: int,
        shift: float = 1.0,  # FLUX uses different shift than video models
    ) -> torch.Tensor:
        """
        Get sigma schedule for FLUX.

        FLUX dev uses shift=1.0, schnell can use shift=0.0.
        """
        from diffusionrl.samplers.log_prob import get_sigma_schedule
        return get_sigma_schedule(num_steps, shift, self.device)


class FluxTextEncoderWrapper:
    """
    Wrapper for FLUX's dual text encoder (CLIP + T5).
    """

    def __init__(
        self,
        clip_encoder: nn.Module,
        clip_tokenizer,
        t5_encoder: nn.Module,
        t5_tokenizer,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,  # T5 max length for FLUX
    ):
        """
        Initialize text encoder wrapper.

        Args:
            clip_encoder: CLIP text encoder
            clip_tokenizer: CLIP tokenizer
            t5_encoder: T5 text encoder
            t5_tokenizer: T5 tokenizer
            device: Device
            dtype: Data type
            max_length: Maximum T5 sequence length
        """
        self.clip_encoder = clip_encoder
        self.clip_tokenizer = clip_tokenizer
        self.t5_encoder = t5_encoder
        self.t5_tokenizer = t5_tokenizer
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
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        batch_size = len(prompt)

        # CLIP encoding for pooled embeddings
        pooled_prompt_embeds = self._encode_clip(prompt)

        # T5 encoding for sequence embeddings
        prompt_embeds = self._encode_t5(prompt)

        # Handle negative prompts for FLUX (typically not used)
        if negative_prompt is not None:
            neg_pooled = self._encode_clip(negative_prompt)
            neg_embeds = self._encode_t5(negative_prompt)
            pooled_prompt_embeds = torch.cat([neg_pooled, pooled_prompt_embeds], dim=0)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds

    def _encode_clip(self, texts: List[str]) -> torch.Tensor:
        """Encode with CLIP for pooled embeddings."""
        if self.clip_encoder is None:
            batch_size = len(texts)
            return torch.zeros(batch_size, 768, dtype=self.dtype, device=self.device)

        with torch.no_grad():
            text_inputs = self.clip_tokenizer(
                texts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)

            clip_output = self.clip_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

            # Use pooled output
            pooled_embeds = clip_output.pooler_output.to(dtype=self.dtype)

        return pooled_embeds

    def _encode_t5(self, texts: List[str]) -> torch.Tensor:
        """Encode with T5 for sequence embeddings."""
        if self.t5_encoder is None:
            batch_size = len(texts)
            return torch.zeros(
                batch_size, self.max_length, 4096,
                dtype=self.dtype, device=self.device
            )

        with torch.no_grad():
            text_inputs = self.t5_tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)

            t5_output = self.t5_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            prompt_embeds = t5_output.last_hidden_state.to(dtype=self.dtype)

        return prompt_embeds

    def to(self, device: Union[str, torch.device]) -> "FluxTextEncoderWrapper":
        """Move encoders to device."""
        self.device = device
        if self.clip_encoder is not None:
            self.clip_encoder.to(device)
        if self.t5_encoder is not None:
            self.t5_encoder.to(device)
        return self


class FluxPipeline:
    """
    Simplified FLUX pipeline for sampling.

    Provides a clean interface for generating image samples.
    """

    def __init__(
        self,
        model_bundle: FluxModelBundle,
        scheduler=None,
    ):
        """
        Initialize pipeline.

        Args:
            model_bundle: FluxModelBundle instance
            scheduler: Optional noise scheduler
        """
        self.model_bundle = model_bundle
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,  # FLUX dev default
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate image from prompt.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Optional negative prompt(s)
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (note: FLUX uses different guidance mechanism)
            generator: Optional random generator

        Returns:
            Generated image tensor [B, C, H, W]
        """
        # Handle string input
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Encode prompt
        prompt_embeds, pooled_prompt_embeds = self.model_bundle.encode_prompt(
            prompt, negative_prompt
        )

        # Calculate latent dimensions (FLUX VAE has 8x compression)
        latent_height = height // 8
        latent_width = width // 8

        # FLUX uses 16 latent channels
        latent_channels = 16

        # Initialize latents
        latents = torch.randn(
            batch_size, latent_channels, latent_height, latent_width,
            device=self.model_bundle.device,
            dtype=self.model_bundle.dtype,
            generator=generator,
        )

        # Get sigma schedule
        sigmas = self.model_bundle.get_sigma_schedule(num_inference_steps)

        # Prepare image ids for positional encoding
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, latent_height, latent_width, self.model_bundle.device, self.model_bundle.dtype
        )

        # Prepare text ids
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.model_bundle.device)

        # Denoising loop
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            # Pack latents for FLUX transformer
            packed_latents = self._pack_latents(latents, latent_height, latent_width)

            # Create timestep
            timestep = sigma.expand(batch_size)

            # Forward pass
            noise_pred = self.model_bundle.transformer(
                hidden_states=packed_latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=timestep,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=torch.tensor([guidance_scale], device=self.model_bundle.device).expand(batch_size),
                return_dict=False,
            )[0]

            # Unpack
            noise_pred = self._unpack_latents(noise_pred, latent_height, latent_width)

            # Euler step
            dt = sigma_next - sigma
            latents = latents + dt * noise_pred

        # Decode
        images = self.model_bundle.decode_latents(latents)

        return images

    def _prepare_latent_image_ids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare latent image IDs for FLUX positional encoding."""
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = torch.arange(height)[:, None]
        latent_image_ids[..., 2] = torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def _pack_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Pack latents into sequence format for FLUX transformer."""
        batch_size, channels, _, _ = latents.shape
        latents = latents.view(batch_size, channels, height, width)
        latents = latents.permute(0, 2, 3, 1)  # B, H, W, C
        latents = latents.reshape(batch_size, height * width, channels)
        return latents

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Unpack latents from sequence format back to spatial format."""
        batch_size, _, channels = latents.shape
        latents = latents.view(batch_size, height, width, channels)
        latents = latents.permute(0, 3, 1, 2)  # B, C, H, W
        return latents
