"""
HunyuanVideo Model Bundle.

Reference: DanceGRPO implementation
"""

import logging
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .base import ModelBundle
from .config import ModelBundleConfig
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(component_name="hunyuan", component_cfg=ModelBundleConfig)
class HunyuanModelBundle(ModelBundle):
    """
    HunyuanVideo model bundle.

    Components:
    - transformer: HunyuanVideo transformer (dual-stream)
    - vae: 3D VAE for video encoding/decoding
    - text_encoder: LLAMA + CLIP text encoder wrapper
    """

    def __init__(
        self,
        config: ModelBundleConfig,
    ):
        """
        Initialize HunyuanVideo model bundle.

        Args:
            config: Typed model-bundle config.
        """
        super().__init__(config)

        self.vae_ckpt_path = config.vae_ckpt_path or config.pretrained_model_ckpt_path
        self.text_encoder_ckpt_path = config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path
        self.load()

    @property
    def model_type(self) -> str:
        return "hunyuan"

    @property
    def media_type(self) -> str:
        return "video"

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        return "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler"

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        return "fsdp"

    @classmethod
    def forward_plugin(cls):
        from diffusionrl.models.forward_plugins import HunyuanForwardPlugin

        return HunyuanForwardPlugin()

    @classmethod
    def supports_sglang_prompt_mode(cls) -> bool:
        return True

    def load(self) -> None:
        """Load all model components."""
        logger.info("Loading HunyuanVideo model bundle...")

        # Load transformer
        self._load_transformer()

        # Load VAE
        self._load_vae()

        # Load text encoder
        self._load_text_encoder()

        logger.info("HunyuanVideo model bundle loaded")

    def _load_transformer(self) -> None:
        """Load the transformer model."""
        try:
            # Try to import from diffusers
            from diffusers.models.transformers.transformer_hunyuan_video import (
                HunyuanVideoTransformer3DModel,
            )

            self._transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                self.pretrained_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            self._transformer.to(self.device)
            logger.info(f"Loaded transformer from {self.pretrained_path}")

        except ImportError:
            logger.warning(
                "Could not import HunyuanVideoTransformer3DModel from diffusers. "
                "Please install diffusers with HunyuanVideo support."
            )
            self._transformer = None

        except Exception as e:
            logger.warning(f"Could not load transformer: {e}")
            self._transformer = None

    def _load_vae(self) -> None:
        """Load the VAE model."""
        try:
            from diffusers import AutoencoderKLHunyuanVideo

            self._vae = AutoencoderKLHunyuanVideo.from_pretrained(
                self.vae_ckpt_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()  # VAE is always in eval mode
            logger.info(f"Loaded VAE from {self.vae_ckpt_path}")

        except ImportError:
            logger.warning("Could not import AutoencoderKLHunyuanVideo from diffusers.")
            self._vae = None

        except Exception as e:
            logger.warning(f"Could not load VAE: {e}")
            self._vae = None

    def _load_text_encoder(self) -> None:
        """Load the text encoder."""
        try:
            # HunyuanVideo uses LLAMA + CLIP dual encoder
            self._text_encoder = HunyuanTextEncoderWrapper(
                pretrained_path=self.text_encoder_ckpt_path,
                device=self.device,
                dtype=self.text_encoder_dtype,
            )
            logger.info(f"Loaded text encoder from {self.text_encoder_ckpt_path}")

        except Exception as e:
            logger.warning(f"Could not load text encoder: {e}")
            self._text_encoder = None

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using the dual text encoder.

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

        For HunyuanVideo, we don't split the transformer blocks.
        """
        try:
            from diffusers.models.transformers.transformer_hunyuan_video import (
                HunyuanVideoSingleTransformerBlock,
                HunyuanVideoTransformerBlock,
            )

            return (HunyuanVideoTransformerBlock, HunyuanVideoSingleTransformerBlock)
        except ImportError:
            return ()


class HunyuanTextEncoderWrapper:
    """
    Wrapper for HunyuanVideo's dual text encoder (LLAMA + CLIP).
    """

    def __init__(
        self,
        pretrained_path: str,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize text encoder wrapper.

        Args:
            pretrained_path: Path to pretrained weights
            device: Device to load on
            dtype: Data type
        """
        self.pretrained_path = pretrained_path
        self.device = device
        self.dtype = dtype

        self.llama_encoder = None
        self.clip_encoder = None
        self.tokenizer = None
        self.clip_tokenizer = None

        self._load_encoders()

    def _load_encoders(self) -> None:
        """Load the LLAMA and CLIP encoders."""
        try:
            from transformers import (
                CLIPTextModel,
                CLIPTokenizer,
                LlamaModel,
                LlamaTokenizer,
            )

            # Load LLAMA
            self.llama_encoder = LlamaModel.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            )
            self.llama_encoder.to(self.device)
            self.llama_encoder.eval()

            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.pretrained_path,
                subfolder="tokenizer",
            )

            # Load CLIP
            self.clip_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder_2",
                torch_dtype=self.dtype,
            )
            self.clip_encoder.to(self.device)
            self.clip_encoder.eval()

            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained_path,
                subfolder="tokenizer_2",
            )

        except Exception as e:
            logger.warning(f"Could not load text encoders: {e}")

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
        if self.llama_encoder is None or self.clip_encoder is None:
            raise RuntimeError("HunyuanTextEncoderWrapper is not initialized: LLAMA/CLIP encoders are unavailable.")

        with torch.no_grad():
            # LLAMA encoding
            llama_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            llama_inputs = {k: v.to(self.device) for k, v in llama_inputs.items()}

            llama_output = self.llama_encoder(**llama_inputs)
            prompt_embeds = llama_output.last_hidden_state

            # CLIP encoding for pooled embeddings
            clip_inputs = self.clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}

            clip_output = self.clip_encoder(**clip_inputs)
            pooled_embeds = clip_output.pooler_output

        return prompt_embeds, pooled_embeds

    def to(self, device: Union[str, torch.device]) -> "HunyuanTextEncoderWrapper":
        """Move encoders to device."""
        self.device = device
        if self.llama_encoder is not None:
            self.llama_encoder.to(device)
        if self.clip_encoder is not None:
            self.clip_encoder.to(device)
        return self
