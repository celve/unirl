"""
diffusionrl Model Bundle Base Class.

Defines the interface for model bundles that package transformer, VAE, and text encoder.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn


class ModelBundle(ABC):
    """
    Base class for model bundles.

    A model bundle packages all components needed for generation:
    - transformer: The main diffusion model
    - vae: Variational autoencoder for encoding/decoding
    - text_encoder: Text encoder for prompt processing

    Each model type (HunyuanVideo, Mochi, FLUX, etc.) implements
    this interface with its specific components.
    """

    def __init__(
        self,
        pretrained_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """
        Initialize model bundle.

        Args:
            pretrained_path: Path to pretrained model weights
            device: Device to load models on
            dtype: Data type for model weights
            **kwargs: Additional model-specific arguments
        """
        self.pretrained_path = pretrained_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Components (initialized by subclasses)
        self._transformer: Optional[nn.Module] = None
        self._vae: Optional[nn.Module] = None
        self._text_encoder: Optional[nn.Module] = None

    @property
    def transformer(self) -> nn.Module:
        """Get the main transformer model."""
        if self._transformer is None:
            raise RuntimeError("Transformer not loaded. Call load() first.")
        return self._transformer

    @property
    def vae(self) -> nn.Module:
        """Get the VAE model."""
        if self._vae is None:
            raise RuntimeError("VAE not loaded. Call load() first.")
        return self._vae

    @property
    def text_encoder(self) -> nn.Module:
        """Get the text encoder."""
        if self._text_encoder is None:
            raise RuntimeError("Text encoder not loaded. Call load() first.")
        return self._text_encoder

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier (e.g., 'hunyuan', 'mochi', 'flux')."""
        ...

    @property
    @abstractmethod
    def media_type(self) -> str:
        """Return the media type identifier ('image' or 'video')."""
        ...

    @classmethod
    def default_sampler_path(cls) -> Optional[str]:
        """Default sampler implementation for this model bundle."""
        return None

    @classmethod
    def default_replay_sampler_path(cls) -> Optional[str]:
        """Default sampler path used to replay old log-probs on training actors.

        By default this matches ``default_sampler_path``. Models can override when
        rollout sampler and replay sampler should differ.
        """
        return cls.default_sampler_path()

    @classmethod
    def declared_model_type(cls) -> Optional[str]:
        """Class-level model type declaration used by model discovery."""
        attr = getattr(cls, "model_type", None)
        if isinstance(attr, property) and callable(attr.fget):
            try:
                value = attr.fget(cls)
            except Exception:
                return None
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        """Default sampler engine type for this model bundle."""
        return None

    @classmethod
    def forward_plugin(cls):
        """Return the forward plugin for this model's loss computation.

        Subclasses should override to provide model-specific forward logic.
        Returns a BaseForwardPlugin instance.
        """
        from diffusionrl.models.forward_plugins import DefaultForwardPlugin
        return DefaultForwardPlugin()

    @classmethod
    def supports_sglang_prompt_mode(cls) -> bool:
        """Whether this model supports SGLang prompt-only rollout mode."""
        return False

    @classmethod
    def validate_config(cls, args: Any) -> None:
        """Model-specific argument normalization/validation hook."""
        return None

    @classmethod
    def embedding_dataset_kwargs(cls) -> Dict[str, Any]:
        """Default kwargs when loading embedding manifests for this model."""
        return {"load_text_ids": False}

    @classmethod
    def create_embedding_dataset(cls, *, json_path: str, **kwargs: Any):
        """Build embedding dataset instance for this model."""
        from diffusionrl.data.datasets import EmbeddingRLDataset

        merged_kwargs = dict(cls.embedding_dataset_kwargs())
        merged_kwargs.update(kwargs)
        return EmbeddingRLDataset(json_path=json_path, **merged_kwargs)

    @abstractmethod
    def load(self) -> None:
        """Load all model components."""
        ...

    @abstractmethod
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to embeddings.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Optional negative prompt(s)

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        ...

    def encode_prompt_for_inference(
        self,
        prompts: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode prompts for sampler inference path.

        Subclasses can override to inject model-specific fields such as
        CFG negative embeddings.
        """
        result = self.encode_prompt(prompts, **kwargs)
        if isinstance(result, tuple):
            output: Dict[str, torch.Tensor] = {"prompt_embeds": result[0]}
            if len(result) > 1:
                output["pooled_prompt_embeds"] = result[1]
            if len(result) > 2:
                output["text_ids"] = result[2]
            return output
        if isinstance(result, dict):
            return dict(result)
        return {"prompt_embeds": result}

    def get_sampler_extra_kwargs(self) -> Dict[str, Any]:
        """
        Return model-specific kwargs for sampler initialization.

        Default behavior is no extra kwargs.
        """
        return {}

    @abstractmethod
    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents to images/videos.

        Args:
            latents: Latent tensor [B, C, H, W] or [B, C, T, H, W]

        Returns:
            Decoded images/videos
        """
        ...

    @abstractmethod
    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images/videos to latents.

        Args:
            images: Image tensor [B, C, H, W] or [B, C, T, H, W]

        Returns:
            Encoded latents
        """
        ...

    def get_no_split_modules(self) -> Tuple[Type[nn.Module], ...]:
        """
        Get module types that should not be split in FSDP.

        Returns:
            Tuple of module types
        """
        return tuple()

    def get_sigma_schedule(
        self,
        num_steps: int,
        shift: float = 3.0,
    ) -> torch.Tensor:
        """
        Get sigma schedule for denoising.

        Args:
            num_steps: Number of denoising steps
            shift: Time shift parameter

        Returns:
            Sigma schedule tensor [num_steps + 1]
        """
        from diffusionrl.samplers.log_prob import get_sigma_schedule
        return get_sigma_schedule(num_steps, shift, self.device)

    def to(self, device: Union[str, torch.device]) -> "ModelBundle":
        """
        Move all components to device.

        Args:
            device: Target device

        Returns:
            self for chaining
        """
        self.device = torch.device(device) if isinstance(device, str) else device

        if self._transformer is not None:
            self._transformer.to(self.device)
        if self._vae is not None:
            self._vae.to(self.device)
        if self._text_encoder is not None:
            self._text_encoder.to(self.device)

        return self

    def train(self, mode: bool = True) -> "ModelBundle":
        """
        Set training mode for transformer only.

        VAE and text encoder are typically frozen during training.

        Args:
            mode: Training mode

        Returns:
            self for chaining
        """
        if self._transformer is not None:
            self._transformer.train(mode)
        return self

    def eval(self) -> "ModelBundle":
        """Set all components to eval mode."""
        if self._transformer is not None:
            self._transformer.eval()
        if self._vae is not None:
            self._vae.eval()
        if self._text_encoder is not None:
            self._text_encoder.eval()
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Get transformer state dict."""
        if self._transformer is None:
            return {}
        return self._transformer.state_dict()

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """Load transformer state dict."""
        if self._transformer is not None:
            self._transformer.load_state_dict(state_dict, strict=strict)

    @property
    def is_video_model(self) -> bool:
        """Whether this is a video generation model."""
        return self.media_type == "video"

    @property
    def is_image_model(self) -> bool:
        """Whether this is an image generation model."""
        return self.media_type == "image"

    def get_config(self) -> Dict[str, Any]:
        """Get model bundle configuration."""
        return {
            "model_type": self.model_type,
            "pretrained_path": self.pretrained_path,
            "dtype": str(self.dtype),
        }

    def iter_offloadable_modules(self, include_transformer: bool = True) -> Iterable[Tuple[str, nn.Module]]:
        """
        Iterate over model components that are safe to move across devices.

        This avoids hardcoding model-specific module names inside inference engines.
        Subclasses can override if they need custom behavior.

        Args:
            include_transformer: If False, skip transformer-like modules.

        Yields:
            (name, module) tuples for offloadable components.
        """
        known_names = {
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
            "image_encoder",
        }
        for name, value in self.__dict__.items():
            if not isinstance(value, nn.Module):
                continue
            base_name = name.lstrip("_").lower()
            if not include_transformer and "transformer" in base_name:
                continue
            if base_name in known_names or any(token in base_name for token in ("encoder", "vae", "transformer")):
                yield name, value

    def load_aux_components(self) -> None:
        """
        Lazily load auxiliary components (VAE, text encoders, scheduler).

        This is used by training actors when they need to sample using the
        training model bundle that was initialized with training_only=True.
        Subclasses may override if they need custom behavior.
        """
        # Load VAE if missing
        if getattr(self, "_vae", None) is None:
            loader = getattr(self, "_load_vae", None)
            if callable(loader):
                loader()

        # Load text encoders if missing
        text_encoder_missing = getattr(self, "_text_encoder", None) is None
        if hasattr(self, "_text_encoder_2") and getattr(self, "_text_encoder_2", None) is None:
            text_encoder_missing = True
        if hasattr(self, "_text_encoder_3") and getattr(self, "_text_encoder_3", None) is None:
            text_encoder_missing = True

        if text_encoder_missing:
            loader = getattr(self, "_load_text_encoders", None)
            if callable(loader):
                loader()
            else:
                loader = getattr(self, "_load_text_encoder", None)
                if callable(loader):
                    loader()

        # Load scheduler if missing
        if hasattr(self, "_scheduler") and getattr(self, "_scheduler", None) is None:
            loader = getattr(self, "_load_scheduler", None)
            if callable(loader):
                loader()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type}, "
            f"pretrained_path={self.pretrained_path})"
        )
