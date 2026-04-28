"""
diffusionrl Model Bundle Base Class.

Defines the interface for model bundles that package transformer, VAE, and text encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, NoReturn, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .config import ModelBundleConfig

if TYPE_CHECKING:
    from diffusionrl.types.forward_context import ForwardContext


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
        config: ModelBundleConfig,
    ):
        """
        Initialize model bundle.

        Args:
            config: Typed model-bundle construction config
        """
        if not isinstance(config, ModelBundleConfig):
            raise TypeError(f"{type(self).__name__} expected {ModelBundleConfig.__name__}, got: {config!r}")
        self.config = config
        self.pretrained_path = config.pretrained_model_ckpt_path
        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = config.model_precision
        self.vae_dtype = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        self.text_encoder_dtype = (
            config.text_encoder_dtype if config.text_encoder_dtype is not None else config.model_precision
        )

        # Components (initialized by subclasses)
        self._transformer: Optional[nn.Module] = None
        self._vae: Optional[nn.Module] = None
        self._text_encoder: Optional[nn.Module] = None
        self._vision_encoder: Optional[nn.Module] = None
        self.training_forward_autocast_dtype: Optional[torch.dtype] = None

    def _raise_aux_component_not_loaded(
        self,
        component: str,
        *,
        missing: Optional[List[str]] = None,
    ) -> NoReturn:
        missing_msg = f" Missing components: {', '.join(missing)}." if missing else ""
        raise RuntimeError(
            f"{type(self).__name__} {component} not loaded.{missing_msg} "
            "This can happen when the bundle was initialized with training_only=True. "
            "Call load_aux_components() before calling encode/decode APIs, or use a bundle initialized with "
            "auxiliary components."
        )

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
            self._raise_aux_component_not_loaded("VAE")
        return self._vae

    @property
    def text_encoder(self) -> nn.Module:
        """Get the text encoder."""
        if self._text_encoder is None:
            self._raise_aux_component_not_loaded("text encoder")
        return self._text_encoder

    @property
    def vision_encoder(self) -> nn.Module:
        """Get the optional vision encoder used by image/video-conditioned models."""
        if self._vision_encoder is None:
            self._raise_aux_component_not_loaded("vision encoder")
        return self._vision_encoder

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier (e.g., 'hunyuan', 'mochi', 'flux')."""
        ...

    @property
    @abstractmethod
    def media_type(self) -> str:
        """Return the task signature, e.g. ``t2i``, ``t2v``, or ``tiv2iv``.

        The compact signature uses ``t`` (text), ``i`` (image), ``v`` (video),
        and ``a`` (audio). Modalities before ``2`` are inputs; modalities after
        ``2`` are outputs.
        """
        ...

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        """Default sampler implementation for this model bundle."""
        return None

    @classmethod
    def default_replay_sampler_dotpath(cls) -> Optional[str]:
        """Default sampler dotpath used to replay old log-probs on training actors.

        By default this matches ``default_sampler_dotpath``. Models can override when
        rollout sampler and replay sampler should differ.
        """
        return cls.default_sampler_dotpath()

    @classmethod
    def default_lora_target_modules(cls) -> Optional[List[str]]:
        """Default LoRA target module names for this model bundle.

        Single source of truth for LoRA target selection. Used by:

        * PEFT adapter injection inside the model bundle (training side)
        * ``EngineConfig.lora_target_modules`` → SGLang ``ServerArgs`` (rollout side)
        * Any checkpointing / logging that enumerates the trained LoRA layers

        Return ``None`` to delegate to the sampler's "wrap everywhere" default
        (NOT recommended in production — produces warnings like ``LoRA adapter
        None does not contain the weights for layer '...'`` when the training
        side only touches a subset of layers).

        Subclasses SHOULD override this with the canonical list for the model.
        CLI ``--training.lora-target-modules`` always wins when provided.
        """
        return None

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
    def supports_sglang_prompt_mode(cls) -> bool:
        """Whether this model supports SGLang prompt-only rollout mode."""
        return False

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

    def _encode_prompt_to_input_dict(
        self,
        prompts: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode prompts and normalize the result for ``encode_inputs``.

        The default tuple interpretation matches SD3/FLUX-style text encoders:
        ``(prompt_embeds, pooled_prompt_embeds, text_ids)``. Models whose
        ``encode_prompt`` tuple carries different semantics should override this
        private helper or ``encode_inputs``.
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

    def encode_inputs(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Encode multimodal conditioning inputs for sampler inference.

        Text-only models use the prompt encoder by default. Multimodal models
        can override this to add image/video conditioning tensors.
        """
        if image is not None and not self.accepts_image_input:
            raise NotImplementedError(f"{type(self).__name__} does not support image conditioning")
        if video is not None and not self.accepts_video_input:
            raise NotImplementedError(f"{type(self).__name__} does not support video conditioning")
        output = self._encode_prompt_to_input_dict(prompts, **kwargs)
        if negative_prompt is None:
            return output

        negative_prompts: Union[str, List[str]]
        if isinstance(negative_prompt, str):
            if isinstance(prompts, str):
                negative_prompts = negative_prompt
            else:
                negative_prompts = [negative_prompt] * len(prompts)
        else:
            prompt_batch_size = 1 if isinstance(prompts, str) else len(prompts)
            if len(negative_prompt) != prompt_batch_size:
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompt)} does not match prompt batch size {prompt_batch_size}"
                )
            negative_prompts = negative_prompt
        negative_output = self._encode_prompt_to_input_dict(negative_prompts, **kwargs)
        if negative_output.get("prompt_embeds") is not None:
            output["negative_prompt_embeds"] = negative_output["prompt_embeds"]
        if negative_output.get("pooled_prompt_embeds") is not None:
            output["negative_pooled_prompt_embeds"] = negative_output["pooled_prompt_embeds"]
        return output

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
        from diffusionrl.sde.runtime import get_sigma_schedule

        return get_sigma_schedule(num_steps, shift, self.device)

    def set_training_forward_autocast_dtype(
        self,
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        """Set the autocast dtype used by training-side forward dispatch."""
        self.training_forward_autocast_dtype = autocast_dtype

    def _build_training_autocast_ctx(self, device: torch.device):
        """Return an autocast context for training forward dispatch."""
        if (
            self.training_forward_autocast_dtype is not None
            and device.type == "cuda"
            and self.training_forward_autocast_dtype in (torch.float16, torch.bfloat16)
        ):
            return torch.autocast("cuda", self.training_forward_autocast_dtype)
        return nullcontext()

    def _prepare_training_timestep(
        self,
        sigma: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Expand per-step sigma to a batch-aligned timestep tensor."""
        if sigma.dim() == 0:
            sigma_expanded = sigma.unsqueeze(0)
        else:
            sigma_expanded = sigma
        return sigma_expanded.expand(batch_size).to(device, dtype=dtype)

    @abstractmethod
    def forward_denoiser(
        self,
        *,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Run the model-family-specific denoiser forward for training."""
        ...

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
        if self._vision_encoder is not None:
            self._vision_encoder.to(self.device)

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
        if self._vision_encoder is not None:
            self._vision_encoder.eval()
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
        """Whether this model can generate video outputs."""
        return self.outputs_video

    @property
    def is_image_model(self) -> bool:
        """Whether this model can generate image outputs."""
        return self.outputs_image

    @property
    def input_media_types(self) -> str:
        """Compact input-modality signature, e.g. ``t`` or ``tiv``."""
        return self._split_media_type()[0]

    @property
    def output_media_types(self) -> str:
        """Compact output-modality signature, e.g. ``i``, ``v``, or ``iv``."""
        return self._split_media_type()[1]

    @property
    def accepts_text_input(self) -> bool:
        return "t" in self.input_media_types

    @property
    def accepts_image_input(self) -> bool:
        return "i" in self.input_media_types

    @property
    def accepts_video_input(self) -> bool:
        return "v" in self.input_media_types

    @property
    def outputs_image(self) -> bool:
        return "i" in self.output_media_types

    @property
    def outputs_video(self) -> bool:
        return "v" in self.output_media_types

    def _split_media_type(self) -> Tuple[str, str]:
        media_type = self.media_type.strip().lower()
        if "2" not in media_type:
            raise ValueError(
                f"{type(self).__name__}.media_type must be a task signature like 't2i' "
                f"or 'tiv2iv'; got {self.media_type!r}"
            )
        inputs, outputs = media_type.split("2", 1)
        valid = set("tiva")
        if not inputs or not outputs or any(ch not in valid for ch in inputs + outputs):
            raise ValueError(f"{type(self).__name__}.media_type has invalid modality signature: {self.media_type!r}")
        return inputs, outputs

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

        # Load vision encoders for image/video-conditioned models when present.
        if getattr(self, "_vision_encoder", None) is None:
            loader = getattr(self, "_load_vision_encoder", None)
            if callable(loader):
                loader()

        # Load scheduler if missing
        if hasattr(self, "_scheduler") and getattr(self, "_scheduler", None) is None:
            loader = getattr(self, "_load_scheduler", None)
            if callable(loader):
                loader()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_type={self.model_type}, pretrained_path={self.pretrained_path})"
