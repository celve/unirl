"""
Inference Engine Interface for GRPO Training.

This module defines the unified interface for inference engines (FSDP, FastVideo, SGLang).
All engines must implement the BaseInferenceEngine interface to work with Ray actors.

Engine Responsibilities:
1. Model loading and initialization
2. Sample generation with log probabilities
3. Weight synchronization from training
4. Memory management (offload/onload)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set
import torch

from diffusionrl.types import SamplerOutput


@dataclass
class EngineConfig:
    """Configuration for inference engine."""

    # Model configuration
    model_path: str = ""
    pretrained_model_saved_path: str = ""

    # Sampler configuration
    num_inference_steps: int = 50
    eta: float = 1.0
    sde_type: str = "sde"
    shift: float = 3.0
    guidance_scale: float = 7.5

    # Video/Image configuration
    height: int = 256
    width: int = 256
    num_frames: int = 16

    # Engine-specific configuration
    engine_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.engine_kwargs is None:
            self.engine_kwargs = {}


@dataclass
class EngineCapabilities:
    """Runtime capabilities exposed by an inference engine."""

    supports_logprob: bool = True
    supports_trajectory: bool = True
    supports_prompt_embeddings: bool = True
    supports_guidance_scale: bool = True
    supports_staged_onload: bool = False
    weight_sync_mode: str = "state_dict"  # state_dict | checkpoint_path | external


class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.

    All inference engines (FSDP, FastVideo, SGLang) must implement this interface
    to be compatible with InferenceActor and Ray scheduling.

    Key Design Principles:
    1. Unified interface for all engines
    2. Support for model weight synchronization
    3. Memory management for GPU offloading
    4. Consistent SamplerOutput format
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize engine with configuration.

        Args:
            config: Engine configuration
        """
        self.config = config
        self._is_initialized = False
        self._is_offloaded = False

    @abstractmethod
    def initialize(self, device: torch.device) -> None:
        """
        Initialize the engine (load models, setup samplers).

        Args:
            device: Target device for inference
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> SamplerOutput:
        """
        Generate samples with log probabilities.

        Args:
            prompts: List of text prompts
            prompt_embeds: Pre-computed prompt embeddings [B, seq, hidden]
            pooled_prompt_embeds: Pooled prompt embeddings [B, hidden]
            encoder_attention_mask: Attention mask [B, seq]
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            height: Output height
            width: Output width
            num_frames: Number of frames (for video)
            latents: Initial latents (if None, sample from noise)
            seed: Random seed for reproducibility
            sde_indices: Set of timestep indices for SDE sampling
            **kwargs: Additional engine-specific arguments

        Returns:
            SamplerOutput with trajectories, log_probs, etc.
        """
        pass

    @abstractmethod
    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to embeddings.

        Args:
            prompts: List of text prompts
            **kwargs: Additional encoding arguments

        Returns:
            Dict containing:
                - prompt_embeds: [B, seq, hidden]
                - pooled_prompt_embeds: [B, hidden] (optional)
                - encoder_attention_mask: [B, seq] (optional)
        """
        pass

    @abstractmethod
    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update model weights from training.

        Args:
            state_dict: New model state dict
        """
        pass

    @abstractmethod
    def offload(self) -> None:
        """Offload model to CPU to free GPU memory."""
        pass

    @abstractmethod
    def onload(self) -> None:
        """Load model back to GPU from CPU."""
        pass

    def onload_weights(self) -> None:
        """Stage 1 onload: bring weights/modules required for weight update."""
        self.onload()

    def onload_post_update(self) -> None:
        """Stage 2 onload: prepare post-weight-update state (default no-op)."""
        return None

    def onload_runtime_cache(self) -> None:
        """Stage 3 onload: prepare runtime caches (KV/CUDA graph)."""
        return None

    def prepare_for_weight_update(self) -> None:
        """Unified stage protocol entrypoint before update_weights/update_weights_from_path."""
        self.onload_weights()

    def finalize_weight_update(self) -> None:
        """Unified stage protocol tail after weight update."""
        self.onload_post_update()
        self.onload_runtime_cache()

    def ensure_ready_for_generate(self) -> None:
        """Ensure generation can safely run from any residency state."""
        if self._is_offloaded:
            self.onload()

    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents to pixel space (optional).

        Args:
            latents: Latent tensor [B, C, H, W] or [B, C, T, H, W]

        Returns:
            Decoded tensor [B, C, H, W] or [B, C, T, H, W]
        """
        raise NotImplementedError("Engine does not support latent decoding")

    @property
    def is_initialized(self) -> bool:
        """Whether engine is initialized."""
        return self._is_initialized

    @property
    def is_offloaded(self) -> bool:
        """Whether engine is offloaded to CPU."""
        return self._is_offloaded

    @property
    @abstractmethod
    def supports_distributed(self) -> bool:
        """Whether engine supports multi-GPU distribution."""
        pass

    @property
    @abstractmethod
    def requires_external_service(self) -> bool:
        """Whether engine requires external service (e.g., SGLang server)."""
        pass

    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
        }

    def health_check(self) -> bool:
        """Check if engine is healthy."""
        if self._is_offloaded:
            return True  # Offloaded state is healthy
        return self._is_initialized

    def get_capabilities(self) -> EngineCapabilities:
        """Get capability snapshot for control-plane scheduling and validation."""
        return EngineCapabilities()

    def get_capabilities_dict(self) -> Dict[str, Any]:
        """Serialize capabilities to plain dict for metadata/RPC usage."""
        return asdict(self.get_capabilities())


# Engine registry for dynamic loading
ENGINE_REGISTRY: Dict[str, type] = {}


def register_engine(name: str):
    """Decorator to register an engine class."""
    def decorator(cls):
        ENGINE_REGISTRY[name] = cls
        return cls
    return decorator


def get_engine(name: str) -> type:
    """Get engine class by name."""
    if name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: {name}. Available: {list(ENGINE_REGISTRY.keys())}")
    return ENGINE_REGISTRY[name]


def create_engine(name: str, config: EngineConfig) -> BaseInferenceEngine:
    """Create engine instance by name."""
    engine_cls = get_engine(name)
    return engine_cls(config)
