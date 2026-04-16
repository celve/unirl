"""Inference engine interface for dedicated rollout-side engines."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.types import RolloutRequest, RolloutSamples
from diffusionrl.types.engine import EngineCapabilities


@dataclass
class EngineConfig:
    """Configuration for rollout-side inference engines."""

    model_dotpath: str = ""
    pretrained_model_ckpt_path: str = ""
    sampler_dotpath: str = ""

    num_inference_steps: int = 50
    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0
    guidance_scale: float = 7.5

    height: int = 256
    width: int = 256
    num_frames: int = 16

    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        self.sde_type = str(self.sde_type or "flow")


class BaseRolloutEngine(ABC):
    """
    Abstract base class for inference engines.

    Dedicated rollout engines (SGLang, future service engines) implement
    this interface to be compatible with RolloutActor and Ray scheduling.

    Key Design Principles:
    1. Unified interface for all engines
    2. Support for model weight synchronization
    3. Memory management via sleep/wake_up
    4. Consistent RolloutSamples format
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
    def generate(self, request: RolloutRequest) -> RolloutSamples:
        """
        Generate samples with log probabilities.

        Args:
            request: A RolloutRequest containing prompts and resolved
                generation parameters.

        Returns:
            RolloutSamples with trajectories, log_probs, etc.
        """
        pass

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
        raise NotImplementedError(
            f"{type(self).__name__} does not support prompt encoding. "
            "Use engines that implement encode_prompt()."
        )

    @abstractmethod
    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update model weights from training.

        Args:
            state_dict: New model state dict
        """
        pass

    def sleep(self) -> None:
        """Release runtime resources when rollout side is inactive."""
        self._is_offloaded = True

    def wake_up(self) -> None:
        """Restore runtime resources required for generation/update."""
        self._is_offloaded = False

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
    def supports_distributed(self) -> bool:
        """Whether engine supports multi-GPU distribution (default: False)."""
        return False

    @property
    def requires_external_service(self) -> bool:
        """Whether engine requires external service (e.g., SGLang server)."""
        return False

    @classmethod
    def declared_capabilities(cls) -> Dict[str, bool]:
        """Config-time capability declaration used by argument validation."""
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

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


class DistributedWeightSyncCapable:
    """Mixin protocol for engines that support advanced weight sync.

    SGLangRolloutEngine implements this for distributed rollout-side weight sync.
    RolloutActor checks isinstance() instead of hasattr().
    """

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        raise NotImplementedError

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list,
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        raise NotImplementedError

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        raise NotImplementedError

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        raise NotImplementedError

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        raise NotImplementedError

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: dict,
    ) -> None:
        """Load LoRA tensors directly into the rollout engine."""
        raise NotImplementedError
