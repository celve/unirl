"""Inference engine interface for rollout-side engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples


class BaseRolloutEngine(ABC):
    """
    Abstract base class for inference engines.

    All rollout engines (SGLang service engines, in-process FSDP engines)
    implement this interface to provide a unified generation contract for
    both RolloutActor and TrainActor.

    Key Design Principles:
    1. Unified interface for all engines
    2. Support for model weight synchronization
    3. Memory management via sleep/wake_up
    4. Consistent RolloutSamples format
    """

    def __init__(self, config: Any):
        """
        Initialize engine with configuration.

        Args:
            config: Engine configuration (e.g. EngineConfig for SGLang,
                SamplingParams for direct FSDP engines).
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
            f"{type(self).__name__} does not support prompt encoding. Use engines that implement encode_prompt()."
        )

    @abstractmethod
    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update model weights from training.

        Args:
            state_dict: New model state dict
        """
        pass

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
