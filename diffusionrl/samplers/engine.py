"""Inference engine interface for rollout-side engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples

__all__ = ["BaseRolloutEngine", "chunked_engine_generate", "chunked_decode_latents"]


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


def chunked_engine_generate(
    engine: BaseRolloutEngine,
    request: RolloutRequest,
    *,
    chunk_size: Optional[int],
) -> RolloutSamples:
    """Call ``engine.generate`` over mini-batch chunks of *request* and concat outputs.

    Splits the request's prompt list into chunks of size ``chunk_size``, calls
    ``engine.generate`` once per chunk, and concatenates the per-chunk
    ``RolloutSamples`` along the batch dim via ``RolloutSamples.concat``.

    Fast path (zero overhead): when ``chunk_size`` is ``None`` or
    ``>= n_prompts``, this is a single direct call to ``engine.generate(request)``.

    Determinism caveats (chunked vs. unchunked):

    - **Initial noise** is bit-identical iff ``init_same_noise=True``: in
      that path ``samplers/noise_utils.py`` keys per-group noise on each
      sample's ``noise_group_id`` (preserved by ``Prompts.slice`` /
      ``Batched``) plus a base seed. With ``init_same_noise=False`` the
      fallback is plain ``torch.randn`` consuming global RNG state, so
      the per-chunk draw schedule differs from a single full-batch draw
      and initial latents will diverge.
    - **Per-step SDE noise** (``eta > 0``) is intentionally unseeded —
      see ``sde/runtime.py:denoising_step`` — so trajectories and
      ``log_probs`` are NOT bit-identical to an unchunked call regardless
      of ``init_same_noise``.

    Outputs remain valid i.i.d. samples; the practical implication is
    that reproducibility depends on ``chunk_size``.
    """
    if request.prompts is None or not request.prompts.prompts:
        raise ValueError(
            f"chunked_engine_generate requires non-empty request.prompts.prompts; got prompts={request.prompts!r}."
        )
    n_prompts = len(request.prompts.prompts)
    if chunk_size is None:
        return engine.generate(request)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive int when set; got {chunk_size!r} (type={type(chunk_size).__name__})."
        )
    if n_prompts <= chunk_size:
        return engine.generate(request)
    outputs: List[RolloutSamples] = []
    for start in range(0, n_prompts, chunk_size):
        end = min(start + chunk_size, n_prompts)
        outputs.append(engine.generate(request.slice(start, end)))
    return RolloutSamples.concat(outputs)


def chunked_decode_latents(
    engine: "BaseRolloutEngine",
    latents: torch.Tensor,
    *,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """Run ``engine.decode_latents`` over mini-batch chunks of *latents* and concat outputs.

    Mirrors :func:`chunked_engine_generate`: fast-path (``chunk_size is None``
    or ``>= batch``) is a single direct call, otherwise slice along dim 0 and
    ``torch.cat`` the per-chunk results. VAE decode is embarrassingly parallel
    along the batch dim so chunking is bit-identical to a single full-batch
    call (no determinism caveat).

    The ``engine`` argument is duck-typed: any object exposing
    ``decode_latents(latents) -> Tensor`` works (e.g. ``BaseRolloutEngine``
    subclasses, ``FSDPBaseSampler``).
    """
    if not isinstance(latents, torch.Tensor) or latents.dim() == 0:
        raise TypeError(f"chunked_decode_latents requires a batched tensor, got {type(latents).__name__}.")
    n = int(latents.shape[0])
    if n == 0:
        raise ValueError("chunked_decode_latents requires non-empty latents (got batch=0).")
    if chunk_size is None:
        return engine.decode_latents(latents)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive int when set; got {chunk_size!r} (type={type(chunk_size).__name__})."
        )
    if n <= chunk_size:
        return engine.decode_latents(latents)
    chunks: List[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(engine.decode_latents(latents[start:end]))
    return torch.cat(chunks, dim=0)
