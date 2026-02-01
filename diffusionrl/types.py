"""
Core types for GRPO training.

These types define the data contracts between different components.

Data Structure Hierarchy:
    LogProbData              - Standardized log probability storage
    PromptEmbeddings         - Bundled embeddings for all model types
    GRPOTrainingBatch        - trajectories + log_probs for GRPO/MixGRPO
    NFTTrainingBatch         - clean_latents only for NFT
    TimestepData             - Per-timestep data for GRPO loss
    TrainingBatch            - Union[GRPOTrainingBatch, NFTTrainingBatch]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, TYPE_CHECKING
from enum import Enum
import torch

if TYPE_CHECKING:
    from torch import device as TorchDevice


class SampleStatus(Enum):
    """Status of a sample in the pipeline."""
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    FAILED = "failed"


SAMPLING_CONTRACT_VERSION = "v1"


# =============================================================================
# Core Data Structures for Type-Safe Pipeline
# =============================================================================

@dataclass
class LogProbData:
    """
    Standardized log probability storage - always dict internally.

    This provides a consistent interface for log probabilities regardless
    of how they were computed (sparse from MixGRPO or dense from full SDE).

    Attributes:
        data: Dictionary mapping step index to log probability tensor [B]
    """
    data: Dict[int, torch.Tensor]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "LogProbData":
        """
        Create from dense tensor [B, T].

        Args:
            tensor: Dense log probability tensor of shape [B, T]

        Returns:
            LogProbData instance with dict representation
        """
        return cls(data={i: tensor[:, i] for i in range(tensor.shape[1])})

    @classmethod
    def from_dict(cls, d: Dict[int, torch.Tensor]) -> "LogProbData":
        """
        Create from existing dictionary.

        Args:
            d: Dictionary mapping step index to tensor [B]

        Returns:
            LogProbData instance
        """
        return cls(data=d.copy())

    @classmethod
    def empty(cls) -> "LogProbData":
        """Create empty LogProbData."""
        return cls(data={})

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        """Get log probability for a specific timestep index."""
        return self.data.get(idx)

    def __contains__(self, idx: int) -> bool:
        """Check if timestep index has log probability."""
        return idx in self.data

    def __len__(self) -> int:
        """Number of timesteps with log probabilities."""
        return len(self.data)

    @property
    def sde_indices(self) -> Set[int]:
        """Get set of timestep indices that have log probabilities (SDE steps)."""
        return set(self.data.keys())

    def to_dict(self) -> Dict[int, torch.Tensor]:
        """Convert to plain dictionary."""
        return self.data.copy()

    def slice(self, start: int, end: int) -> "LogProbData":
        """
        Slice log probabilities along batch dimension.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New LogProbData with sliced tensors
        """
        return LogProbData(
            data={k: v[start:end] for k, v in self.data.items()}
        )

    def to_device(self, device: Union[str, "TorchDevice"]) -> "LogProbData":
        """Move all tensors to specified device."""
        return LogProbData(
            data={k: v.to(device) for k, v in self.data.items()}
        )


@dataclass
class PromptEmbeddings:
    """
    Bundled embeddings for different model architectures.

    This provides a unified container for all embedding types needed
    across Flux, SD3, and other models.

    Attributes:
        prompt_embeds: Text encoder hidden states [B, seq, hidden]
        pooled_prompt_embeds: Pooled text embeddings [B, hidden] (optional)
        encoder_attention_mask: Attention mask for encoder tokens (optional)
        negative_prompt_embeds: Negative prompt hidden states [B, seq, hidden] (optional)
        negative_pooled_prompt_embeds: Negative pooled text embeddings [B, hidden] (optional)
        text_ids: Text position IDs for Flux (optional)
        image_ids: Image position IDs for Flux (optional)
    """
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    image_ids: Optional[torch.Tensor] = None

    def to_device(self, device: Union[str, "TorchDevice"]) -> "PromptEmbeddings":
        """Move all tensors to specified device."""
        return PromptEmbeddings(
            prompt_embeds=self.prompt_embeds.to(device),
            pooled_prompt_embeds=(
                self.pooled_prompt_embeds.to(device)
                if self.pooled_prompt_embeds is not None else None
            ),
            encoder_attention_mask=(
                self.encoder_attention_mask.to(device)
                if self.encoder_attention_mask is not None else None
            ),
            negative_prompt_embeds=(
                self.negative_prompt_embeds.to(device)
                if self.negative_prompt_embeds is not None else None
            ),
            negative_pooled_prompt_embeds=(
                self.negative_pooled_prompt_embeds.to(device)
                if self.negative_pooled_prompt_embeds is not None else None
            ),
            text_ids=(
                self.text_ids.to(device)
                if self.text_ids is not None else None
            ),
            image_ids=(
                self.image_ids.to(device)
                if self.image_ids is not None else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {"prompt_embeds": self.prompt_embeds}
        if self.pooled_prompt_embeds is not None:
            result["pooled_prompt_embeds"] = self.pooled_prompt_embeds
        if self.encoder_attention_mask is not None:
            result["encoder_attention_mask"] = self.encoder_attention_mask
        if self.negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = self.negative_prompt_embeds
        if self.negative_pooled_prompt_embeds is not None:
            result["negative_pooled_prompt_embeds"] = self.negative_pooled_prompt_embeds
        if self.text_ids is not None:
            result["text_ids"] = self.text_ids
        if self.image_ids is not None:
            result["image_ids"] = self.image_ids
        return result

    def slice(self, start: int, end: int) -> "PromptEmbeddings":
        """
        Slice embeddings along batch dimension.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New PromptEmbeddings with sliced tensors

        Note:
            For FLUX models, image_ids are NOT batched (shape [H*W, 3], shared
            across batch). This method detects unbatched tensors by comparing
            their first dimension with prompt_embeds batch size and keeps them
            as-is (shared) instead of slicing.
        """
        batch_size = self.prompt_embeds.shape[0]

        # Helper to slice only if tensor is batched (first dim == batch_size)
        def _slice_if_batched(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if tensor.shape[0] == batch_size:
                return tensor[start:end]
            # Unbatched (e.g., FLUX image_ids): keep as-is (shared)
            return tensor

        return PromptEmbeddings(
            prompt_embeds=self.prompt_embeds[start:end],
            pooled_prompt_embeds=_slice_if_batched(self.pooled_prompt_embeds),
            encoder_attention_mask=_slice_if_batched(self.encoder_attention_mask),
            negative_prompt_embeds=_slice_if_batched(self.negative_prompt_embeds),
            negative_pooled_prompt_embeds=_slice_if_batched(self.negative_pooled_prompt_embeds),
            text_ids=_slice_if_batched(self.text_ids),
            image_ids=_slice_if_batched(self.image_ids),
        )


@dataclass
class TimestepData:
    """
    Single timestep data for GRPO loss computation.

    Contains all data needed to compute loss at a specific timestep.

    Attributes:
        latents: Current noisy latents x_t [B, C, H, W]
        next_latents: Next step latents x_{t-1} [B, C, H, W]
        log_prob: Old log probability [B] (None for ODE steps)
        sigma: Current sigma value
        sigma_next: Next sigma value
        timestep_idx: Index of this timestep
    """
    latents: torch.Tensor           # x_t
    next_latents: torch.Tensor      # x_{t-1}
    log_prob: Optional[torch.Tensor]
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    timestep_idx: int = 0

    def to_device(self, device: Union[str, "TorchDevice"]) -> "TimestepData":
        """Move all tensors to specified device."""
        return TimestepData(
            latents=self.latents.to(device),
            next_latents=self.next_latents.to(device),
            log_prob=self.log_prob.to(device) if self.log_prob is not None else None,
            sigma=self.sigma.to(device) if isinstance(self.sigma, torch.Tensor) else self.sigma,
            sigma_next=self.sigma_next.to(device) if isinstance(self.sigma_next, torch.Tensor) else self.sigma_next,
            timestep_idx=self.timestep_idx,
        )


@dataclass
class GRPOTrainingBatch:
    """
    GRPO/MixGRPO training batch with trajectories and log probabilities.

    This batch type is used for trajectory-based algorithms that require
    the full sampling path and importance sampling ratios.

    Attributes:
        trajectories: Full sampling trajectory [B, T+1, C, H, W]
        log_probs: Log probabilities for each SDE step
        timesteps: Sigma schedule [T+1]
        advantages: Per-sample advantages [B]
        embeddings: Bundled prompt embeddings
        rewards: Raw rewards [B] (optional, for logging)
        prompts: Original text prompts (optional, for logging)
        num_steps: Number of inference steps
    """
    trajectories: torch.Tensor      # [B, T+1, C, H, W]
    log_probs: LogProbData
    timesteps: torch.Tensor         # [T+1]
    advantages: torch.Tensor        # [B]
    embeddings: PromptEmbeddings
    rewards: Optional[torch.Tensor] = None
    prompts: Optional[List[str]] = None
    num_steps: int = 50
    is_partitioned: bool = False
    step_indices: Optional[torch.Tensor] = None  # [T+1], explicit step labels
    # Experimental/ad-hoc bridge for engines that cannot emit old log_probs
    # during rollout (e.g., FastVideo). Training actors may replay these steps.
    target_sde_indices: Optional[Set[int]] = None
    sampling_weight_version: Optional[int] = None

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.trajectories.shape[0]

    @property
    def sde_indices(self) -> Set[int]:
        """Get set of timestep indices that used SDE (have log_probs)."""
        if len(self.log_probs) > 0:
            return self.log_probs.sde_indices
        if self.target_sde_indices is not None:
            return set(int(i) for i in self.target_sde_indices)
        return set()

    @property
    def device(self) -> torch.device:
        """Get device of the batch tensors."""
        return self.trajectories.device

    @property
    def resolved_step_indices(self) -> torch.Tensor:
        """Get explicit step labels aligned with timesteps/trajectory axis."""
        if self.step_indices is not None:
            return self.step_indices
        return torch.arange(self.timesteps.shape[0], device=self.timesteps.device, dtype=torch.long)

    def _is_contiguous_step_index(self) -> bool:
        steps = self.resolved_step_indices
        expected = torch.arange(steps.shape[0], device=steps.device, dtype=steps.dtype)
        return bool(torch.equal(steps, expected))

    def get_position_for_step(self, step_idx: int) -> int:
        """Map a logical step label to trajectory position index."""
        steps = self.resolved_step_indices
        hits = (steps == int(step_idx)).nonzero(as_tuple=False)
        if hits.numel() == 0:
            raise ValueError(
                f"step_idx={step_idx} not present in step_indices={steps.tolist()}"
            )
        pos = int(hits[0].item())
        if pos >= self.trajectories.shape[1] - 1:
            raise ValueError(
                f"step_idx={step_idx} maps to terminal position {pos} (no t+1 pair available)"
            )
        return pos

    def validate(self) -> None:
        """
        Validate batch consistency.

        Raises:
            ValueError: If batch dimensions are inconsistent
        """
        B = self.trajectories.shape[0]
        T = self.trajectories.shape[1] - 1

        if self.advantages.shape[0] != B:
            raise ValueError(
                f"Advantages batch size {self.advantages.shape[0]} != "
                f"trajectories batch size {B}"
            )

        if self.timesteps.shape[0] != T + 1:
            raise ValueError(
                f"Timesteps length {self.timesteps.shape[0]} != "
                f"expected {T + 1}"
            )

        step_indices = self.resolved_step_indices
        if int(step_indices.shape[0]) != T + 1:
            raise ValueError(
                f"Step indices length {step_indices.shape[0]} != expected {T + 1}"
            )
        if step_indices.numel() > 1 and not bool(torch.all(step_indices[1:] > step_indices[:-1])):
            raise ValueError(f"step_indices must be strictly increasing, got: {step_indices.tolist()}")
        if self.target_sde_indices is not None:
            allowed_steps = set(int(v) for v in step_indices[:-1].tolist())
            bad = sorted(int(i) for i in self.target_sde_indices if int(i) not in allowed_steps)
            if bad:
                raise ValueError(
                    f"target_sde_indices contain out-of-range steps: {bad}, allowed={sorted(allowed_steps)}"
                )

        if self.embeddings.prompt_embeds.shape[0] != B:
            raise ValueError(
                f"Embeddings batch size {self.embeddings.prompt_embeds.shape[0]} != "
                f"batch size {B}"
            )

        # Validate log_probs batch sizes
        for idx, lp in self.log_probs.data.items():
            if lp.shape[0] != B:
                raise ValueError(
                    f"Log prob at index {idx} has batch size {lp.shape[0]} != {B}"
                )

    def to_device(self, device: Union[str, "TorchDevice"]) -> "GRPOTrainingBatch":
        """Move all tensors to specified device."""
        return GRPOTrainingBatch(
            trajectories=self.trajectories.to(device),
            log_probs=self.log_probs.to_device(device),
            timesteps=self.timesteps.to(device),
            advantages=self.advantages.to(device),
            embeddings=self.embeddings.to_device(device),
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            prompts=self.prompts,
            num_steps=self.num_steps,
            is_partitioned=self.is_partitioned,
            step_indices=self.step_indices.to(device) if self.step_indices is not None else None,
            target_sde_indices=self.target_sde_indices,
            sampling_weight_version=self.sampling_weight_version,
        )

    def get_timestep_data(self, t_idx: int) -> TimestepData:
        """
        Extract data for a specific trajectory position index.

        Args:
            t_idx: Position index along trajectory axis [0, T-1]

        Returns:
            TimestepData for the specified timestep
        """
        if not self._is_contiguous_step_index():
            raise ValueError(
                "Non-contiguous step_indices detected. "
                "Use get_timestep_data_by_step(step_idx) instead of positional get_timestep_data()."
            )
        return TimestepData(
            latents=self.trajectories[:, t_idx],
            next_latents=self.trajectories[:, t_idx + 1],
            log_prob=self.log_probs[t_idx],
            sigma=self.timesteps[t_idx],
            sigma_next=self.timesteps[t_idx + 1],
            timestep_idx=t_idx,
        )

    def get_timestep_data_by_step(self, step_idx: int) -> TimestepData:
        """
        Extract data for a logical step label from step_indices.

        Args:
            step_idx: Logical step label from sampling contract.
        """
        pos = self.get_position_for_step(step_idx)
        return TimestepData(
            latents=self.trajectories[:, pos],
            next_latents=self.trajectories[:, pos + 1],
            log_prob=self.log_probs[int(step_idx)],
            sigma=self.timesteps[pos],
            sigma_next=self.timesteps[pos + 1],
            timestep_idx=int(step_idx),
        )

    def slice(self, start: int, end: int) -> "GRPOTrainingBatch":
        """
        Slice batch along sample dimension for micro-batch gradient accumulation.

        This method is dimension-agnostic and supports both image and video:
        - Image trajectories: [B, T+1, C, H, W]
        - Video trajectories: [B, T+1, C, T_frames, H, W]

        Only the batch dimension (first axis) is sliced.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New GRPOTrainingBatch with sliced data
        """
        return GRPOTrainingBatch(
            trajectories=self.trajectories[start:end],
            log_probs=self.log_probs.slice(start, end),
            timesteps=self.timesteps,  # Shared across samples
            advantages=self.advantages[start:end],
            embeddings=self.embeddings.slice(start, end),
            rewards=self.rewards[start:end] if self.rewards is not None else None,
            prompts=self.prompts[start:end] if self.prompts is not None else None,
            num_steps=self.num_steps,
            is_partitioned=True,
            step_indices=self.step_indices,
            target_sde_indices=self.target_sde_indices,
            sampling_weight_version=self.sampling_weight_version,
        )

    def to_loss_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for legacy loss interfaces.

        Returns:
            Dictionary compatible with GRPOLoss.compute()
        """
        return {
            "trajectories": self.trajectories,
            "latents": self.trajectories,  # Alias for compatibility
            "log_probs_dict": self.log_probs.to_dict(),
            "timesteps": self.timesteps,
            "sigmas": self.timesteps,  # Alias for compatibility
            "sde_indices": self.sde_indices,
            "target_sde_indices": self.target_sde_indices,
            "step_indices": self.resolved_step_indices,
            **self.embeddings.to_dict(),
        }


@dataclass
class NFTTrainingBatch:
    """
    NFT training batch with clean latents only.

    NFT uses forward diffusion in the loss function, so only clean
    latents (x0) are needed - no trajectories or log probabilities.

    Attributes:
        clean_latents: Clean image latents [B, C, H, W]
        advantages: Per-sample advantages [B]
        embeddings: Bundled prompt embeddings
        rewards: Raw rewards [B] (optional, for logging)
        prompts: Original text prompts (optional, for logging)
        timesteps: Optional timestep schedule (e.g., sigmas) [T+1] or [T]
    """
    clean_latents: torch.Tensor     # [B, C, H, W]
    advantages: torch.Tensor        # [B]
    embeddings: PromptEmbeddings
    rewards: Optional[torch.Tensor] = None
    prompts: Optional[List[str]] = None
    timesteps: Optional[torch.Tensor] = None
    is_partitioned: bool = False
    sampling_weight_version: Optional[int] = None

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.clean_latents.shape[0]

    @property
    def device(self) -> torch.device:
        """Get device of the batch tensors."""
        return self.clean_latents.device

    def validate(self) -> None:
        """
        Validate batch consistency.

        Raises:
            ValueError: If batch dimensions are inconsistent
        """
        B = self.clean_latents.shape[0]

        if self.advantages.shape[0] != B:
            raise ValueError(
                f"Advantages batch size {self.advantages.shape[0]} != "
                f"clean_latents batch size {B}"
            )

        if self.embeddings.prompt_embeds.shape[0] != B:
            raise ValueError(
                f"Embeddings batch size {self.embeddings.prompt_embeds.shape[0]} != "
                f"batch size {B}"
            )

    def to_device(self, device: Union[str, "TorchDevice"]) -> "NFTTrainingBatch":
        """Move all tensors to specified device."""
        return NFTTrainingBatch(
            clean_latents=self.clean_latents.to(device),
            advantages=self.advantages.to(device),
            embeddings=self.embeddings.to_device(device),
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            prompts=self.prompts,
            timesteps=self.timesteps.to(device) if self.timesteps is not None else None,
            is_partitioned=self.is_partitioned,
            sampling_weight_version=self.sampling_weight_version,
        )

    def slice(self, start: int, end: int) -> "NFTTrainingBatch":
        """
        Slice batch along sample dimension.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New NFTTrainingBatch with sliced data
        """
        return NFTTrainingBatch(
            clean_latents=self.clean_latents[start:end],
            advantages=self.advantages[start:end],
            embeddings=self.embeddings.slice(start, end),
            rewards=self.rewards[start:end] if self.rewards is not None else None,
            prompts=self.prompts[start:end] if self.prompts is not None else None,
            timesteps=self.timesteps if self.timesteps is not None else None,
            is_partitioned=True,
            sampling_weight_version=self.sampling_weight_version,
        )

    def to_loss_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for legacy loss interfaces.

        Returns:
            Dictionary compatible with NFTLoss.compute()
        """
        return {
            "clean_latents": self.clean_latents,
            "timesteps": self.timesteps,
            **self.embeddings.to_dict(),
        }


# Type alias for union of training batch types
TrainingBatch = Union[GRPOTrainingBatch, NFTTrainingBatch]



def is_grpo_batch(batch: TrainingBatch) -> bool:
    """Check if batch is a GRPO training batch."""
    return isinstance(batch, GRPOTrainingBatch)


def is_nft_batch(batch: TrainingBatch) -> bool:
    """Check if batch is an NFT training batch."""
    return isinstance(batch, NFTTrainingBatch)


# =============================================================================
# Legacy Types (kept for backward compatibility)
# =============================================================================

@dataclass
class SamplerOutput:
    """
    Output from a sampler.

    This is the unified interface for all samplers (FastVideo, image models, etc.)

    Attributes:
        latents: Final denoised latents [B, C, H, W] or [B, C, T, H, W] for video
        timesteps: Sigma schedule [num_steps+1]
        trajectories: Full sampling trajectory [B, num_steps+1, C, ...] (optional)
        log_probs: Typed log probabilities for each SDE step (optional)
        embeddings: Bundled prompt embeddings (optional)
        decoded_images: Already decoded PIL images for reward computation (optional, for image models)
        metadata: Additional sampler-specific data
    """
    latents: torch.Tensor
    timesteps: torch.Tensor
    # Optional fields - not all samplers/algorithms need these
    trajectories: Optional[torch.Tensor] = None
    log_probs: Optional[LogProbData] = None
    embeddings: Optional[PromptEmbeddings] = None
    decoded_images: Optional[List[Any]] = None  # List[PIL.Image.Image]
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = SAMPLING_CONTRACT_VERSION
    step_indices: Optional[torch.Tensor] = None

    @property
    def num_steps(self) -> int:
        """Number of denoising steps."""
        if self.trajectories is not None:
            return self.trajectories.shape[1] - 1
        return self.timesteps.shape[0] - 1

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self.latents.shape[0]

    @property
    def sde_indices(self) -> Set[int]:
        """Get set of timesteps that used SDE (derived from log_probs)."""
        if self.log_probs is not None:
            return self.log_probs.sde_indices
        if isinstance(self.metadata, dict):
            meta_indices = self.metadata.get("sde_indices")
            if meta_indices is not None:
                return set(int(i) for i in meta_indices)
        return set()

    @property
    def resolved_step_indices(self) -> torch.Tensor:
        """Get explicit step indices aligned with timesteps."""
        if self.step_indices is not None:
            return self.step_indices
        return torch.arange(self.timesteps.shape[0], device=self.timesteps.device, dtype=torch.long)

    @property
    def has_trajectories(self) -> bool:
        """Whether this output includes full trajectories."""
        return self.trajectories is not None

    @property
    def has_log_probs(self) -> bool:
        """Whether this output includes log probabilities."""
        return self.log_probs is not None and len(self.log_probs) > 0

    @property
    def has_decoded_images(self) -> bool:
        """Whether this output includes decoded images for reward computation."""
        return self.decoded_images is not None and len(self.decoded_images) > 0

    def to_device(self, device: Union[str, "TorchDevice"]) -> "SamplerOutput":
        """Move all tensors to specified device."""
        return SamplerOutput(
            latents=self.latents.to(device),
            timesteps=self.timesteps.to(device),
            trajectories=self.trajectories.to(device) if self.trajectories is not None else None,
            log_probs=self.log_probs.to_device(device) if self.log_probs is not None else None,
            embeddings=self.embeddings.to_device(device) if self.embeddings is not None else None,
            decoded_images=self.decoded_images,  # PIL images don't need device transfer
            metadata=self.metadata,
            contract_version=self.contract_version,
            step_indices=self.step_indices.to(device) if self.step_indices is not None else None,
        )

    def validate_contract(
        self,
        *,
        requires_log_probs: bool = False,
        requires_trajectory: bool = False,
        requires_embeddings: bool = False,
    ) -> None:
        """Validate SamplingContractV1 consistency."""
        if self.contract_version != SAMPLING_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported sampling contract version: {self.contract_version} "
                f"(expected {SAMPLING_CONTRACT_VERSION})"
            )

        if self.latents is None or self.timesteps is None:
            raise ValueError("SamplerOutput contract violation: latents and timesteps must be present.")

        if self.timesteps.ndim != 1:
            raise ValueError(f"SamplerOutput timesteps must be 1D, got shape={tuple(self.timesteps.shape)}")

        B = int(self.latents.shape[0])
        T_plus_1 = int(self.timesteps.shape[0])

        step_indices = self.resolved_step_indices
        if step_indices.shape[0] != T_plus_1:
            raise ValueError(
                f"SamplerOutput step_indices length {step_indices.shape[0]} != timesteps length {T_plus_1}"
            )
        if step_indices.numel() > 1 and not bool(torch.all(step_indices[1:] > step_indices[:-1])):
            raise ValueError(f"SamplerOutput step_indices must be strictly increasing, got {step_indices.tolist()}")

        if requires_trajectory and self.trajectories is None:
            raise ValueError("SamplerOutput contract violation: trajectories required but missing.")

        if self.trajectories is not None:
            if int(self.trajectories.shape[0]) != B:
                raise ValueError(
                    f"SamplerOutput trajectories batch {self.trajectories.shape[0]} != latents batch {B}"
                )
            if int(self.trajectories.shape[1]) != T_plus_1:
                raise ValueError(
                    f"SamplerOutput trajectories T+1 {self.trajectories.shape[1]} != timesteps length {T_plus_1}"
                )

        if requires_log_probs and not self.has_log_probs:
            raise ValueError("SamplerOutput contract violation: log_probs required but missing.")

        if self.log_probs is not None:
            allowed_steps = set(int(v) for v in step_indices[:-1].tolist())
            for idx, lp in self.log_probs.data.items():
                if idx not in allowed_steps:
                    raise ValueError(
                        f"SamplerOutput contract violation: log_prob index {idx} not in step_indices[:-1]={sorted(allowed_steps)}"
                    )
                if int(lp.shape[0]) != B:
                    raise ValueError(
                        f"SamplerOutput contract violation: log_prob[{idx}] batch {lp.shape[0]} != {B}"
                    )
            if self.sde_indices and set(int(i) for i in self.sde_indices) != set(self.log_probs.data.keys()):
                raise ValueError(
                    "SamplerOutput contract violation: sde_indices must exactly match log_probs keys"
                )

        if requires_embeddings and self.embeddings is None:
            raise ValueError("SamplerOutput contract violation: embeddings required but missing.")

        if self.embeddings is not None and int(self.embeddings.prompt_embeds.shape[0]) != B:
            raise ValueError(
                f"SamplerOutput contract violation: embeddings batch {self.embeddings.prompt_embeds.shape[0]} != {B}"
            )

        if self.metadata is not None:
            trajectory_format = self.metadata.get("trajectory_format")
            timestep_type = self.metadata.get("timestep_type")
            if trajectory_format is None:
                raise ValueError("SamplerOutput contract violation: metadata.trajectory_format is required.")
            if timestep_type is None:
                raise ValueError("SamplerOutput contract violation: metadata.timestep_type is required.")
            valid_formats = {"dense_latent", "video_dense_latent", "packed_seq_c4"}
            if trajectory_format not in valid_formats:
                raise ValueError(
                    f"SamplerOutput contract violation: unknown trajectory_format={trajectory_format}"
                )
            if timestep_type not in {"sigma", "timestep"}:
                raise ValueError(
                    f"SamplerOutput contract violation: unknown timestep_type={timestep_type}"
                )


@dataclass
class InferenceRequest:
    """Request for inference/sampling."""
    prompts: List[str]
    prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None

    # Sampling parameters
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    eta: float = 1.0
    sde_type: str = "sde"

    # Output control
    return_trajectories: bool = True
    return_log_probs: bool = True

    # Reproducibility
    seed: Optional[int] = None
    latents: Optional[torch.Tensor] = None

    # Additional kwargs for specific models
    kwargs: Dict[str, Any] = field(default_factory=dict)


# RewardRequest and RewardResponse are now defined in workers/reward/base.py
# Re-export for backward compatibility
from diffusionrl.workers.reward.base import RewardRequest, RewardResponse, RewardType
