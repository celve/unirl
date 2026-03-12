"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import torch

if TYPE_CHECKING:
    from torch import device as TorchDevice

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
        return LogProbData(data={k: v[start:end] for k, v in self.data.items()})

    def reindex(self, indices: torch.Tensor) -> "LogProbData":
        """
        Reindex log probabilities along batch dimension using a permutation tensor.

        Args:
            indices: 1-D LongTensor of sample indices (e.g. from torch.randperm)

        Returns:
            New LogProbData with reindexed tensors
        """
        return LogProbData(data={k: v[indices] for k, v in self.data.items()})

    def to_device(self, device: Union[str, "TorchDevice"]) -> "LogProbData":
        """Move all tensors to specified device."""
        return LogProbData(data={k: v.to(device) for k, v in self.data.items()})


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
                if self.pooled_prompt_embeds is not None
                else None
            ),
            encoder_attention_mask=(
                self.encoder_attention_mask.to(device)
                if self.encoder_attention_mask is not None
                else None
            ),
            negative_prompt_embeds=(
                self.negative_prompt_embeds.to(device)
                if self.negative_prompt_embeds is not None
                else None
            ),
            negative_pooled_prompt_embeds=(
                self.negative_pooled_prompt_embeds.to(device)
                if self.negative_pooled_prompt_embeds is not None
                else None
            ),
            text_ids=self.text_ids.to(device) if self.text_ids is not None else None,
            image_ids=self.image_ids.to(device) if self.image_ids is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for downstream consumers."""
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

        def _slice_if_batched(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if tensor.shape[0] == batch_size:
                return tensor[start:end]
            return tensor

        return PromptEmbeddings(
            prompt_embeds=self.prompt_embeds[start:end],
            pooled_prompt_embeds=_slice_if_batched(self.pooled_prompt_embeds),
            encoder_attention_mask=_slice_if_batched(self.encoder_attention_mask),
            negative_prompt_embeds=_slice_if_batched(self.negative_prompt_embeds),
            negative_pooled_prompt_embeds=_slice_if_batched(
                self.negative_pooled_prompt_embeds
            ),
            text_ids=_slice_if_batched(self.text_ids),
            image_ids=_slice_if_batched(self.image_ids),
        )

    def reindex(self, indices: torch.Tensor) -> "PromptEmbeddings":
        """
        Reindex embeddings along batch dimension using a permutation tensor.

        For FLUX models, image_ids are NOT batched (shared across batch)
        and are kept as-is.

        Args:
            indices: 1-D LongTensor of sample indices (e.g. from torch.randperm)

        Returns:
            New PromptEmbeddings with reindexed tensors
        """
        batch_size = self.prompt_embeds.shape[0]

        def _reindex_if_batched(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if tensor.shape[0] == batch_size:
                return tensor[indices]
            return tensor

        return PromptEmbeddings(
            prompt_embeds=self.prompt_embeds[indices],
            pooled_prompt_embeds=_reindex_if_batched(self.pooled_prompt_embeds),
            encoder_attention_mask=_reindex_if_batched(self.encoder_attention_mask),
            negative_prompt_embeds=_reindex_if_batched(self.negative_prompt_embeds),
            negative_pooled_prompt_embeds=_reindex_if_batched(
                self.negative_pooled_prompt_embeds
            ),
            text_ids=_reindex_if_batched(self.text_ids),
            image_ids=_reindex_if_batched(self.image_ids),
        )


@dataclass
class RolloutOutput:
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
    trajectories: Optional[torch.Tensor] = None
    log_probs: Optional[LogProbData] = None
    embeddings: Optional[PromptEmbeddings] = None
    decoded_images: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
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
        return torch.arange(
            self.timesteps.shape[0], device=self.timesteps.device, dtype=torch.long
        )

    @property
    def has_log_probs(self) -> bool:
        """Whether this output includes log probabilities."""
        return self.log_probs is not None and len(self.log_probs) > 0

    @property
    def has_decoded_images(self) -> bool:
        """Whether this output includes decoded images for reward computation."""
        return self.decoded_images is not None and len(self.decoded_images) > 0

    def to_device(self, device: Union[str, "TorchDevice"]) -> "RolloutOutput":
        """Move all tensors to specified device."""
        return RolloutOutput(
            latents=self.latents.to(device),
            timesteps=self.timesteps.to(device),
            trajectories=self.trajectories.to(device)
            if self.trajectories is not None
            else None,
            log_probs=self.log_probs.to_device(device) if self.log_probs is not None else None,
            embeddings=self.embeddings.to_device(device)
            if self.embeddings is not None
            else None,
            decoded_images=self.decoded_images,
            metadata=self.metadata,
            step_indices=self.step_indices.to(device)
            if self.step_indices is not None
            else None,
        )

    def validate_contract(
        self,
        *,
        requires_log_probs: bool = False,
        requires_trajectory: bool = False,
        requires_embeddings: bool = False,
    ) -> None:
        """Validate SamplingContractV1 consistency."""
        if self.latents is None or self.timesteps is None:
            raise ValueError(
                "RolloutOutput contract violation: latents and timesteps must be present."
            )

        if self.timesteps.ndim != 1:
            raise ValueError(
                f"RolloutOutput timesteps must be 1D, got shape={tuple(self.timesteps.shape)}"
            )

        batch_size = int(self.latents.shape[0])
        t_plus_1 = int(self.timesteps.shape[0])

        step_indices = self.resolved_step_indices
        if step_indices.shape[0] != t_plus_1:
            raise ValueError(
                f"RolloutOutput step_indices length {step_indices.shape[0]} != timesteps length {t_plus_1}"
            )
        if step_indices.numel() > 1 and not bool(
            torch.all(step_indices[1:] > step_indices[:-1])
        ):
            raise ValueError(
                f"RolloutOutput step_indices must be strictly increasing, got {step_indices.tolist()}"
            )

        if requires_trajectory and self.trajectories is None:
            raise ValueError(
                "RolloutOutput contract violation: trajectories required but missing."
            )

        if self.trajectories is not None:
            if int(self.trajectories.shape[0]) != batch_size:
                raise ValueError(
                    f"RolloutOutput trajectories batch {self.trajectories.shape[0]} != latents batch {batch_size}"
                )
            if int(self.trajectories.shape[1]) != t_plus_1:
                raise ValueError(
                    f"RolloutOutput trajectories T+1 {self.trajectories.shape[1]} != timesteps length {t_plus_1}"
                )

        if requires_log_probs and not self.has_log_probs:
            raise ValueError(
                "RolloutOutput contract violation: log_probs required but missing."
            )

        if self.log_probs is not None:
            allowed_steps = set(int(v) for v in step_indices[:-1].tolist())
            for idx, lp in self.log_probs.data.items():
                if idx not in allowed_steps:
                    raise ValueError(
                        f"RolloutOutput contract violation: log_prob index {idx} not in step_indices[:-1]={sorted(allowed_steps)}"
                    )
                if int(lp.shape[0]) != batch_size:
                    raise ValueError(
                        f"RolloutOutput contract violation: log_prob[{idx}] batch {lp.shape[0]} != {batch_size}"
                    )
            if self.sde_indices and set(int(i) for i in self.sde_indices) != set(
                self.log_probs.data.keys()
            ):
                raise ValueError(
                    "RolloutOutput contract violation: sde_indices must exactly match log_probs keys"
                )

        if requires_embeddings and self.embeddings is None:
            raise ValueError(
                "RolloutOutput contract violation: embeddings required but missing."
            )

        if (
            self.embeddings is not None
            and int(self.embeddings.prompt_embeds.shape[0]) != batch_size
        ):
            raise ValueError(
                f"RolloutOutput contract violation: embeddings batch {self.embeddings.prompt_embeds.shape[0]} != {batch_size}"
            )

        if self.metadata is not None:
            trajectory_format = self.metadata.get("trajectory_format")
            timestep_type = self.metadata.get("timestep_type")
            if trajectory_format is None:
                raise ValueError(
                    "RolloutOutput contract violation: metadata.trajectory_format is required."
                )
            if timestep_type is None:
                raise ValueError(
                    "RolloutOutput contract violation: metadata.timestep_type is required."
                )
            valid_formats = {"dense_latent", "video_dense_latent", "packed_seq_c4"}
            if trajectory_format not in valid_formats:
                raise ValueError(
                    f"RolloutOutput contract violation: unknown trajectory_format={trajectory_format}"
                )
            if timestep_type not in {"sigma", "timestep"}:
                raise ValueError(
                    f"RolloutOutput contract violation: unknown timestep_type={timestep_type}"
                )


@dataclass
class RolloutRequest:
    """Request for rollout generation.

    This is the single interface contract for all ``generate()`` calls
    throughout the rollout pipeline (engines, actors, actor-groups,
    distributed helpers).

    External callers are expected to provide text prompts. Optional embedding
    tensor fields remain available only for internal compatibility paths and
    fallback plumbing.
    """

    prompts: List[str]
    prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    eta: float = 1.0
    sde_type: str = "sde"
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = None
    latents: Optional[torch.Tensor] = None
    sde_indices: Optional[Set[int]] = None
    decode_for_reward: bool = False
    sampling_adapter: Optional[str] = None
    return_trajectories: bool = True
    return_log_probs: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def slice_prompts(self, start: int, end: int) -> "RolloutRequest":
        """Create a sub-request with sliced prompts for distributed actors.

        Tensor fields that have a batch dimension matching len(prompts)
        are sliced accordingly.  Scalar / None fields are copied as-is.
        """
        import copy
        req = copy.copy(self)
        req.prompts = self.prompts[start:end]

        # Slice tensor fields that may be batched along dim-0
        for attr in (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "encoder_attention_mask",
            "text_ids",
            "latents",
        ):
            val = getattr(self, attr, None)
            if val is not None and isinstance(val, torch.Tensor) and val.shape[0] == len(self.prompts):
                setattr(req, attr, val[start:end])

        # kwargs is shallow-copied to avoid mutation
        req.kwargs = dict(self.kwargs)
        return req


__all__ = [
    "RolloutOutput",
    "RolloutRequest",
    "LogProbData",
    "PromptEmbeddings",
]
