"""Training batch data types for algorithms and loss modules.

`BackwardTrainingBatch` is for trajectory-based GRPO/MixGRPO optimization.
`ForwardTrainingBatch` is for NFT optimization on clean latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import torch

from .sampling import LogProbData, PromptEmbeddings

if TYPE_CHECKING:
    from torch import device as TorchDevice


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
        sigmas: Full sigma schedule [T+1] for all timesteps.
            Used to derive sigma_max (sigmas[1]) for log_prob boundary handling,
            ensuring training-inference consistency.
    """

    latents: torch.Tensor
    next_latents: torch.Tensor
    log_prob: Optional[torch.Tensor]
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    timestep_idx: int = 0
    sigmas: Optional[torch.Tensor] = None

    def to_device(self, device: Union[str, "TorchDevice"]) -> "TimestepData":
        """Move all tensors to specified device."""
        return TimestepData(
            latents=self.latents.to(device),
            next_latents=self.next_latents.to(device),
            log_prob=self.log_prob.to(device) if self.log_prob is not None else None,
            sigma=self.sigma.to(device)
            if isinstance(self.sigma, torch.Tensor)
            else self.sigma,
            sigma_next=self.sigma_next.to(device)
            if isinstance(self.sigma_next, torch.Tensor)
            else self.sigma_next,
            timestep_idx=self.timestep_idx,
            sigmas=self.sigmas.to(device)
            if self.sigmas is not None
            else None,
        )


# Used by GRPO/MixGRPO: requires full trajectory + old log_probs for IS/PPO-style objectives.
@dataclass
class BackwardTrainingBatch:
    """
    GRPO/MixGRPO training batch with trajectories and log probabilities.

    This batch type is used for trajectory-based algorithms that require
    the full sampling path and importance sampling ratios.
    """

    trajectories: torch.Tensor
    log_probs: LogProbData
    timesteps: torch.Tensor
    advantages: torch.Tensor
    embeddings: PromptEmbeddings
    rewards: Optional[torch.Tensor] = None
    prompts: Optional[List[str]] = None
    num_steps: int = 50
    is_partitioned: bool = False
    step_indices: Optional[torch.Tensor] = None
    target_sde_indices: Optional[Set[int]] = None

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
        return torch.arange(
            self.timesteps.shape[0], device=self.timesteps.device, dtype=torch.long
        )

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
        batch_size = self.trajectories.shape[0]
        steps_count = self.trajectories.shape[1] - 1

        if self.advantages.shape[0] != batch_size:
            raise ValueError(
                f"Advantages batch size {self.advantages.shape[0]} != "
                f"trajectories batch size {batch_size}"
            )

        if self.timesteps.shape[0] != steps_count + 1:
            raise ValueError(
                f"Timesteps length {self.timesteps.shape[0]} != "
                f"expected {steps_count + 1}"
            )

        step_indices = self.resolved_step_indices
        if int(step_indices.shape[0]) != steps_count + 1:
            raise ValueError(
                f"Step indices length {step_indices.shape[0]} != expected {steps_count + 1}"
            )
        if step_indices.numel() > 1 and not bool(
            torch.all(step_indices[1:] > step_indices[:-1])
        ):
            raise ValueError(
                f"step_indices must be strictly increasing, got: {step_indices.tolist()}"
            )
        if self.target_sde_indices is not None:
            allowed_steps = set(int(v) for v in step_indices[:-1].tolist())
            bad = sorted(
                int(i) for i in self.target_sde_indices if int(i) not in allowed_steps
            )
            if bad:
                raise ValueError(
                    f"target_sde_indices contain out-of-range steps: {bad}, allowed={sorted(allowed_steps)}"
                )

        if self.embeddings.prompt_embeds.shape[0] != batch_size:
            raise ValueError(
                f"Embeddings batch size {self.embeddings.prompt_embeds.shape[0]} != "
                f"batch size {batch_size}"
            )

        for idx, log_prob in self.log_probs.data.items():
            if log_prob.shape[0] != batch_size:
                raise ValueError(
                    f"Log prob at index {idx} has batch size {log_prob.shape[0]} != {batch_size}"
                )

    def to_device(self, device: Union[str, "TorchDevice"]) -> "BackwardTrainingBatch":
        """Move all tensors to specified device."""
        return BackwardTrainingBatch(
            trajectories=self.trajectories.to(device),
            log_probs=self.log_probs.to_device(device),
            timesteps=self.timesteps.to(device),
            advantages=self.advantages.to(device),
            embeddings=self.embeddings.to_device(device),
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            prompts=self.prompts,
            num_steps=self.num_steps,
            is_partitioned=self.is_partitioned,
            step_indices=self.step_indices.to(device)
            if self.step_indices is not None
            else None,
            target_sde_indices=self.target_sde_indices,
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
            sigmas=self.timesteps,
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
            sigmas=self.timesteps,
        )

    def slice(self, start: int, end: int) -> "BackwardTrainingBatch":
        """
        Slice batch along sample dimension for micro-batch gradient accumulation.

        This method is dimension-agnostic and supports both image and video:
        - Image trajectories: [B, T+1, C, H, W]
        - Video trajectories: [B, T+1, C, T_frames, H, W]
        """
        return BackwardTrainingBatch(
            trajectories=self.trajectories[start:end],
            log_probs=self.log_probs.slice(start, end),
            timesteps=self.timesteps,
            advantages=self.advantages[start:end],
            embeddings=self.embeddings.slice(start, end),
            rewards=self.rewards[start:end] if self.rewards is not None else None,
            prompts=self.prompts[start:end] if self.prompts is not None else None,
            num_steps=self.num_steps,
            is_partitioned=True,
            step_indices=self.step_indices,
            target_sde_indices=self.target_sde_indices,
        )

    def to_loss_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for loss interfaces.

        Returns:
            Dictionary compatible with GRPOLoss.compute_timestep()
        """
        return {
            "trajectories": self.trajectories,
            "latents": self.trajectories,
            "log_probs_dict": self.log_probs.to_dict(),
            "timesteps": self.timesteps,
            "sigmas": self.timesteps,
            "sde_indices": self.sde_indices,
            "target_sde_indices": self.target_sde_indices,
            "step_indices": self.resolved_step_indices,
            **self.embeddings.to_dict(),
        }


# Used by NFT: consumes clean latents (x0) and constructs forward diffusion states in loss.
@dataclass
class ForwardTrainingBatch:
    """
    NFT training batch with clean latents only.

    NFT uses forward diffusion in the loss function, so only clean
    latents (x0) are needed - no trajectories or log probabilities.
    """

    clean_latents: torch.Tensor
    advantages: torch.Tensor
    embeddings: PromptEmbeddings
    rewards: Optional[torch.Tensor] = None
    prompts: Optional[List[str]] = None
    timesteps: Optional[torch.Tensor] = None
    is_partitioned: bool = False

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
        batch_size = self.clean_latents.shape[0]

        if self.advantages.shape[0] != batch_size:
            raise ValueError(
                f"Advantages batch size {self.advantages.shape[0]} != "
                f"clean_latents batch size {batch_size}"
            )

        if self.embeddings.prompt_embeds.shape[0] != batch_size:
            raise ValueError(
                f"Embeddings batch size {self.embeddings.prompt_embeds.shape[0]} != "
                f"batch size {batch_size}"
            )

    def to_device(self, device: Union[str, "TorchDevice"]) -> "ForwardTrainingBatch":
        """Move all tensors to specified device."""
        return ForwardTrainingBatch(
            clean_latents=self.clean_latents.to(device),
            advantages=self.advantages.to(device),
            embeddings=self.embeddings.to_device(device),
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            prompts=self.prompts,
            timesteps=self.timesteps.to(device) if self.timesteps is not None else None,
            is_partitioned=self.is_partitioned,
        )

    def slice(self, start: int, end: int) -> "ForwardTrainingBatch":
        """
        Slice batch along sample dimension.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New ForwardTrainingBatch with sliced data
        """
        return ForwardTrainingBatch(
            clean_latents=self.clean_latents[start:end],
            advantages=self.advantages[start:end],
            embeddings=self.embeddings.slice(start, end),
            rewards=self.rewards[start:end] if self.rewards is not None else None,
            prompts=self.prompts[start:end] if self.prompts is not None else None,
            timesteps=self.timesteps if self.timesteps is not None else None,
            is_partitioned=True,
        )

    def to_loss_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for loss interfaces.

        Returns:
            Dictionary compatible with NFTLoss.compute_batch()
        """
        return {
            "clean_latents": self.clean_latents,
            "timesteps": self.timesteps,
            **self.embeddings.to_dict(),
        }


TrainingBatch = Union[BackwardTrainingBatch, ForwardTrainingBatch]


def is_backward_batch(batch: TrainingBatch) -> bool:
    """Check if batch is a BackwardTrainingBatch (used by GRPO/MixGRPO)."""
    return isinstance(batch, BackwardTrainingBatch)


def is_forward_batch(batch: TrainingBatch) -> bool:
    """Check if batch is a ForwardTrainingBatch (used by NFT)."""
    return isinstance(batch, ForwardTrainingBatch)

__all__ = [
    "BackwardTrainingBatch",
    "ForwardTrainingBatch",
    "TimestepData",
    "TrainingBatch",
    "is_backward_batch",
    "is_forward_batch",
]
