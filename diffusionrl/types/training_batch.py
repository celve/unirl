"""Unified training batch data type for all algorithms.

Replaces the previous ``BackwardTrainingBatch`` / ``ForwardTrainingBatch``
split with a single ``TrainingBatch`` class that stores:

- ``trajectory_store``: compact latent storage via :class:`TrajectoryStore`
- ``forward_context``: model forward parameters via :class:`ForwardContext`
- ``log_probs``, ``advantages``, ``timesteps``, etc.

NFT compatibility:

- ``batch.clean_latents`` returns ``trajectory_store.clean_latents``
- ``batch.has_trajectory_rl_data`` returns ``False`` when log_probs is empty

GRPO compatibility:

- ``batch.trajectories`` returns ``trajectory_store.data`` (full trajectory)
- ``batch.get_timestep_data_by_step(step_idx)`` works as before
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import torch

from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.sample import LogProbData
from diffusionrl.utils.batched import FieldKind, concat_field, field, shared_field

from .forward_context import ForwardContext
from .trajectory_store import TrajectoryStore

if TYPE_CHECKING:
    from torch import device as TorchDevice


@dataclass
class TimestepData:
    """Single timestep data for GRPO loss computation."""

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
            sigma=self.sigma.to(device) if isinstance(self.sigma, torch.Tensor) else self.sigma,
            sigma_next=self.sigma_next.to(device) if isinstance(self.sigma_next, torch.Tensor) else self.sigma_next,
            timestep_idx=self.timestep_idx,
            sigmas=self.sigmas.to(device) if self.sigmas is not None else None,
        )


@dataclass
class TrainingBatch(Transportable):
    """Unified training batch for all algorithms.

    Combines what was previously ``BackwardTrainingBatch`` (GRPO)
    and ``ForwardTrainingBatch`` (NFT) into a single type.

    ``trajectory_store`` and ``forward_context`` are tagged
    ``transport=True`` so a single ``hydrate`` walk fetches both subtrees in
    one TQ round-trip.
    """

    # Required fields (no defaults) — must come first for dataclass
    trajectory_store: TrajectoryStore = field(kind=FieldKind.CONCAT, transport=True)
    timesteps: torch.Tensor = shared_field()
    advantages: torch.Tensor = concat_field()
    forward_context: ForwardContext = field(kind=FieldKind.CONCAT, transport=True)

    # Optional per-sample fields
    log_probs: Optional[LogProbData] = concat_field(default=None)
    rewards: Optional[torch.Tensor] = concat_field(default=None)
    # Per-sample-per-component breakdown of `rewards`; observability-only
    # (compute_rollout_batch_metrics), unused by loss/advantage code.
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    prompts: Optional[Prompts] = concat_field(default=None)
    extras: Dict[str, Any] = concat_field(default_factory=dict)

    # Optional shared fields
    step_indices: Optional[torch.Tensor] = shared_field(default=None)
    target_sde_indices: Optional[Set[int]] = shared_field(default=None)

    # ---- core properties ----------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.trajectory_store.batch_size

    @property
    def device(self) -> torch.device:
        return self.trajectory_store.device

    # ---- prompt ID compat properties -----------------------------------------

    @property
    def prompt_ids(self) -> Optional[List[str]]:
        return self.prompts.prompt_ids if self.prompts is not None else None

    @property
    def sample_ids(self) -> Optional[List[str]]:
        return self.prompts.sample_ids if self.prompts is not None else None

    @property
    def group_ids(self) -> Optional[List[str]]:
        return self.prompts.group_ids if self.prompts is not None else None

    # ---- trajectory access (GRPO compat) ------------------------------------

    @property
    def trajectories(self) -> torch.Tensor:
        """Full trajectory tensor [B, T+1, ...] — GRPO compatibility.

        .. warning:: When ``trajectory_store.is_selective`` is ``True``, the
           data tensor is **not** a dense ``[B, T+1, ...]`` tensor.  Use
           ``trajectory_store.get_pair(pos)`` for position-safe access.
        """
        if self.trajectory_store.is_selective:
            warnings.warn(
                "Accessing batch.trajectories on a selective TrajectoryStore. "
                "The data tensor has fewer columns than T+1. "
                "Use trajectory_store.get_pair(pos) for safe access.",
                stacklevel=2,
            )
        return self.trajectory_store.data

    @property
    def clean_latents(self) -> torch.Tensor:
        """Final denoised latents — NFT compatibility."""
        return self.trajectory_store.clean_latents

    @property
    def has_trajectory_rl_data(self) -> bool:
        """True when this batch carries trajectory RL data (GRPO-style)."""
        return (
            self.log_probs is not None
            and len(self.log_probs.data) > 0
            and not self.trajectory_store.is_clean_latents_only
        )

    # ---- SDE step indexing --------------------------------------------------

    @property
    def sde_indices(self) -> Set[int]:
        """Timestep indices that used SDE (have log_probs)."""
        if self.log_probs is not None and len(self.log_probs.data) > 0:
            return set(int(k) for k in self.log_probs.data.keys())
        if self.target_sde_indices is not None:
            return set(int(i) for i in self.target_sde_indices)
        return set()

    @property
    def resolved_step_indices(self) -> torch.Tensor:
        """Explicit step labels aligned with timesteps/trajectory axis."""
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
            raise ValueError(f"step_idx={step_idx} not present in step_indices={steps.tolist()}")
        pos = int(hits[0].item())
        if not self.trajectory_store.has_position(pos) or not self.trajectory_store.has_position(pos + 1):
            raise ValueError(
                f"step_idx={step_idx} maps to position {pos} but the (pos, pos+1) pair "
                f"is not available in trajectory_store "
                f"(stored positions: {self.trajectory_store.stored_positions})"
            )
        return pos

    def get_timestep_data(self, t_idx: int) -> TimestepData:
        """Extract data for a specific trajectory position index."""
        if not self._is_contiguous_step_index():
            raise ValueError("Non-contiguous step_indices detected. Use get_timestep_data_by_step(step_idx) instead.")
        log_prob = self.log_probs[t_idx] if self.log_probs is not None else None
        latents, next_latents = self.trajectory_store.get_pair(t_idx)
        return TimestepData(
            latents=latents,
            next_latents=next_latents,
            log_prob=log_prob,
            sigma=self.timesteps[t_idx],
            sigma_next=self.timesteps[t_idx + 1],
            timestep_idx=t_idx,
            sigmas=self.timesteps,
        )

    def get_timestep_data_by_step(self, step_idx: int) -> TimestepData:
        """Extract data for a logical step label from step_indices."""
        pos = self.get_position_for_step(step_idx)
        log_prob = self.log_probs[int(step_idx)] if self.log_probs is not None else None
        latents, next_latents = self.trajectory_store.get_pair(pos)
        return TimestepData(
            latents=latents,
            next_latents=next_latents,
            log_prob=log_prob,
            sigma=self.timesteps[pos],
            sigma_next=self.timesteps[pos + 1],
            timestep_idx=int(step_idx),
            sigmas=self.timesteps,
        )

    # ---- timestep lookup by value -------------------------------------------

    def get_timestep_data_by_timestep(self, timestep: Any) -> TimestepData:
        """Extract data for a timestep value from the batch timestep schedule."""
        step_idx = self.get_step_for_timestep(timestep)
        return self.get_timestep_data_by_step(step_idx)

    def get_step_for_timestep(self, timestep: Any) -> int:
        """Resolve the logical step label corresponding to a timestep value."""
        timestep_tensor = torch.as_tensor(
            timestep,
            device=self.timesteps.device,
            dtype=self.timesteps.dtype,
        )
        hits = (self.timesteps[:-1] == timestep_tensor).nonzero(as_tuple=False)
        if hits.numel() == 0:
            raise ValueError(
                f"timestep={timestep_tensor.item()!r} not present in timesteps={self.timesteps[:-1].tolist()}"
            )
        pos = int(hits[0].item())
        return int(self.resolved_step_indices[pos].item())

    # ---- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Validate batch consistency."""
        bs = self.batch_size

        if self.advantages.shape[0] != bs:
            raise ValueError(f"Advantages batch size {self.advantages.shape[0]} != trajectory batch size {bs}")

        if self.trajectory_store.is_full:
            steps_count = self.trajectory_store.num_stored - 1
            if self.timesteps.shape[0] != steps_count + 1:
                raise ValueError(f"Timesteps length {self.timesteps.shape[0]} != expected {steps_count + 1}")

            step_indices = self.resolved_step_indices
            if int(step_indices.shape[0]) != steps_count + 1:
                raise ValueError(f"Step indices length {step_indices.shape[0]} != expected {steps_count + 1}")
            if step_indices.numel() > 1 and not bool(torch.all(step_indices[1:] > step_indices[:-1])):
                raise ValueError(f"step_indices must be strictly increasing, got: {step_indices.tolist()}")
            if self.target_sde_indices is not None:
                allowed_steps = set(int(v) for v in step_indices[:-1].tolist())
                bad = sorted(int(i) for i in self.target_sde_indices if int(i) not in allowed_steps)
                if bad:
                    raise ValueError(
                        f"target_sde_indices contain out-of-range steps: {bad}, allowed={sorted(allowed_steps)}"
                    )

        elif self.trajectory_store.is_selective:
            # Selective store: only validate that target_sde_indices
            # have both (pos, pos+1) positions stored.
            if self.target_sde_indices is not None:
                for idx in sorted(self.target_sde_indices):
                    if not self.trajectory_store.has_position(idx):
                        raise ValueError(
                            f"target_sde_indices step {idx} not stored in "
                            f"selective trajectory (stored: {self.trajectory_store.stored_positions})"
                        )
                    if not self.trajectory_store.has_position(idx + 1):
                        raise ValueError(
                            f"target_sde_indices step {idx} requires position {idx + 1} "
                            f"but it is not stored in selective trajectory "
                            f"(stored: {self.trajectory_store.stored_positions})"
                        )

        ctx_bs = self.forward_context.batch_size
        if ctx_bs > 0 and ctx_bs != bs:
            raise ValueError(f"ForwardContext batch size {ctx_bs} != trajectory batch size {bs}")
        for name in ("sample_ids", "prompt_ids", "group_ids"):
            ids = getattr(self, name)
            if ids is not None and len(ids) != bs:
                raise ValueError(f"{name} length {len(ids)} != batch size {bs}")

        if self.log_probs is not None:
            for idx, log_prob in self.log_probs.data.items():
                if log_prob.shape[0] != bs:
                    raise ValueError(f"Log prob at index {idx} has batch size {log_prob.shape[0]} != {bs}")

    # ---- modality detection -------------------------------------------------

    def detect_modality(self) -> str:
        """Detect media modality from trajectory data shape."""
        return self.trajectory_store.detect_modality()


__all__ = [
    "TimestepData",
    "TrainingBatch",
    "TrajectoryStore",
]
