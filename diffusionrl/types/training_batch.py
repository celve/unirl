"""Unified training batch data type for all algorithms.

Replaces the previous ``BackwardTrainingBatch`` / ``ForwardTrainingBatch``
split with a single ``TrainingBatch`` class that stores:

- ``trajectory_store``: compact latent storage via :class:`TrajectoryStore`
- ``forward_context``: model forward parameters via :class:`ForwardContext`
- ``log_probs``, ``advantages``, ``timesteps``, etc.

Coordinate system
-----------------

There is a single index space for denoising steps:

  ``step_idx`` ∈ ``{0, 1, ..., T-1}``

where ``T = timesteps.shape[0] - 1`` (the number of denoising steps).
``step_idx`` directly indexes into:

- ``timesteps[step_idx]`` → sigma for step ``step_idx``
- ``timesteps[step_idx + 1]`` → sigma_next
- ``trajectory_store.get_pair(step_idx)`` → ``(latents, next_latents)``
- ``log_probs[step_idx]`` → old log_prob

``step_idx`` IS the trajectory position — there is no separate "position"
coordinate.  This is a structural guarantee, not a runtime assertion.

SSOT contract (trajectory-RL path)
----------------------------------

For any batch whose trajectory store is **not** clean-latents-only, the
following invariants hold and are enforced by :meth:`TrainingBatch.validate`:

1. ``target_sde_indices`` is the **single source of truth** for the set of
   step indices this batch trains on.  It is populated by
   ``assemble_training_batch`` from the scheduler-provided ``sde_indices``,
   narrowed only by ``get_filtered_training_indices`` (skip_last_timestep /
   skip_initial_timesteps).  Any mismatch with ``trajectory_store`` or
   ``log_probs`` raises — no silent clamping.
2. When ``log_probs`` is present and non-empty,
   ``log_probs.data.keys() == target_sde_indices``.  On the normal
   trajectory-RL path ``assemble_training_batch`` guarantees ``log_probs``
   is always populated, so this is effectively unconditional in practice.
3. For every ``idx`` in ``target_sde_indices``, both position ``idx`` and
   position ``idx + 1`` are stored in ``trajectory_store``.
4. ``batch.sde_indices`` returns ``target_sde_indices`` directly, so all
   downstream consumers (GRPO loss, replay, metrics) see the same step set.

NFT compatibility:

- ``batch.clean_latents`` returns ``trajectory_store.clean_latents``
- ``batch.has_trajectory_rl_data`` returns ``False`` when log_probs is empty
- NFT batches skip the SSOT invariants above (clean-latents-only path)

GRPO compatibility:

- ``batch.trajectories`` returns ``trajectory_store.data`` (full trajectory)
- ``batch.get_timestep_data_by_step(step_idx)`` works as before
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import torch

from .batch_ops import (
    batch_clone,
    batch_concat,
    batch_move,
    batch_reindex,
    batch_slice,
)
from .forward_context import ForwardContext
from .sampling import LogProbData
from .trajectory_store import TrajectoryStore

if TYPE_CHECKING:
    from torch import device as TorchDevice


def build_rollout_extras(*, request: Any, sampler_outputs: List[Any]) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}

    request_meta = getattr(request, "meta", None)
    if isinstance(request_meta, dict) and request_meta:
        extras["request_meta"] = batch_clone(request_meta)

    sample_meta_values: List[Any] = []
    sample_meta_batch_sizes: List[int] = []
    for output in sampler_outputs:
        meta = getattr(output, "meta", None)
        latents = getattr(output, "latents", None)
        if not isinstance(meta, dict) or not meta:
            continue
        if not torch.is_tensor(latents):
            continue
        sample_meta_values.append(meta)
        sample_meta_batch_sizes.append(int(latents.shape[0]))
    if sample_meta_values:
        extras["sample_meta"] = batch_concat(
            sample_meta_values,
            batch_sizes=sample_meta_batch_sizes,
            deep_clone=True,
            strict=False,
        )

    return extras


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


@dataclass
class TrainingBatch:
    """Unified training batch for all algorithms.

    Combines what was previously ``BackwardTrainingBatch`` (GRPO)
    and ``ForwardTrainingBatch`` (NFT) into a single type.
    """

    trajectory_store: TrajectoryStore
    timesteps: torch.Tensor
    advantages: torch.Tensor
    forward_context: ForwardContext

    log_probs: Optional[LogProbData] = None
    rewards: Optional[torch.Tensor] = None
    prompts: Optional[List[str]] = None
    prompt_ids: Optional[List[str]] = None
    sample_ids: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
    is_partitioned: bool = False
    target_sde_indices: Optional[Set[int]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---- core properties ----------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.trajectory_store.batch_size

    @property
    def device(self) -> torch.device:
        return self.trajectory_store.device

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
            and len(self.log_probs) > 0
            and not self.trajectory_store.is_clean_latents_only
        )

    # ---- SDE step indexing --------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Number of denoising steps T.  ``timesteps`` has T+1 elements."""
        return int(self.timesteps.shape[0]) - 1

    @property
    def sde_indices(self) -> Set[int]:
        """Step indices this batch trains on (single source of truth).

        Returns ``target_sde_indices`` for the trajectory-RL path (populated
        by ``assemble_training_batch`` from ``scheduler.resolve_rollout_sde_indices``).
        Returns an empty set on the NFT / clean-latents-only path, which does
        not train per step.
        """
        if self.target_sde_indices is None:
            return set()
        return set(int(i) for i in self.target_sde_indices)

    @property
    def resolved_step_indices(self) -> torch.Tensor:
        """Compatibility shim — returns ``[0, 1, ..., T]``.

        .. deprecated:: Derived from ``num_steps``.  Prefer ``num_steps``
           and ``step_labels`` for new code.  Do not extend this property
           to support non-contiguous indices.
        """
        return torch.arange(
            self.num_steps + 1,
            device=self.timesteps.device,
            dtype=torch.long,
        )

    @property
    def step_labels(self) -> Set[int]:
        """Trainable step labels ``{0, 1, ..., T-1}``.

        Each step ``i`` uses trajectory positions ``i`` and ``i+1``.
        Derived from ``num_steps``.
        """
        return set(range(self.num_steps))

    def get_position_for_step(self, step_idx: int) -> int:
        """Validate and return step_idx as a trajectory position.

        ``step_idx`` IS the position (structural guarantee).  This method
        only does bounds checking and trajectory-pair validation.
        """
        pos = int(step_idx)
        T = self.num_steps
        if pos < 0 or pos >= T:
            raise ValueError(
                f"step_idx={step_idx} out of range [0, {T})"
            )
        if not self.trajectory_store.has_position(pos) or not self.trajectory_store.has_position(pos + 1):
            raise ValueError(
                f"step_idx={step_idx} requires positions {pos} and {pos + 1} "
                f"but trajectory_store only has: "
                f"{self.trajectory_store.stored_positions}"
            )
        return pos

    def get_timestep_data(self, t_idx: int) -> TimestepData:
        """Extract data for a specific step index."""
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
        """Extract all data needed for loss computation at a given step.

        Validates step_idx bounds and trajectory-pair availability,
        then delegates to ``get_timestep_data``.
        """
        pos = self.get_position_for_step(step_idx)
        return self.get_timestep_data(pos)

    # ---- timestep lookup by value -------------------------------------------

    def get_timestep_data_by_timestep(self, timestep: Any) -> TimestepData:
        """Extract data for a timestep value from the batch timestep schedule."""
        step_idx = self.get_step_for_timestep(timestep)
        return self.get_timestep_data_by_step(step_idx)

    def get_step_for_timestep(self, timestep: Any) -> int:
        """Resolve the step label corresponding to a timestep sigma value.

        Searches ``timesteps[:-1]`` (the T denoising sigmas) for a match.
        Since step_idx == position, the array index IS the step label.
        """
        timestep_tensor = torch.as_tensor(
            timestep,
            device=self.timesteps.device,
            dtype=self.timesteps.dtype,
        )
        if timestep_tensor.ndim > 0:
            timestep_tensor = timestep_tensor.flatten()[0]
        matches = (self.timesteps[:-1] - timestep_tensor).abs() < 1e-6
        hits = matches.nonzero(as_tuple=False)
        if hits.numel() == 0:
            raise ValueError(
                f"timestep={timestep_tensor.item()!r} not found in "
                f"timesteps={self.timesteps[:-1].tolist()}"
            )
        return int(hits[0].item())

    # ---- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Validate batch consistency.

        Structural invariant:
          ``step_idx == position``.  ``T = timesteps.shape[0] - 1`` is the
          sole source for the number of denoising steps.

        Trajectory-RL invariants (enforced on non-NFT paths):
          (I1) ``target_sde_indices`` is set and ``⊆ {0..T-1}``.
          (I2) When ``log_probs`` is present and non-empty,
               ``log_probs.data.keys() == target_sde_indices``.
               (On the normal trajectory-RL path, ``assemble_training_batch``
               guarantees ``log_probs`` is always populated, so this is
               effectively unconditional in practice.)
          (I3) every idx in ``target_sde_indices`` has both position ``idx``
               and ``idx + 1`` stored in ``trajectory_store``.

        NFT / clean-latents-only batches skip the trajectory-RL invariants.
        """
        bs = self.batch_size
        T = self.num_steps

        if self.advantages.shape[0] != bs:
            raise ValueError(
                f"Advantages batch size {self.advantages.shape[0]} != "
                f"trajectory batch size {bs}"
            )

        # Structural guard: timesteps length must agree with full trajectory.
        if self.trajectory_store.is_full:
            if self.trajectory_store.num_stored != T + 1:
                raise ValueError(
                    f"Timesteps imply T={T} (len={T + 1}) but trajectory_store "
                    f"has {self.trajectory_store.num_stored} positions."
                )
            if self.target_sde_indices is not None:
                bad = sorted(
                    int(i) for i in self.target_sde_indices
                    if int(i) < 0 or int(i) >= T
                )
                if bad:
                    raise ValueError(
                        f"target_sde_indices contain out-of-range steps: {bad}, "
                        f"allowed=range(0, {T})"
                    )

        # Trajectory-RL SSOT invariants. Skip for NFT (clean_latents_only) path.
        if not self.trajectory_store.is_clean_latents_only:
            if self.target_sde_indices is None:
                raise ValueError(
                    "Trajectory-RL TrainingBatch requires target_sde_indices to be set "
                    "(it is the single source of truth for trainable step indices)."
                )
            target = set(int(i) for i in self.target_sde_indices)
            if self.log_probs is not None and len(self.log_probs) > 0:
                lp_keys = set(int(k) for k in self.log_probs.data.keys())
                extra_keys = sorted(lp_keys - target)
                if extra_keys:
                    raise ValueError(
                        f"log_probs has keys outside target_sde_indices: "
                        f"extra={extra_keys}, target={sorted(target)}"
                    )
                missing_keys = sorted(target - lp_keys)
                if missing_keys:
                    raise ValueError(
                        f"log_probs missing keys required by target_sde_indices: "
                        f"missing={missing_keys}, target={sorted(target)}, "
                        f"log_probs_keys={sorted(lp_keys)}"
                    )
            for idx in sorted(target):
                if not self.trajectory_store.has_position(idx):
                    raise ValueError(
                        f"target_sde_indices step {idx} not stored in trajectory "
                        f"(stored: {self.trajectory_store.stored_positions})"
                    )
                if not self.trajectory_store.has_position(idx + 1):
                    raise ValueError(
                        f"target_sde_indices step {idx} requires position {idx + 1} "
                        f"but it is not stored in trajectory "
                        f"(stored: {self.trajectory_store.stored_positions})"
                    )

        ctx_bs = self.forward_context.batch_size
        if ctx_bs > 0 and ctx_bs != bs:
            raise ValueError(
                f"ForwardContext batch size {ctx_bs} != "
                f"trajectory batch size {bs}"
            )
        for name in ("sample_ids", "prompt_ids", "group_ids"):
            ids = getattr(self, name)
            if ids is not None and len(ids) != bs:
                raise ValueError(f"{name} length {len(ids)} != batch size {bs}")

        if self.log_probs is not None:
            for idx, log_prob in self.log_probs.data.items():
                if log_prob.shape[0] != bs:
                    raise ValueError(
                        f"Log prob at index {idx} has batch size "
                        f"{log_prob.shape[0]} != {bs}"
                    )

    # ---- batch operations ---------------------------------------------------

    def to_device(self, device: Union[str, "TorchDevice"]) -> "TrainingBatch":
        """Move all tensors to specified device."""
        return TrainingBatch(
            trajectory_store=self.trajectory_store.to_device(device),
            timesteps=self.timesteps.to(device),
            advantages=self.advantages.to(device),
            forward_context=self.forward_context.to_device(device),
            log_probs=self.log_probs.to_device(device) if self.log_probs is not None else None,
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            prompts=self.prompts,
            prompt_ids=self.prompt_ids,
            sample_ids=self.sample_ids,
            group_ids=self.group_ids,
            is_partitioned=self.is_partitioned,
            target_sde_indices=self.target_sde_indices,
            extras=batch_move(self.extras, device),
        )

    def slice(self, start: int, end: int) -> "TrainingBatch":
        """Slice batch along sample dimension for micro-batch gradient accumulation."""
        return TrainingBatch(
            trajectory_store=self.trajectory_store.slice_batch(start, end),
            timesteps=self.timesteps,
            advantages=self.advantages[start:end].clone(),
            forward_context=self.forward_context.slice(start, end),
            log_probs=self.log_probs.slice(start, end) if self.log_probs is not None else None,
            rewards=self.rewards[start:end].clone() if self.rewards is not None else None,
            prompts=self.prompts[start:end] if self.prompts is not None else None,
            prompt_ids=self.prompt_ids[start:end] if self.prompt_ids is not None else None,
            sample_ids=self.sample_ids[start:end] if self.sample_ids is not None else None,
            group_ids=self.group_ids[start:end] if self.group_ids is not None else None,
            is_partitioned=True,
            target_sde_indices=self.target_sde_indices,
            extras=batch_slice(
                self.extras,
                batch_size=int(self.batch_size),
                start=start,
                end=end,
                recursive=True,
                deep_clone=True,
            ),
        )

    def shuffle(self, indices: torch.Tensor) -> "TrainingBatch":
        """Shuffle (reindex) batch along sample dimension."""
        return TrainingBatch(
            trajectory_store=self.trajectory_store.reindex_batch(indices),
            timesteps=self.timesteps,
            advantages=self.advantages[indices],
            forward_context=self.forward_context.reindex(indices),
            log_probs=self.log_probs.reindex(indices) if self.log_probs is not None else None,
            rewards=self.rewards[indices] if self.rewards is not None else None,
            prompts=(
                [self.prompts[i] for i in indices.tolist()]
                if self.prompts is not None
                else None
            ),
            prompt_ids=(
                [self.prompt_ids[i] for i in indices.tolist()]
                if self.prompt_ids is not None
                else None
            ),
            sample_ids=(
                [self.sample_ids[i] for i in indices.tolist()]
                if self.sample_ids is not None
                else None
            ),
            group_ids=(
                [self.group_ids[i] for i in indices.tolist()]
                if self.group_ids is not None
                else None
            ),
            is_partitioned=self.is_partitioned,
            target_sde_indices=self.target_sde_indices,
            extras=batch_reindex(
                self.extras,
                indices=indices,
                batch_size=int(self.batch_size),
                recursive=True,
                deep_clone=True,
            ),
        )

    # ---- modality detection -------------------------------------------------

    def detect_modality(self) -> str:
        """Detect media modality from trajectory data shape."""
        return self.trajectory_store.detect_modality()


__all__ = [
    "TimestepData",
    "TrainingBatch",
    "TrajectoryStore",
    "build_rollout_extras",
]
