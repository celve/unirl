"""Typed training-batch indexing and concatenation helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from diffusionrl.types.batch_ops import concat_payload_values, reindex_payload_value
from diffusionrl.types.forward_context import ForwardContext
from diffusionrl.types.sampling import LogProbData
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.types.trajectory_store import TrajectoryStore


def _as_index_tensor(indices: Sequence[int], *, device: torch.device) -> torch.Tensor:
    if len(indices) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(list(indices), dtype=torch.long, device=device)


def index_training_batch(batch: TrainingBatch, keep_indices: Sequence[int]) -> TrainingBatch:
    """Select arbitrary sample indices from a training batch."""
    idx = _as_index_tensor(keep_indices, device=batch.device)
    if idx.numel() == 0:
        raise ValueError("Cannot index TrainingBatch with empty indices.")

    log_probs = None
    if batch.log_probs is not None and len(batch.log_probs) > 0:
        log_probs = LogProbData.from_dict(
            {
                int(k): v.index_select(0, idx.to(v.device))
                for k, v in batch.log_probs.to_dict().items()
            }
        )

    prompts = None
    if batch.prompts is not None:
        prompts = [batch.prompts[i] for i in keep_indices]

    return TrainingBatch(
        trajectory_store=batch.trajectory_store.index_select_batch(idx),
        log_probs=log_probs,
        timesteps=batch.timesteps,
        advantages=batch.advantages.index_select(0, idx.to(batch.advantages.device)),
        forward_context=batch.forward_context.reindex(idx),
        rewards=(
            batch.rewards.index_select(0, idx.to(batch.rewards.device))
            if batch.rewards is not None
            else None
        ),
        prompts=prompts,
        prompt_ids=(
            [batch.prompt_ids[i] for i in keep_indices]
            if batch.prompt_ids is not None
            else None
        ),
        sample_ids=(
            [batch.sample_ids[i] for i in keep_indices]
            if batch.sample_ids is not None
            else None
        ),
        group_ids=(
            [batch.group_ids[i] for i in keep_indices]
            if batch.group_ids is not None
            else None
        ),
        is_partitioned=batch.is_partitioned,
        step_indices=batch.step_indices,
        target_sde_indices=batch.target_sde_indices,
        extras=reindex_payload_value(
            batch.extras,
            indices=idx,
            batch_size=int(batch.batch_size),
        ),
    )


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a.to(b.device), b))


def concat_training_batches(batches: Sequence[TrainingBatch]) -> TrainingBatch:
    """Concatenate training batches along batch dimension."""
    if not batches:
        raise ValueError("Cannot concatenate empty batch list.")

    first = batches[0]
    typed = list(batches)

    for b in typed[1:]:
        if not _tensor_equal(first.timesteps, b.timesteps):
            raise ValueError("Inconsistent timesteps across batches.")

    log_probs: Optional[LogProbData] = None
    if first.log_probs is not None and len(first.log_probs) > 0:
        key_set = set(int(k) for k in first.log_probs.to_dict().keys())
        for b in typed[1:]:
            if b.log_probs is None:
                raise ValueError("Inconsistent log_probs presence across batches.")
            if set(int(k) for k in b.log_probs.to_dict().keys()) != key_set:
                raise ValueError("Inconsistent log_prob step indices across batches.")
        log_probs = LogProbData.from_dict(
            {
                int(step): torch.cat(
                    [b.log_probs.to_dict()[int(step)] for b in typed],
                    dim=0,
                )
                for step in sorted(key_set)
            }
        )

    prompts = None
    if all(b.prompts is not None for b in typed):
        prompts = [p for b in typed for p in (b.prompts or [])]
    prompt_ids = None
    if all(b.prompt_ids is not None for b in typed):
        prompt_ids = [p for b in typed for p in (b.prompt_ids or [])]
    sample_ids = None
    if all(b.sample_ids is not None for b in typed):
        sample_ids = [s for b in typed for s in (b.sample_ids or [])]
    group_ids = None
    if all(b.group_ids is not None for b in typed):
        group_ids = [g for b in typed for g in (b.group_ids or [])]

    rewards = None
    if all(b.rewards is not None for b in typed):
        rewards = torch.cat([b.rewards for b in typed if b.rewards is not None], dim=0)
    extras = concat_payload_values(
        [b.extras for b in typed],
        batch_sizes=[int(b.batch_size) for b in typed],
    ) or {}

    merged_ctx = type(first.forward_context).cat([b.forward_context for b in typed])

    return TrainingBatch(
        trajectory_store=TrajectoryStore.concat([b.trajectory_store for b in typed]),
        log_probs=log_probs,
        timesteps=first.timesteps,
        advantages=torch.cat([b.advantages for b in typed], dim=0),
        forward_context=merged_ctx,
        rewards=rewards,
        prompts=prompts,
        prompt_ids=prompt_ids,
        sample_ids=sample_ids,
        group_ids=group_ids,
        is_partitioned=False,
        step_indices=first.step_indices,
        target_sde_indices=first.target_sde_indices,
        extras=extras,
    )


__all__ = [
    "concat_training_batches",
    "index_training_batch",
]
