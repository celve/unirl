"""Typed training-batch indexing and concatenation helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from diffusionrl.types.sampling import LogProbData, PromptEmbeddings
from diffusionrl.types.training_batch import (
    BackwardTrainingBatch,
    ForwardTrainingBatch,
    TrainingBatch,
    _concat_extra_payload,
    _reindex_extra_payload,
)


def _as_index_tensor(indices: Sequence[int], *, device: torch.device) -> torch.Tensor:
    if len(indices) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(list(indices), dtype=torch.long, device=device)


def _index_if_batched(
    tensor: Optional[torch.Tensor],
    idx: torch.Tensor,
    batch_size: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if int(tensor.shape[0]) != int(batch_size):
        return tensor
    return tensor.index_select(0, idx.to(tensor.device))


def _index_prompt_embeddings(emb: PromptEmbeddings, idx: torch.Tensor) -> PromptEmbeddings:
    batch_size = int(emb.prompt_embeds.shape[0])
    return PromptEmbeddings(
        prompt_embeds=emb.prompt_embeds.index_select(0, idx.to(emb.prompt_embeds.device)),
        pooled_prompt_embeds=_index_if_batched(emb.pooled_prompt_embeds, idx, batch_size),
        encoder_attention_mask=_index_if_batched(emb.encoder_attention_mask, idx, batch_size),
        negative_prompt_embeds=_index_if_batched(emb.negative_prompt_embeds, idx, batch_size),
        negative_pooled_prompt_embeds=_index_if_batched(
            emb.negative_pooled_prompt_embeds,
            idx,
            batch_size,
        ),
        text_ids=_index_if_batched(emb.text_ids, idx, batch_size),
        image_ids=_index_if_batched(emb.image_ids, idx, batch_size),
    )


def index_training_batch(batch: TrainingBatch, keep_indices: Sequence[int]) -> TrainingBatch:
    """Select arbitrary sample indices from a typed training batch."""
    if isinstance(batch, BackwardTrainingBatch):
        idx = _as_index_tensor(keep_indices, device=batch.trajectories.device)
        if idx.numel() == 0:
            raise ValueError("Cannot index BackwardTrainingBatch with empty indices.")

        log_probs = LogProbData.from_dict(
            {
                int(k): v.index_select(0, idx.to(v.device))
                for k, v in batch.log_probs.to_dict().items()
            }
        )

        prompts = None
        if batch.prompts is not None:
            prompts = [batch.prompts[i] for i in keep_indices]

        return BackwardTrainingBatch(
            trajectories=batch.trajectories.index_select(0, idx.to(batch.trajectories.device)),
            log_probs=log_probs,
            timesteps=batch.timesteps,
            advantages=batch.advantages.index_select(0, idx.to(batch.advantages.device)),
            embeddings=_index_prompt_embeddings(batch.embeddings, idx),
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
            extras=_reindex_extra_payload(
                batch.extras,
                indices=idx,
                batch_size=int(batch.batch_size),
            ),
        )

    if isinstance(batch, ForwardTrainingBatch):
        idx = _as_index_tensor(keep_indices, device=batch.clean_latents.device)
        if idx.numel() == 0:
            raise ValueError("Cannot index ForwardTrainingBatch with empty indices.")

        prompts = None
        if batch.prompts is not None:
            prompts = [batch.prompts[i] for i in keep_indices]

        return ForwardTrainingBatch(
            clean_latents=batch.clean_latents.index_select(0, idx.to(batch.clean_latents.device)),
            advantages=batch.advantages.index_select(0, idx.to(batch.advantages.device)),
            embeddings=_index_prompt_embeddings(batch.embeddings, idx),
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
            timesteps=batch.timesteps,
            is_partitioned=batch.is_partitioned,
            extras=_reindex_extra_payload(
                batch.extras,
                indices=idx,
                batch_size=int(batch.batch_size),
            ),
        )

    raise TypeError(f"Unsupported training batch type: {type(batch).__name__}")


def _concat_optional_embedding_field(
    tensors: Sequence[Optional[torch.Tensor]],
    *,
    batch_sizes: Sequence[int],
) -> Optional[torch.Tensor]:
    if not tensors or all(t is None for t in tensors):
        return None
    if any(t is None for t in tensors):
        raise ValueError("Inconsistent optional embedding field presence across batches.")

    casted: List[torch.Tensor] = [t for t in tensors if t is not None]
    is_batched = [int(t.shape[0]) == int(bs) for t, bs in zip(casted, batch_sizes)]
    if all(is_batched):
        return torch.cat(casted, dim=0)
    if any(is_batched):
        raise ValueError("Mixed batched/non-batched embedding tensor shapes across batches.")
    return casted[0]


def _concat_prompt_embeddings(embeddings: Sequence[PromptEmbeddings]) -> PromptEmbeddings:
    if not embeddings:
        raise ValueError("Cannot concatenate empty embedding list.")

    batch_sizes = [int(e.prompt_embeds.shape[0]) for e in embeddings]
    return PromptEmbeddings(
        prompt_embeds=torch.cat([e.prompt_embeds for e in embeddings], dim=0),
        pooled_prompt_embeds=_concat_optional_embedding_field(
            [e.pooled_prompt_embeds for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        encoder_attention_mask=_concat_optional_embedding_field(
            [e.encoder_attention_mask for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        negative_prompt_embeds=_concat_optional_embedding_field(
            [e.negative_prompt_embeds for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        negative_pooled_prompt_embeds=_concat_optional_embedding_field(
            [e.negative_pooled_prompt_embeds for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        text_ids=_concat_optional_embedding_field(
            [e.text_ids for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        image_ids=_concat_optional_embedding_field(
            [e.image_ids for e in embeddings],
            batch_sizes=batch_sizes,
        ),
    )


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a.to(b.device), b))


def concat_training_batches(batches: Sequence[TrainingBatch]) -> TrainingBatch:
    """Concatenate typed training batches along batch dimension."""
    if not batches:
        raise ValueError("Cannot concatenate empty batch list.")

    first = batches[0]
    if isinstance(first, BackwardTrainingBatch):
        if not all(isinstance(b, BackwardTrainingBatch) for b in batches):
            raise TypeError("Cannot mix BackwardTrainingBatch and ForwardTrainingBatch in concat.")

        typed: List[BackwardTrainingBatch] = [b for b in batches if isinstance(b, BackwardTrainingBatch)]
        key_set = set(int(k) for k in typed[0].log_probs.to_dict().keys())
        for b in typed[1:]:
            if set(int(k) for k in b.log_probs.to_dict().keys()) != key_set:
                raise ValueError("Inconsistent log_prob step indices across batches.")
            if not _tensor_equal(typed[0].timesteps, b.timesteps):
                raise ValueError("Inconsistent timesteps across BackwardTrainingBatch items.")

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
        extras = _concat_extra_payload(
            [b.extras for b in typed],
            batch_sizes=[int(b.batch_size) for b in typed],
        ) or {}

        return BackwardTrainingBatch(
            trajectories=torch.cat([b.trajectories for b in typed], dim=0),
            log_probs=log_probs,
            timesteps=typed[0].timesteps,
            advantages=torch.cat([b.advantages for b in typed], dim=0),
            embeddings=_concat_prompt_embeddings([b.embeddings for b in typed]),
            rewards=rewards,
            prompts=prompts,
            prompt_ids=prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            is_partitioned=False,
            step_indices=typed[0].step_indices,
            target_sde_indices=typed[0].target_sde_indices,
            extras=extras,
        )

    if isinstance(first, ForwardTrainingBatch):
        if not all(isinstance(b, ForwardTrainingBatch) for b in batches):
            raise TypeError("Cannot mix ForwardTrainingBatch and BackwardTrainingBatch in concat.")

        typed_f: List[ForwardTrainingBatch] = [b for b in batches if isinstance(b, ForwardTrainingBatch)]
        base_timesteps = typed_f[0].timesteps
        if base_timesteps is not None:
            for b in typed_f[1:]:
                if b.timesteps is None or not _tensor_equal(base_timesteps, b.timesteps):
                    raise ValueError("Inconsistent timesteps across ForwardTrainingBatch items.")

        prompts_f = None
        if all(b.prompts is not None for b in typed_f):
            prompts_f = [p for b in typed_f for p in (b.prompts or [])]
        prompt_ids_f = None
        if all(b.prompt_ids is not None for b in typed_f):
            prompt_ids_f = [p for b in typed_f for p in (b.prompt_ids or [])]
        sample_ids_f = None
        if all(b.sample_ids is not None for b in typed_f):
            sample_ids_f = [s for b in typed_f for s in (b.sample_ids or [])]
        group_ids_f = None
        if all(b.group_ids is not None for b in typed_f):
            group_ids_f = [g for b in typed_f for g in (b.group_ids or [])]

        rewards_f = None
        if all(b.rewards is not None for b in typed_f):
            rewards_f = torch.cat([b.rewards for b in typed_f if b.rewards is not None], dim=0)
        extras_f = _concat_extra_payload(
            [b.extras for b in typed_f],
            batch_sizes=[int(b.batch_size) for b in typed_f],
        ) or {}

        return ForwardTrainingBatch(
            clean_latents=torch.cat([b.clean_latents for b in typed_f], dim=0),
            advantages=torch.cat([b.advantages for b in typed_f], dim=0),
            embeddings=_concat_prompt_embeddings([b.embeddings for b in typed_f]),
            rewards=rewards_f,
            prompts=prompts_f,
            prompt_ids=prompt_ids_f,
            sample_ids=sample_ids_f,
            group_ids=group_ids_f,
            timesteps=base_timesteps,
            is_partitioned=False,
            extras=extras_f,
        )

    raise TypeError(f"Unsupported training batch type: {type(first).__name__}")


__all__ = [
    "concat_training_batches",
    "index_training_batch",
]
