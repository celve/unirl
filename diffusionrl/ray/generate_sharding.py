"""Shared rollout-request sharding helpers for actor-group generate()."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch

from diffusionrl.types import RolloutRequest
from diffusionrl.types.sampling import RolloutOutput


@dataclass(frozen=True)
class GenerateShardPlan:
    """Resolved shard plan for one generate() request."""

    shards: List[Optional[RolloutRequest]]
    shard_sizes: List[int]
    original_batch_size: int
    effective_batch_size: int

    @property
    def was_padded(self) -> bool:
        return self.effective_batch_size > self.original_batch_size


def _balanced_shard_sizes(total_items: int, num_shards: int) -> List[int]:
    base = total_items // num_shards
    remainder = total_items % num_shards
    return [base + (1 if index < remainder else 0) for index in range(num_shards)]


def _pad_batched_value(value: Any, batch_size: int, target_size: int) -> Any:
    if value is None or target_size <= batch_size:
        return value

    pad_count = target_size - batch_size
    if isinstance(value, list) and len(value) == batch_size:
        return value + [value[-1]] * pad_count
    if isinstance(value, tuple) and len(value) == batch_size:
        return value + (value[-1],) * pad_count
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        pad = value[-1:].repeat(pad_count, *([1] * (value.dim() - 1)))
        return torch.cat([value, pad], dim=0)
    return value


def _pad_rollout_request(
    request: RolloutRequest,
    *,
    batch_size: int,
    target_size: int,
) -> RolloutRequest:
    if target_size <= batch_size:
        return request

    padded = copy.copy(request)
    padded.prompts = list(_pad_batched_value(list(request.prompts), batch_size, target_size))
    for attr in (
        "prompt_ids",
        "sample_ids",
        "group_ids",
        "noise_group_ids",
        "prompt_metadata",
        "latents",
    ):
        setattr(
            padded,
            attr,
            _pad_batched_value(getattr(request, attr, None), batch_size, target_size),
        )

    padded.kwargs = {
        key: _pad_batched_value(value, batch_size, target_size)
        for key, value in dict(request.kwargs).items()
    }
    return padded


def build_generate_shard_plan(
    request: RolloutRequest,
    *,
    num_actors: int,
    pad_to_actor_count: bool = False,
) -> GenerateShardPlan:
    """Build evenly-balanced RolloutRequest shards for actor-group sampling."""
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors}")
    if not isinstance(request.prompts, list) or len(request.prompts) == 0:
        raise ValueError("generate requires a non-empty prompt list")

    original_batch_size = len(request.prompts)
    effective_batch_size = (
        max(original_batch_size, num_actors)
        if pad_to_actor_count
        else original_batch_size
    )
    effective_request = _pad_rollout_request(
        request,
        batch_size=original_batch_size,
        target_size=effective_batch_size,
    )

    shard_sizes = _balanced_shard_sizes(effective_batch_size, num_actors)
    shards: List[Optional[RolloutRequest]] = []
    start = 0
    for count in shard_sizes:
        end = start + count
        shards.append(effective_request.slice_prompts(start, end) if count > 0 else None)
        start = end

    return GenerateShardPlan(
        shards=shards,
        shard_sizes=shard_sizes,
        original_batch_size=original_batch_size,
        effective_batch_size=effective_batch_size,
    )


def slice_sampler_output(output: Any, start: int, end: int) -> Any:
    """Slice a batched sampler output along its leading batch dimension."""
    if not isinstance(output, RolloutOutput):
        return output
    return RolloutOutput(
        latents=output.latents[start:end],
        timesteps=output.timesteps,
        trajectories=output.trajectories[start:end] if output.trajectories is not None else None,
        log_probs=output.log_probs.slice(start, end) if output.log_probs is not None else None,
        embeddings=output.embeddings.slice(start, end) if output.embeddings is not None else None,
        decoded_images=output.decoded_images[start:end] if output.decoded_images is not None else None,
        metadata=output.metadata,
        step_indices=output.step_indices,
    )


def trim_generate_outputs(
    outputs: Sequence[Any],
    *,
    plan: GenerateShardPlan,
) -> List[Any]:
    """Drop or slice padded outputs back to the original batch size."""
    resolved_outputs = list(outputs)
    if not plan.was_padded:
        return resolved_outputs

    trimmed: List[Any] = []
    remaining = plan.original_batch_size
    active_shard_sizes = [count for count in plan.shard_sizes if count > 0]
    for output, count in zip(resolved_outputs, active_shard_sizes):
        if remaining <= 0:
            break
        keep = min(count, remaining)
        if keep < count:
            output = slice_sampler_output(output, 0, keep)
        trimmed.append(output)
        remaining -= keep
    return trimmed


__all__ = [
    "GenerateShardPlan",
    "build_generate_shard_plan",
    "slice_sampler_output",
    "trim_generate_outputs",
]
