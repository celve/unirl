"""Shared rollout-request sharding helpers for actor-group generate()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from diffusionrl.types import RolloutRequest
from diffusionrl.types.sampling import RolloutSamples


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
    effective_request = request.pad_to(effective_batch_size)

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
    if not isinstance(output, RolloutSamples):
        return output
    return output.slice(start, end)


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
