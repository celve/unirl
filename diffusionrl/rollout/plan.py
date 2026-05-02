"""Authoritative rollout-distribution plan.

Single home for inter-actor sharding, intra-actor per-call chunking, and
post-scatter trim. Exposed at ``cfg.rollout.plan`` and materialized via
``materialize()`` to a real ``RolloutPlan`` instance whose methods are bound
to the configured chunk size.

Replaces the previous ``GenerateShardPlan`` runtime dataclass and the free
helper functions in ``diffusionrl.ray.generate_sharding``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples


def _balanced_shard_sizes(total_items: int, num_shards: int) -> List[int]:
    base = total_items // num_shards
    remainder = total_items % num_shards
    return [base + (1 if index < remainder else 0) for index in range(num_shards)]


def _slice_sampler_output(output: Any, start: int, end: int) -> Any:
    """Slice a batched sampler output along its leading batch dimension."""
    if not isinstance(output, RolloutSamples):
        return output
    return output.slice(start, end)


def _output_batch_size(output: Any) -> Optional[int]:
    """Return the batch dim of a sampler output, or ``None`` if non-batched.

    ``None`` outputs (no-work shards) report 0; ``RolloutSamples`` reports
    its inferred ``batch_size``; objects exposing ``__len__`` report it; everything
    else returns ``None`` so the caller falls back to a no-op trim path.
    """
    if output is None:
        return 0
    if isinstance(output, RolloutSamples):
        return int(output.batch_size)
    try:
        return int(len(output))
    except TypeError:
        return None


@register_config(group="rollout/plan", name="default")
@dataclass
class RolloutPlan:
    """Rollout-distribution plan — sharding, chunking, and trim in one class.

    ``forward_batch_size`` drives intra-actor per-call chunking (consumed by
    ``chunked_engine_generate`` / ``chunked_decode_latents`` in
    ``diffusionrl.samplers.engine``). The ``shard``/``shard_grouped``/``trim``
    methods drive inter-actor sharding for actor-group ``generate()`` calls.
    """

    forward_batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        require(
            self.forward_batch_size is None or self.forward_batch_size >= 1,
            f"RolloutPlan.forward_batch_size must be >= 1 when set; got {self.forward_batch_size!r}",
        )

    # ------------------------------------------------------------------
    # Inter-actor sharding (replaces build_generate_shard_plan)
    # ------------------------------------------------------------------
    def shard(
        self,
        request: RolloutRequest,
        *,
        num_actors: int,
        pad_to_actor_count: bool = False,
    ) -> List[Optional[RolloutRequest]]:
        """Evenly-balanced ``RolloutRequest`` shards for actor-group sampling.

        Operates on the legacy ``request.prompts: List[str]`` shape. Pads to
        ``num_actors`` when ``pad_to_actor_count=True``; ``trim()`` undoes
        the padding by slicing each output back against the original request.
        """
        if num_actors < 1:
            raise ValueError(f"num_actors must be >= 1, got {num_actors}")
        if not isinstance(request.prompts, list) or len(request.prompts) == 0:
            raise ValueError("generate requires a non-empty prompt list")

        original_batch_size = len(request.prompts)
        effective_batch_size = max(original_batch_size, num_actors) if pad_to_actor_count else original_batch_size
        effective_request = request.pad_to(effective_batch_size)

        shard_sizes = _balanced_shard_sizes(effective_batch_size, num_actors)
        shards: List[Optional[RolloutRequest]] = []
        start = 0
        for count in shard_sizes:
            end = start + count
            shards.append(effective_request.slice_prompts(start, end) if count > 0 else None)
            start = end
        return shards

    def shard_grouped(
        self,
        request: RolloutRequest,
        *,
        num_actors: int,
        samples_per_prompt: int,
    ) -> List[Optional[RolloutRequest]]:
        """Group-aligned shards across ``num_actors`` for the typed
        ``RolloutRequest(prompts: Prompts, ...)`` shape.

        Splits at multiples of ``samples_per_prompt`` so each shard contains
        whole groups. Required because per-actor ``generate_buffered`` calls
        ``RolloutResponse.split()`` (groups by ``prompts.group_ids``) and
        ``compute_advantages`` z-scores within one group — splitting a group
        across actors corrupts the normalization.

        Returns ``List[Optional[RolloutRequest]]`` of length ``num_actors``;
        entries are ``None`` for actors that get no work.
        """
        if num_actors < 1:
            raise ValueError(f"num_actors must be >= 1, got {num_actors}")
        p = request.prompts
        total_samples = len(p.prompts)
        if total_samples == 0:
            return [None] * num_actors
        if samples_per_prompt < 1:
            raise ValueError(f"samples_per_prompt must be >= 1 for sharding, got {samples_per_prompt}")
        if total_samples % samples_per_prompt != 0:
            raise ValueError(
                f"total samples ({total_samples}) is not a multiple of "
                f"samples_per_prompt ({samples_per_prompt}); cannot shard at group boundaries."
            )
        num_groups = total_samples // samples_per_prompt
        base_groups = num_groups // num_actors
        remainder = num_groups % num_actors

        shards: List[Optional[RolloutRequest]] = []
        cursor = 0
        for actor_idx in range(num_actors):
            my_groups = base_groups + (1 if actor_idx < remainder else 0)
            my_samples = my_groups * samples_per_prompt
            if my_samples == 0:
                shards.append(None)
                continue
            start = cursor
            end = cursor + my_samples
            sub_prompts = Prompts(
                prompts=list(p.prompts[start:end]),
                prompt_ids=list(p.prompt_ids[start:end]),
                sample_ids=list(p.sample_ids[start:end]),
                group_ids=list(p.group_ids[start:end]),
                noise_group_ids=list(p.noise_group_ids[start:end]),
                prompt_metadata=list(p.prompt_metadata[start:end]),
            )
            shards.append(
                RolloutRequest(
                    prompts=sub_prompts,
                    sampling_params=request.sampling_params,
                    collect_media_preview=bool(request.collect_media_preview),
                    media_max_items=int(request.media_max_items),
                )
            )
            cursor = end
        return shards

    # ------------------------------------------------------------------
    # Post-scatter trim (replaces trim_generate_outputs)
    # ------------------------------------------------------------------
    def trim(
        self,
        outputs: Sequence[Any],
        *,
        request: RolloutRequest,
    ) -> List[Any]:
        """Drop padding from per-shard outputs back to ``request``'s prompt count.

        Walks outputs in shard order, slicing each ``RolloutSamples`` to the
        smaller of its own batch dim or the remaining target. Reads target total
        from the ``request`` directly (no separate metadata struct needed).
        For non-batched outputs (e.g. dicts) trim is a no-op."""
        resolved_outputs = list(outputs)
        target_total = len(request.prompts) if isinstance(request.prompts, list) else len(request.prompts.prompts)

        sizes: List[Optional[int]] = []
        for out in resolved_outputs:
            sizes.append(_output_batch_size(out))
        if any(s is None for s in sizes):
            # Non-batched outputs in the mix — trim doesn't apply uniformly.
            return resolved_outputs
        total_size = sum(int(s) for s in sizes if s is not None)
        if total_size <= target_total:
            return resolved_outputs

        trimmed: List[Any] = []
        remaining = target_total
        for output, count in zip(resolved_outputs, sizes):
            if output is None:
                continue
            if remaining <= 0:
                break
            count_int = int(count) if count is not None else 0
            keep = min(count_int, remaining)
            if keep < count_int:
                output = _slice_sampler_output(output, 0, keep)
            trimmed.append(output)
            remaining -= keep
        return trimmed


__all__ = ["RolloutPlan"]
