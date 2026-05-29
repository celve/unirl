"""Authoritative rollout-distribution plan.

Single home for inter-actor sharding and intra-actor per-call chunking.
Exposed at ``cfg.rollout.plan`` and materialized via ``materialize()`` to
a real ``RolloutPlan`` instance whose methods are bound to the configured
chunk size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.types.rollout_req import RolloutReq


@register_config(group="rollout/plan", name="default")
@dataclass
class RolloutPlan:
    """Rollout-distribution plan — sharding and chunking.

    ``forward_batch_size`` drives intra-actor per-call chunking (consumed by
    :func:`diffusionrl.rollout.engine.chunked_engine_generate_req`).
    :meth:`shard_grouped_req` drives inter-actor sharding for actor-group
    ``generate()`` calls.
    """

    forward_batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        require(
            self.forward_batch_size is None or self.forward_batch_size >= 1,
            f"RolloutPlan.forward_batch_size must be >= 1 when set; got {self.forward_batch_size!r}",
        )

    # ------------------------------------------------------------------
    # Inter-actor sharding for the RolloutReq path
    # ------------------------------------------------------------------
    def shard_grouped_req(
        self,
        req: RolloutReq,
        *,
        num_actors: int,
        samples_per_prompt: int,
    ) -> List[Optional[RolloutReq]]:
        """Group-aligned ``RolloutReq`` shards across ``num_actors``.

        Splits at multiples of ``samples_per_prompt`` so each shard contains
        whole groups (required for per-actor advantage normalization to see
        complete groups).

        Slicing delegates to :meth:`Batch.slice` (``distributed/tensor/batch.py``),
        which understands the ``concat_field`` annotations on ``RolloutReq``
        — ``sample_ids`` / ``group_ids`` slice as lists, ``primitives`` recurses
        into the dict and slices each ``Batch`` primitive, ``stage_params``
        is shared and copied as-is.

        Returns ``List[Optional[RolloutReq]]`` of length ``num_actors``;
        entries are ``None`` for actors that get no work.
        """
        if num_actors < 1:
            raise ValueError(f"num_actors must be >= 1, got {num_actors}")
        total_samples = int(req.batch_size)
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

        shards: List[Optional[RolloutReq]] = []
        cursor = 0
        for actor_idx in range(num_actors):
            my_groups = base_groups + (1 if actor_idx < remainder else 0)
            my_samples = my_groups * samples_per_prompt
            if my_samples == 0:
                shards.append(None)
                continue
            start = cursor
            end = cursor + my_samples
            shards.append(req.slice(start, end))
            cursor = end
        return shards


__all__ = ["RolloutPlan"]
