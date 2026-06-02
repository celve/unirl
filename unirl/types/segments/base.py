"""Segment base class and SegmentStatus enum.

A ``Segment`` is the modality-tagged, SoA-batched container for one
modality's generation outputs across a rollout. There is no per-sample
``Segment`` — segments are always batched. Per-segment metadata
(``sample_indices``, ``positions``) maps each segment back to its sample
and position in that sample's segment sequence.

Two promotion methods (``as_condition`` / ``as_condition_with``) let a
finished segment become input for the next stage in interleaved unified
rollouts (Show-o / Transfusion / Janus pattern). The split exists because
some promotions are encoder-free (latent → latent-condition) while others
require a re-embedding pass (tokens → text-embedding-condition).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions.base import Condition, Modality


class SegmentStatus(int, Enum):
    PENDING = 0
    COMPLETED = 1
    TRUNCATED = 2
    ABORTED = 3


@dataclass
class Segment(Batch):
    """SoA batched container for one modality's generation outputs.

    ``sample_indices[k]`` gives the sample index that segment ``k`` belongs
    to; ``positions[k]`` gives its order within that sample's segment
    sequence (0 for first segment, 1 for second, …).

    ``log_probs`` and ``loss_mask`` are intentionally NOT declared on the
    base because their shape and semantics differ per modality:

    - :class:`LatentSegment` uses ``log_probs`` / ``loss_mask`` as
      ``[N_segs, S]`` per-step / per-segment tensors (CONCAT field).
    - :class:`TextSegment` uses them as ``[total_tokens]`` per-token
      tensors packed along dim 0 (PACKED field, framework-managed
      ``cu_seqlens``).

    Each subclass declares its own version with the right field kind.
    """

    modality: ClassVar[Modality]

    # Keep small routing metadata inline. The driver mutates/sample-offsets
    # these fields while aggregating per-actor RolloutResp shards; moving them
    # behind transport would leave only TensorMeta handles at that boundary.
    sample_indices: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    positions: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    status: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    # ---- promotion to Condition ---------------------------------------------

    def as_condition(self) -> Optional[Condition]:
        """Encoder-free promotion. Override on subclasses where it makes sense.

        Default returns ``None`` — subclasses that need an encoder pass
        should leave this returning ``None`` and override
        ``as_condition_with`` instead.
        """
        return None

    def as_condition_with(self, encoder: Callable[..., Any]) -> Condition:
        """Encoder-mediated promotion (e.g. token → embedding).

        Default raises ``NotImplementedError``; modalities whose promotion
        is encoder-free should leave this and use ``as_condition`` instead.
        """
        raise NotImplementedError(f"{type(self).__name__}.as_condition_with(encoder) is not implemented")


__all__ = ["Segment", "SegmentStatus"]
