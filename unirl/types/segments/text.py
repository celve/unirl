"""TextSegment — SoA container for AR token rollouts (varlen-packed).

All token-aligned fields are :func:`packed_field`s with shape
``[total_tokens]`` along dim 0. In addition to rollout ``tokens``,
``log_probs``, and ``loss_mask``, the optional ``token_advantages``,
``value_targets``, and ``value_mask`` fields carry SAO learner signals without
losing alignment through concat/select/slice operations. The framework manages the
``cu_seqlens`` metadata behind a hidden ``_packed_cu_seqlens`` attribute on
the instance — read it via the inherited :attr:`Batch.cu_seqlens` property
and per-sample sizes via :attr:`Batch.lengths`. Segment ``k``'s tokens are
``tokens[cu_seqlens[k]:cu_seqlens[k+1]]``.

Construct via :meth:`TextSegment.pack` (the framework-provided constructor)
passing per-sample tensor lists, e.g. ``TextSegment.pack(tokens=[t0, t1, t2],
log_probs=[lp0, lp1, lp2])``. The default
``TextSegment(...)`` dataclass constructor still works for already-packed
inputs but leaves cu_seqlens at None until set externally (used internally
by ``concat`` / ``slice`` / ``select``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import packed_field
from unirl.types.conditions.base import Condition, Modality
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.segments.base import Segment


@dataclass
class TextSegment(Segment):
    """AR text segment with packed varlen tokens."""

    modality: ClassVar[Modality] = Modality.TEXT

    tokens: Optional[torch.Tensor] = packed_field(default=None)
    log_probs: Optional[torch.Tensor] = packed_field(default=None)
    loss_mask: Optional[torch.Tensor] = packed_field(default=None)
    # Learner-side, action-token-aligned SAO signals.  They deliberately live on
    # the segment instead of Part.advantages (which is sequence-aligned) so
    # variable-length token values survive Batch packing and DP slicing exactly.
    token_advantages: Optional[torch.Tensor] = packed_field(default=None)
    value_targets: Optional[torch.Tensor] = packed_field(default=None)
    value_mask: Optional[torch.Tensor] = packed_field(default=None)

    def as_condition_with(self, encoder: Callable[..., Any]) -> Condition:
        """Re-embed packed tokens via the supplied encoder into a TextEmbedCondition.

        ``encoder`` is invoked as ``encoder(tokens)`` and is expected to return
        a tensor (or object with an ``embeds`` field) shaped to feed
        ``TextEmbedCondition.embeds``. The contract is intentionally loose so
        unified-model bundles can pass shared-embedding-table lookups or
        full encoder forwards transparently.
        """
        if self.tokens is None:
            raise ValueError("TextSegment.as_condition_with: tokens is None")
        out = encoder(self.tokens)
        embeds = getattr(out, "embeds", out)
        return TextEmbedCondition(embeds=embeds)


__all__ = ["TextSegment"]
