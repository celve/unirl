"""RolloutResp — top-level SoA container for one rollout's worth of data.

Holds three dicts keyed by free-form modality+role strings:

- ``conditions: Dict[str, Condition]`` — encoded inputs the rollout used.
- ``rollout_traces: Dict[str, Segment]`` — encoded outputs the rollout produced.
- ``decoded: Dict[str, Texts | Images | Videos | Audios]`` — decoded outputs
  ready for reward / display / external consumption.

Plus per-sample fields (sample_ids, group_ids, rewards, advantages, status).

The ``concat`` override is the only custom logic: when merging two rollout
shards, the second shard's ``Segment.sample_indices`` must be offset-shifted
by ``len(first_shard.sample_ids)`` so rollout traces resolve to the correct
sample in the merged container. Everything else (dict merging, tensor concat)
is handled by ``Batched.concat`` recursion.

Per-sample access is via raw indexing (no ``SegmentView``). For
per-segment-row fields (like LatentSegment's latents)::

    mask = resp.rollout_traces["image"].sample_indices == i
    latents_i = resp.rollout_traces["image"].latents[mask]

For packed varlen fields (like TextSegment's tokens), use the
framework-managed ``cu_seqlens`` to slice each sample's chunk::

    cu = resp.rollout_traces["text"].cu_seqlens
    tokens_i = resp.rollout_traces["text"].tokens[cu[i]:cu[i + 1]]

If a richer per-sample API is needed later, helpers can be added without
changing the storage layout.

Pairs with ``RolloutReq`` (in ``diffusionrl/types/rollout_req.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

import torch

from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.types.conditions import Condition
from diffusionrl.types.primitives import Audios, Images, Texts, Videos
from diffusionrl.types.sample import MediaPreview
from diffusionrl.types.segments import Segment
from diffusionrl.utils.batched import Batched, FieldKind, concat_field, field, max_field

T = TypeVar("T", bound="RolloutResp")

Decoded = Union[Texts, Images, Videos, Audios]


@dataclass
class RolloutResp(Transportable):
    conditions: Dict[str, Condition] = field(kind=FieldKind.CONCAT, transport=True, default_factory=dict)
    rollout_traces: Dict[str, Segment] = field(kind=FieldKind.CONCAT, transport=True, default_factory=dict)
    decoded: Dict[str, Decoded] = field(kind=FieldKind.CONCAT, transport=True, default_factory=dict)

    sample_ids: List[str] = concat_field(default_factory=list)
    group_ids: List[str] = concat_field(default_factory=list)
    rewards: Optional[torch.Tensor] = concat_field(default=None)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    status: Optional[torch.Tensor] = concat_field(default=None)
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    media_preview: Optional[MediaPreview] = concat_field(default=None)
    reward_compute_s: float = max_field(default=0.0)

    # ---- batch_size override ------------------------------------------------

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size

    # ---- per-group split ---------------------------------------------------

    def split(self) -> List["RolloutResp"]:
        """Split into one ``RolloutResp`` per unique group id, preserving order.

        Mirrors ``RolloutResponse.split`` (``diffusionrl/types/response.py:21``)
        but reads ``group_ids`` directly off this container (the legacy variant
        reads it off ``self.request.prompts.group_ids``). Builds per-group
        shards via ``RolloutResp.select`` (the override below) so per-sample
        fields go through ``Batched.select`` and per-token rollout traces get
        the sample-aware filter+remap.
        """
        if not self.group_ids:
            return [self]
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(self.group_ids):
            groups.setdefault(gid, []).append(i)
        results: List[RolloutResp] = []
        for gid in dict.fromkeys(self.group_ids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(self.select(indices))
        return results

    # ---- media-preview cap -------------------------------------------------

    def cap_media_preview(self, max_items: int) -> None:
        """Truncate ``media_preview`` to at most ``max_items`` entries.

        Mirrors ``RolloutSamples.cap_media_preview``
        (``diffusionrl/types/sample.py:142``). Driver-side hook used after
        ``aggregate`` to enforce the per-rollout cap regardless of how many
        shards contributed.
        """
        if self.media_preview is None:
            return
        limit = max(1, int(max_items))
        if len(self.media_preview) <= limit:
            return
        self.media_preview = self.media_preview.slice(0, limit)

    # ---- concat with sample_indices remap ----------------------------------

    @classmethod
    def concat(cls: Type[T], items: Sequence[T]) -> T:
        if not items:
            raise ValueError(f"Cannot concat empty sequence of {cls.__name__}")
        if len(items) == 1:
            return items[0]

        # Cumulative sample-count offsets, one per shard.
        offsets: List[int] = [0]
        for shard in items[:-1]:
            offsets.append(offsets[-1] + len(shard.sample_ids))

        shifted: List[T] = []
        for shard, off in zip(items, offsets):
            if off == 0:
                shifted.append(shard)
                continue
            shifted.append(_shift_sample_indices(shard, off))

        # Delegate to the generic Batched concat now that rollout traces are aligned.
        return Batched.concat.__func__(cls, shifted)


def _shift_sample_indices(resp: T, offset: int) -> T:
    """Return a copy of ``resp`` whose traces' ``sample_indices`` are shifted by ``offset``."""
    if offset == 0 or not resp.rollout_traces:
        return resp

    new_rollout_traces: Dict[str, Segment] = {}
    for key, seg in resp.rollout_traces.items():
        new_seg = seg.clone()
        if new_seg.sample_indices is not None:
            new_seg.sample_indices = new_seg.sample_indices + offset
        new_rollout_traces[key] = new_seg

    # Rebuild via the dataclass init so all other fields are preserved.
    kwargs: Dict[str, Any] = {}
    from dataclasses import fields as dc_fields

    for f in dc_fields(resp):
        kwargs[f.name] = getattr(resp, f.name)
    kwargs["rollout_traces"] = new_rollout_traces
    return type(resp)(**kwargs)


__all__ = ["RolloutResp", "Decoded"]
