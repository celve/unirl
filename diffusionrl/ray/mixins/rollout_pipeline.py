"""Rollout pipeline mixin for the ``RolloutReq`` / ``RolloutResp`` path.

Operates on ``RolloutResp`` directly. Reward scoring consumes each
scorable track through
:meth:`diffusionrl.reward.pipeline.RewardPipeline.score_and_attach`,
which takes ``(req, track)`` and writes rewards onto the track in
place.

Host class contract
-------------------
The host must provide:

- ``self.engine`` — a :class:`BaseRolloutEngine`
- ``self._rollout_plan`` — a ``RolloutPlan`` (forward_batch_size only used by chunking)
- ``self._adv_scope`` — ``str``: ``"global"`` or ``"group"``
- ``self._adv_use_global_std`` — ``bool``
- ``self._adv_samples_per_prompt`` — ``int``
- ``self.generate(req: RolloutReq) → RolloutResp`` — provided by the host actor
- ``self.put_buffer`` / ``self.get_buffer`` / ``self.pop_buffer`` (from ``Buffer``)
- ``self._ensure_reward_pipeline()`` — returns ``RewardPipeline``

Each ``generate_buffered`` call splits the resp by group and pairs each shard
with a per-group ``RolloutReq`` shard. The pairing is held on a per-actor
``_handle_state`` dict keyed by handle id so we don't mutate ``RolloutResp`` to
carry runtime metadata across Ray.

Track dispatch: scorable tracks are discovered by segment type via
:data:`SCORER_BY_SEGMENT_TYPE` — the single mapping ``LatentSegment →
"default"`` today, generalizable to multi-modality (TextSegment, …) when
more reward services land. Reward scoring writes to each scorable track;
:meth:`RolloutResp.propagate_rewards` then fills parent-track rewards
from their children. Sharding identity (sample_ids/group_ids for the
buffer handle and per-shard ``RolloutReq``) comes from the root track
(the unique track with ``parent_track=None``). Single-track resps are
the trivial case: root == only == scorable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Type

import torch

from diffusionrl.algorithms.normalizers import (
    build_group_index_map,
    normalize_global,
    normalize_grouped,
    require_expected_group_sizes,
    require_valid_group_ids,
)
from diffusionrl.distributed.transfer_queue import TransferQueueRuntime, tqbridge
from diffusionrl.types.media_preview import build_media_preview_for_track
from diffusionrl.types.sampling import get_ar_params, get_diffusion_params
from diffusionrl.types.segments.base import Segment
from diffusionrl.types.segments.latent import LatentSegment
from diffusionrl.utils.batched import Batched, concat_field

if TYPE_CHECKING:
    from diffusionrl.transfer.buffer import BufferHandle
    from diffusionrl.types.rollout_req import RolloutReq
    from diffusionrl.types.rollout_resp import RolloutResp


logger = logging.getLogger(__name__)


SCORER_BY_SEGMENT_TYPE: Dict[Type[Segment], str] = {
    LatentSegment: "default",
}
"""Maps a track's segment type to a scorer-registry key.

Today the actor holds one ``RewardPipeline`` and the key ``"default"``
points at it; multi-service support (LLM-judge for refined, CLIP for
image, …) is a follow-up that lifts the key into an actual service
handle. Structural track-lookup ("which tracks have this segment type?")
lives on :class:`RolloutResp` as
:meth:`~diffusionrl.types.rollout_resp.RolloutResp.tracks_with_segment_types`;
the registry here is the caller-side dispatch policy that consults it.
"""


@dataclass
class _RolloutRespMeta(Batched):
    """Per-handle key for buffered RolloutResp shards.

    Carried on ``BufferHandle.key`` for handle introspection and any
    future driver-side routing.
    """

    sample_ids: List[str] = concat_field(default_factory=list)
    group_ids: List[str] = concat_field(default_factory=list)


def _make_meta(resp_shard: "RolloutResp") -> _RolloutRespMeta:
    track = resp_shard.root_track()
    return _RolloutRespMeta(
        sample_ids=list(track.sample_ids),
        group_ids=list(track.group_ids),
    )


def _stamp_actor_reward_total(responses: List["RolloutResp"]) -> None:
    """Replace each per-handle ``reward_compute_s`` with the per-actor sum.

    Mirrors the legacy ``_stamp_actor_reward_total`` in
    ``ray/mixins/rollout_pipeline.py`` for the new shape.
    """
    actor_total = sum(float(r.reward_compute_s) for r in responses)
    for r in responses:
        r.reward_compute_s = actor_total


def _responses_to_cpu(responses: List["RolloutResp"]) -> List["RolloutResp"]:
    """Stage rollout payloads on CPU before Ray serializes them.

    With TransferQueue off (the default), returned CUDA tensors deserialize
    onto the driver's GPU 0 — every actor's shard collapses there under
    ``CUDA_VISIBLE_DEVICES`` isolation — so the driver gather + ``aggregate``
    OOMs as the global batch grows. Staging on CPU keeps that in host RAM;
    train actors move their shard back to device in ``_train_resp``. With TQ
    on, ``@tqbridge(put=True)`` dehydrates these already-CPU payloads into the
    store, so the two paths compose.
    """
    return [response.to_device("cpu") for response in responses]


class RolloutPipelineMixin:
    """Reusable generate → reward → advantage pipeline for ``RolloutReq``/``RolloutResp``."""

    # ---- per-handle state --------------------------------------------------

    def _ensure_handle_state(self) -> Dict[str, "RolloutReq"]:
        if not hasattr(self, "_handle_state"):
            self._handle_state: Dict[str, "RolloutReq"] = {}
        return self._handle_state

    def _split_req_by_group(self, req: "RolloutReq") -> List["RolloutReq"]:
        """Mirror :meth:`RolloutResp.split` for ``RolloutReq``.

        Inlined here (not a method on ``RolloutReq``) because the symmetry
        is local to the buffered-pipeline lifecycle — the public
        ``RolloutReq`` API stays leaner.
        """
        if not req.group_ids:
            return [req]
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(req.group_ids):
            groups.setdefault(gid, []).append(i)
        results: List["RolloutReq"] = []
        for gid in dict.fromkeys(req.group_ids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(req.select(indices))
        return results

    # ---- buffered generation -----------------------------------------------

    def generate_buffered(self, req: "RolloutReq") -> List["BufferHandle"]:
        """Generate, split per group, buffer per shard. Pairs each handle with
        its originating per-group ``RolloutReq`` shard via ``_handle_state``.
        """
        full_resp = self.generate(req)
        resp_shards = full_resp.split()
        req_shards = self._split_req_by_group(req)
        # PE-joint case: resp is grouped by LLM-rewrite parents (P*N groups)
        # while req is grouped by original prompts (P groups). Each req shard
        # corresponds to N consecutive resp shards (the N LLM rewrites of
        # that prompt; group-by-parent contiguous ordering is guaranteed by
        # RolloutReq.make_root_track / RolloutTrack.fork_track). Replicate
        # each req shard N times so the per-handle pairing works. Non-PE
        # (1:1) case: factor=1, replicate is a no-op.
        if len(req_shards) > 0 and len(resp_shards) > len(req_shards) and len(resp_shards) % len(req_shards) == 0:
            factor = len(resp_shards) // len(req_shards)
            # Cross-check the divisibility heuristic against the request's
            # explicit branching factors (N=PE-rewrites × M=samples-per-PE).
            # Without this, an accidentally divisible factor (e.g. 2× when
            # N×M=4) would silently pair shards to the wrong request.
            ar_params = get_ar_params(req.sampling_params)
            diff_params = get_diffusion_params(req.sampling_params)
            _N = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            _M = int(diff_params.samples_per_prompt) if diff_params is not None else 1
            expected_factor = _N * _M
            if factor != expected_factor:
                raise RuntimeError(
                    f"RolloutPipelineMixin.generate_buffered: implicit resp/req shard "
                    f"factor {factor} (len(resp_shards)={len(resp_shards)} / "
                    f"len(req_shards)={len(req_shards)}) does not match sampling_params "
                    f"N*M={expected_factor} (N={_N}, M={_M}). "
                    f"Shard pairing is ambiguous."
                )
            expanded: List["RolloutReq"] = []
            for r in req_shards:
                expanded.extend([r] * factor)
            req_shards = expanded
        if len(req_shards) != len(resp_shards):
            raise RuntimeError(
                f"RolloutPipelineMixin.generate_buffered: req split has "
                f"{len(req_shards)} shards but resp split has {len(resp_shards)}; "
                f"req.group_ids and resp track group_ids must align."
            )

        state = self._ensure_handle_state()
        handles: List["BufferHandle"] = []
        for req_shard, resp_shard in zip(req_shards, resp_shards):
            handle = self.put_buffer(_make_meta(resp_shard), resp_shard)
            state[handle.id] = req_shard
            handles.append(handle)
        return handles

    # ---- reward attachment -------------------------------------------------

    def attach_reward(self, handle: "BufferHandle") -> None:
        """Score the buffered ``RolloutResp`` directly off ``(req, track)``.

        Iterates :meth:`RolloutResp.tracks_with_segment_types` against
        :data:`SCORER_BY_SEGMENT_TYPE` to find tracks whose segment type
        has a registered scorer (today: ``LatentSegment``). Each scorable
        track is handed to :meth:`RewardPipeline.score_and_attach` along
        with the per-shard ``RolloutReq``; the reward pipeline reads
        texts off ``req.primitives['text']`` and writes
        ``rewards`` / ``component_rewards`` onto the track in place.
        After per-track scoring, :meth:`RolloutResp.propagate_rewards`
        fills parent-track rewards from their children (mean over each
        group). Single-track resps reduce to today's "score the one
        LatentSegment track" behavior; no parent exists, so propagate
        is a no-op.

        No ``decode_latents`` call — engines ship the decoded pixels on
        each scorable track's ``decoded`` already (``Images`` / ``Videos``
        primitive). Media preview is captured only on scored leaves (the
        only tracks with decoded pixels).
        """
        resp: "RolloutResp" = self.get_buffer(handle)
        state = self._ensure_handle_state()
        try:
            req_shard = state[handle.id]
        except KeyError as exc:
            raise RuntimeError(
                f"RolloutPipelineMixin.attach_reward: handle {handle.id} has "
                f"no recorded state. Was generate_buffered called on this actor?"
            ) from exc

        collect_media = bool(req_shard.collect_media_preview)
        max_items = int(req_shard.media_max_items)

        score_t0 = time.perf_counter()
        for _name, track in resp.tracks_with_segment_types(SCORER_BY_SEGMENT_TYPE.keys()):
            self._ensure_reward_pipeline().score_and_attach(req=req_shard, track=track)
            if collect_media:
                track.media_preview = build_media_preview_for_track(
                    req=req_shard,
                    track=track,
                    max_items=max_items,
                )
            else:
                track.media_preview = None
        resp.reward_compute_s = float(time.perf_counter() - score_t0)

        # Fill parent-track rewards from their children. ``propagate_rewards``
        # returns a new resp where tracks with already-set rewards reuse the
        # same instance (direct-rewards-win), so we only need to copy back
        # newly-filled parent rewards onto the buffered track instances.
        propagated = resp.propagate_rewards(op="mean")
        for tname, t_new in propagated.tracks.items():
            if resp.tracks[tname].rewards is None and t_new.rewards is not None:
                resp.tracks[tname].rewards = t_new.rewards

    # ---- advantage computation --------------------------------------------

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: List[str],
    ) -> torch.Tensor:
        scope = str(self._adv_scope)
        if scope == "global":
            return normalize_global(rewards)
        if scope == "group":
            normalized_ids = require_valid_group_ids(group_ids)
            group_index_map = build_group_index_map(normalized_ids)
            groups = require_expected_group_sizes(group_index_map, self._adv_samples_per_prompt)
            if not groups:
                raise ValueError(
                    "adv_normalization_scope='group' could not find any valid group; "
                    "all group_ids were empty after normalization."
                )
            return normalize_grouped(
                rewards,
                groups,
                use_global_std=bool(self._adv_use_global_std),
            )
        raise ValueError(f"Unknown adv_normalization_scope={scope!r}. Expected 'global' or 'group'.")

    def compute_advantages(self, handle: "BufferHandle") -> None:
        """Compute advantages for one buffered shard, per-track.

        Iterates every track with rewards attached (scored leaves + tracks
        whose rewards were propagated from children). Used outside the
        fused ``run_rollout_pipeline`` flow when callers attach reward +
        advantages per handle; the fused entrypoint normalizes across all
        shards instead.
        """
        resp: "RolloutResp" = self.get_buffer(handle)
        any_rewarded = False
        for track in resp.tracks.values():
            if track.rewards is None:
                continue
            any_rewarded = True
            track.advantages = self._compute_advantages(
                rewards=track.rewards,
                group_ids=list(track.group_ids),
            )
        if not any_rewarded:
            raise RuntimeError("Cannot compute advantages: rewards not attached.")

    # ---- fused pipelines ---------------------------------------------------

    @tqbridge(get=False, put=True)
    def run_rollout_pipeline(self, req: "RolloutReq") -> List["RolloutResp"]:
        """Fused actor-side rollout: generate + reward + cross-shard advantages.

        Per-track cross-shard GRPO: for each track name present in the
        responses with rewards attached, concat that track's rewards and
        group_ids across all shards, compute advantages once, then scatter
        back. Single-track resps reduce to today's behavior. Multi-track
        resps get one cross-shard advantage compute per track name —
        each track has its own group equivalence classes (e.g. refined's
        groups are per-prompt, image's groups are per-refined-parent).

        ``algorithm.use_global_std=True`` sees the full reward distribution
        for this actor on each track. Per-group mean is preserved via
        ``group_ids``.
        """
        handles = self.generate_buffered(req)
        for h in handles:
            self.attach_reward(h)
        responses: List["RolloutResp"] = [self.pop_buffer(h) for h in handles]
        # Free per-handle state regardless of downstream success.
        state = self._ensure_handle_state()
        for h in handles:
            state.pop(h.id, None)

        _stamp_actor_reward_total(responses)

        # Discover all track names that carry rewards across this actor's
        # shards. Preserve first-seen insertion order so multi-track resps
        # keep parent-before-child ordering (matters only for logging /
        # debugging; advantage compute is per-track-independent).
        track_names: List[str] = []
        seen: set = set()
        for r in responses:
            for name, t in r.tracks.items():
                if name in seen:
                    continue
                if t.rewards is not None:
                    seen.add(name)
                    track_names.append(name)
        if not track_names:
            raise RuntimeError("Cannot compute advantages: rewards not attached on any track.")

        for name in track_names:
            shards = [r.tracks[name] for r in responses if name in r.tracks]
            missing = [i for i, t in enumerate(shards) if t.rewards is None]
            if missing:
                raise RuntimeError(
                    f"Cannot compute advantages for track {name!r}: rewards missing on shard(s) {missing}."
                )
            all_rewards = torch.cat([t.rewards for t in shards])
            all_group_ids: List[str] = [gid for t in shards for gid in t.group_ids]
            all_advantages = self._compute_advantages(
                rewards=all_rewards,
                group_ids=all_group_ids,
            )
            offset = 0
            for t in shards:
                n = int(t.rewards.shape[0])
                t.advantages = all_advantages[offset : offset + n]
                offset += n
        return _responses_to_cpu(responses)

    def run_eval_pipeline(self, req: "RolloutReq") -> List["RolloutResp"]:
        """Eval pipeline: generate + reward, no advantages."""
        handles = self.generate_buffered(req)
        for h in handles:
            self.attach_reward(h)
        responses: List["RolloutResp"] = [self.pop_buffer(h) for h in handles]
        state = self._ensure_handle_state()
        for h in handles:
            state.pop(h.id, None)
        _stamp_actor_reward_total(responses)
        return _responses_to_cpu(responses)

    # ---- TransferQueue lifecycle (parity with legacy mixin) ---------------

    def init_transferqueue_client(self, *, handoff: dict) -> None:
        """Install per-actor TransferQueue client (called from the driver)."""
        self.tq_handoff = handoff
        self.tq_runtime = TransferQueueRuntime().install()
        self.tq_client = self.tq_runtime.create_client("Actor", handoff)

    def reset_zero_copy_buffer_free(self) -> None:
        """Reset the actor's zero-copy buffer-free state at the top of each rollout step."""
        self.tq_runtime.reset_zero_copy_buffer_free()


__all__ = ["RolloutPipelineMixin"]
