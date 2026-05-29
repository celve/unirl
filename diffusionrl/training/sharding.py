"""Per-actor sharding of multi-track :class:`RolloutResp`.

The training-side DP fan-out: a multi-track :class:`RolloutResp` arrives
on the driver with one ``RolloutTrack`` per track (e.g. ``refined`` and
``image`` in PE-style joint training, where each track has its own
``batch_size`` and the ``image`` samples reference ``refined`` parents
via ``parent_ids``). This module splits the resp across N actors such
that:

1. Lineage stays coherent per shard — for a leaf sample on actor A,
   that sample's full ancestor chain (refined parent, root prompt, ...)
   also lives on actor A. Cross-actor parent lookups never happen.
2. Per-shard ``RolloutTrack.compute_advantages`` keeps working — the
   ``(n_groups, branch)`` reshape invariance is preserved because each
   actor sees complete root-group blocks.

The strategy is **shard by root track**: identify the unique track with
``parent_track is None`` (or the only track for single-track resps),
balanced-split its sample indices across actors using
``floor / remainder``, then walk the lineage tree downward to build
per-actor index sets for every child track. ``RolloutTrack.select`` is
the underlying primitive (inherited from ``Batch.select``).

Fail-fast on:

- More than one root track (ambiguous sharding decision).
- Root ``batch_size < num_actors`` (every FSDP rank must see at least
  one sample; collectives deadlock otherwise).
- Non-uniform branching (a leaf sample whose ancestor chain references
  a sample id that no parent track exposes — indicates malformed
  lineage).
"""

from __future__ import annotations

from typing import Dict, List, Set

from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack


def _find_root_track_name(tracks: Dict[str, RolloutTrack]) -> str:
    """Return the name of the unique track with ``parent_track is None``.

    A multi-track resp may carry multiple root tracks in theory (e.g. two
    independent modalities each rooted at the request). Sharding by
    "the root" is ambiguous in that case; this function raises so callers
    surface the structural mismatch rather than silently picking one.
    """
    roots = [name for name, t in tracks.items() if t.parent_track is None]
    if len(roots) == 0:
        raise ValueError(
            "shard_resp_per_actor: no root track found (every track has a "
            "parent_track); the lineage tree must have at least one root."
        )
    if len(roots) > 1:
        raise ValueError(
            f"shard_resp_per_actor: multiple root tracks {sorted(roots)} — "
            "ambiguous sharding choice. Restructure the lineage to have a "
            "single root, or shard manually upstream."
        )
    return roots[0]


def _compute_root_ancestor_per_sample(
    tracks: Dict[str, RolloutTrack],
    root_name: str,
) -> Dict[str, List[str]]:
    """For each track, return a list (parallel to track.sample_ids) of
    the root track's sample_id that each sample descends from.

    For the root track itself this is just ``list(track.sample_ids)``.
    Other tracks walk their ``parent_ids`` up the lineage to root.

    Topological walk: process tracks once their parent's ancestor mapping
    is already resolved. Raises if a track's lineage is dangling (parent
    track absent from the resp, or a parent_id absent from the parent's
    sample_ids).
    """
    resolved: Dict[str, List[str]] = {root_name: list(tracks[root_name].sample_ids)}
    pending: Dict[str, RolloutTrack] = {n: t for n, t in tracks.items() if n != root_name}

    while pending:
        progressed = False
        for name in list(pending.keys()):
            track = pending[name]
            parent_name = track.parent_track
            if parent_name is None:
                # Another root track — caught earlier in _find_root_track_name,
                # but defensively raise here too.
                raise ValueError(
                    f"shard_resp_per_actor: track {name!r} is rooted but "
                    f"_find_root_track_name returned {root_name!r}; multi-root "
                    "resps are not supported."
                )
            if parent_name not in resolved:
                # Parent not yet resolved; try the next pending track in this pass.
                continue
            if parent_name not in tracks:
                raise ValueError(
                    f"shard_resp_per_actor: track {name!r}.parent_track="
                    f"{parent_name!r} is not present in resp.tracks "
                    f"(keys: {sorted(tracks)})."
                )
            parent_track = tracks[parent_name]
            parent_sid_to_idx = {sid: i for i, sid in enumerate(parent_track.sample_ids)}
            parent_root_ancestors = resolved[parent_name]
            track_parent_ids = track.parent_ids or []
            if len(track_parent_ids) != len(track.sample_ids):
                raise ValueError(
                    f"shard_resp_per_actor: track {name!r}.parent_ids has "
                    f"length {len(track_parent_ids)} but sample_ids has "
                    f"length {len(track.sample_ids)}."
                )
            ancestors: List[str] = []
            for pid in track_parent_ids:
                if pid not in parent_sid_to_idx:
                    raise ValueError(
                        f"shard_resp_per_actor: track {name!r} references "
                        f"parent_id={pid!r} that is absent from parent track "
                        f"{parent_name!r}.sample_ids."
                    )
                ancestors.append(parent_root_ancestors[parent_sid_to_idx[pid]])
            resolved[name] = ancestors
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(
                f"shard_resp_per_actor: cycle or unresolved parents in lineage tree; pending tracks: {sorted(pending)}."
            )
    return resolved


def _balanced_split(*, total_size: int, num_shards: int) -> List[range]:
    """Balanced-split ``[0, total_size)`` into ``num_shards`` contiguous ranges.

    Every shard gets ``floor(total_size / num_shards)``; the first
    ``total_size % num_shards`` shards get one extra. Same allocation as
    today's single-track ``TrainActorGroup.train`` (preserves the
    deterministic mapping recipes rely on).
    """
    base = total_size // num_shards
    remainder = total_size % num_shards
    cursor = 0
    out: List[range] = []
    for i in range(num_shards):
        size = base + (1 if i < remainder else 0)
        out.append(range(cursor, cursor + size))
        cursor += size
    return out


def shard_resp_per_actor(resp: RolloutResp, num_actors: int) -> List[RolloutResp]:
    """Shard a multi-track ``RolloutResp`` across ``num_actors`` DP ranks.

    Returns a list of length ``num_actors``; each entry is a new
    ``RolloutResp`` whose tracks are coherent sub-slices of the input
    tracks, preserving lineage:

    - Identifies the unique root track (``parent_track is None``).
    - Balanced-splits the root's sample indices across actors.
    - For each non-root track, builds per-actor index sets by walking
      the lineage up to root: a sample lands on actor A iff its root
      ancestor is one of actor A's root samples.
    - Per-actor ``RolloutResp`` is built via per-track
      :meth:`RolloutTrack.select` (inherited from ``Batch.select``).

    Raises ``ValueError`` if:

    - ``num_actors < 1`` or the resp has no tracks.
    - The root track's ``batch_size < num_actors`` (every FSDP rank
      must receive at least one root sample).
    - Multi-root resps (ambiguous sharding choice).
    """
    if num_actors < 1:
        raise ValueError(f"shard_resp_per_actor: num_actors must be >= 1; got {num_actors}.")
    if not resp.tracks:
        raise ValueError("shard_resp_per_actor: resp has no tracks.")
    if num_actors == 1:
        return [resp]

    root_name = _find_root_track_name(resp.tracks)
    root_track = resp.tracks[root_name]
    root_bs = int(root_track.batch_size)
    if root_bs < num_actors:
        raise ValueError(
            f"shard_resp_per_actor: root track {root_name!r}.batch_size="
            f"{root_bs} is smaller than num_actors={num_actors}; every "
            "FSDP rank must receive at least one root sample (collectives "
            "require all ranks to participate)."
        )

    ancestors_per_track = _compute_root_ancestor_per_sample(resp.tracks, root_name)
    root_index_ranges = _balanced_split(total_size=root_bs, num_shards=num_actors)

    import torch

    shards: List[RolloutResp] = []
    for actor_idx, root_range in enumerate(root_index_ranges):
        # Set of root sample_ids this actor owns.
        actor_root_sids: Set[str] = {root_track.sample_ids[i] for i in root_range}

        actor_tracks: Dict[str, RolloutTrack] = {}
        for track_name, track in resp.tracks.items():
            ancestors = ancestors_per_track[track_name]
            indices = [i for i, root_sid in enumerate(ancestors) if root_sid in actor_root_sids]
            if not indices:
                # An empty per-actor track on a downstream track means the
                # branching is non-uniform vs the root split — fail-fast.
                # (Uniform branching guarantees every actor gets at least one
                # sample per track when root_bs >= num_actors.)
                raise ValueError(
                    f"shard_resp_per_actor: actor {actor_idx} would receive 0 "
                    f"samples on track {track_name!r}; this means the lineage "
                    f"branching is non-uniform vs the root split. Use uniform "
                    "branching (RolloutReq.make_root_track / "
                    "RolloutTrack.fork_track guarantee this)."
                )
            actor_tracks[track_name] = track.select(torch.tensor(indices, dtype=torch.long))

        shards.append(type(resp)(tracks=actor_tracks, reward_compute_s=resp.reward_compute_s))

    return shards


__all__ = ["shard_resp_per_actor"]
