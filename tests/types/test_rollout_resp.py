"""Tests for RolloutResp container and its concat sample_indices remap."""

from __future__ import annotations

import pytest
import torch

from diffusionrl.types.conditions import TextEmbedCondition
from diffusionrl.types.primitives import Image, Images
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments import LatentSegment


def _make_shard(
    sample_ids,
    *,
    text_embeds,
    image_seg_sample_indices,
    image_seg_positions,
    image_latents,
    decoded_pixels,
    rewards,
):
    return RolloutResp(
        tracks={
            "image_latent": RolloutTrack(
                sample_ids=sample_ids,
                parent_ids=["g"] * len(sample_ids),
                conditions={"text": TextEmbedCondition(embeds=text_embeds)},
                segment=LatentSegment(
                    sample_indices=image_seg_sample_indices,
                    positions=image_seg_positions,
                    latents=image_latents,
                ),
                decoded=Images.from_list([Image(pixels=p) for p in decoded_pixels]),
                rewards=rewards,
            ),
        }
    )


def test_concat_remaps_sample_indices_via_offset_shift():
    a = _make_shard(
        sample_ids=["s0", "s1"],
        text_embeds=torch.zeros(2, 4, 8),
        image_seg_sample_indices=torch.tensor([0, 1]),
        image_seg_positions=torch.tensor([0, 0]),
        image_latents=torch.zeros(2, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.5, 0.7]),
    )
    b = _make_shard(
        sample_ids=["s2"],
        text_embeds=torch.ones(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.ones(1, 2, 4, 4, 4),
        decoded_pixels=[torch.ones(3, 4, 4)],
        rewards=torch.tensor([0.9]),
    )

    merged = RolloutResp.concat([a, b])
    assert merged.batch_size == 3
    assert merged.tracks["image_latent"].sample_ids == ["s0", "s1", "s2"]

    # Segment sample_indices are offset-shifted in the second shard.
    seg = merged.tracks["image_latent"].segment
    assert seg.sample_indices.tolist() == [0, 1, 2]
    assert seg.positions.tolist() == [0, 0, 0]
    assert seg.latents.shape == (3, 2, 4, 4, 4)


def test_concat_merges_dict_keyed_fields_by_key():
    a = _make_shard(
        sample_ids=["s0"],
        text_embeds=torch.zeros(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.zeros(1, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.1]),
    )
    b = _make_shard(
        sample_ids=["s1"],
        text_embeds=torch.ones(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.ones(1, 2, 4, 4, 4),
        decoded_pixels=[torch.ones(3, 4, 4)],
        rewards=torch.tensor([0.2]),
    )

    merged = RolloutResp.concat([a, b])

    # Conditions merged by key.
    assert "text" in merged.tracks["image_latent"].conditions
    assert merged.tracks["image_latent"].conditions["text"].embeds.shape == (2, 4, 8)

    # Decoded values concatenated along batch dim.
    assert merged.tracks["image_latent"].decoded.pixels.shape == (2, 3, 4, 4)


def test_concat_does_not_mutate_input_shards():
    a = _make_shard(
        sample_ids=["s0"],
        text_embeds=torch.zeros(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.zeros(1, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.1]),
    )
    b = _make_shard(
        sample_ids=["s1"],
        text_embeds=torch.ones(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.ones(1, 2, 4, 4, 4),
        decoded_pixels=[torch.ones(3, 4, 4)],
        rewards=torch.tensor([0.2]),
    )
    RolloutResp.concat([a, b])
    # Second shard's segment sample_indices unchanged.
    assert b.tracks["image_latent"].segment.sample_indices.tolist() == [0]


def test_slice_picks_subset_along_sample_axis():
    a = _make_shard(
        sample_ids=["s0", "s1", "s2"],
        text_embeds=torch.zeros(3, 4, 8),
        image_seg_sample_indices=torch.tensor([0, 1, 2]),
        image_seg_positions=torch.tensor([0, 0, 0]),
        image_latents=torch.zeros(3, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4)] * 3,
        rewards=torch.tensor([0.1, 0.2, 0.3]),
    )
    sub = a.slice(1, 3)
    assert sub.batch_size == 2
    assert sub.tracks["image_latent"].sample_ids == ["s1", "s2"]


def test_per_sample_raw_indexing():
    """Confirm the no-SegmentView access pattern works."""
    a = _make_shard(
        sample_ids=["s0", "s1"],
        text_embeds=torch.zeros(2, 4, 8),
        image_seg_sample_indices=torch.tensor([0, 0, 1]),  # sample 0 has two segments
        image_seg_positions=torch.tensor([0, 1, 0]),
        image_latents=torch.arange(3 * 2 * 4 * 4 * 4, dtype=torch.float).reshape(3, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.1, 0.2]),
    )

    seg = a.tracks["image_latent"].segment
    mask = seg.sample_indices == 0
    sample_0_latents = seg.latents[mask]
    assert sample_0_latents.shape == (2, 2, 4, 4, 4)

    mask = seg.sample_indices == 1
    sample_1_latents = seg.latents[mask]
    assert sample_1_latents.shape == (1, 2, 4, 4, 4)


# ---- multi-track behavior (Step 1 of the tracks refactor) ------------------


def _make_track(name, sample_ids, *, parent_ids=None, parent_track=None, n_segs=None):
    """Helper: minimal RolloutTrack with a LatentSegment carrying sample_indices."""
    n = len(sample_ids) if n_segs is None else n_segs
    return RolloutTrack(
        sample_ids=list(sample_ids),
        parent_ids=list(parent_ids) if parent_ids is not None else None,
        parent_track=parent_track,
        segment=LatentSegment(
            sample_indices=torch.arange(n, dtype=torch.long),
            positions=torch.zeros(n, dtype=torch.long),
            latents=torch.zeros(n, 2, 4, 4, 4),
        ),
    )


def test_two_track_resp_with_lineage_constructs_cleanly():
    """Happy path: refined parent + image child with valid foreign-key lineage."""
    refined = _make_track(
        "refined",
        ["p0/r0", "p0/r1", "p1/r0", "p1/r1"],
        parent_ids=["p0", "p0", "p1", "p1"],
    )
    image = _make_track(
        "image",
        ["p0/r0/i0", "p0/r0/i1", "p0/r1/i0", "p0/r1/i1", "p1/r0/i0", "p1/r0/i1", "p1/r1/i0", "p1/r1/i1"],
        parent_ids=["p0/r0", "p0/r0", "p0/r1", "p0/r1", "p1/r0", "p1/r0", "p1/r1", "p1/r1"],
        parent_track="refined",
    )
    resp = RolloutResp(tracks={"refined": refined, "image": image})
    assert set(resp.tracks.keys()) == {"refined", "image"}
    # group_ids derived per-track from parent_ids:
    assert resp.tracks["refined"].group_ids == ["p0", "p0", "p1", "p1"]
    assert resp.tracks["image"].group_ids == [
        "p0/r0",
        "p0/r0",
        "p0/r1",
        "p0/r1",
        "p1/r0",
        "p1/r0",
        "p1/r1",
        "p1/r1",
    ]


def test_post_init_rejects_dangling_parent_track():
    """parent_track must reference a sibling track key in the same RolloutResp."""
    image = _make_track("image", ["a", "b"], parent_ids=["x", "y"], parent_track="nonexistent")
    with pytest.raises(ValueError, match="parent_track='nonexistent' not in tracks"):
        RolloutResp(tracks={"image": image})


def test_post_init_rejects_mismatched_parent_ids_length():
    """parent_ids length must equal sample_ids length."""
    track = RolloutTrack(
        sample_ids=["a", "b", "c"],
        parent_ids=["p", "p"],  # too short
    )
    with pytest.raises(ValueError, match="parent_ids length 2 != sample_ids length 3"):
        RolloutResp(tracks={"t": track})


def test_post_init_rejects_unknown_parent_id():
    """Foreign key: parent_ids must all be in parent track's sample_ids."""
    refined = _make_track("refined", ["p0/r0", "p0/r1"], parent_ids=["p0", "p0"])
    image = _make_track(
        "image",
        ["p0/r0/i0", "p0/r0/i1"],
        parent_ids=["p0/r0", "phantom_id"],  # phantom not in refined
        parent_track="refined",
    )
    with pytest.raises(ValueError, match="parent_ids: 1 ids not in parent track"):
        RolloutResp(tracks={"refined": refined, "image": image})


def test_two_track_concat_per_track_offset_shift():
    """Each track's segment.sample_indices is offset-shifted by THAT track's
    cumulative sample count, independent of other tracks' sizes."""
    # Shard A: refined has 2 samples, image has 3 samples.
    refined_a = _make_track("refined", ["p0/r0", "p0/r1"], parent_ids=["p0", "p0"])
    image_a = _make_track(
        "image",
        ["p0/r0/i0", "p0/r0/i1", "p0/r1/i0"],
        parent_ids=["p0/r0", "p0/r0", "p0/r1"],
        parent_track="refined",
    )
    a = RolloutResp(tracks={"refined": refined_a, "image": image_a})

    # Shard B: refined has 1 sample, image has 2 samples.
    refined_b = _make_track("refined", ["p1/r0"], parent_ids=["p1"])
    image_b = _make_track(
        "image",
        ["p1/r0/i0", "p1/r0/i1"],
        parent_ids=["p1/r0", "p1/r0"],
        parent_track="refined",
    )
    b = RolloutResp(tracks={"refined": refined_b, "image": image_b})

    merged = RolloutResp.concat([a, b])

    # Refined: 3 samples, sample_indices = [0, 1, 2] (B's [0] shifted by 2).
    assert merged.tracks["refined"].sample_ids == ["p0/r0", "p0/r1", "p1/r0"]
    assert merged.tracks["refined"].segment.sample_indices.tolist() == [0, 1, 2]

    # Image: 5 samples, sample_indices = [0, 1, 2, 3, 4] (B's [0, 1] shifted by 3 — image's own count).
    assert merged.tracks["image"].sample_ids == [
        "p0/r0/i0",
        "p0/r0/i1",
        "p0/r1/i0",
        "p1/r0/i0",
        "p1/r0/i1",
    ]
    assert merged.tracks["image"].segment.sample_indices.tolist() == [0, 1, 2, 3, 4]


def test_two_track_concat_does_not_mutate_input_shards():
    """Per-track concat must clone segments before shifting; inputs untouched."""
    refined_a = _make_track("refined", ["p0/r0"], parent_ids=["p0"])
    refined_b = _make_track("refined", ["p1/r0"], parent_ids=["p1"])
    a = RolloutResp(tracks={"refined": refined_a})
    b = RolloutResp(tracks={"refined": refined_b})
    RolloutResp.concat([a, b])
    # b's segment unchanged.
    assert b.tracks["refined"].segment.sample_indices.tolist() == [0]


def test_track_split_by_group_ids():
    """RolloutTrack.split() partitions samples by parent_ids equivalence class."""
    track = _make_track(
        "image",
        ["p0/r0/i0", "p0/r0/i1", "p0/r1/i0", "p1/r0/i0", "p1/r0/i1"],
        parent_ids=["p0/r0", "p0/r0", "p0/r1", "p1/r0", "p1/r0"],
    )
    splits = track.split()
    assert len(splits) == 3
    assert [t.sample_ids for t in splits] == [
        ["p0/r0/i0", "p0/r0/i1"],
        ["p0/r1/i0"],
        ["p1/r0/i0", "p1/r0/i1"],
    ]


def test_root_track_group_ids_fall_back_to_sample_ids():
    """When parent_ids is None, group_ids = sample_ids (each sample its own group)."""
    track = _make_track("root", ["a", "b", "c"])
    assert track.parent_ids is None
    assert track.group_ids == ["a", "b", "c"]


# ---- structural lookups (root_track, tracks_with_segment_types) ------------


def test_root_track_returns_only_track_for_single_track_resp():
    """Single-track resp: the only track is the root."""
    track = _make_track("image", ["a", "b"], parent_ids=["g", "g"])
    resp = RolloutResp(tracks={"image": track})
    assert resp.root_track() is track


def test_root_track_returns_root_for_multi_track_resp():
    """Multi-track resp: the track with parent_track=None wins."""
    refined = _make_track("refined", ["p0/r0", "p0/r1"], parent_ids=["p0", "p0"])
    image = _make_track(
        "image",
        ["p0/r0/i0", "p0/r0/i1"],
        parent_ids=["p0/r0", "p0/r1"],
        parent_track="refined",
    )
    resp = RolloutResp(tracks={"refined": refined, "image": image})
    assert resp.root_track() is refined


def test_root_track_raises_on_multiple_roots():
    """Two tracks with parent_track=None is ambiguous."""
    a = _make_track("a", ["a0"])
    b = _make_track("b", ["b0"])
    resp = RolloutResp(tracks={"a": a, "b": b})
    with pytest.raises(RuntimeError, match="expected exactly one root track"):
        resp.root_track()


def test_tracks_with_segment_types_filters_by_exact_type():
    """Returns only tracks whose segment type matches one of the requested types."""
    refined = RolloutTrack(
        sample_ids=["p0/r0", "p0/r1"],
        parent_ids=["p0", "p0"],
        segment=None,  # refined has no LatentSegment in this fixture
    )
    image = _make_track(
        "image",
        ["p0/r0/i0"],
        parent_ids=["p0/r0"],
        parent_track="refined",
    )
    resp = RolloutResp(tracks={"refined": refined, "image": image})
    scorable = resp.tracks_with_segment_types([LatentSegment])
    assert [name for name, _ in scorable] == ["image"]


def test_tracks_with_segment_types_skips_none_segments():
    """Tracks with segment=None are skipped regardless of requested types."""
    track = RolloutTrack(sample_ids=["a"], parent_ids=["g"], segment=None)
    resp = RolloutResp(tracks={"only": track})
    assert resp.tracks_with_segment_types([LatentSegment]) == []


# ---- multi-track split (root-group / "first dimension" partitioning) -------


def _refined_image_resp(prompt_ids, y, z):
    """Build a refined+image RolloutResp.

    refined: root track, ``len(prompt_ids) * y`` samples, parent_ids replicated
    from prompt_ids.
    image: child of refined, ``len(prompt_ids) * y * z`` samples, parent_ids
    replicated from refined.sample_ids.
    """
    refined_sids = [f"{p}/r{j}" for p in prompt_ids for j in range(y)]
    refined_pids = [p for p in prompt_ids for _ in range(y)]
    image_sids = [f"{rs}/i{k}" for rs in refined_sids for k in range(z)]
    image_pids = [rs for rs in refined_sids for _ in range(z)]
    refined = _make_track("refined", refined_sids, parent_ids=refined_pids)
    image = _make_track(
        "image",
        image_sids,
        parent_ids=image_pids,
        parent_track="refined",
    )
    return RolloutResp(tracks={"refined": refined, "image": image})


def test_multi_track_split_partitions_by_root_group():
    """Two prompts → two shards; each shard holds one prompt's subtree."""
    resp = _refined_image_resp(prompt_ids=["p0", "p1"], y=2, z=3)
    shards = resp.split()

    assert len(shards) == 2
    assert set(shards[0].tracks.keys()) == {"refined", "image"}
    assert set(shards[1].tracks.keys()) == {"refined", "image"}

    # Shard 0: p0 subtree.
    assert shards[0].tracks["refined"].sample_ids == ["p0/r0", "p0/r1"]
    assert shards[0].tracks["refined"].parent_ids == ["p0", "p0"]
    assert shards[0].tracks["image"].sample_ids == [
        "p0/r0/i0",
        "p0/r0/i1",
        "p0/r0/i2",
        "p0/r1/i0",
        "p0/r1/i1",
        "p0/r1/i2",
    ]
    assert shards[0].tracks["image"].parent_ids == [
        "p0/r0",
        "p0/r0",
        "p0/r0",
        "p0/r1",
        "p0/r1",
        "p0/r1",
    ]
    # Shard 1: p1 subtree.
    assert shards[1].tracks["refined"].sample_ids == ["p1/r0", "p1/r1"]
    assert shards[1].tracks["image"].sample_ids == [
        "p1/r0/i0",
        "p1/r0/i1",
        "p1/r0/i2",
        "p1/r1/i0",
        "p1/r1/i1",
        "p1/r1/i2",
    ]


def test_multi_track_split_preserves_post_init_invariants():
    """Each shard must satisfy parent_ids ⊆ parent track sample_ids."""
    resp = _refined_image_resp(prompt_ids=["p0", "p1", "p2"], y=2, z=2)
    shards = resp.split()
    # Constructing each shard again to re-run __post_init__ — must not raise.
    for shard in shards:
        RolloutResp(tracks=dict(shard.tracks))


def test_multi_track_split_rejects_multiple_roots():
    """Two tracks with parent_track=None is ambiguous; split must raise."""
    a = _make_track("a", ["a0", "a1"])
    b = _make_track("b", ["b0", "b1"])
    resp = RolloutResp(tracks={"a": a, "b": b})
    with pytest.raises(RuntimeError, match="expected exactly one root track"):
        resp.split()


def test_multi_track_split_single_track_matches_legacy_behavior():
    """Single-track resp reduces to splitting that one track by group_ids."""
    track = _make_track(
        "image",
        ["p0/r0/i0", "p0/r0/i1", "p0/r1/i0", "p1/r0/i0"],
        parent_ids=["p0/r0", "p0/r0", "p0/r1", "p1/r0"],
    )
    resp = RolloutResp(tracks={"image": track})
    shards = resp.split()
    assert [list(s.tracks.keys()) for s in shards] == [["image"], ["image"], ["image"]]
    assert [s.tracks["image"].sample_ids for s in shards] == [
        ["p0/r0/i0", "p0/r0/i1"],
        ["p0/r1/i0"],
        ["p1/r0/i0"],
    ]


# ---- multi-track behavior (Step 1 of the tracks refactor) ------------------


def _make_track(name, sample_ids, *, parent_ids=None, parent_track=None, n_segs=None):
    """Helper: minimal RolloutTrack with a LatentSegment carrying sample_indices."""
    n = len(sample_ids) if n_segs is None else n_segs
    return RolloutTrack(
        sample_ids=list(sample_ids),
        parent_ids=list(parent_ids) if parent_ids is not None else None,
        parent_track=parent_track,
        segment=LatentSegment(
            sample_indices=torch.arange(n, dtype=torch.long),
            positions=torch.zeros(n, dtype=torch.long),
            latents=torch.zeros(n, 2, 4, 4, 4),
        ),
    )
