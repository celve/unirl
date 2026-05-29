"""Tests for the tracks fan-out helpers — RolloutReq.make_root_track and
RolloutTrack.fork_track — plus the full x*y*z prompt-enhancement tree shape.

Covers:

- Hierarchical sample_ids and parent_ids in group-by-parent order.
- Conditions replicated via Batch.repeat_interleave(branch).
- Lineage invariants enforced by RolloutResp.__post_init__ on tree assembly.
- Two-level fan-out (req → refined → image) with the full x*y*z shape.
- DP-shard concat round-trip with per-track sample_indices offset shift.
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.types.conditions import TextEmbedCondition
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments import LatentSegment

# ---- helpers ---------------------------------------------------------------


def _make_req(sample_ids, *, embeds=None):
    """Minimal RolloutReq with optional request_conditions text embeds."""
    request_conditions = {}
    if embeds is not None:
        request_conditions["text"] = TextEmbedCondition(embeds=embeds)
    return RolloutReq(
        sample_ids=list(sample_ids),
        group_ids=["g"] * len(sample_ids),
        request_conditions=request_conditions,
    )


def _no_conditions(_):
    """decode_to_condition stub for tests that don't care about conditions."""
    return {}


# ---- make_root_track -------------------------------------------------------


def test_make_root_track_basic_tree_shape():
    """req.make_root_track creates a track with hierarchical IDs in group-by-parent order."""
    req = _make_req(["p0", "p1"])
    track = req.make_root_track("refined", branch=3, decode_to_condition=_no_conditions)
    assert track.sample_ids == ["p0/r0", "p0/r1", "p0/r2", "p1/r0", "p1/r1", "p1/r2"]
    assert track.parent_ids == ["p0", "p0", "p0", "p1", "p1", "p1"]
    assert track.parent_track is None
    assert track.conditions == {}
    assert track.segment is None
    assert track.decoded is None
    assert track.group_ids == track.parent_ids  # property derives from parent_ids


def test_make_root_track_replicates_conditions_group_by_parent():
    """Conditions returned by decode_to_condition are repeat_interleaved by branch."""
    req = _make_req(["p0", "p1"])

    def fake_encode(r):
        # 2 prompts, embeds shape (2, 4): row 0 is p0's, row 1 is p1's.
        return {"text": TextEmbedCondition(embeds=torch.tensor([[1.0] * 4, [2.0] * 4]))}

    track = req.make_root_track("refined", branch=3, decode_to_condition=fake_encode)
    # After branch=3 in group-by-parent order: rows 0-2 = p0's embed, rows 3-5 = p1's embed.
    assert track.conditions["text"].embeds.shape == (6, 4)
    assert torch.allclose(track.conditions["text"].embeds[:3], torch.full((3, 4), 1.0))
    assert torch.allclose(track.conditions["text"].embeds[3:], torch.full((3, 4), 2.0))


def test_make_root_track_default_inherits_request_conditions():
    """When decode_to_condition is None, default = use req.request_conditions."""
    embeds = torch.tensor([[1.0] * 4, [2.0] * 4])
    req = _make_req(["p0", "p1"], embeds=embeds)
    track = req.make_root_track("refined", branch=2)  # no decode_to_condition
    # Default path: request_conditions["text"] is replicated.
    assert track.conditions["text"].embeds.shape == (4, 4)
    assert torch.allclose(track.conditions["text"].embeds[:2], torch.full((2, 4), 1.0))
    assert torch.allclose(track.conditions["text"].embeds[2:], torch.full((2, 4), 2.0))


def test_make_root_track_branch_one_is_identity_in_count():
    """branch=1 returns one child per parent (no replication)."""
    req = _make_req(["p0", "p1", "p2"])
    track = req.make_root_track("refined", branch=1, decode_to_condition=_no_conditions)
    assert track.sample_ids == ["p0/r0", "p1/r0", "p2/r0"]
    assert track.parent_ids == ["p0", "p1", "p2"]


def test_make_root_track_rejects_empty_req_or_zero_branch():
    """Empty sample_ids or non-positive branch should fail loudly."""
    empty_req = RolloutReq(sample_ids=[], group_ids=[])
    with pytest.raises(ValueError, match="no sample_ids"):
        empty_req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)

    req = _make_req(["p0"])
    with pytest.raises(ValueError, match="branch must be >= 1"):
        req.make_root_track("refined", branch=0, decode_to_condition=_no_conditions)


# ---- fork_track ------------------------------------------------------------


def test_fork_track_basic_tree_shape():
    """track.fork_track creates a child with hierarchical IDs from parent's sample_ids."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)
    image = refined.fork_track("refined", "image", branch=2, decode_to_condition=_no_conditions)
    # x=2, y=2 → refined = 4; z=2 → image = 8.
    assert image.sample_ids == [
        "p0/r0/i0",
        "p0/r0/i1",
        "p0/r1/i0",
        "p0/r1/i1",
        "p1/r0/i0",
        "p1/r0/i1",
        "p1/r1/i0",
        "p1/r1/i1",
    ]
    assert image.parent_ids == [
        "p0/r0",
        "p0/r0",
        "p0/r1",
        "p0/r1",
        "p1/r0",
        "p1/r0",
        "p1/r1",
        "p1/r1",
    ]
    assert image.parent_track == "refined"


def test_fork_track_replicates_conditions_group_by_parent():
    """fork_track's decode_to_condition output is repeat_interleaved by branch."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=1, decode_to_condition=_no_conditions)
    # refined now has 2 samples: p0/r0, p1/r0.

    def fake_encode(t):
        # 2 parent samples, embeds shape (2, 4).
        return {"text": TextEmbedCondition(embeds=torch.tensor([[1.0] * 4, [2.0] * 4]))}

    image = refined.fork_track("refined", "image", branch=3, decode_to_condition=fake_encode)
    # Group-by-parent: rows 0-2 = parent 0 (p0/r0), rows 3-5 = parent 1 (p1/r0).
    assert image.conditions["text"].embeds.shape == (6, 4)
    assert torch.allclose(image.conditions["text"].embeds[:3], torch.full((3, 4), 1.0))
    assert torch.allclose(image.conditions["text"].embeds[3:], torch.full((3, 4), 2.0))


def test_fork_track_default_no_conditions():
    """fork_track with decode_to_condition=None gives an empty conditions dict."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=1, decode_to_condition=_no_conditions)
    image = refined.fork_track("refined", "image", branch=2)
    assert image.conditions == {}


def test_fork_track_rejects_empty_track_or_zero_branch():
    """Empty sample_ids or non-positive branch should fail loudly."""
    empty_track = RolloutTrack(sample_ids=[])
    with pytest.raises(ValueError, match="no sample_ids"):
        empty_track.fork_track("p", "c", branch=2, decode_to_condition=_no_conditions)

    req = _make_req(["p0"])
    refined = req.make_root_track("refined", branch=1, decode_to_condition=_no_conditions)
    with pytest.raises(ValueError, match="branch must be >= 1"):
        refined.fork_track("refined", "image", branch=0, decode_to_condition=_no_conditions)


# ---- full x*y*z tree -------------------------------------------------------


def test_full_x_y_z_tree_construction():
    """Build the full prompt-enhancement tree: x=2 prompts, y=3 refined, z=2 images."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=3, decode_to_condition=_no_conditions)
    image = refined.fork_track("refined", "image", branch=2, decode_to_condition=_no_conditions)

    assert len(refined.sample_ids) == 2 * 3  # x*y
    assert len(image.sample_ids) == 2 * 3 * 2  # x*y*z

    # The tree assembles in a RolloutResp without lineage violations
    # (parent_track foreign-key + parent_ids subset-of-parent both pass).
    resp = RolloutResp(tracks={"refined": refined, "image": image})
    assert set(resp.tracks.keys()) == {"refined", "image"}

    # Group sizes: refined groups by p0 / p1 (3 samples each); image groups
    # by p0/r0, p0/r1, p0/r2, p1/r0, p1/r1, p1/r2 (2 samples each).
    refined_groups = {gid: 0 for gid in dict.fromkeys(refined.group_ids)}
    for gid in refined.group_ids:
        refined_groups[gid] += 1
    assert all(count == 3 for count in refined_groups.values())
    assert len(refined_groups) == 2

    image_groups = {gid: 0 for gid in dict.fromkeys(image.group_ids)}
    for gid in image.group_ids:
        image_groups[gid] += 1
    assert all(count == 2 for count in image_groups.values())
    assert len(image_groups) == 6


def test_full_x_y_z_tree_concat_round_trip():
    """Concat two DP shards of the x*y*z tree; verify per-track sample_indices remap."""

    def _build_resp(prompt_ids):
        req = _make_req(prompt_ids)
        refined = req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)
        # Attach a LatentSegment to refined so the per-track concat remap fires.
        refined.segment = LatentSegment(
            sample_indices=torch.arange(len(refined.sample_ids), dtype=torch.long),
            positions=torch.zeros(len(refined.sample_ids), dtype=torch.long),
            latents=torch.zeros(len(refined.sample_ids), 2, 4, 4, 4),
        )
        image = refined.fork_track("refined", "image", branch=2, decode_to_condition=_no_conditions)
        image.segment = LatentSegment(
            sample_indices=torch.arange(len(image.sample_ids), dtype=torch.long),
            positions=torch.zeros(len(image.sample_ids), dtype=torch.long),
            latents=torch.zeros(len(image.sample_ids), 2, 4, 4, 4),
        )
        return RolloutResp(tracks={"refined": refined, "image": image})

    # Two DP shards with disjoint prompt IDs (avoids parent_ids foreign-key
    # collisions on merge).
    a = _build_resp(["p0", "p1"])
    b = _build_resp(["p2", "p3"])
    merged = RolloutResp.concat([a, b])

    # Refined: 2 prompts × 2 refined per shard = 4 per shard, 8 merged.
    # sample_indices = [0, 1, 2, 3, 4, 5, 6, 7] (B's [0..3] shifted by 4).
    assert len(merged.tracks["refined"].sample_ids) == 8
    assert merged.tracks["refined"].segment.sample_indices.tolist() == list(range(8))

    # Image: 4 refined × 2 image per shard = 8 per shard, 16 merged.
    # sample_indices = [0..15] (B's [0..7] shifted by 8 — image's own count).
    assert len(merged.tracks["image"].sample_ids) == 16
    assert merged.tracks["image"].segment.sample_indices.tolist() == list(range(16))

    # Lineage survives the concat: every image's parent_id is in refined's sample_ids.
    refined_ids = set(merged.tracks["refined"].sample_ids)
    assert all(p in refined_ids for p in merged.tracks["image"].parent_ids)


# ---- propagate_rewards (Step 3) --------------------------------------------


def _full_tree_with_image_rewards(prompt_ids, branch_y, branch_z, image_rewards):
    """Build a refined+image RolloutResp with image_rewards set on the leaf."""
    req = _make_req(prompt_ids)
    refined = req.make_root_track("refined", branch=branch_y, decode_to_condition=_no_conditions)
    image = refined.fork_track("refined", "image", branch=branch_z, decode_to_condition=_no_conditions)
    image.rewards = torch.as_tensor(image_rewards, dtype=torch.float32)
    return RolloutResp(tracks={"refined": refined, "image": image})


def test_propagate_rewards_mean_aggregates_image_to_refined():
    """image rewards aggregate to refined via mean over z-sized groups."""
    # x=2, y=2, z=3 → 4 refined, 12 image.
    resp = _full_tree_with_image_rewards(
        prompt_ids=["p0", "p1"],
        branch_y=2,
        branch_z=3,
        image_rewards=list(range(12)),
    )
    propagated = resp.propagate_rewards(op="mean")
    # mean of [0,1,2], [3,4,5], [6,7,8], [9,10,11] = [1, 4, 7, 10]
    expected = torch.tensor([1.0, 4.0, 7.0, 10.0])
    assert torch.allclose(propagated.tracks["refined"].rewards, expected)
    # image rewards unchanged.
    assert torch.allclose(
        propagated.tracks["image"].rewards,
        torch.arange(12, dtype=torch.float32),
    )


def test_propagate_rewards_max_aggregation():
    """op='max' picks the best child per group."""
    resp = _full_tree_with_image_rewards(
        prompt_ids=["p0", "p1"],
        branch_y=2,
        branch_z=3,
        image_rewards=list(range(12)),
    )
    propagated = resp.propagate_rewards(op="max")
    # max of [0,1,2], [3,4,5], [6,7,8], [9,10,11] = [2, 5, 8, 11]
    expected = torch.tensor([2.0, 5.0, 8.0, 11.0])
    assert torch.allclose(propagated.tracks["refined"].rewards, expected)


def test_propagate_rewards_sum_aggregation():
    """op='sum' totals child rewards per group."""
    resp = _full_tree_with_image_rewards(
        prompt_ids=["p0", "p1"],
        branch_y=2,
        branch_z=3,
        image_rewards=list(range(12)),
    )
    propagated = resp.propagate_rewards(op="sum")
    # sum of [0,1,2], [3,4,5], [6,7,8], [9,10,11] = [3, 12, 21, 30]
    expected = torch.tensor([3.0, 12.0, 21.0, 30.0])
    assert torch.allclose(propagated.tracks["refined"].rewards, expected)


def test_propagate_rewards_direct_rewards_win():
    """Pre-set parent rewards are NOT overwritten by aggregation."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)
    refined.rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])  # direct (e.g. from LLM judge)
    image = refined.fork_track("refined", "image", branch=3, decode_to_condition=_no_conditions)
    image.rewards = torch.arange(12, dtype=torch.float32)
    resp = RolloutResp(tracks={"refined": refined, "image": image})

    propagated = resp.propagate_rewards(op="mean")
    # Refined unchanged — direct wins over inherited.
    assert torch.allclose(propagated.tracks["refined"].rewards, torch.tensor([10.0, 20.0, 30.0, 40.0]))


def test_propagate_rewards_raises_on_none_child_rewards():
    """If a child has rewards=None, propagation cannot proceed and must raise."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)
    image = refined.fork_track("refined", "image", branch=3, decode_to_condition=_no_conditions)
    # image not scored yet: rewards=None.
    resp = RolloutResp(tracks={"refined": refined, "image": image})
    with pytest.raises(ValueError, match="rewards is None"):
        resp.propagate_rewards(op="mean")


def test_propagate_rewards_unknown_op_raises():
    """Typo'd op should fail loudly."""
    resp = _full_tree_with_image_rewards(["p0"], 2, 2, list(range(4)))
    with pytest.raises(ValueError, match="unknown op 'median'"):
        resp.propagate_rewards(op="median")  # type: ignore[arg-type]


def test_propagate_rewards_does_not_mutate_input():
    """propagate_rewards returns a new RolloutResp; input is untouched."""
    resp = _full_tree_with_image_rewards(["p0"], 2, 2, list(range(4)))
    assert resp.tracks["refined"].rewards is None
    _ = resp.propagate_rewards(op="mean")
    # Original input still has refined.rewards=None.
    assert resp.tracks["refined"].rewards is None


# ---- compute_advantages (Step 3) -------------------------------------------


def test_compute_advantages_per_group_zero_mean():
    """Each group's advantages have mean 0 after normalization."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=3, decode_to_condition=_no_conditions)
    refined.rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2 groups of 3
    out = refined.compute_advantages()
    # Group 0: rewards [1,2,3] — mean is 2, advantages mean ≈ 0
    # Group 1: rewards [4,5,6] — mean is 5, advantages mean ≈ 0
    assert torch.allclose(out.advantages[:3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out.advantages[3:].mean(), torch.tensor(0.0), atol=1e-6)


def test_compute_advantages_normalize_unit_std_per_group():
    """With normalize=True, each group's std (population) is ~1."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=3, decode_to_condition=_no_conditions)
    refined.rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = refined.compute_advantages(normalize=True)
    # population std of [1,2,3] is sqrt(2/3) ≈ 0.816, normalized to 1.
    g0_std = out.advantages[:3].std(unbiased=False)
    g1_std = out.advantages[3:].std(unbiased=False)
    assert torch.allclose(g0_std, torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(g1_std, torch.tensor(1.0), atol=1e-3)


def test_compute_advantages_no_normalize_is_centering():
    """normalize=False returns reward - group_mean (no division by std)."""
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=3, decode_to_condition=_no_conditions)
    refined.rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = refined.compute_advantages(normalize=False)
    expected = torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])  # centered per group
    assert torch.allclose(out.advantages, expected)


def test_compute_advantages_uniform_rewards_branch_one_handled_safely():
    """Single-sample groups (branch=1) and uniform-reward groups don't NaN."""
    # Branch=1: every sample is its own group.
    req = _make_req(["p0", "p1"])
    refined = req.make_root_track("refined", branch=1, decode_to_condition=_no_conditions)
    refined.rewards = torch.tensor([3.0, 7.0])
    out = refined.compute_advantages(normalize=True)
    # Each group has one sample → mean = sample, advantage = 0.
    assert torch.allclose(out.advantages, torch.zeros_like(out.advantages))


def test_compute_advantages_root_track_zero():
    """Root track with parent_ids=None: each sample its own group → advantages all 0."""
    track = RolloutTrack(
        sample_ids=["a", "b", "c"],
        rewards=torch.tensor([1.0, 5.0, 9.0]),
    )
    out = track.compute_advantages()
    assert torch.allclose(out.advantages, torch.zeros_like(out.advantages))


def test_compute_advantages_raises_when_no_rewards():
    """Calling compute_advantages on a track with rewards=None must raise."""
    req = _make_req(["p0"])
    refined = req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)
    with pytest.raises(ValueError, match="no rewards"):
        refined.compute_advantages()


def test_compute_advantages_does_not_mutate_input():
    """compute_advantages returns a new track; input is untouched."""
    req = _make_req(["p0"])
    refined = req.make_root_track("refined", branch=2, decode_to_condition=_no_conditions)
    refined.rewards = torch.tensor([1.0, 3.0])
    assert refined.advantages is None
    _ = refined.compute_advantages()
    assert refined.advantages is None  # still None


# ---- end-to-end: full pipeline (propagate then advantages) -----------------


def test_full_tree_propagate_then_compute_advantages_per_track():
    """End-to-end: image scored, propagate up to refined, advantages per track."""
    resp = _full_tree_with_image_rewards(
        prompt_ids=["p0", "p1"],
        branch_y=2,
        branch_z=3,
        image_rewards=list(range(12)),
    )
    propagated = resp.propagate_rewards(op="mean")

    image_track = propagated.tracks["image"]
    refined_track = propagated.tracks["refined"]

    # Image track: 4 groups of 3 (one per refined parent).
    image_with_adv = image_track.compute_advantages()
    # Each group of 3 has mean 0 advantages.
    for g in range(4):
        assert torch.allclose(
            image_with_adv.advantages[g * 3 : (g + 1) * 3].mean(),
            torch.tensor(0.0),
            atol=1e-6,
        )

    # Refined track: 2 groups of 2 (one per original prompt).
    refined_with_adv = refined_track.compute_advantages()
    for g in range(2):
        assert torch.allclose(
            refined_with_adv.advantages[g * 2 : (g + 1) * 2].mean(),
            torch.tensor(0.0),
            atol=1e-6,
        )


# ---- propagate_rewards (Step 3) --------------------------------------------


def _full_tree_with_image_rewards(prompt_ids, branch_y, branch_z, image_rewards):
    """Build a refined+image RolloutResp with image_rewards set on the leaf."""
    req = _make_req(prompt_ids)
    refined = req.make_root_track("refined", branch=branch_y, decode_to_condition=_no_conditions)
    image = refined.fork_track("refined", "image", branch=branch_z, decode_to_condition=_no_conditions)
    image.rewards = torch.as_tensor(image_rewards, dtype=torch.float32)
    return RolloutResp(tracks={"refined": refined, "image": image})
