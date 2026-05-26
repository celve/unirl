"""Tests for ``diffusionrl.training.sharding.shard_resp_per_actor``.

Synthetic multi-track ``RolloutResp`` (no real model bundle, no Ray) —
verifies per-actor lineage coherence, the balanced split invariant, the
multi-root rejection, and the root-batch < num-actors guard.
"""

from __future__ import annotations

from typing import List, Optional

import pytest
import torch

from diffusionrl.training.sharding import shard_resp_per_actor
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments import LatentSegment


def _latent_segment(*, batch_size: int, num_steps: int = 2) -> LatentSegment:
    return LatentSegment(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        latents=torch.zeros(batch_size, num_steps + 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
        sde_logp=torch.full((batch_size, num_steps), -1.0, dtype=torch.float32),
        sde_indices=torch.arange(num_steps, dtype=torch.long),
    )


def _make_track(
    *,
    sample_ids: List[str],
    parent_ids: Optional[List[str]] = None,
    parent_track: Optional[str] = None,
) -> RolloutTrack:
    bs = len(sample_ids)
    return RolloutTrack(
        sample_ids=list(sample_ids),
        parent_ids=parent_ids,
        parent_track=parent_track,
        conditions={},
        segment=_latent_segment(batch_size=bs),
        advantages=torch.zeros(bs, dtype=torch.float32),
    )


def _make_pe_resp(*, num_prompts: int, branch_refined: int, branch_image: int) -> RolloutResp:
    """Build a 3-level lineage resp: prompts → refined → image (uniform branching)."""
    prompt_sids = [f"p{i}" for i in range(num_prompts)]
    refined_sids = [f"{pid}/r{j}" for pid in prompt_sids for j in range(branch_refined)]
    refined_parents = [pid for pid in prompt_sids for _ in range(branch_refined)]
    image_sids = [f"{rsid}/i{k}" for rsid in refined_sids for k in range(branch_image)]
    image_parents = [rsid for rsid in refined_sids for _ in range(branch_image)]

    prompt = _make_track(sample_ids=prompt_sids)  # root: parent_track=None
    refined = _make_track(sample_ids=refined_sids, parent_ids=refined_parents, parent_track="prompt")
    image = _make_track(sample_ids=image_sids, parent_ids=image_parents, parent_track="refined")
    return RolloutResp(tracks={"prompt": prompt, "refined": refined, "image": image})


# ---------------------------------------------------------------------------
# Single-track resps degenerate cleanly
# ---------------------------------------------------------------------------


def test_shard_single_track_resp_one_actor_returns_input_unchanged():
    track = _make_track(sample_ids=["s0", "s1", "s2", "s3"])
    resp = RolloutResp(tracks={"image": track})
    out = shard_resp_per_actor(resp, num_actors=1)
    assert out == [resp]


def test_shard_single_track_resp_balanced_split():
    """4 samples / 2 actors → 2 + 2."""
    track = _make_track(sample_ids=["s0", "s1", "s2", "s3"])
    resp = RolloutResp(tracks={"image": track})
    out = shard_resp_per_actor(resp, num_actors=2)
    assert len(out) == 2
    assert out[0].tracks["image"].sample_ids == ["s0", "s1"]
    assert out[1].tracks["image"].sample_ids == ["s2", "s3"]


def test_shard_single_track_resp_uneven_split_first_gets_remainder():
    """5 samples / 2 actors → 3 + 2 (first gets +1)."""
    track = _make_track(sample_ids=[f"s{i}" for i in range(5)])
    resp = RolloutResp(tracks={"image": track})
    out = shard_resp_per_actor(resp, num_actors=2)
    assert out[0].tracks["image"].sample_ids == ["s0", "s1", "s2"]
    assert out[1].tracks["image"].sample_ids == ["s3", "s4"]


# ---------------------------------------------------------------------------
# Multi-track lineage coherence — the load-bearing invariant
# ---------------------------------------------------------------------------


def test_shard_three_level_lineage_preserves_coherence():
    """4 prompts × 2 refined × 2 image, 2 actors → each actor owns 2 prompts → 4 refined → 8 image."""
    resp = _make_pe_resp(num_prompts=4, branch_refined=2, branch_image=2)
    out = shard_resp_per_actor(resp, num_actors=2)
    assert len(out) == 2

    # Actor 0 owns prompts p0, p1 → refined p0/r0..p1/r1 → image p0/r0/i0..p1/r1/i1.
    actor0 = out[0]
    assert actor0.tracks["prompt"].sample_ids == ["p0", "p1"]
    assert actor0.tracks["refined"].sample_ids == ["p0/r0", "p0/r1", "p1/r0", "p1/r1"]
    assert actor0.tracks["image"].sample_ids == [
        "p0/r0/i0",
        "p0/r0/i1",
        "p0/r1/i0",
        "p0/r1/i1",
        "p1/r0/i0",
        "p1/r0/i1",
        "p1/r1/i0",
        "p1/r1/i1",
    ]

    # Actor 1 owns prompts p2, p3 and the rest of the tree.
    actor1 = out[1]
    assert actor1.tracks["prompt"].sample_ids == ["p2", "p3"]
    assert actor1.tracks["refined"].sample_ids == ["p2/r0", "p2/r1", "p3/r0", "p3/r1"]
    assert actor1.tracks["image"].sample_ids == [
        "p2/r0/i0",
        "p2/r0/i1",
        "p2/r1/i0",
        "p2/r1/i1",
        "p3/r0/i0",
        "p3/r0/i1",
        "p3/r1/i0",
        "p3/r1/i1",
    ]


def test_shard_three_level_lineage_post_shard_parent_ids_still_consistent():
    """After sharding, each shard's tracks must still pass RolloutResp.__post_init__.

    The post-init validator (in types/rollout_resp.py) checks foreign-key
    correctness: every child track's parent_ids must be a subset of its
    parent track's sample_ids. The sharding must preserve this.
    """
    resp = _make_pe_resp(num_prompts=4, branch_refined=2, branch_image=2)
    out = shard_resp_per_actor(resp, num_actors=2)

    for shard in out:
        refined = shard.tracks["refined"]
        prompt = shard.tracks["prompt"]
        # Each refined sample's parent_id must be in prompt.sample_ids.
        assert set(refined.parent_ids) <= set(prompt.sample_ids)
        image = shard.tracks["image"]
        assert set(image.parent_ids) <= set(refined.sample_ids)


def test_shard_three_level_lineage_compute_advantages_post_shard():
    """After sharding, per-shard compute_advantages must yield non-zero advantages.

    This validates the reshape invariance: each actor sees complete
    (n_groups_per_actor, branch) blocks on its child tracks. If sharding
    violated this (e.g. fragmented a group across actors), then
    compute_advantages on the actor's refined track would be single-sample
    groups and produce all-zero advantages (the safe degenerate case).
    """
    resp = _make_pe_resp(num_prompts=4, branch_refined=2, branch_image=2)
    out = shard_resp_per_actor(resp, num_actors=2)

    for shard in out:
        refined = shard.tracks["refined"]
        # Inject distinct per-sample rewards so std > 0 and advantages are non-zero.
        refined_with_rewards = type(refined)(
            sample_ids=refined.sample_ids,
            parent_ids=refined.parent_ids,
            parent_track=refined.parent_track,
            conditions=refined.conditions,
            segment=refined.segment,
            decoded=refined.decoded,
            media_preview=refined.media_preview,
            rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),  # 4 refined per shard (2 per parent)
            component_rewards=refined.component_rewards,
            advantages=refined.advantages,
            status=refined.status,
        )
        adv_track = refined_with_rewards.compute_advantages(normalize=True)
        assert adv_track.advantages is not None
        # Two groups of two siblings each → group mean = 1.5 and 3.5, std > 0.
        # Advantages are non-zero except potentially exactly at the means.
        assert adv_track.advantages.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Fail-fast guards
# ---------------------------------------------------------------------------


def test_shard_rejects_root_batch_less_than_num_actors():
    track = _make_track(sample_ids=["s0", "s1"])
    resp = RolloutResp(tracks={"image": track})
    with pytest.raises(ValueError, match="smaller than num_actors"):
        shard_resp_per_actor(resp, num_actors=4)


def test_shard_rejects_multi_root_resp():
    """Two tracks each with parent_track=None → ambiguous root."""
    track_a = _make_track(sample_ids=["a0", "a1", "a2", "a3"])
    track_b = _make_track(sample_ids=["b0", "b1", "b2", "b3"])
    resp = RolloutResp(tracks={"a": track_a, "b": track_b})
    with pytest.raises(ValueError, match="multiple root tracks"):
        shard_resp_per_actor(resp, num_actors=2)


def test_shard_rejects_zero_actors():
    track = _make_track(sample_ids=["s0", "s1"])
    resp = RolloutResp(tracks={"image": track})
    with pytest.raises(ValueError, match="num_actors must be >= 1"):
        shard_resp_per_actor(resp, num_actors=0)


def test_shard_num_actors_one_short_circuits():
    """num_actors=1 returns the input unchanged regardless of multi-root."""
    track_a = _make_track(sample_ids=["a0", "a1"])
    track_b = _make_track(sample_ids=["b0", "b1"])
    resp = RolloutResp(tracks={"a": track_a, "b": track_b})
    out = shard_resp_per_actor(resp, num_actors=1)
    assert out == [resp]
