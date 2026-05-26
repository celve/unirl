"""Tests for ``compute_rollout_resp_metrics`` — track-aware metric builder.

Replaces the legacy ``compute_rollout_batch_metrics`` tests that
consumed ``TrainingBatch`` payloads. The new entrypoint reads
per-track rewards / advantages / component_rewards / group_ids
directly off a :class:`RolloutResp`.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.utils.wandb_metrics import compute_rollout_resp_metrics


def _make_track(
    *,
    n: int,
    parent_ids: Optional[List[str]],
    rewards: Optional[List[float]] = None,
    advantages: Optional[List[float]] = None,
    component_rewards: Optional[dict] = None,
) -> RolloutTrack:
    return RolloutTrack(
        sample_ids=[f"s{i}" for i in range(n)],
        parent_ids=list(parent_ids) if parent_ids is not None else None,
        rewards=torch.tensor(rewards, dtype=torch.float32) if rewards is not None else None,
        advantages=torch.tensor(advantages, dtype=torch.float32) if advantages is not None else None,
        component_rewards=(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in component_rewards.items()}
            if component_rewards
            else None
        ),
    )


def test_single_track_keys_unprefixed():
    track = _make_track(
        n=4,
        parent_ids=["g0", "g0", "g1", "g1"],
        rewards=[0.0, 2.0, 4.0, 6.0],
        advantages=[-1.0, 1.0, -1.0, 1.0],
    )
    resp = RolloutResp(tracks={"image": track})

    metrics = compute_rollout_resp_metrics(resp=resp)
    assert metrics["num_samples"] == 4.0
    assert metrics["reward_mean"] == 3.0
    assert metrics["reward_min"] == 0.0
    assert metrics["reward_max"] == 6.0
    assert metrics["advantage_mean"] == 0.0
    # No keys are namespaced under the track name in single-track mode.
    assert not any(k.startswith("image_") for k in metrics)


def test_multi_track_keys_namespaced():
    image = _make_track(
        n=2,
        parent_ids=["g", "g"],
        rewards=[1.0, 3.0],
    )
    refined = _make_track(
        n=2,
        parent_ids=["p", "p"],
        rewards=[5.0, 7.0],
    )
    resp = RolloutResp(tracks={"refined": refined, "image": image})

    metrics = compute_rollout_resp_metrics(resp=resp)
    # Per-track prefixes are used when more than one track is present.
    assert metrics["image_reward_mean"] == 2.0
    assert metrics["refined_reward_mean"] == 6.0
    # No unprefixed reward_mean — every track is namespaced.
    assert "reward_mean" not in metrics


def test_zero_std_groups_counted_when_group_ids_align():
    # Two groups of two; first group has std 0 (both rewards equal).
    track = _make_track(
        n=4,
        parent_ids=["g0", "g0", "g1", "g1"],
        rewards=[1.0, 1.0, 1.0, 3.0],
    )
    resp = RolloutResp(tracks={"image": track})
    metrics = compute_rollout_resp_metrics(resp=resp)
    assert metrics["group_count"] == 2.0
    assert metrics["zero_std_group_count"] == 1.0
    assert metrics["zero_std_group_ratio"] == 0.5


def test_component_rewards_emit_per_metric_stats():
    track = _make_track(
        n=3,
        parent_ids=["g", "g", "g"],
        rewards=[0.0, 2.0, 4.0],
        component_rewards={"clip/score": [0.5, 1.0, 1.5]},
    )
    resp = RolloutResp(tracks={"image": track})
    metrics = compute_rollout_resp_metrics(resp=resp)
    # '/' in component name flattens to '_' under the rollout/ prefix.
    assert "reward_clip_score_mean" in metrics
    assert metrics["reward_clip_score_mean"] == 1.0


def test_no_rewards_yields_only_num_samples():
    track = _make_track(n=2, parent_ids=["g", "g"])
    resp = RolloutResp(tracks={"image": track})
    metrics = compute_rollout_resp_metrics(resp=resp)
    assert metrics == {"num_samples": 2.0}
