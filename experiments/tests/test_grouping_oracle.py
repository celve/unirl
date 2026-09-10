"""E0 gate: the grouping oracle against UniRL's own reshape path.

Covers the description's "validate unique root IDs, matching part depth, uniform
parent-contiguous rows" checks and the review's structural counterexample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oracle.flat_row_oracle import compare_to_oracle, default_layer, flat_id_join, group_key  # noqa: E402


def _part(sample_ids, rewards):
    from unirl.types.sample import Part

    return Part(sample_ids=list(sample_ids), rewards=torch.tensor(rewards, dtype=torch.float32))


def test_group_key_walks_the_lineage_path():
    assert group_key("a/0/1", 0) == "a"
    assert group_key("a/0/1", 1) == "a/0"
    assert group_key("a/0/1", 2) == "a/0/1"
    with pytest.raises(ValueError):
        group_key("a/0", 5)


def test_default_layer_is_the_parent_of_the_first_id():
    assert default_layer(["a/0", "a/1"]) == 0
    assert default_layer(["a/0/0", "a/0/1"]) == 1
    assert default_layer(["a"]) == 0


def test_oracle_matches_unirl_on_parent_contiguous_rows():
    sample_ids = ["a/0", "a/1", "b/0", "b/1"]
    rewards = [1.0, 3.0, 10.0, 30.0]
    part = _part(sample_ids, rewards).compute_advantages(normalize=False)
    verdict = compare_to_oracle(sample_ids, rewards, part.advantages.tolist(), normalize=False)
    assert verdict.passed, verdict.findings
    assert verdict.max_abs_advantage_diff < 1e-6


def test_interleaved_rows_are_rejected_not_miscomputed():
    """The review's counterexample: at this commit UniRL raises rather than mixing groups."""
    sample_ids = ["a/0", "b/0", "a/1", "b/1"]
    rewards = [1.0, 10.0, 3.0, 30.0]

    with pytest.raises(ValueError, match="not in group-by-parent contiguous order"):
        _part(sample_ids, rewards).compute_advantages(normalize=False)

    oracle = flat_id_join(sample_ids, rewards, normalize=False)
    assert not oracle.parent_contiguous
    assert oracle.group_means == {"a": 2.0, "b": 20.0}


def test_oracle_flags_a_reshape_that_crossed_groups():
    """A parent-major reshape on interleaved rows yields 5.5/16.5; the join must reject it."""
    sample_ids = ["a/0", "b/0", "a/1", "b/1"]
    rewards = [1.0, 10.0, 3.0, 30.0]
    reshape_advantages = [1.0 - 5.5, 10.0 - 5.5, 3.0 - 16.5, 30.0 - 16.5]
    verdict = compare_to_oracle(sample_ids, rewards, reshape_advantages, normalize=False)
    assert not verdict.passed
    assert len(verdict.mismatched_ids) == 4


def test_non_uniform_groups_are_rejected_by_both_paths():
    sample_ids = ["a/0", "a/1", "a/2", "b/0"]
    rewards = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="non-uniform group sizes"):
        _part(sample_ids, rewards).compute_advantages(normalize=False)
    oracle = flat_id_join(sample_ids, rewards, normalize=False)
    assert not oracle.uniform_branch


def test_duplicate_root_ids_are_reported():
    sample_ids = ["a/0", "a/0", "b/0", "b/1"]
    oracle = flat_id_join(sample_ids, [1.0, 2.0, 3.0, 4.0], normalize=False)
    assert oracle.duplicate_ids == ["a/0"]


def test_normalized_advantages_match_including_std():
    sample_ids = ["a/0", "a/1", "a/2", "a/3", "b/0", "b/1", "b/2", "b/3"]
    rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]
    part = _part(sample_ids, rewards).compute_advantages(normalize=True)
    verdict = compare_to_oracle(sample_ids, rewards, part.advantages.tolist(), normalize=True)
    assert verdict.passed, verdict.findings


def test_non_finite_rewards_are_excluded_from_the_group_mean():
    sample_ids = ["a/0", "a/1", "b/0", "b/1"]
    rewards = [1.0, float("nan"), 10.0, 30.0]
    part = _part(sample_ids, rewards).compute_advantages(normalize=False)
    verdict = compare_to_oracle(sample_ids, rewards, part.advantages.tolist(), normalize=False)
    assert verdict.passed, verdict.findings
