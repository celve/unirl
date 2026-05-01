"""Tests for compute_rollout_batch_metrics per-component reward stats."""

from __future__ import annotations

import pytest
import torch

from diffusionrl.utils.wandb_metrics import compute_rollout_batch_metrics


class _BatchStub:
    """Duck-typed stand-in for `TrainingBatch` -- fed via monkeypatched `_iter_batches`."""

    def __init__(self, *, rewards=None, advantages=None, component_rewards=None, batch_size=0):
        self.rewards = rewards
        self.advantages = advantages
        self.component_rewards = component_rewards
        self.batch_size = batch_size
        self.group_ids = None
        self.has_trajectory_rl_data = False


def _patch_iter(monkeypatch, batches):
    from diffusionrl.utils import wandb_metrics

    monkeypatch.setattr(wandb_metrics, "_iter_batches", lambda training_data: batches)


class TestComputeRolloutBatchMetrics:
    def test_no_components_emits_only_aggregate(self, monkeypatch):
        b = _BatchStub(
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            component_rewards=None,
            batch_size=3,
        )
        _patch_iter(monkeypatch, [b])
        m = compute_rollout_batch_metrics(training_data=None)
        assert m["reward_mean"] == pytest.approx(2.0)
        per_comp = [k for k in m if k.startswith("reward_") and k.endswith("_mean") and k != "reward_mean"]
        assert per_comp == []

    def test_with_components_emits_both(self, monkeypatch):
        b = _BatchStub(
            rewards=torch.tensor([0.7, 0.5]),
            component_rewards={
                "pickscore": torch.tensor([0.8, 0.6]),
                "hpsv2": torch.tensor([0.6, 0.4]),
            },
            batch_size=2,
        )
        _patch_iter(monkeypatch, [b])
        m = compute_rollout_batch_metrics(training_data=None)
        assert m["reward_mean"] == pytest.approx(0.6)
        assert m["reward_pickscore_mean"] == pytest.approx(0.7)
        assert m["reward_pickscore_min"] == pytest.approx(0.6)
        assert m["reward_pickscore_max"] == pytest.approx(0.8)
        assert m["reward_hpsv2_mean"] == pytest.approx(0.5)

    def test_multi_batch_concat(self, monkeypatch):
        b1 = _BatchStub(
            rewards=torch.tensor([1.0]),
            component_rewards={"pickscore": torch.tensor([1.0])},
            batch_size=1,
        )
        b2 = _BatchStub(
            rewards=torch.tensor([3.0]),
            component_rewards={"pickscore": torch.tensor([3.0])},
            batch_size=1,
        )
        _patch_iter(monkeypatch, [b1, b2])
        m = compute_rollout_batch_metrics(training_data=None)
        assert m["reward_mean"] == pytest.approx(2.0)
        assert m["reward_pickscore_mean"] == pytest.approx(2.0)

    def test_slash_in_component_name_normalized(self, monkeypatch):
        b = _BatchStub(
            rewards=torch.tensor([0.5]),
            component_rewards={"a/b": torch.tensor([0.5])},
            batch_size=1,
        )
        _patch_iter(monkeypatch, [b])
        m = compute_rollout_batch_metrics(training_data=None)
        assert "reward_a_b_mean" in m
        assert "reward_a/b_mean" not in m

    def test_heterogeneous_keys_across_batches(self, monkeypatch):
        b1 = _BatchStub(
            rewards=torch.tensor([0.5, 0.7]),
            component_rewards={"pickscore": torch.tensor([0.5, 0.7])},
            batch_size=2,
        )
        b2 = _BatchStub(
            rewards=torch.tensor([0.2]),
            component_rewards={"hpsv2": torch.tensor([0.2])},
            batch_size=1,
        )
        _patch_iter(monkeypatch, [b1, b2])
        m = compute_rollout_batch_metrics(training_data=None)
        assert m["reward_pickscore_mean"] == pytest.approx(0.6)
        assert m["reward_hpsv2_mean"] == pytest.approx(0.2)

    def test_empty_tensor_skipped(self, monkeypatch):
        # An empty per-component tensor is silently skipped (no metric, no crash).
        b = _BatchStub(
            rewards=torch.tensor([0.5]),
            component_rewards={
                "pickscore": torch.tensor([]),
                "hpsv2": torch.tensor([0.5]),
            },
            batch_size=1,
        )
        _patch_iter(monkeypatch, [b])
        m = compute_rollout_batch_metrics(training_data=None)
        assert "reward_pickscore_mean" not in m
        assert m["reward_hpsv2_mean"] == pytest.approx(0.5)
