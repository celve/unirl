"""Unit tests for ``BaseTrainer`` wandb-logging assembly (no GPU / no real wandb).

Bypasses the pool-opening ``__init__`` via ``__new__`` and drives
:meth:`BaseTrainer._log_rollout` with a fake recording logger, asserting the
``rollout/`` / ``train/`` / ``perf/`` panels get the expected keys for both the
single-track (DiffusionTrainer / VLMTrainer) and multi-track (PETrainer) shapes,
and that it no-ops when logging is off.
"""

from types import SimpleNamespace

import torch

from diffusionrl.trainer.base import BaseTrainer


class _RecordingWandb:
    """Stand-in for ``DiffusionRLWandBLogger`` that records calls."""

    def __init__(self) -> None:
        self.initialized = True
        self.rollout: list = []
        self.step: list = []
        self.perf: list = []

    def log_rollout(self, step, metrics):
        self.rollout.append((step, dict(metrics)))

    def log_step(self, step, metrics):
        self.step.append((step, dict(metrics)))

    def log_perf(self, step, metrics):
        self.perf.append((step, dict(metrics)))


def _result(*, loss, has_backward=True):
    """A duck-typed ``TrainStepResult`` (only the fields the logger reads)."""
    return SimpleNamespace(
        loss=loss,
        grad_norm=0.5,
        lr=3.0e-4,
        has_backward=has_backward,
        metrics={"policy_loss": loss / 10.0},
    )


def _track(rewards, advantages):
    """A duck-typed ``RolloutTrack`` for ``compute_rollout_resp_metrics``."""
    return SimpleNamespace(
        rewards=torch.tensor(rewards),
        advantages=torch.tensor(advantages),
        group_ids=None,
        component_rewards=None,
    )


def _trainer(wandb_logger):
    trainer = BaseTrainer.__new__(BaseTrainer)  # bypass the pool-opening __init__
    trainer.wandb_logger = wandb_logger
    trainer._optimizer_step = 0
    return trainer


def test_log_rollout_single_track() -> None:
    wb = _RecordingWandb()
    trainer = _trainer(wb)
    resp = SimpleNamespace(batch_size=4, tracks={"main": _track([1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 0.0, 1.0])})

    trainer._log_rollout(0, _result(loss=1.0), resp, step_time_s=0.25)

    # rollout panel: step = rollout_id + 1; single-track keys are unprefixed.
    r_step, r_metrics = wb.rollout[0]
    assert r_step == 1
    assert r_metrics["reward_mean"] == 2.5
    assert {"reward_std", "reward_min", "reward_max", "advantage_mean"} <= set(r_metrics)
    # train panel: optimizer step incremented; scalars + algorithm metrics.
    s_step, s_metrics = wb.step[0]
    assert s_step == 1
    assert s_metrics["loss"] == 1.0 and s_metrics["lr"] == 3.0e-4
    assert "policy_loss" in s_metrics
    # perf panel.
    assert wb.perf[0] == (1, {"rollout_time_s": 0.25})


def test_log_rollout_multi_track_namespaced() -> None:
    wb = _RecordingWandb()
    trainer = _trainer(wb)
    results = {"ar": _result(loss=2.0), "diffusion": _result(loss=1.0)}
    resp = SimpleNamespace(
        batch_size=8,
        tracks={
            "ar": _track([2.0, 2.0, 2.0, 5.0], [0.0, 0.0, -1.0, 1.0]),
            "diffusion": _track([1.0, 3.0, 2.0, 2.0], [-1.0, 1.0, 0.0, 0.0]),
        },
    )

    trainer._log_rollout(2, results, resp, step_time_s=0.5)

    _, r_metrics = wb.rollout[0]
    # multi-track → reward stats prefixed by track name.
    assert "ar_reward_mean" in r_metrics and "diffusion_reward_mean" in r_metrics
    _, s_metrics = wb.step[0]
    # train scalars per-track namespaced.
    assert s_metrics["ar/loss"] == 2.0 and s_metrics["diffusion/loss"] == 1.0
    assert "ar/policy_loss" in s_metrics and "diffusion/policy_loss" in s_metrics


def test_log_rollout_skips_train_panel_without_backward() -> None:
    wb = _RecordingWandb()
    trainer = _trainer(wb)
    resp = SimpleNamespace(batch_size=2, tracks={"main": _track([1.0, 2.0], [0.0, 0.0])})

    trainer._log_rollout(0, _result(loss=1.0, has_backward=False), resp)

    assert wb.rollout  # rollout panel still logged
    assert not wb.step  # no optimizer step → no train panel
    assert trainer._optimizer_step == 0


def test_log_rollout_noop_when_disabled() -> None:
    trainer = _trainer(None)
    resp = SimpleNamespace(batch_size=1, tracks={"main": _track([1.0], [0.0])})
    trainer._log_rollout(0, _result(loss=1.0), resp, step_time_s=0.1)  # must not raise
