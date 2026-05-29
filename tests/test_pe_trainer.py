"""Unit test for :meth:`PETrainer.train_step` orchestration.

CPU-only, no pool / Ray / models. The trainer is built via ``__new__`` (the
heavy ``__init__`` opens a placement block) with stub ``rollout`` / ``reward``
/ per-side ``stack`` handles, so we exercise the pure orchestration logic:

    score image track → propagate_rewards → per-track advantages → route to stacks

with a hand-computed P=2, N=2, M=2 scenario whose advantages are known.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from diffusionrl.trainer.pe import PETrainer
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, _track_with_field
from diffusionrl.types.sampling import (
    ARSamplingParams,
    ComposedSamplingParams,
    DiffusionSamplingParams,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RolloutStub:
    def __init__(self, resp: RolloutResp) -> None:
        self._resp = resp
        self.generated = 0

    def wake_up(self) -> None:  # no-op (trainside)
        pass

    def sleep(self) -> None:
        pass

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.generated += 1
        return self._resp


class _RewardStub:
    """Scores whatever track it is handed by attaching a fixed reward vector."""

    def __init__(self, rewards: torch.Tensor) -> None:
        self._rewards = rewards
        self.calls = 0
        self.scored_track = None

    def score_and_attach(self, *, req, track):
        self.calls += 1
        self.scored_track = track
        return _track_with_field(track, "rewards", self._rewards.clone())


class _StackStub:
    def __init__(self) -> None:
        self.received = None
        self.training_progress = None

    def train_track(self, track, *, training_progress):
        self.received = track
        self.training_progress = training_progress
        return SimpleNamespace(loss=1.0, grad_norm=2.0, lr=3.0e-4)


def _composed(n_rewrites: int, n_images: int) -> ComposedSamplingParams:
    return ComposedSamplingParams(
        diffusion=DiffusionSamplingParams(samples_per_prompt=n_images),
        ar=ARSamplingParams(samples_per_prompt=n_rewrites),
    )


def _two_track_resp(*, P: int, N: int, M: int) -> tuple[RolloutResp, RolloutReq]:
    """Build a lineage-correct 2-track resp via make_root_track + fork_track."""
    req = RolloutReq(
        sample_ids=[f"p{i}" for i in range(P)],
        group_ids=[f"p{i}" for i in range(P)],
        primitives={"text": Texts(texts=[f"raw{i}" for i in range(P)])},
        sampling_params=_composed(N, M),
    )
    ar_track = req.make_root_track(track_name="ar", branch=N)  # P*N, parent_ids=prompt
    diff_track = ar_track.fork_track(parent_name="ar", child_name="diffusion", branch=M)  # P*N*M
    return RolloutResp(tracks={"ar": ar_track, "diffusion": diff_track}), req


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_train_step_scores_image_propagates_and_routes() -> None:
    P, N, M = 2, 2, 2
    resp, req = _two_track_resp(P=P, N=N, M=M)
    diff_track = resp.tracks["diffusion"]  # capture pre-scoring identity

    # 8 image rewards, grouped by rewrite (M=2 consecutive per rewrite):
    #   p0/a0:[1,3]  p0/a1:[2,2]  p1/a0:[0,4]  p1/a1:[5,5]
    image_rewards = torch.tensor([1.0, 3.0, 2.0, 2.0, 0.0, 4.0, 5.0, 5.0])

    rollout = _RolloutStub(resp)
    reward = _RewardStub(image_rewards)
    ar_stack, diff_stack = _StackStub(), _StackStub()

    trainer = PETrainer.__new__(PETrainer)  # bypass the pool-opening __init__
    trainer.rollout = rollout
    trainer.reward = reward
    trainer.ar = SimpleNamespace(stack=ar_stack)
    trainer.diffusion = SimpleNamespace(stack=diff_stack)

    results, mean_reward = trainer.train_step(req, training_progress=0.5)

    # ── reward scored the IMAGE track only (8 samples), exactly once ──
    assert reward.calls == 1
    assert reward.scored_track is diff_track
    assert len(reward.scored_track.sample_ids) == P * N * M == 8
    assert reward.scored_track.parent_track == "ar"

    # ── routing: each track went to its own stack at the right size ──
    assert ar_stack.received is not None and diff_stack.received is not None
    assert len(ar_stack.received.sample_ids) == P * N == 4
    assert len(diff_stack.received.sample_ids) == P * N * M == 8
    assert ar_stack.training_progress == 0.5 and diff_stack.training_progress == 0.5

    # ── credit assignment: image reward mean over M → "ar" rewards ──
    #   means: [mean(1,3), mean(2,2), mean(0,4), mean(5,5)] = [2,2,2,5]
    torch.testing.assert_close(
        ar_stack.received.rewards.to(torch.float32),
        torch.tensor([2.0, 2.0, 2.0, 5.0]),
    )

    # ── advantages: "ar" grouped by prompt (N=2 rewrites/prompt) ──
    #   p0=[2,2]→[0,0]; p1=[2,5]→mean3.5,std1.5→[-1,1]
    torch.testing.assert_close(
        ar_stack.received.advantages.to(torch.float32),
        torch.tensor([0.0, 0.0, -1.0, 1.0]),
        atol=1e-4,
        rtol=0,
    )

    # ── advantages: "diffusion" grouped by rewrite (M=2 images/rewrite) ──
    #   [1,3]→[-1,1]; [2,2]→[0,0]; [0,4]→[-1,1]; [5,5]→[0,0]
    torch.testing.assert_close(
        diff_stack.received.advantages.to(torch.float32),
        torch.tensor([-1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]),
        atol=1e-4,
        rtol=0,
    )

    # ── results + mean reward ──
    assert set(results) == {"ar", "diffusion"}
    assert results["ar"].loss == 1.0 and results["diffusion"].lr == 3.0e-4
    assert abs(mean_reward - 22.0 / 8.0) < 1e-6  # mean of image_rewards = 2.75
    assert rollout.generated == 1
