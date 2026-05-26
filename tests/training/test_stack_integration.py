"""End-to-end integration test for the multi-track StageTrainStack.

Drives multiple rollout cycles through ``train_track`` using TrainTrack
grouping.  Confirms per-track counters, grad flow, LR schedules, and
on_rollout_end dispatch.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

import torch
import torch.nn as nn

from diffusionrl.algorithms import ARGRPO, DiffusionGRPO
from diffusionrl.models.types.replay_result import ReplayResult
from diffusionrl.training import StageTrainStack
from diffusionrl.training.train_track import TrainTrack
from diffusionrl.types.conditions import Condition, TextEmbedCondition
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments import LatentSegment, TextSegment

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeDiffusionStage:
    def __init__(self, init_value: float = 0.5) -> None:
        self._model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self._model.weight.fill_(init_value)
        self.param = self._model.weight

    def trainable_module(self) -> nn.Module:
        return self._model

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: LatentSegment,
        params: object = None,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        text = conditions["text"]
        B = int(text.embeds.shape[0])
        if step_indices is None:
            S = 0 if segment.sde_indices is None else int(segment.sde_indices.shape[0])
        else:
            S = len(step_indices)
        feat = text.embeds.float().reshape(B, -1).mean(dim=1)
        out = self.param.squeeze() * feat
        log_probs = out.unsqueeze(1).expand(B, max(S, 1))[:, :S].contiguous()
        return ReplayResult(log_probs=log_probs)


class _FakeARStage:
    def __init__(self, init_value: float = 0.5) -> None:
        self._model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self._model.weight.fill_(init_value)
        self.param = self._model.weight

    def trainable_module(self) -> nn.Module:
        return self._model

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: TextSegment,
    ) -> torch.Tensor:
        assert segment.tokens is not None
        return self.param.squeeze() * (segment.tokens.float() + 1.0)


def _conditions(batch_size: int) -> Mapping[str, Condition]:
    return {"text": TextEmbedCondition(embeds=torch.randn(batch_size, 4, 8))}


def _diffusion_track(batch_size: int = 2, num_steps: int = 4) -> RolloutTrack:
    sde_logp = torch.full((batch_size, num_steps), -1.0, dtype=torch.float32)
    segment = LatentSegment(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        latents=torch.zeros(batch_size, num_steps + 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
        sde_logp=sde_logp,
        sde_indices=torch.arange(num_steps, dtype=torch.long),
    )
    return RolloutTrack(
        sample_ids=[f"d{i}" for i in range(batch_size)],
        parent_ids=None,
        parent_track=None,
        conditions=_conditions(batch_size),
        segment=segment,
        advantages=torch.tensor([0.5, -0.5][:batch_size], dtype=torch.float32),
    )


def _ar_track(batch_size: int = 2, tokens_per_sample: int = 3) -> RolloutTrack:
    segment = TextSegment.pack(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        tokens=[
            torch.arange(k * tokens_per_sample, (k + 1) * tokens_per_sample, dtype=torch.long)
            for k in range(batch_size)
        ],
        log_probs=[torch.full((tokens_per_sample,), -2.0, dtype=torch.float32) for _ in range(batch_size)],
    )
    return RolloutTrack(
        sample_ids=[f"a{i}" for i in range(batch_size)],
        parent_ids=None,
        parent_track=None,
        conditions=_conditions(batch_size),
        segment=segment,
        advantages=torch.tensor([0.4, -0.6][:batch_size], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Multi-rollout integration
# ---------------------------------------------------------------------------


def test_multi_track_stack_advances_per_track_counters_across_rollouts():
    diff_stage = _FakeDiffusionStage(init_value=0.7)
    ar_stage = _FakeARStage(init_value=0.3)

    diff_optim = torch.optim.SGD(diff_stage.trainable_module().parameters(), lr=0.1)
    ar_optim = torch.optim.SGD(ar_stage.trainable_module().parameters(), lr=0.05)
    diff_sched = torch.optim.lr_scheduler.LambdaLR(diff_optim, lr_lambda=lambda step: 0.5**step)
    ar_sched = torch.optim.lr_scheduler.LambdaLR(ar_optim, lr_lambda=lambda step: 0.5**step)

    tracks = {
        "image": TrainTrack(
            stage=diff_stage,
            ema=None,
            optimizer=diff_optim,
            scheduler=diff_sched,
            algorithm=DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            micro_batch_size=2,
        ),
        "ar": TrainTrack(
            stage=ar_stage,
            ema=None,
            optimizer=ar_optim,
            scheduler=ar_sched,
            algorithm=ARGRPO(stage=ar_stage, conditions_cls=None),
            micro_batch_size=2,
        ),
    }

    stack = StageTrainStack(tracks=tracks, max_grad_norm=1.0)

    diff_param_before = diff_stage.param.detach().clone()
    ar_param_before = ar_stage.param.detach().clone()

    rollouts_results = []
    for rollout_id in range(3):
        resp = RolloutResp(tracks={"image": _diffusion_track(), "ar": _ar_track()})
        results = {
            "image": stack.train_track(resp, "image", training_progress=rollout_id / 3.0),
            "ar": stack.train_track(resp, "ar", training_progress=rollout_id / 3.0),
        }
        assert results["image"].has_backward
        assert results["ar"].has_backward
        rollouts_results.append(results)
        stack.on_rollout_end()

    assert stack._optimizer_steps == {"image": 3, "ar": 3}
    assert not torch.allclose(diff_stage.param.detach(), diff_param_before)
    assert not torch.allclose(ar_stage.param.detach(), ar_param_before)
    assert abs(rollouts_results[-1]["image"].lr - 0.1 * (0.5**3)) < 1e-9
    assert abs(rollouts_results[-1]["ar"].lr - 0.05 * (0.5**3)) < 1e-9
