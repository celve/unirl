"""Runnable smoke for the multi-track ``StageTrainStack``.

Demonstrates end-to-end per-track optimizer steps with two independent
tracks (``image`` = diffusion, ``ar`` = AR text) on a tiny fake-stage
setup. No real model bundle, no FSDP, no Ray — runs in a few seconds
on CPU or a single GPU.

Phases:

1. Build two ``_FakeStage`` instances each holding a leaf ``nn.Parameter``.
2. Wrap each in a ``_FakePolicy`` adapter (exposes ``parameters()`` so
   the optimizer can step the leaf param).
3. Construct one :class:`StageTrainStack` over both tracks with
   per-track ``(policy, optimizer, scheduler, algorithm)`` tuples and a
   shared ``max_grad_norm``.
4. Drive 5 rollouts: each rollout builds a fresh multi-track
   :class:`RolloutResp` and calls ``train_track`` once per track.
5. Assert per-track step counters advance to 5, both leaf params moved,
   and ``on_rollout_end`` fires once per rollout per track.

Run::

    cd ~/diffusionrl && source .venv/bin/activate
    python scripts/smoke_multi_track_stack.py

Exits 0 on success, non-zero with a traceback on any failure (so it
slots into CI or a manual sanity run).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, List, Mapping, Optional

import torch
import torch.nn as nn

from diffusionrl.algorithms import ARGRPO, DiffusionGRPO
from diffusionrl.models.types.replay_result import ReplayResult
from diffusionrl.training import StageTrainStack
from diffusionrl.types.conditions import Condition, TextEmbedCondition
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments import LatentSegment, TextSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("smoke_multi_track_stack")


# ---------------------------------------------------------------------------
# Fakes — minimal Stage + Policy implementations
# ---------------------------------------------------------------------------


class _FakeDiffusionStage:
    """Minimal DiffusionStage: leaf nn.Parameter; replay returns ``param * sum(text.embeds)``."""

    def __init__(self, init_value: float = 0.5) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))

    def diffuse(self, *args, **kwargs):
        raise NotImplementedError("smoke: not used")

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: LatentSegment,
        params: Any = None,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        text = conditions["text"]
        assert text.embeds is not None
        B = int(text.embeds.shape[0])
        if step_indices is None:
            S = 0 if segment.sde_indices is None else int(segment.sde_indices.shape[0])
        else:
            S = len(step_indices)
        feat = text.embeds.float().reshape(B, -1).mean(dim=1)
        out = self.param * feat
        log_probs = out.unsqueeze(1).expand(B, max(S, 1))[:, :S].contiguous()
        return ReplayResult(log_probs=log_probs)


class _FakeARStage:
    """Minimal ARStage: leaf nn.Parameter; replay returns ``param * (tokens.float() + 1)``."""

    def __init__(self, init_value: float = 0.5) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))

    def autoregress(self, *args, **kwargs):
        raise NotImplementedError("smoke: not used")

    def replay(self, conditions: Mapping[str, Condition], *, segment: TextSegment) -> torch.Tensor:
        assert segment.tokens is not None
        return self.param * (segment.tokens.float() + 1.0)


class _FakePolicy:
    """Thin Policy adapter exposing the stage's leaf parameter to the optimizer."""

    def __init__(self, stage: Any) -> None:
        self.source = stage
        self.model = None

    def trainable_module(self):
        return self.model

    def parameters(self):
        return iter([self.source.param])

    def replay(self, *args, **kwargs):
        return self.source.replay(*args, **kwargs)


# ---------------------------------------------------------------------------
# RolloutResp builders
# ---------------------------------------------------------------------------


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
# Smoke main
# ---------------------------------------------------------------------------


def main() -> int:
    num_rollouts = 5
    logger.info("=== Multi-track StageTrainStack smoke (rollouts=%d) ===", num_rollouts)

    # Stages + policies.
    diff_stage = _FakeDiffusionStage(init_value=0.7)
    ar_stage = _FakeARStage(init_value=0.3)
    diff_policy = _FakePolicy(diff_stage)
    ar_policy = _FakePolicy(ar_stage)

    diff_param_before = diff_stage.param.detach().clone()
    ar_param_before = ar_stage.param.detach().clone()

    # Per-track optimizer + scheduler (independent cadence).
    diff_optim = torch.optim.SGD([diff_stage.param], lr=0.1)
    ar_optim = torch.optim.SGD([ar_stage.param], lr=0.05)
    diff_sched = torch.optim.lr_scheduler.LambdaLR(diff_optim, lr_lambda=lambda step: 1.0)
    ar_sched = torch.optim.lr_scheduler.LambdaLR(ar_optim, lr_lambda=lambda step: 1.0)

    # Composed stack over both tracks.
    stack = StageTrainStack(
        policies={"image": diff_policy, "ar": ar_policy},
        optimizers={"image": diff_optim, "ar": ar_optim},
        schedulers={"image": diff_sched, "ar": ar_sched},
        algorithms={
            "image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            "ar": ARGRPO(stage=ar_stage, conditions_cls=None),
        },
        micro_batch_sizes={"image": 2, "ar": 2},
        max_grad_norm=1.0,
    )
    logger.info("PHASE 1 ✓ Built StageTrainStack with tracks=%s", sorted(stack.algorithms))

    # Drive rollouts.
    for rollout_id in range(num_rollouts):
        resp = RolloutResp(tracks={"image": _diffusion_track(), "ar": _ar_track()})
        image_result = stack.train_track(resp, "image", training_progress=rollout_id / num_rollouts)
        ar_result = stack.train_track(resp, "ar", training_progress=rollout_id / num_rollouts)
        stack.on_rollout_end()
        logger.info(
            "PHASE 2 ✓ rollout %d: image(loss=%.4f, grad=%.4f, lr=%.4f) ar(loss=%.4f, grad=%.4f, lr=%.4f)",
            rollout_id,
            image_result.loss,
            image_result.grad_norm,
            image_result.lr,
            ar_result.loss,
            ar_result.grad_norm,
            ar_result.lr,
        )

    # Assertions.
    if stack._optimizer_steps != {"image": num_rollouts, "ar": num_rollouts}:
        logger.error(
            "FAIL: per-track step counters %s != expected {image: %d, ar: %d}",
            stack._optimizer_steps,
            num_rollouts,
            num_rollouts,
        )
        return 1
    logger.info("PHASE 3 ✓ Per-track step counters advanced to %d each", num_rollouts)

    if torch.allclose(diff_stage.param.detach(), diff_param_before):
        logger.error("FAIL: image track param did not move across %d rollouts", num_rollouts)
        return 1
    if torch.allclose(ar_stage.param.detach(), ar_param_before):
        logger.error("FAIL: ar track param did not move across %d rollouts", num_rollouts)
        return 1
    logger.info(
        "PHASE 4 ✓ Both track params moved: image %.4f → %.4f; ar %.4f → %.4f",
        diff_param_before.item(),
        diff_stage.param.item(),
        ar_param_before.item(),
        ar_stage.param.item(),
    )

    logger.info("=== ALL PHASES PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
