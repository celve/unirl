"""Tests for the stage-driven training contract.

Validates :class:`StageAlgorithm` (and its concrete :class:`DiffusionGRPO` /
:class:`ARGRPO`) plus :class:`StageTrainStack.train_microbatch` against fake
stages that hold a leaf :class:`torch.nn.Parameter`. The fakes return the
expected shape from ``replay`` so the algorithm's loss is well-defined; the
parameter's ``.grad`` is the ground truth that backward fired.

No real model is loaded; no FSDP; no Ray; no rollout pipeline.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

import torch
import torch.nn as nn

from diffusionrl.algorithms_new import (
    ARGRPO,
    AlgorithmStepResult,
    DiffusionGRPO,
    StageAlgorithm,
)
from diffusionrl.training_new import StageTrainStack
from diffusionrl.types.conditions import Condition, TextEmbedCondition
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.segments import LatentSegment, TextSegment

# ---------------------------------------------------------------------------
# Fake stages
# ---------------------------------------------------------------------------


class _FakeDiffusionStage:
    """Minimal :class:`DiffusionStage` impl with a leaf ``nn.Parameter``.

    ``replay`` returns ``param * sum(text.embeds)`` broadcast to ``[B, S']``;
    that wires the param into the loss so we can assert ``param.grad`` is
    populated after ``compute_loss_and_backward``.
    """

    def __init__(self, *, init_value: float = 0.5) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))

    def diffuse(self, conditions: Mapping[str, Condition], *, schedule, params=None) -> LatentSegment:
        raise NotImplementedError("not used in tests")

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: LatentSegment,
        params: Any = None,
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        text = conditions["text"]
        assert text.embeds is not None
        B = int(text.embeds.shape[0])
        if step_indices is None:
            S = 0 if segment.sde_indices is None else int(segment.sde_indices.shape[0])
        else:
            S = len(step_indices)
        feat = text.embeds.float().reshape(B, -1).mean(dim=1)  # [B]
        out = self.param * feat
        return out.unsqueeze(1).expand(B, max(S, 1))[:, :S].contiguous()


class _FakeARStage:
    """Minimal :class:`ARStage` impl with a leaf ``nn.Parameter``.

    ``replay`` returns ``param * (segment.tokens.float() + 1)`` so the
    parameter is wired into a ``[total_tokens]`` log-prob tensor.
    """

    def __init__(self, *, init_value: float = 0.5) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))

    def autoregress(self, conditions, *, sampling_params, **kwargs):
        raise NotImplementedError("not used in tests")

    def replay(
        self,
        conditions: Mapping[str, Condition],
        *,
        segment: TextSegment,
    ) -> torch.Tensor:
        assert segment.tokens is not None
        return self.param * (segment.tokens.float() + 1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_latent_segment(*, batch_size: int = 2, num_steps: int = 4) -> LatentSegment:
    """Synthetic LatentSegment with full SDE log-probs at every step."""
    sde_indices_list = list(range(num_steps))
    sde_logp = torch.full((batch_size, num_steps), -1.0, dtype=torch.float32)
    return LatentSegment(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        latents=torch.zeros(batch_size, num_steps + 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
        sde_logp=sde_logp,
        sde_indices=torch.tensor(sde_indices_list, dtype=torch.long),
    )


def _make_text_segment(*, batch_size: int = 2, tokens_per_sample: int = 3) -> TextSegment:
    """Synthetic TextSegment with packed varlen tokens + per-token log-probs."""
    return TextSegment.pack(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        tokens=[
            torch.arange(
                k * tokens_per_sample,
                (k + 1) * tokens_per_sample,
                dtype=torch.long,
            )
            for k in range(batch_size)
        ],
        log_probs=[torch.full((tokens_per_sample,), -2.0, dtype=torch.float32) for _ in range(batch_size)],
    )


def _conditions_with_text(*, batch_size: int) -> Mapping[str, Condition]:
    return {"text": TextEmbedCondition(embeds=torch.randn(batch_size, 4, 8))}


# ---------------------------------------------------------------------------
# DiffusionGRPO
# ---------------------------------------------------------------------------


def test_diffusion_grpo_shape() -> None:
    """DiffusionGRPO: backward fires, leaf param has grad, result keys present."""
    stage = _FakeDiffusionStage(init_value=0.7)
    alg = DiffusionGRPO(stage=stage, params=None, clip_range=0.2, conditions_cls=None)

    seg = _make_latent_segment(batch_size=2, num_steps=4)
    conds = _conditions_with_text(batch_size=2)
    advantages = torch.tensor([0.5, -0.3])

    assert isinstance(alg, StageAlgorithm)
    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )

    assert isinstance(result, AlgorithmStepResult)
    assert result.has_backward
    assert result.num_steps_or_tokens == 4
    assert torch.isfinite(torch.tensor(result.loss)).item()
    # Param grad fired.
    assert stage.param.grad is not None
    assert stage.param.grad.abs().item() > 0.0
    # Required metrics keys are present.
    expected_keys = {
        "policy_loss",
        "clip_range",
        "ratio_mean",
        "ratio_std",
        "ratio_min",
        "ratio_max",
        "clip_fraction",
        "approx_kl",
    }
    assert expected_keys.issubset(result.metrics.keys())


def test_diffusion_grpo_target_step_subset() -> None:
    """Subclass-style override of ``_resolve_target_steps`` trains a subset."""

    class _Subset(DiffusionGRPO):
        def _resolve_target_steps(self, segment: LatentSegment) -> List[int]:
            return [1, 3]

    stage = _FakeDiffusionStage()
    alg = _Subset(stage=stage, params=None, conditions_cls=None)
    seg = _make_latent_segment(batch_size=2, num_steps=4)
    conds = _conditions_with_text(batch_size=2)
    advantages = torch.tensor([1.0, -1.0])

    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert result.has_backward
    assert result.num_steps_or_tokens == 2


def test_diffusion_grpo_no_sde_indices_returns_no_backward() -> None:
    """A segment without sde_indices yields a no-op step (defensive)."""
    stage = _FakeDiffusionStage()
    alg = DiffusionGRPO(stage=stage, params=None, conditions_cls=None)
    seg = LatentSegment(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        latents=torch.zeros(2, 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, 2),
        indices=torch.arange(2, dtype=torch.long),
        sde_logp=None,
        sde_indices=None,
    )
    result = alg.compute_loss_and_backward(
        conditions=_conditions_with_text(batch_size=2),
        segment=seg,
        advantages=torch.tensor([0.0, 0.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert not result.has_backward
    assert result.num_steps_or_tokens == 0
    assert stage.param.grad is None


# ---------------------------------------------------------------------------
# ARGRPO
# ---------------------------------------------------------------------------


def test_ar_grpo_shape() -> None:
    """ARGRPO: backward fires, leaf param has grad, num_steps_or_tokens = total tokens."""
    stage = _FakeARStage(init_value=0.7)
    alg = ARGRPO(stage=stage, clip_range=0.2, conditions_cls=None)

    seg = _make_text_segment(batch_size=2, tokens_per_sample=3)
    conds = _conditions_with_text(batch_size=2)
    advantages = torch.tensor([0.4, -0.6])

    assert isinstance(alg, StageAlgorithm)
    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )

    assert isinstance(result, AlgorithmStepResult)
    assert result.has_backward
    assert result.num_steps_or_tokens == 6  # 2 samples × 3 tokens
    assert torch.isfinite(torch.tensor(result.loss)).item()
    assert stage.param.grad is not None
    assert stage.param.grad.abs().item() > 0.0


def test_ar_grpo_advantage_expansion_via_lengths() -> None:
    """Per-sample advantages expand to per-token via per-sample lengths."""
    lengths = torch.tensor([2, 3], dtype=torch.long)  # sample 0 → 2 tokens; sample 1 → 3 tokens
    advantages = torch.tensor([0.1, 0.9])
    expanded = ARGRPO._expand_advantages_to_tokens(advantages, lengths, dtype=torch.float32, device=torch.device("cpu"))
    expected = torch.tensor([0.1, 0.1, 0.9, 0.9, 0.9], dtype=torch.float32)
    assert torch.allclose(expanded, expected)


def test_ar_grpo_empty_segment_returns_no_backward() -> None:
    """An empty TextSegment is a no-op (defensive)."""
    stage = _FakeARStage()
    alg = ARGRPO(stage=stage, conditions_cls=None)
    # Two samples, both with zero response tokens.
    seg = TextSegment.pack(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        tokens=[
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ],
        log_probs=[
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        ],
    )
    result = alg.compute_loss_and_backward(
        conditions=_conditions_with_text(batch_size=2),
        segment=seg,
        advantages=torch.tensor([0.0, 0.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert not result.has_backward
    assert result.num_steps_or_tokens == 0
    assert stage.param.grad is None


# ---------------------------------------------------------------------------
# StageTrainStack dispatch
# ---------------------------------------------------------------------------


def test_stage_train_stack_dispatch_fires_both_slots() -> None:
    """Two algorithms registered under different slots both run; both grads populate."""
    diff_stage = _FakeDiffusionStage(init_value=0.7)
    ar_stage = _FakeARStage(init_value=0.3)
    diff_alg = DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None)
    ar_alg = ARGRPO(stage=ar_stage, conditions_cls=None)

    # policy / optimizer / scheduler / cfg are unused by train_microbatch;
    # we exercise only the dispatch path here.
    stack = StageTrainStack(
        policy=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        algorithms={"image": diff_alg, "ar": ar_alg},
        cfg=None,  # type: ignore[arg-type]
    )

    resp = RolloutResp(
        sample_ids=["s0", "s1"],
        group_ids=["g", "g"],
        conditions=_conditions_with_text(batch_size=2),
        rollout_traces={
            "image": _make_latent_segment(batch_size=2, num_steps=4),
            "ar": _make_text_segment(batch_size=2, tokens_per_sample=3),
        },
        advantages=torch.tensor([0.5, -0.5]),
    )

    results = stack.train_microbatch(resp, training_progress=0.0, loss_scale=1.0)
    assert set(results.keys()) == {"image", "ar"}
    assert all(r.has_backward for r in results.values())
    assert diff_stage.param.grad is not None and diff_stage.param.grad.abs().item() > 0.0
    assert ar_stage.param.grad is not None and ar_stage.param.grad.abs().item() > 0.0


def test_stage_train_stack_absent_slot_raises_by_default() -> None:
    """A registered algorithm whose slot is absent from resp.rollout_traces
    raises ValueError by default — silent skipping is a multi-modal-RL
    silent-no-train risk."""
    import pytest

    diff_stage = _FakeDiffusionStage()
    ar_stage = _FakeARStage()
    stack = StageTrainStack(
        policy=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        algorithms={
            "image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            "ar": ARGRPO(stage=ar_stage, conditions_cls=None),
        },
        cfg=None,  # type: ignore[arg-type]
        # No optional_slots → fail-fast on absent slot.
    )
    resp = RolloutResp(
        sample_ids=["s0", "s1"],
        group_ids=["g", "g"],
        conditions=_conditions_with_text(batch_size=2),
        rollout_traces={"image": _make_latent_segment(batch_size=2, num_steps=4)},
        advantages=torch.tensor([0.5, -0.5]),
    )
    with pytest.raises(ValueError, match=r"slot 'ar' is registered .* but absent"):
        stack.train_microbatch(resp, training_progress=0.0, loss_scale=1.0)
    # AR grad must still be untouched after the raise.
    assert ar_stage.param.grad is None


def test_stage_train_stack_optional_slot_silently_skipped() -> None:
    """Slots listed in ``optional_slots`` are silently skipped when absent
    (legacy per-task-topology behavior, now opt-in)."""
    diff_stage = _FakeDiffusionStage()
    ar_stage = _FakeARStage()
    stack = StageTrainStack(
        policy=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        algorithms={
            "image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            "ar": ARGRPO(stage=ar_stage, conditions_cls=None),
        },
        cfg=None,  # type: ignore[arg-type]
        optional_slots=frozenset({"ar"}),
    )
    # Image-only response (e.g. SD3 t2i topology); the AR slot is absent
    # but explicitly marked optional.
    resp = RolloutResp(
        sample_ids=["s0", "s1"],
        group_ids=["g", "g"],
        conditions=_conditions_with_text(batch_size=2),
        rollout_traces={"image": _make_latent_segment(batch_size=2, num_steps=4)},
        advantages=torch.tensor([0.5, -0.5]),
    )
    results = stack.train_microbatch(resp, training_progress=0.0, loss_scale=1.0)
    assert set(results.keys()) == {"image"}
    assert ar_stage.param.grad is None  # AR was not invoked


def test_stage_train_stack_requires_advantages() -> None:
    """``resp.advantages is None`` is a fail-fast precondition."""
    import pytest

    stage = _FakeDiffusionStage()
    stack = StageTrainStack(
        policy=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        algorithms={"image": DiffusionGRPO(stage=stage, params=None, conditions_cls=None)},
        cfg=None,  # type: ignore[arg-type]
    )
    resp = RolloutResp(
        sample_ids=["s0"],
        group_ids=["g"],
        conditions=_conditions_with_text(batch_size=1),
        rollout_traces={"image": _make_latent_segment(batch_size=1, num_steps=2)},
        advantages=None,
    )
    with pytest.raises(ValueError, match="advantages"):
        stack.train_microbatch(resp, training_progress=0.0, loss_scale=1.0)
