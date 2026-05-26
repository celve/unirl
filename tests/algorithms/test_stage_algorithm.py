"""Tests for the stage-driven training contract.

Validates :class:`StageAlgorithm` (and its concrete :class:`DiffusionGRPO` /
:class:`ARGRPO`) plus :class:`StageTrainStack.train_track` against fake
stages that hold a leaf :class:`torch.nn.Parameter`. The fakes return the
expected shape from ``replay`` so the algorithm's loss is well-defined; the
parameter's ``.grad`` is the ground truth that backward fired.

No real model is loaded; no FSDP; no Ray; no rollout pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from diffusionrl.algorithms import (
    ARGRPO,
    AlgorithmStepResult,
    DiffusionGRPO,
    StageAlgorithm,
)
from diffusionrl.models.types.replay_result import ReplayResult
from diffusionrl.training import StageTrainStack, TrackMiniBatchResult
from diffusionrl.types.conditions import Condition, TextEmbedCondition
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments import LatentSegment, TextSegment

# ---------------------------------------------------------------------------
# Fake stages
# ---------------------------------------------------------------------------


class _FakeDiffusionStage:
    """Minimal :class:`DiffusionStage` impl with a leaf ``nn.Parameter``.

    ``replay`` returns a ``ReplayResult`` whose ``log_probs`` are
    ``param * sum(text.embeds)`` broadcast to ``[B, S']``; that wires the
    param into the loss so we can assert ``param.grad`` is populated after
    ``compute_loss_and_backward``.
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
    ) -> ReplayResult:
        text = conditions["text"]
        assert text.embeds is not None
        B = int(text.embeds.shape[0])
        if step_indices is None:
            S = 0 if segment.sde_indices is None else int(segment.sde_indices.shape[0])
        else:
            S = len(step_indices)
        feat = text.embeds.float().reshape(B, -1).mean(dim=1)  # [B]
        out = self.param * feat
        log_probs = out.unsqueeze(1).expand(B, max(S, 1))[:, :S].contiguous()
        return ReplayResult(log_probs=log_probs)


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
# StageTrainStack — per-track composed stack
# ---------------------------------------------------------------------------


class _FakePolicy:
    """Thin Policy adapter wrapping a fake stage's leaf ``nn.Parameter``.

    Implements only the surface ``StageTrainStack.train_track``
    actually touches: ``parameters()`` (for the no-FSDP fallback path of
    ``clip_grad_norm_``) and ``replay()`` delegation (so the algorithm's
    stage reference still works when wrapped). ``walk_source_chain`` is
    satisfied via the ``source`` attribute.
    """

    def __init__(self, stage: Any) -> None:
        self.source = stage
        # ``model`` is part of the Policy protocol but the tests don't
        # exercise it; setting to None keeps the surface honest.
        self.model = None

    def trainable_module(self):
        return self.model

    def parameters(self):
        # Yield the underlying stage's leaf parameter so the optimizer +
        # clip_grad_norm fallback work end-to-end.
        return iter([self.source.param])

    def replay(self, *args, **kwargs):
        return self.source.replay(*args, **kwargs)


def _make_diffusion_track(*, batch_size: int = 2, num_steps: int = 4) -> RolloutTrack:
    """Build a single-track RolloutTrack wrapping a synthetic LatentSegment."""
    return RolloutTrack(
        sample_ids=[f"d{i}" for i in range(batch_size)],
        parent_ids=None,
        parent_track=None,
        conditions=_conditions_with_text(batch_size=batch_size),
        segment=_make_latent_segment(batch_size=batch_size, num_steps=num_steps),
        advantages=torch.tensor([0.5, -0.5][:batch_size]),
    )


def _make_ar_track(*, batch_size: int = 2, tokens_per_sample: int = 3) -> RolloutTrack:
    """Build a single-track RolloutTrack wrapping a synthetic TextSegment."""
    return RolloutTrack(
        sample_ids=[f"a{i}" for i in range(batch_size)],
        parent_ids=None,
        parent_track=None,
        conditions=_conditions_with_text(batch_size=batch_size),
        segment=_make_text_segment(batch_size=batch_size, tokens_per_sample=tokens_per_sample),
        advantages=torch.tensor([0.4, -0.6][:batch_size]),
    )


def _build_two_track_stack(
    *,
    diff_init: float = 0.7,
    ar_init: float = 0.3,
    micro_batch_size: int = 2,
    max_grad_norm: float = 1.0,
    optional_tracks: frozenset = frozenset(),
):
    """Construct a two-track stack + the underlying stages (for grad inspection)."""
    diff_stage = _FakeDiffusionStage(init_value=diff_init)
    ar_stage = _FakeARStage(init_value=ar_init)
    diff_policy = _FakePolicy(diff_stage)
    ar_policy = _FakePolicy(ar_stage)
    diff_optim = torch.optim.SGD([diff_stage.param], lr=0.1)
    ar_optim = torch.optim.SGD([ar_stage.param], lr=0.1)
    stack = StageTrainStack(
        policies={"image": diff_policy, "ar": ar_policy},
        optimizers={"image": diff_optim, "ar": ar_optim},
        schedulers={"image": None, "ar": None},
        algorithms={
            "image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            "ar": ARGRPO(stage=ar_stage, conditions_cls=None),
        },
        micro_batch_sizes={"image": micro_batch_size, "ar": micro_batch_size},
        max_grad_norm=max_grad_norm,
        optional_tracks=optional_tracks,
    )
    return stack, diff_stage, ar_stage


def test_multi_track_train_track_steps_each_track_independently() -> None:
    """Calling train_track twice (once per track) advances both grads and both step counters."""
    stack, diff_stage, ar_stage = _build_two_track_stack()
    resp = RolloutResp(tracks={"image": _make_diffusion_track(), "ar": _make_ar_track()})

    image_result = stack.train_track(resp, "image", training_progress=0.0)
    ar_result = stack.train_track(resp, "ar", training_progress=0.0)

    assert isinstance(image_result, TrackMiniBatchResult)
    assert isinstance(ar_result, TrackMiniBatchResult)
    assert image_result.track_name == "image"
    assert ar_result.track_name == "ar"
    assert image_result.has_backward and ar_result.has_backward
    assert diff_stage.param.grad is not None and diff_stage.param.grad.abs().item() > 0.0
    assert ar_stage.param.grad is not None and ar_stage.param.grad.abs().item() > 0.0
    # Per-track optimizer-step counters advance independently.
    assert stack._optimizer_steps == {"image": 1, "ar": 1}


def test_multi_track_train_track_optional_absent_returns_noop() -> None:
    """An optional track absent from resp.tracks returns a has_backward=False result, no raise."""
    stack, diff_stage, ar_stage = _build_two_track_stack(
        optional_tracks=frozenset({"ar"}),
    )
    resp = RolloutResp(tracks={"image": _make_diffusion_track()})  # ar absent

    # The image track trains normally.
    image_result = stack.train_track(resp, "image", training_progress=0.0)
    assert image_result.has_backward
    assert diff_stage.param.grad is not None and diff_stage.param.grad.abs().item() > 0.0

    # The ar track is optional + absent → no-op result, no raise.
    ar_result = stack.train_track(resp, "ar", training_progress=0.0)
    assert ar_result.track_name == "ar"
    assert not ar_result.has_backward
    assert ar_result.loss == 0.0
    assert ar_result.micros == []
    # AR stage's grad must remain untouched.
    assert ar_stage.param.grad is None
    # Per-track step counter advanced only for image.
    assert stack._optimizer_steps == {"image": 1, "ar": 0}


def test_multi_track_train_track_required_absent_raises() -> None:
    """A non-optional registered track that's missing from resp.tracks raises."""
    import pytest

    stack, _diff_stage, ar_stage = _build_two_track_stack()  # no optional_tracks
    resp = RolloutResp(tracks={"image": _make_diffusion_track()})  # ar absent

    with pytest.raises(ValueError, match=r"track 'ar' is registered but absent"):
        stack.train_track(resp, "ar", training_progress=0.0)
    # AR grad must remain untouched after the raise.
    assert ar_stage.param.grad is None


def test_multi_track_train_track_requires_track_advantages() -> None:
    """``track.advantages is None`` is a per-track fail-fast precondition."""
    import pytest

    stack, _diff_stage, _ar_stage = _build_two_track_stack()
    no_adv_track = RolloutTrack(
        sample_ids=["d0", "d1"],
        parent_ids=None,
        parent_track=None,
        conditions=_conditions_with_text(batch_size=2),
        segment=_make_latent_segment(batch_size=2, num_steps=4),
        advantages=None,
    )
    resp = RolloutResp(tracks={"image": no_adv_track, "ar": _make_ar_track()})

    with pytest.raises(ValueError, match=r"track 'image' has advantages=None"):
        stack.train_track(resp, "image", training_progress=0.0)


def test_multi_track_train_track_unknown_track_raises() -> None:
    """Calling train_track with a name not in self.algorithms raises."""
    import pytest

    stack, _diff_stage, _ar_stage = _build_two_track_stack()
    resp = RolloutResp(tracks={"image": _make_diffusion_track(), "ar": _make_ar_track()})

    with pytest.raises(ValueError, match=r"track 'unknown' is not registered"):
        stack.train_track(resp, "unknown", training_progress=0.0)


def test_multi_track_stack_post_init_invariants() -> None:
    """All parallel dicts must share the same key set."""
    import pytest

    diff_stage = _FakeDiffusionStage()
    diff_policy = _FakePolicy(diff_stage)
    diff_optim = torch.optim.SGD([diff_stage.param], lr=0.1)

    # Mismatched keys: optimizers has 'image', algorithms has 'image' + 'ar'.
    with pytest.raises(ValueError, match=r"optimizers keys.*algorithms keys"):
        StageTrainStack(
            policies={"image": diff_policy, "ar": diff_policy},
            optimizers={"image": diff_optim},  # missing 'ar'
            schedulers={"image": None, "ar": None},
            algorithms={
                "image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
                "ar": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            },
            micro_batch_sizes={"image": 2, "ar": 2},
            max_grad_norm=1.0,
        )

    # optional_tracks must be a subset of algorithm keys.
    with pytest.raises(ValueError, match=r"optional_tracks.*not in algorithms"):
        StageTrainStack(
            policies={"image": diff_policy},
            optimizers={"image": diff_optim},
            schedulers={"image": None},
            algorithms={"image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None)},
            micro_batch_sizes={"image": 2},
            max_grad_norm=1.0,
            optional_tracks=frozenset({"image", "phantom"}),
        )


def test_multi_track_stack_on_rollout_end_per_track() -> None:
    """on_rollout_end walks every track's Policy chain and forwards each track's step counter."""
    diff_stage = _FakeDiffusionStage()
    ar_stage = _FakeARStage()

    captured: Dict[str, List[Optional[int]]] = {"image": [], "ar": []}

    class _CaptureOnRolloutEnd:
        def __init__(self, name: str, source: Any) -> None:
            self._name = name
            self.source = source
            self.model = None

        def trainable_module(self):
            return None

        def parameters(self):
            return iter([self.source.param])

        def replay(self, *args, **kwargs):
            return self.source.replay(*args, **kwargs)

        def on_rollout_end(self, step: Optional[int]) -> None:
            captured[self._name].append(step)

    stack = StageTrainStack(
        policies={
            "image": _CaptureOnRolloutEnd("image", diff_stage),
            "ar": _CaptureOnRolloutEnd("ar", ar_stage),
        },
        optimizers={
            "image": torch.optim.SGD([diff_stage.param], lr=0.1),
            "ar": torch.optim.SGD([ar_stage.param], lr=0.1),
        },
        schedulers={"image": None, "ar": None},
        algorithms={
            "image": DiffusionGRPO(stage=diff_stage, params=None, conditions_cls=None),
            "ar": ARGRPO(stage=ar_stage, conditions_cls=None),
        },
        micro_batch_sizes={"image": 2, "ar": 2},
        max_grad_norm=1.0,
    )
    # Bump only the image track's counter so we can assert per-track forwarding.
    stack._optimizer_steps["image"] = 7
    stack._optimizer_steps["ar"] = 3
    stack.on_rollout_end()
    assert captured == {"image": [7], "ar": [3]}
