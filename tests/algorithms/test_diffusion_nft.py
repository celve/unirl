"""CPU-only unit tests for :class:`DiffusionNFT`.

The algorithm class is fed:
- a fake :class:`DiffusionStage` whose ``predict_noise_at_step`` returns
  ``param * sample`` so the leaf ``nn.Parameter`` is wired into the loss
  graph;
- a fake :class:`NFTLoRAPolicy`-shaped mock that exposes a real
  context-manager ``with_old_adapter()`` plus inert ``step`` /
  ``on_rollout_end`` methods.

No peft, no FSDP, no Ray. The :class:`NFTLoRAPolicy` itself is exercised
in ``tests/test_nft_lora_policy.py`` (which requires peft).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Optional

import pytest
import torch
import torch.nn as nn

from diffusionrl.algorithms import (
    AlgorithmStepResult,
    DiffusionNFT,
    DiffusionNFTConfig,
    StageAlgorithm,
)
from diffusionrl.types.conditions import Condition, TextEmbedCondition
from diffusionrl.types.segments import LatentSegment

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeNFTLoRAPolicy:
    """Pure-Python stand-in for :class:`NFTLoRAPolicy`.

    Records the active "adapter" so tests can assert that
    ``with_old_adapter`` swaps and restores. ``step`` / ``on_rollout_end``
    are inert counters — the algorithm doesn't call them; they only need
    to exist so ``DiffusionNFT.__init__`` passes its method-presence check.
    """

    def __init__(self) -> None:
        self.active_adapter = "default"
        self.step_calls = 0
        self.rollout_end_calls = 0

    @contextmanager
    def with_old_adapter(self):
        prev = self.active_adapter
        self.active_adapter = "old"
        try:
            yield
        finally:
            self.active_adapter = prev

    def step(self, optimization_step: Optional[int] = None) -> None:
        self.step_calls += 1

    def on_rollout_end(self, step: Optional[int] = None) -> None:
        self.rollout_end_calls += 1


class _FakeDiffusionStage:
    """Minimal stage exposing :meth:`predict_noise_at_step`.

    The return value ``param * sample`` couples the leaf
    ``nn.Parameter`` into the loss graph through the forward path the
    algorithm calls. ``predict_noise_at_step`` records which "adapter"
    was active at call time so tests can verify the algorithm executes
    one forward under ``default`` and one under ``old``.
    """

    def __init__(self, *, lora_policy: _FakeNFTLoRAPolicy, init_value: float = 0.5) -> None:
        # nn.Module wrapper so the parameter is a real leaf with grad.
        self._mod = nn.Module()
        self._mod.param = nn.Parameter(torch.tensor(float(init_value)))
        self._lora_policy = lora_policy
        self.predict_calls: list[str] = []

    @property
    def model(self) -> nn.Module:  # noqa: D401 - mirror real stage attr
        return self._mod

    @property
    def param(self) -> nn.Parameter:
        return self._mod.param

    def predict_noise_at_step(
        self,
        conditions: Mapping[str, Condition],
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: Any,
    ) -> torch.Tensor:
        del conditions, sigma, params
        self.predict_calls.append(self._lora_policy.active_adapter)
        return self._mod.param * sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_segment(
    *,
    batch_size: int = 4,
    num_inference_steps: int = 4,
    channels: int = 4,
    h: int = 8,
    w: int = 8,
) -> LatentSegment:
    # Dense latents path: ``sde_indices=None`` + ``sde_logp=None`` is what
    # NFT expects (rollout was forward-process). Sigma schedule has
    # ``num_inference_steps + 1`` entries; the terminal zero is dropped
    # by NFT's resolver so K_max == num_inference_steps.
    latents = torch.randn(batch_size, 1, channels, h, w)
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    return LatentSegment(
        sample_indices=torch.arange(batch_size, dtype=torch.long),
        positions=torch.zeros(batch_size, dtype=torch.long),
        latents=latents,
        sigmas=sigmas,
        indices=torch.arange(num_inference_steps + 1, dtype=torch.long),
        sde_logp=None,
        sde_indices=None,
    )


def _conditions(*, batch_size: int) -> Mapping[str, Condition]:
    return {"text": TextEmbedCondition(embeds=torch.randn(batch_size, 4, 8))}


def _build_alg(
    *,
    lora_policy: _FakeNFTLoRAPolicy,
    stage: _FakeDiffusionStage,
    **kwargs,
) -> DiffusionNFT:
    """Construct DiffusionNFT with sensible NFT defaults; ``**kwargs`` overrides."""
    defaults = dict(
        stage=stage,
        params=None,
        nft_lora_policy=lora_policy,
        beta=1.0,
        adv_clip_max=5.0,
        adv_mode="raw",
        use_adaptive_weight=False,  # default off in tests; toggled where checked
        train_timestep_mode="all",
        shuffle_train_timesteps=False,
        apply_time_shift_in_loss=False,
        training_timestep_fraction=1.0,
        kl_coef=0.0,
        conditions_cls=None,
    )
    defaults.update(kwargs)
    return DiffusionNFT(**defaults)


# ---------------------------------------------------------------------------
# DiffusionNFT.__init__ fail-fast
# ---------------------------------------------------------------------------


def test_nft_rejects_unsupported_adv_mode() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    with pytest.raises(ValueError, match="adv_mode"):
        _build_alg(lora_policy=lora, stage=stage, adv_mode="sign")


def test_nft_rejects_unsupported_timestep_mode() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    with pytest.raises(ValueError, match="train_timestep_mode"):
        _build_alg(lora_policy=lora, stage=stage, train_timestep_mode="fraction")


def test_nft_rejects_apply_time_shift() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    with pytest.raises(ValueError, match="apply_time_shift_in_loss"):
        _build_alg(lora_policy=lora, stage=stage, apply_time_shift_in_loss=True)


def test_nft_rejects_kl_coef_gt_zero() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    with pytest.raises(ValueError, match="kl_coef"):
        _build_alg(lora_policy=lora, stage=stage, kl_coef=0.01)


def test_nft_rejects_out_of_range_timestep_fraction() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    with pytest.raises(ValueError, match="training_timestep_fraction"):
        _build_alg(lora_policy=lora, stage=stage, training_timestep_fraction=0.0)
    with pytest.raises(ValueError, match="training_timestep_fraction"):
        _build_alg(lora_policy=lora, stage=stage, training_timestep_fraction=1.5)


def test_nft_rejects_missing_lora_policy_surface() -> None:
    stage = _FakeDiffusionStage(lora_policy=_FakeNFTLoRAPolicy())

    class _BadPolicy:  # missing with_old_adapter
        def step(self, *_a, **_k): ...
        def on_rollout_end(self, *_a, **_k): ...

    with pytest.raises(TypeError, match="with_old_adapter"):
        _build_alg(lora_policy=_BadPolicy(), stage=stage)


# ---------------------------------------------------------------------------
# DiffusionNFT.compute_loss_and_backward
# ---------------------------------------------------------------------------


def test_nft_loss_finite_and_backward_with_adapter_switch() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora, init_value=0.5)
    # mode=all with K=4 (segment has 5 sigmas → drop terminal zero → 4 iters)
    alg = _build_alg(lora_policy=lora, stage=stage)

    assert isinstance(alg, StageAlgorithm)

    seg = _make_segment(batch_size=4, num_inference_steps=4)
    conds = _conditions(batch_size=4)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )

    assert isinstance(result, AlgorithmStepResult)
    assert result.has_backward
    # K iterations (after dropping terminal zero from sigmas).
    assert result.num_steps_or_tokens == 4
    assert torch.isfinite(torch.tensor(result.loss)).item()
    # Param grad fired (the loss flowed through stage.param).
    assert stage.param.grad is not None
    assert stage.param.grad.abs().item() > 0.0

    # Each of K iterations does (default, old) — so 2K forwards total.
    expected_calls = ["default", "old"] * 4
    assert stage.predict_calls == expected_calls, stage.predict_calls
    # Active adapter restored after final iteration.
    assert lora.active_adapter == "default"

    # Required metrics keys (from aggregated per-iter metrics + summary).
    expected = {
        "policy_loss",
        "total_loss",
        "loss_per_iter",
        "num_timesteps",
        "pos_loss_mean",
        "neg_loss_mean",
        "r_mean",
        "advantage_mean",
        "advantage_std",
        "prediction_deviation",
        "x0_norm",
        "t_value",
    }
    assert expected.issubset(result.metrics.keys())
    assert result.metrics["num_timesteps"] == 4.0


def test_nft_mode_all_uses_segment_sigmas() -> None:
    """mode='all' must consume segment.sigmas (drop terminal zero)."""
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage, train_timestep_mode="all")
    seg = _make_segment(batch_size=2, num_inference_steps=5)
    result = alg.compute_loss_and_backward(
        conditions=_conditions(batch_size=2),
        segment=seg,
        advantages=torch.tensor([1.0, -1.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    # 5 + 1 sigmas → drop terminal zero → K=5 iterations.
    assert result.num_steps_or_tokens == 5
    assert len(stage.predict_calls) == 2 * 5  # default + old per K


def test_nft_mode_random_synthesizes_b_timesteps() -> None:
    """mode='random' must use B random scalars regardless of segment.sigmas."""
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage, train_timestep_mode="random")
    B = 7
    seg = _make_segment(batch_size=B, num_inference_steps=3)
    result = alg.compute_loss_and_backward(
        conditions=_conditions(batch_size=B),
        segment=seg,
        advantages=torch.randn(B),
        training_progress=0.0,
        loss_scale=1.0,
    )
    # random mode → K = B (one scalar per sample, broadcast in loop).
    assert result.num_steps_or_tokens == B
    assert len(stage.predict_calls) == 2 * B


def test_nft_mode_all_raises_when_sigmas_missing() -> None:
    """mode='all' must fail-fast if segment.sigmas is None (no schedule)."""
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage, train_timestep_mode="all")
    seg = _make_segment(batch_size=2)
    seg.sigmas = None  # drop the schedule
    with pytest.raises(ValueError, match="segment.sigmas"):
        alg.compute_loss_and_backward(
            conditions=_conditions(batch_size=2),
            segment=seg,
            advantages=torch.tensor([1.0, -1.0]),
            training_progress=0.0,
            loss_scale=1.0,
        )


def test_nft_fraction_slice_shortens_k() -> None:
    """training_timestep_fraction=0.5 should halve K in mode=all."""
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(
        lora_policy=lora,
        stage=stage,
        train_timestep_mode="all",
        training_timestep_fraction=0.5,
    )
    seg = _make_segment(batch_size=2, num_inference_steps=10)
    result = alg.compute_loss_and_backward(
        conditions=_conditions(batch_size=2),
        segment=seg,
        advantages=torch.tensor([1.0, -1.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )
    # 10 + 1 sigmas → drop terminal zero → 10 candidate ts → fraction 0.5 → K=5.
    assert result.num_steps_or_tokens == 5


def test_nft_advantage_remap_to_unit_interval() -> None:
    """Advantages span the full clip range → r should span [0, 1]."""
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage, adv_clip_max=5.0)

    seg = _make_segment(batch_size=3)
    conds = _conditions(batch_size=3)
    # -5 → 0, 0 → 0.5, +5 → 1.0
    advantages = torch.tensor([-5.0, 0.0, 5.0])

    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert result.metrics["r_mean"] == pytest.approx(0.5, abs=1e-6)


def test_nft_advantage_clipping_caps_r() -> None:
    """Advantages outside [-clip, clip] should be clipped, capping r at [0, 1]."""
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage, adv_clip_max=5.0)

    seg = _make_segment(batch_size=2)
    conds = _conditions(batch_size=2)
    # Both should saturate.
    advantages = torch.tensor([-10.0, 10.0])

    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    # r values: 0.0 + 1.0 → mean 0.5.
    assert result.metrics["r_mean"] == pytest.approx(0.5, abs=1e-6)


def test_nft_adaptive_weight_changes_loss() -> None:
    """Toggling use_adaptive_weight should change the loss value."""
    lora_a = _FakeNFTLoRAPolicy()
    stage_a = _FakeDiffusionStage(lora_policy=lora_a, init_value=0.5)
    alg_a = _build_alg(lora_policy=lora_a, stage=stage_a, use_adaptive_weight=False)

    lora_b = _FakeNFTLoRAPolicy()
    stage_b = _FakeDiffusionStage(lora_policy=lora_b, init_value=0.5)
    alg_b = _build_alg(lora_policy=lora_b, stage=stage_b, use_adaptive_weight=True)

    torch.manual_seed(0)
    seg = _make_segment(batch_size=4)
    conds = _conditions(batch_size=4)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    torch.manual_seed(123)
    res_a = alg_a.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    torch.manual_seed(123)
    res_b = alg_b.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    # Same RNG, same advantages, same stage init — only differ by adaptive
    # weighting. The weighted version should land at a different value.
    assert res_a.loss != pytest.approx(res_b.loss, abs=1e-6)


def test_nft_zero_batch_returns_no_backward() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage)

    # Build a segment with batch=0 (no clean latents).
    seg = LatentSegment(
        sample_indices=torch.empty(0, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
        latents=torch.empty(0, 1, 4, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, 2),
        indices=torch.arange(2, dtype=torch.long),
        sde_logp=None,
        sde_indices=None,
    )
    conds = {"text": TextEmbedCondition(embeds=torch.empty(0, 4, 8))}

    result = alg.compute_loss_and_backward(
        conditions=conds,
        segment=seg,
        advantages=torch.empty(0),
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert not result.has_backward
    assert result.num_steps_or_tokens == 0


def test_nft_rejects_segment_without_latents() -> None:
    lora = _FakeNFTLoRAPolicy()
    stage = _FakeDiffusionStage(lora_policy=lora)
    alg = _build_alg(lora_policy=lora, stage=stage)

    seg = LatentSegment(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        latents=None,
        sigmas=torch.linspace(1.0, 0.0, 2),
        indices=torch.arange(2, dtype=torch.long),
        sde_logp=None,
        sde_indices=None,
    )
    conds = _conditions(batch_size=2)

    with pytest.raises(ValueError, match="segment.latents"):
        alg.compute_loss_and_backward(
            conditions=conds,
            segment=seg,
            advantages=torch.zeros(2),
            training_progress=0.0,
            loss_scale=1.0,
        )


def test_nft_config_dataclass_roundtrip() -> None:
    """DiffusionNFTConfig fields match DiffusionNFT.__init__ kwargs."""
    cfg = DiffusionNFTConfig(
        beta=1.5,
        adv_clip_max=3.0,
        adv_mode="raw",
        use_adaptive_weight=False,
        train_timestep_mode="random",
        shuffle_train_timesteps=True,
        apply_time_shift_in_loss=False,
        training_timestep_fraction=0.8,
        kl_coef=0.0,
    )
    assert cfg.beta == pytest.approx(1.5)
    assert cfg.adv_clip_max == pytest.approx(3.0)
    assert cfg.train_timestep_mode == "random"
    assert cfg.training_timestep_fraction == pytest.approx(0.8)
