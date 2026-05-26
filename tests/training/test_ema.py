"""Tests for EMA handle + Shadow integration.

Exercises EMA step/on_rollout_end/use_shadow with both NFT (dual-adapter)
and Mirror (shadow params) shadow strategies.  CPU-only, no FSDP.
"""

from __future__ import annotations

import pytest

peft = pytest.importorskip("peft")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from diffusionrl.training.configs import EmaFullConfig, EmaLoraConfig  # noqa: E402
from diffusionrl.training.ema import EMA, make_decay_fn  # noqa: E402
from diffusionrl.training.inject import (  # noqa: E402
    apply_deferred_ops,
    inject_mirror,
    inject_nft,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)


# ---------------------------------------------------------------------------
# EMA with NFT shadow (dual peft adapter)
# ---------------------------------------------------------------------------


def _build_nft_ema(timing: str = "optimizer_step") -> tuple[EMA, nn.Module]:
    model = _TinyModel()
    shadow = inject_nft(model, rank=4, alpha=8, target_modules=["q_proj"])
    apply_deferred_ops(model)
    cfg = EmaLoraConfig(ema_decay=0.5, timing=timing, ema_decay_type="constant")
    ema = EMA(shadow=shadow, decay_fn=make_decay_fn(cfg), timing=timing)
    return ema, model


def test_nft_ema_step_updates_shadow():
    ema, model = _build_nft_ema(timing="optimizer_step")
    pairs_before = [(lv.clone(), s.clone()) for lv, s in ema.shadow.iter_pairs()]

    with torch.no_grad():
        for lv, _ in ema.shadow.iter_pairs():
            lv.fill_(10.0)

    ema.step(0)

    for (_, s_before), (_, s_after) in zip(pairs_before, ema.shadow.iter_pairs()):
        assert not torch.allclose(s_after, s_before)


def test_nft_ema_on_rollout_end_noop_when_timing_is_optimizer_step():
    ema, model = _build_nft_ema(timing="optimizer_step")
    pairs_before = [(s.clone()) for _, s in ema.shadow.iter_pairs()]

    with torch.no_grad():
        for lv, _ in ema.shadow.iter_pairs():
            lv.fill_(10.0)

    ema.on_rollout_end(0)

    for s_before, (_, s_after) in zip(pairs_before, ema.shadow.iter_pairs()):
        assert torch.allclose(s_after, s_before)


def test_nft_ema_use_shadow_swaps_and_restores():
    ema, model = _build_nft_ema()

    with torch.no_grad():
        for lv, _ in ema.shadow.iter_pairs():
            lv.fill_(99.0)

    live_before = [lv.clone() for lv, _ in ema.shadow.iter_pairs()]

    with ema.use_shadow():
        for lv, _ in ema.shadow.iter_pairs():
            assert not torch.allclose(lv, torch.tensor(99.0))

    for l_before, (l_after, _) in zip(live_before, ema.shadow.iter_pairs()):
        assert torch.allclose(l_after, l_before)


# ---------------------------------------------------------------------------
# EMA with mirror shadow (shadow_ params)
# ---------------------------------------------------------------------------


def _build_mirror_ema(timing: str = "optimizer_step") -> tuple[EMA, nn.Module]:
    model = _TinyModel()
    with torch.no_grad():
        model.q_proj.weight.fill_(1.0)
    shadow = inject_mirror(model, prefix="shadow_")
    apply_deferred_ops(model)
    cfg = EmaFullConfig(target_decay=0.9, timing=timing)
    ema = EMA(shadow=shadow, decay_fn=make_decay_fn(cfg), timing=timing)
    return ema, model


def test_mirror_ema_step_moves_shadow_toward_live():
    ema, model = _build_mirror_ema()

    with torch.no_grad():
        model.q_proj.weight.fill_(10.0)

    ema.step(0)

    for live, shd in ema.shadow.iter_pairs():
        assert not torch.allclose(shd, torch.tensor(1.0))
        assert shd.mean().item() > 1.0


def test_mirror_ema_use_shadow_swaps_params():
    ema, model = _build_mirror_ema()

    with torch.no_grad():
        model.q_proj.weight.fill_(99.0)

    with ema.use_shadow():
        assert torch.allclose(model.q_proj.weight, torch.tensor(1.0))

    assert torch.allclose(model.q_proj.weight, torch.tensor(99.0))


def test_mirror_ema_use_shadow_restores_on_exception():
    ema, model = _build_mirror_ema()

    with torch.no_grad():
        model.q_proj.weight.fill_(42.0)

    with pytest.raises(RuntimeError, match="deliberate"):
        with ema.use_shadow():
            raise RuntimeError("deliberate")

    assert torch.allclose(model.q_proj.weight, torch.tensor(42.0))


def test_mirror_ema_rollout_end_timing():
    ema, model = _build_mirror_ema(timing="rollout_end")

    with torch.no_grad():
        model.q_proj.weight.fill_(10.0)

    ema.step(0)
    for _, shd in ema.shadow.iter_pairs():
        assert torch.allclose(shd, torch.tensor(1.0))

    ema.on_rollout_end(0)
    for _, shd in ema.shadow.iter_pairs():
        assert not torch.allclose(shd, torch.tensor(1.0))


# ---------------------------------------------------------------------------
# make_decay_fn
# ---------------------------------------------------------------------------


def test_make_decay_fn_constant():
    cfg = EmaLoraConfig(ema_decay=0.42, ema_decay_type="constant")
    fn = make_decay_fn(cfg)
    assert fn(0) == pytest.approx(0.42)
    assert fn(100) == pytest.approx(0.42)


def test_make_decay_fn_linear():
    cfg = EmaLoraConfig(ema_decay_type="linear", ema_uprate=0.01, ema_uphold=0.5)
    fn = make_decay_fn(cfg)
    assert fn(0) == pytest.approx(0.0)
    assert fn(10) == pytest.approx(0.1)
    assert fn(1000) == pytest.approx(0.5)


def test_make_decay_fn_warmup():
    cfg = EmaLoraConfig(ema_decay_type="warmup", ema_flat_steps=5, ema_uprate=0.01, ema_uphold=0.5)
    fn = make_decay_fn(cfg)
    assert fn(0) == 0.0
    assert fn(4) == 0.0
    assert fn(5) == pytest.approx(0.0)
    assert fn(15) == pytest.approx(0.1)


def test_make_decay_fn_ema_full():
    cfg = EmaFullConfig(target_decay=0.999)
    fn = make_decay_fn(cfg)
    assert fn(0) == pytest.approx(min(1 / 10, 0.999))
    assert fn(100) == pytest.approx(min(101 / 110, 0.999))


# ---------------------------------------------------------------------------
# apply_shadow / restore_shadow (RPC-friendly)
# ---------------------------------------------------------------------------


def test_apply_and_restore_shadow():
    ema, model = _build_mirror_ema()

    with torch.no_grad():
        model.q_proj.weight.fill_(50.0)

    ema.apply_shadow()
    assert torch.allclose(model.q_proj.weight, torch.tensor(1.0))

    ema.restore_shadow()
    assert torch.allclose(model.q_proj.weight, torch.tensor(50.0))
