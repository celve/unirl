"""Tests for inject functions + deferred ops.

Exercises inject_lora, inject_nft, inject_mirror, apply_deferred_ops,
and the Shadow data bundle — all CPU-only, no FSDP, no Ray.
"""

from __future__ import annotations

import pytest

peft = pytest.importorskip("peft")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from diffusionrl.training.inject import (  # noqa: E402
    apply_deferred_ops,
    inject_lora,
    inject_mirror,
    inject_nft,
)
from diffusionrl.training.shadow import Shadow  # noqa: E402


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)

    def trainable_module(self):
        return self


def _make_model() -> _TinyModel:
    return _TinyModel()


# ---------------------------------------------------------------------------
# inject_lora
# ---------------------------------------------------------------------------


def test_inject_lora_freezes_base_and_adds_adapter():
    model = _make_model()
    inject_lora(model, rank=4, alpha=8, target_modules=["q_proj", "k_proj"])

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert len(trainable) > 0
    assert len(frozen) > 0
    assert all("lora" in n for n in trainable)


def test_inject_lora_stamps_deferred_ops():
    model = _make_model()
    inject_lora(model, rank=4, alpha=8, target_modules=["q_proj"])
    assert hasattr(model, "_deferred_ops")
    assert len(model._deferred_ops) == 1


def test_inject_lora_deferred_ops_drain():
    model = _make_model()
    inject_lora(model, rank=4, alpha=8, target_modules=["q_proj"])
    apply_deferred_ops(model)
    assert model._deferred_ops == []


# ---------------------------------------------------------------------------
# inject_nft
# ---------------------------------------------------------------------------


def test_inject_nft_returns_shadow():
    model = _make_model()
    shadow = inject_nft(model, rank=4, alpha=8, target_modules=["q_proj", "k_proj"])
    assert isinstance(shadow, Shadow)


def test_inject_nft_freezes_shadow_adapter():
    model = _make_model()
    inject_nft(model, rank=4, alpha=8, target_modules=["q_proj"], default="default", shadow="old")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert all("old" not in n for n in trainable)
    assert any("old" in n for n in frozen)


def test_inject_nft_stamps_three_deferred_ops():
    model = _make_model()
    inject_nft(model, rank=4, alpha=8, target_modules=["q_proj"])
    assert len(model._deferred_ops) == 3


def test_inject_nft_shadow_iter_pairs():
    model = _make_model()
    shadow = inject_nft(model, rank=4, alpha=8, target_modules=["q_proj"])
    apply_deferred_ops(model)
    pairs = list(shadow.iter_pairs())
    assert len(pairs) > 0
    for live, shd in pairs:
        assert live.shape == shd.shape


def test_inject_nft_shadow_swap():
    model = _make_model()
    shadow = inject_nft(model, rank=4, alpha=8, target_modules=["q_proj"])
    apply_deferred_ops(model)

    pairs_before = [(lv.clone(), s.clone()) for lv, s in shadow.iter_pairs()]

    with torch.no_grad():
        for lv, _ in shadow.iter_pairs():
            lv.fill_(99.0)

    shadow.swap_in()
    for (_, s_before), (active, _) in zip(pairs_before, shadow.iter_pairs()):
        assert torch.allclose(active, s_before)

    shadow.swap_out()


# ---------------------------------------------------------------------------
# inject_mirror
# ---------------------------------------------------------------------------


def test_inject_mirror_returns_shadow():
    model = _make_model()
    shadow = inject_mirror(model, prefix="shadow_")
    assert isinstance(shadow, Shadow)


def test_inject_mirror_registers_shadow_params():
    model = _make_model()
    n_trainable_before = sum(1 for p in model.parameters() if p.requires_grad)
    inject_mirror(model, prefix="shadow_")
    n_total_after = sum(1 for _ in model.parameters())
    assert n_total_after == n_trainable_before * 2


def test_inject_mirror_shadow_params_frozen():
    model = _make_model()
    inject_mirror(model, prefix="shadow_")
    for n, p in model.named_parameters():
        if "shadow_" in n:
            assert not p.requires_grad


def test_inject_mirror_deferred_sync():
    model = _make_model()
    with torch.no_grad():
        model.q_proj.weight.fill_(3.0)
        model.k_proj.weight.fill_(5.0)
    shadow = inject_mirror(model, prefix="shadow_")
    apply_deferred_ops(model)

    for live, shd in shadow.iter_pairs():
        assert torch.allclose(live, shd)


def test_inject_mirror_swap_is_self_inverse():
    model = _make_model()
    with torch.no_grad():
        model.q_proj.weight.fill_(1.0)
        model.k_proj.weight.fill_(2.0)
    shadow = inject_mirror(model, prefix="shadow_")
    apply_deferred_ops(model)

    with torch.no_grad():
        model.q_proj.weight.fill_(10.0)

    shadow.swap_in()
    assert torch.allclose(model.q_proj.weight, torch.tensor(1.0))

    shadow.swap_out()
    assert torch.allclose(model.q_proj.weight, torch.tensor(10.0))


# ---------------------------------------------------------------------------
# apply_deferred_ops — generic
# ---------------------------------------------------------------------------


def test_apply_deferred_ops_noop_on_clean_model():
    model = _make_model()
    apply_deferred_ops(model)
    assert model._deferred_ops == []


def test_apply_deferred_ops_runs_in_order():
    model = _make_model()
    order: list[str] = []
    model._deferred_ops = [
        lambda m: order.append("first"),
        lambda m: order.append("second"),
    ]
    apply_deferred_ops(model)
    assert order == ["first", "second"]
