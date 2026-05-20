"""Tests for :class:`NFTLoRAPolicy`.

Requires ``peft`` — gracefully skipped if the import fails. Exercises:

- dual-adapter install (both ``default`` and ``old`` LoRA adapters present
  on each :class:`peft.tuners.lora.LoraLayer`)
- ``parameters()`` filtering — only ``default``-side LoRA params trainable
- ``with_old_adapter`` context manager — switches and restores
- ``step`` / ``on_rollout_end`` dispatch follows ``ema_update_timing``
- ``_sync_default_to_old`` after ``post_materialize_init`` — old matches default
- EMA update mutates ``old`` adapter weights toward ``default`` per the
  configured decay schedule

CPU-only; no FSDP, no Ray.
"""

from __future__ import annotations

import pytest

peft = pytest.importorskip("peft")

import torch  # noqa: E402  (after importorskip gate)
import torch.nn as nn  # noqa: E402

from diffusionrl.training_new.nft_lora_policy import NFTLoRAPolicy, NFTLoRAPolicyConfig  # noqa: E402


class _TinySource:
    """Minimal Stage-shaped source: exposes ``trainable_module``."""

    def __init__(self, module: nn.Module) -> None:
        self._module = module

    def trainable_module(self) -> nn.Module:
        return self._module

    def post_materialize_init(self) -> None:
        pass


class _TinyModel(nn.Module):
    """nn.Module with named Linear children that peft can target."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


def _build_policy(**cfg_overrides) -> NFTLoRAPolicy:
    cfg = NFTLoRAPolicyConfig(
        rank=4,
        alpha=8,
        dropout=0.0,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        bias="none",
        task_type="FEATURE_EXTRACTION",
        **cfg_overrides,
    )
    return NFTLoRAPolicy(cfg, _TinySource(_TinyModel()))


def _count_lora_layers(model: nn.Module) -> int:
    from peft.tuners.lora import LoraLayer

    return sum(1 for m in model.modules() if isinstance(m, LoraLayer))


def _adapter_names_present(model: nn.Module) -> set[str]:
    """Collect adapter names that any LoraLayer has installed."""
    from peft.tuners.lora import LoraLayer

    names: set[str] = set()
    for m in model.modules():
        if isinstance(m, LoraLayer):
            for matrix in ("lora_A", "lora_B"):
                bank = getattr(m, matrix, None)
                if bank is not None:
                    names.update(bank.keys())
    return names


def test_install_creates_two_adapters() -> None:
    policy = _build_policy()
    assert _count_lora_layers(policy.model) == 4
    assert _adapter_names_present(policy.model) == {"default", "old"}


def test_only_default_adapter_is_trainable() -> None:
    policy = _build_policy()
    # Every trainable param must live under the ``default`` adapter.
    for name, p in policy.named_parameters():
        assert ".default." in name, name
    # ``old`` adapter params exist on the raw model but are frozen.
    for name, p in policy.model.named_parameters():
        if ".old." in name:
            assert not p.requires_grad, name


def test_with_old_adapter_switches_and_restores() -> None:
    policy = _build_policy()
    # Outside context: default is active.
    initial = getattr(policy.model, "active_adapter", None)
    with policy.with_old_adapter():
        active = getattr(policy.model, "active_adapter", None)
        # peft may store active_adapter as str or list[str]; handle both.
        if isinstance(active, list):
            active = active[0]
        assert active == "old"
    # Restored.
    final = getattr(policy.model, "active_adapter", None)
    if isinstance(final, list):
        final = final[0]
    assert final == (initial if not isinstance(initial, list) else initial[0])


def test_post_materialize_syncs_default_to_old() -> None:
    policy = _build_policy()
    # Run post_materialize: should reset both adapters + hard-copy default→old.
    policy.post_materialize_init()
    from peft.tuners.lora import LoraLayer

    for m in policy.model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for matrix in ("lora_A", "lora_B"):
            bank = getattr(m, matrix)
            assert torch.equal(bank["default"].weight, bank["old"].weight)


def test_step_no_op_when_timing_is_rollout_end() -> None:
    """ema_update_timing='rollout_end' means step() should not advance EMA."""
    policy = _build_policy(ema_update_timing="rollout_end", ema_decay=0.5)
    policy.post_materialize_init()

    from peft.tuners.lora import LoraLayer

    # Mutate default so old != default.
    for m in policy.model.modules():
        if isinstance(m, LoraLayer):
            m.lora_A["default"].weight.data.fill_(1.0)

    # Capture old before step.
    snap_before = {}
    for i, m in enumerate(policy.model.modules()):
        if isinstance(m, LoraLayer):
            snap_before[i] = m.lora_A["old"].weight.data.clone()

    policy.step(optimization_step=0)

    # Old should be unchanged (step was a no-op).
    for i, m in enumerate(policy.model.modules()):
        if isinstance(m, LoraLayer):
            assert torch.equal(snap_before[i], m.lora_A["old"].weight.data)


def test_on_rollout_end_runs_ema_update_when_timing_matches() -> None:
    policy = _build_policy(
        ema_update_timing="rollout_end",
        ema_decay_type="constant",
        ema_decay=0.5,
    )
    policy.post_materialize_init()

    from peft.tuners.lora import LoraLayer

    # Make default a known non-zero value; old starts equal to default
    # (post_materialize sync), so after EMA with decay=0.5 the new old
    # should be:  0.5 * old_before + 0.5 * default = 0.5 * default + 0.5 * default
    # — i.e. still equal to default. Mutate default to 2.0 to make the
    # update detectable.
    for m in policy.model.modules():
        if isinstance(m, LoraLayer):
            m.lora_A["default"].weight.data.fill_(2.0)

    # Capture old before update.
    snap_before = {}
    for i, m in enumerate(policy.model.modules()):
        if isinstance(m, LoraLayer):
            snap_before[i] = m.lora_A["old"].weight.data.clone()

    policy.on_rollout_end(step=0)

    # Expected: old_new = 0.5 * old_before + 0.5 * 2.0
    for i, m in enumerate(policy.model.modules()):
        if isinstance(m, LoraLayer):
            expected = 0.5 * snap_before[i] + 0.5 * 2.0
            assert torch.allclose(m.lora_A["old"].weight.data, expected, atol=1e-6)


def test_warmup_decay_starts_at_zero() -> None:
    """ema_decay_type='warmup' with flat_steps>0 should yield decay=0 early
    → hard copy default→old (no slow EMA)."""
    policy = _build_policy(
        ema_update_timing="rollout_end",
        ema_decay_type="warmup",
        ema_flat_steps=5,
        ema_uprate=0.1,
        ema_uphold=0.9,
    )
    policy.post_materialize_init()

    from peft.tuners.lora import LoraLayer

    for m in policy.model.modules():
        if isinstance(m, LoraLayer):
            m.lora_A["default"].weight.data.fill_(3.0)
            m.lora_A["old"].weight.data.fill_(1.0)

    # During warmup-flat phase (step < flat_steps), update_old should hard-copy.
    policy.on_rollout_end(step=0)
    for m in policy.model.modules():
        if isinstance(m, LoraLayer):
            assert torch.allclose(m.lora_A["old"].weight.data, m.lora_A["default"].weight.data)


def test_rejects_invalid_decay_type() -> None:
    with pytest.raises(ValueError, match="ema_decay_type"):
        _build_policy(ema_decay_type="exponential")


def test_rejects_invalid_update_timing() -> None:
    with pytest.raises(ValueError, match="ema_update_timing"):
        _build_policy(ema_update_timing="every_microbatch")


def test_rejects_old_adapter_name_collision() -> None:
    with pytest.raises(ValueError, match="old_adapter_name"):
        _build_policy(old_adapter_name="default")
