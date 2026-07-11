"""Dual-expert LoRA wire-name routing for the wan22 vLLM-Omni sync (CPU).

The engine's ``DiffusionLoRAManager`` activates adapters by full
component-qualified module name (``transformer.blocks.N...`` /
``transformer_2.blocks.N...``, module ``to_out`` not ``to_out.0``), while the
trainer's checksum verify is NAME-INSENSITIVE — a wrong substitution map
verifies green while every engine layer silently runs base weights. These
tests pin the exact wire-name set so that failure mode is caught on CPU.

Covers:
- ``apply_name_substitutions``: routing, unmatched-pattern raise, collision
  raise (pure dict logic);
- end-to-end PEFT → ``extract_lora_tensors`` → substitutions on a miniature
  ``WanDualTransformer`` composite, asserting the exact engine wire names the
  wan22 parity recipe's sync block must produce.

Run: ``pytest tests/test_wan22_lora_name_map.py -q``
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unirl.distributed.weight_sync.lora.base import apply_name_substitutions

# The wan22 parity recipe's sync block, verbatim.
WAN22_SUBSTITUTIONS = {
    "high_noise.": "transformer.",
    "low_noise.": "transformer_2.",
    ".to_out.0.": ".to_out.",
}
WAN22_TARGETS = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
]


def _canonical(expert: str, module: str, ab: str) -> str:
    return f"{expert}.blocks.0.{module}.{ab}.weight"


def test_substitutions_route_dual_experts():
    tensors = {
        _canonical(expert, module, ab): torch.zeros(1)
        for expert in ("high_noise", "low_noise")
        for module in ("attn1.to_q", "attn1.to_out.0", "attn2.to_v")
        for ab in ("lora_A", "lora_B")
    }
    routed = apply_name_substitutions(tensors, WAN22_SUBSTITUTIONS)
    assert set(routed) == {
        f"{component}.blocks.0.{module}.{ab}.weight"
        for component in ("transformer", "transformer_2")
        for module in ("attn1.to_q", "attn1.to_out", "attn2.to_v")
        for ab in ("lora_A", "lora_B")
    }
    assert len(routed) == len(tensors)


def test_substitution_unmatched_pattern_raises():
    tensors = {_canonical("high_noise", "attn1.to_q", "lora_A"): torch.zeros(1)}
    with pytest.raises(RuntimeError, match="matched no extracted key"):
        apply_name_substitutions(tensors, {"low_noise.": "transformer_2."})


def test_substitution_collision_raises():
    tensors = {
        "high_noise.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(1),
        "low_noise.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(1),
    }
    with pytest.raises(RuntimeError, match="colliding keys"):
        apply_name_substitutions(tensors, {"high_noise.": "transformer.", "low_noise.": "transformer."})


# --------------------------------------------------------------------------
# End-to-end: PEFT names on a miniature dual composite → engine wire names.
# --------------------------------------------------------------------------


class _TinyAttn(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _TinyAttn()
        self.attn2 = _TinyAttn()


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock()])


def test_peft_extract_names_end_to_end():
    peft = pytest.importorskip("peft")

    from unirl.models.wan22.bundle import WanDualTransformer
    from unirl.utils.peft_merge import extract_lora_tensors

    composite = WanDualTransformer(high_noise=_TinyExpert(), low_noise=_TinyExpert())
    lora_cfg = peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=list(WAN22_TARGETS)
    )
    model = peft.get_peft_model(composite, lora_cfg)

    tensors = extract_lora_tensors(model, param_prefix="", adapter_name="default", dtype=torch.bfloat16)
    routed = apply_name_substitutions(tensors, WAN22_SUBSTITUTIONS)

    expected_modules = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out",
                        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"]
    expected = {
        f"{component}.blocks.0.{module}.{ab}.weight"
        for component in ("transformer", "transformer_2")
        for module in expected_modules
        for ab in ("lora_A", "lora_B")
    }
    assert set(routed) == expected, (
        f"engine wire-name mismatch:\n  missing={sorted(expected - set(routed))[:4]}"
        f"\n  extra={sorted(set(routed) - expected)[:4]}"
    )
