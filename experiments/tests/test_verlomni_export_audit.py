"""E0 gate: the VeRL-Omni export audit rejects the ways an export goes wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.audit_verlomni_export import compare  # noqa: E402


def _native(scale: float = 1.0):
    return {
        "module.transformer.blocks.0.attn.lora_A.default.weight": torch.full((4, 8), 0.5 * scale),
        "module.transformer.blocks.0.attn.lora_B.default.weight": torch.full((8, 4), 0.25 * scale),
    }


def _exported(scale: float = 1.0):
    return {
        "base_model.model.blocks.0.attn.lora_a.weight": torch.full((4, 8), 0.5 * scale),
        "base_model.model.blocks.0.attn.lora_b.weight": torch.full((8, 4), 0.25 * scale),
    }


def test_a_faithful_export_passes():
    passed, findings, report = compare(_native(), _exported(), tolerance=1e-5)
    assert passed, findings
    assert report["shared_tensors"] == 2


def test_a_dropped_tensor_fails():
    exported = _exported()
    exported.pop("base_model.model.blocks.0.attn.lora_b.weight")
    passed, findings, _ = compare(_native(), exported, tolerance=1e-5)
    assert not passed
    assert any("dropped by the export" in f for f in findings)


def test_changed_values_fail():
    passed, findings, _ = compare(_native(), _exported(scale=2.0), tolerance=1e-5)
    assert not passed
    assert any("differ beyond" in f for f in findings)


def test_an_all_zero_export_fails():
    exported = {name: torch.zeros_like(t) for name, t in _exported().items()}
    passed, findings, _ = compare(_native(), exported, tolerance=1e-5)
    assert not passed
    assert any("all-zero" in f for f in findings)


def test_a_shape_change_fails():
    exported = _exported()
    exported["base_model.model.blocks.0.attn.lora_a.weight"] = torch.full((4, 16), 0.5)
    passed, findings, _ = compare(_native(), exported, tolerance=1e-5)
    assert not passed
    assert any("shape" in f for f in findings)


def test_a_dtype_change_is_reported_but_tolerated_within_tolerance():
    exported = {name: t.to(torch.bfloat16) for name, t in _exported().items()}
    passed, _, report = compare(_native(), exported, tolerance=1e-2)
    assert passed
    assert len(report["dtype_changes"]) == 2


def test_a_full_weight_checkpoint_is_rejected_as_the_wrong_path():
    native = {"module.transformer.blocks.0.attn.weight": torch.ones(4, 4)}
    passed, findings, _ = compare(native, _exported(), tolerance=1e-5)
    assert not passed
    assert any("no lora_* tensors" in f for f in findings)
