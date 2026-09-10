#!/usr/bin/env python
"""Audit a VeRL-Omni native checkpoint export against its PEFT/diffusers output.

E2-R's admission depends on the baseline's exported adapter, not a guessed native
checkpoint path: the runbook requires "its verified exported PEFT adapter". This
script is the verification. It compares the tensors the exporter wrote against the
tensors the native checkpoint holds, so a silently mis-keyed or dtype-truncated
export fails here rather than becoming a published reference number.

Read-only with respect to both inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _load_state_dict(path: Path) -> Dict[str, "object"]:
    """Load a safetensors or torch checkpoint into a flat name→tensor mapping."""
    import torch

    if path.is_dir():
        candidates = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"{path}: no .safetensors/.bin/.pt found")
        merged: Dict[str, object] = {}
        for candidate in candidates:
            merged.update(_load_state_dict(candidate))
        return merged

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("state_dict", "module", "model"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            blob = blob[key]
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(blob).__name__}")
    return blob


def _lora_tensors(state: Dict[str, object]) -> Dict[str, object]:
    """The LoRA A/B tensors only — the exporter's subject."""
    return {name: tensor for name, tensor in state.items() if "lora_" in name.lower()}


def _normalize(name: str) -> str:
    """Strip the wrapper prefixes and suffixes that differ across the two formats."""
    for prefix in ("module.", "_orig_mod.", "base_model.model.", "transformer.", "model."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in (".default.weight", ".weight"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("lora_A", "lora_a").replace("lora_B", "lora_b")


def compare(
    native: Dict[str, object],
    exported: Dict[str, object],
    *,
    tolerance: float,
) -> Tuple[bool, List[str], Dict[str, object]]:
    """Compare the two LoRA tensor sets by normalized name, then by value."""
    import torch

    native_lora = {_normalize(k): v for k, v in _lora_tensors(native).items()}
    exported_lora = {_normalize(k): v for k, v in _lora_tensors(exported).items()}

    findings: List[str] = []
    if not native_lora:
        findings.append("native checkpoint holds no lora_* tensors — wrong path, or a full-weight checkpoint")
    if not exported_lora:
        findings.append("export holds no lora_* tensors — the exporter produced no adapter")

    only_native = sorted(set(native_lora) - set(exported_lora))
    only_exported = sorted(set(exported_lora) - set(native_lora))
    if only_native:
        findings.append(f"{len(only_native)} tensors dropped by the export: {only_native[:6]}")
    if only_exported:
        findings.append(f"{len(only_exported)} tensors invented by the export: {only_exported[:6]}")

    shared = sorted(set(native_lora) & set(exported_lora))
    max_diff = 0.0
    mismatched: List[str] = []
    dtype_changes: List[str] = []
    for name in shared:
        a, b = native_lora[name], exported_lora[name]
        if tuple(a.shape) != tuple(b.shape):
            findings.append(f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}")
            continue
        if a.dtype != b.dtype:
            dtype_changes.append(f"{name}: {a.dtype} -> {b.dtype}")
        diff = float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            mismatched.append(f"{name} (max|Δ|={diff:.3e})")

    if mismatched:
        findings.append(f"{len(mismatched)} tensors differ beyond {tolerance:.1e}: {mismatched[:6]}")
    if all(float(exported_lora[n].to(torch.float32).abs().max().item()) == 0.0 for n in shared) and shared:
        findings.append("every exported tensor is all-zero — the export folded an uninitialized adapter")

    report: Dict[str, object] = {
        "native_lora_tensors": len(native_lora),
        "exported_lora_tensors": len(exported_lora),
        "shared_tensors": len(shared),
        "only_in_native": only_native,
        "only_in_export": only_exported,
        "max_abs_diff": max_diff,
        "tolerance": tolerance,
        "dtype_changes": dtype_changes,
        "mismatched_tensors": mismatched,
        "findings": findings,
    }
    return not findings, findings, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True, help="VeRL-Omni native checkpoint dir or file")
    parser.add_argument("--exported-adapter", type=Path, required=True, help="PEFT/diffusers adapter the exporter wrote")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here as well as stdout")
    args = parser.parse_args(argv)

    native = _load_state_dict(args.native_checkpoint)
    exported = _load_state_dict(args.exported_adapter)
    passed, findings, report = compare(native, exported, tolerance=args.tolerance)

    report["native_checkpoint"] = str(args.native_checkpoint)
    report["exported_adapter"] = str(args.exported_adapter)
    report["passed"] = passed

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)

    if not passed:
        print("\nEXPORT AUDIT FAILED — E2-R is inadmissible until these are resolved:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print("\nEXPORT AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
