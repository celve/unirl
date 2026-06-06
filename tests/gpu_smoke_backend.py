#!/usr/bin/env python
"""1-GPU backend smoke + checksum parity: VeOmniBackend vs FSDPBackend.

Not a pytest module (no ``test_`` prefix) — a standalone GPU-pod script for
the P1.8 validation gates. Each backend run constructs the bundle + backend
from its real recipe, asserts structural invariants, and dumps trainable
counts + parameter fingerprints to JSON; ``compare`` asserts the two runs
loaded identical weights (proving VeOmniBackend's post-parallelize load
lands the same bytes as FSDPBackend's eager load).

Usage (inside the unirl venv on a GPU pod):

    export QWEN_IMAGE_PATH=/dockerdata/Qwen-Image
    python tests/gpu_smoke_backend.py fsdp   /tmp/fp_fsdp.json
    python tests/gpu_smoke_backend.py veomni /tmp/fp_veomni.json
    python tests/gpu_smoke_backend.py compare /tmp/fp_fsdp.json /tmp/fp_veomni.json

Run the two backend modes as separate processes (each owns the process
group / parallel state for its lifetime).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RECIPES = {
    "fsdp": "diffusion/qwen_image_trainside",
    "veomni": "diffusion/qwen_image_trainside_veomni",
}
N_FINGERPRINTS = 12


def _ensure_single_process_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _fingerprint(t) -> dict:
    import torch

    data = t.data if hasattr(t, "data") else t
    if hasattr(data, "full_tensor"):
        data = data.full_tensor()
    f = data.detach().float()
    return {
        "shape": list(f.shape),
        "sum": float(torch.sum(f).item()),
        "abs_sum": float(torch.sum(torch.abs(f)).item()),
    }


def run_backend(mode: str, out_path: str) -> None:
    _ensure_single_process_dist_env()

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    examples = Path(__file__).resolve().parent.parent / "examples"
    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name=RECIPES[mode])

    print(f"[{mode}] building bundle ...", flush=True)
    bundle = instantiate(cfg.bundle)
    print(f"[{mode}] building backend {cfg.backend._target_} ...", flush=True)
    backend = instantiate(cfg.backend, bundle=bundle)

    model = backend.model
    params = dict(model.named_parameters())

    n_trainable = sum(1 for p in params.values() if p.requires_grad)
    n_total = len(params)
    dtypes = {str(p.dtype) for p in params.values() if p.is_floating_point()}
    metas = [n for n, p in params.items() if p.is_meta]

    assert not metas, f"params still on meta after construction: {metas[:5]}"
    assert dtypes == {"torch.bfloat16"}, f"expected pure bf16 master weights, got {dtypes}"
    assert n_trainable > 0, "no trainable params (LoRA injection failed?)"

    # Plain tensor ATTRIBUTES (not params/buffers — e.g. diffusers'
    # QwenEmbedRope complex rope tables) are invisible to to_empty and
    # named_parameters; a meta one only explodes at forward time.
    meta_attrs = [
        f"{mod_name}.{attr}"
        for mod_name, mod in model.named_modules()
        for attr, val in vars(mod).items()
        if isinstance(val, torch.Tensor) and val.is_meta
    ]
    assert not meta_attrs, f"plain tensor attrs still on meta: {meta_attrs[:5]}"

    # Fingerprint a deterministic sample of BASE weights (shared names across
    # both backends): first N sorted non-trainable 2D+ params.
    base_names = sorted(n for n, p in params.items() if not p.requires_grad and p.ndim >= 2)
    sample = base_names[:: max(1, len(base_names) // N_FINGERPRINTS)][:N_FINGERPRINTS]
    fingerprints = {n: _fingerprint(params[n]) for n in sample}

    # And one trainable LoRA fingerprint class: lora_B must be all-zero at
    # init (identity adapter => step-0 importance ratio 1).
    lora_b = [n for n, p in params.items() if p.requires_grad and "lora_b" in n.lower()]
    lora_b_nonzero = []
    for n in lora_b:
        fp = _fingerprint(params[n])
        if fp["abs_sum"] != 0.0:
            lora_b_nonzero.append(n)
    assert not lora_b_nonzero, f"lora_B params not zero-init: {lora_b_nonzero[:5]}"

    result = {
        "mode": mode,
        "n_total": n_total,
        "n_trainable": n_trainable,
        "n_lora_b": len(lora_b),
        "fingerprints": fingerprints,
        "device": str(next(iter(params.values())).device),
        "torch": torch.__version__,
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[{mode}] OK: total={n_total} trainable={n_trainable} lora_B={len(lora_b)} -> {out_path}", flush=True)


def compare(path_a: str, path_b: str) -> None:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())

    failures = []
    for key in ("n_total", "n_trainable", "n_lora_b"):
        if a[key] != b[key]:
            failures.append(f"{key}: {a['mode']}={a[key]} vs {b['mode']}={b[key]}")

    names_a, names_b = set(a["fingerprints"]), set(b["fingerprints"])
    if names_a != names_b:
        failures.append(f"fingerprint name sets differ: {sorted(names_a ^ names_b)[:6]}")
    for n in sorted(names_a & names_b):
        fa, fb = a["fingerprints"][n], b["fingerprints"][n]
        if fa != fb:
            failures.append(f"{n}: {fa} vs {fb}")

    if failures:
        print("CHECKSUM PARITY: FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"CHECKSUM PARITY: PASS ({len(names_a)} fingerprints bit-identical, counts equal)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] in RECIPES:
        run_backend(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
