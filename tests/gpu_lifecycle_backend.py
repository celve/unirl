#!/usr/bin/env python
"""VeOmniBackend lifecycle micro-test: offload/onload + save/load (P1.8 gate 4).

The trainside pilot recipe never exercises offload or checkpointing, so this
validates them standalone on 1 GPU:

    export QWEN_IMAGE_PATH=/dockerdata/Qwen-Image
    CUDA_VISIBLE_DEVICES=0 python tests/gpu_lifecycle_backend.py /tmp/ckpt_veomni

Sequence: construct -> fingerprint -> offload (assert CPU + GPU freed) ->
onload (assert fingerprints unchanged) -> fake-grad optimizer_step ->
save -> perturb -> load -> assert fingerprints restored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fingerprints(model, k: int = 8) -> dict:
    import torch

    out = {}
    names = sorted(n for n, p in model.named_parameters() if p.ndim >= 2)
    for n in names[:: max(1, len(names) // k)][:k]:
        p = dict(model.named_parameters())[n].data
        if hasattr(p, "full_tensor"):
            p = p.full_tensor()
        out[n] = float(torch.sum(p.detach().float()).item())
    return out


def main(ckpt_dir: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    examples = Path(__file__).resolve().parent.parent / "examples"
    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name="diffusion/qwen_image_trainside_veomni")

    print("[gate4] constructing backend ...", flush=True)
    bundle = instantiate(cfg.bundle)
    backend = instantiate(cfg.backend, bundle=bundle)
    fp0 = _fingerprints(backend.model)

    # -- offload --------------------------------------------------------
    used_before = torch.cuda.memory_allocated()
    backend.offload()
    used_after = torch.cuda.memory_allocated()
    on_gpu = [n for n, p in backend.model.named_parameters() if p.device.type == "cuda"]
    assert not on_gpu, f"params still on cuda after offload: {on_gpu[:4]}"
    freed_gb = (used_before - used_after) / 2**30
    assert freed_gb > 10, f"offload freed only {freed_gb:.1f} GiB (transformer should be >10)"
    print(f"[gate4] offload OK (freed {freed_gb:.1f} GiB)", flush=True)

    # -- onload ---------------------------------------------------------
    backend.onload()
    off_gpu = [n for n, p in backend.model.named_parameters() if p.device.type != "cuda"]
    assert not off_gpu, f"params not back on cuda after onload: {off_gpu[:4]}"
    assert _fingerprints(backend.model) == fp0, "fingerprints changed across offload/onload"
    print("[gate4] onload OK (fingerprints stable)", flush=True)

    # -- optimizer step with fake grads ----------------------------------
    backend.zero_grad()
    for p in backend.model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    norm = backend.optimizer_step(max_grad_norm=1.0)
    assert norm == 0.0, f"zero grads must clip to norm 0, got {norm}"
    print("[gate4] optimizer_step OK (norm=0, machinery exercised)", flush=True)

    # -- save / perturb / load -------------------------------------------
    fp1 = _fingerprints(backend.model)
    backend.save(ckpt_dir)
    backend.randomize_weights_for_smoke(seed=7)
    assert _fingerprints(backend.model) != fp1, "randomize did nothing — perturb step invalid"
    backend.load(ckpt_dir)
    assert _fingerprints(backend.model) == fp1, "fingerprints not restored by load"
    print("[gate4] save/load round-trip OK", flush=True)

    print("GATE4 PASS", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
