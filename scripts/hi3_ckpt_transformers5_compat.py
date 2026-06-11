#!/usr/bin/env python3
"""Patch a HunyuanImage-3 (base) checkpoint's trust_remote_code files for
transformers 5.x compatibility, so the unirl bundle's standalone
``HunyuanImage3Pipeline.generate()`` path runs end-to-end.

The checkpoint's vendored modeling/image-processor was written against
transformers 4.x; on transformers 5.10.x three API drifts break the bundle's
direct generate path (found via scripts/hi3_modality_smoke.py, LIN-415):

  A. hunyuan.py KV-cache: transformers 5.x ``StaticLayer.lazy_initialization``
     now requires ``(key_states, value_states)``; the checkpoint calls it with
     ``key_states`` only  -> TypeError on AR generation (t2t/i2t).
  B. image_processor.py: the Siglip2 vision processor returns list-valued
     ``pixel_values`` in transformers 5.x; ``.squeeze(0)`` then fails. Forcing
     ``return_tensors="pt"`` restores tensor outputs (i2t/it2i).
  C. hunyuan.py HunyuanImage3ForCausalMM.forward(): has no **kwargs and rejects
     the ``rope_image_info`` kwarg the unirl diffusion stage passes. The forward
     derives rope from ``custom_pos_emb`` (also passed), so accept+ignore it
     (t2i/it2i).

Idempotent: re-running is a no-op once patched. Operates on a WRITABLE copy of
the checkpoint (e.g. /dockerdata/HunyuanImage-3), not the read-only ceph share.

Usage:
    python hi3_ckpt_transformers5_compat.py [CKPT_DIR]   # default /dockerdata/HunyuanImage-3
After patching, clear the trust_remote_code cache so the patched files reload:
    rm -rf ~/.cache/huggingface/modules/transformers_modules/HunyuanImage*
"""
import os
import sys

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/dockerdata/HunyuanImage-3"


def patch(path: str, old: str, new: str, label: str) -> bool:
    if not os.path.exists(path):
        print(f"[compat] {label}: MISSING {path} — skip")
        return False
    s = open(path).read()
    if new in s and old not in s:
        print(f"[compat] {label}: already patched")
        return True
    n = s.count(old)
    if n == 0:
        print(f"[compat] {label}: ANCHOR NOT FOUND in {os.path.basename(path)} — manual review needed")
        return False
    open(path, "w").write(s.replace(old, new))
    print(f"[compat] {label}: patched {n}x in {os.path.basename(path)}")
    return True


H = os.path.join(CKPT, "hunyuan.py")
IP = os.path.join(CKPT, "image_processor.py")
print(f"[compat] patching checkpoint at {CKPT}")

ok = True

# A — KV-cache lazy_initialization signature
ok &= patch(
    H,
    "self.layers[layer_idx].lazy_initialization(key_states)",
    "self.layers[layer_idx].lazy_initialization(key_states, value_states)",
    "A/kv-cache lazy_initialization",
)

# B — Siglip2 image processor list -> tensor
ok &= patch(
    IP,
    "inputs = self.vision_encoder_processor(image)",
    'inputs = self.vision_encoder_processor(image, return_tensors="pt")',
    "B/image-proc return_tensors",
)

# C — forward() accept+ignore rope_image_info
ok &= patch(
    H,
    "            first_step: Optional[bool] = None,\n            # for gen image",
    "            first_step: Optional[bool] = None,\n"
    "            rope_image_info: Optional[Any] = None,  # accept+ignore: rope comes from custom_pos_emb\n"
    "            # for gen image",
    "C/forward accept rope_image_info",
)

print("[compat] done" + ("" if ok else " (WITH UNRESOLVED ANCHORS — review above)"))
sys.exit(0 if ok else 3)
