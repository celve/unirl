"""Offline analyzer for the wan22 in-vivo temb divergence.

Loads the one-shot tensor dumps written at the first divergent step
(/root/parity_dump/{engine,trainer}_{in,cond}.pt), verifies the forward
inputs really are bitwise-equal, pins WHICH condition_embedder output
diverges, then recomputes the embedder from the raw checkpoint in a clean
process as ground truth. Also probes the two call-context theories that
survive the SHA evidence:

  1. batch-shape GEMM tiling — the same row through a [1,*] vs [8,*] bf16
     GEMM can reduce in different order;
  2. torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
     differing between the engine (vLLM worker) and trainer processes.

Usage (pod, 1 GPU):
  python scripts/wan22_dump_analyze.py \
      --model /root/unirl/models/local/Wan2.2-T2V-A14B-Diffusers \
      --dumps /root/parity_dump
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


def cmp(name: str, a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    if a is None and b is None:
        print(f"  {name:24s} both None")
        return True
    if a is None or b is None:
        print(f"  {name:24s} ONE-SIDED: engine={a is not None} trainer={b is not None}")
        return False
    if a.shape != b.shape:
        print(f"  {name:24s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    eq = torch.equal(a, b)
    if eq:
        print(f"  {name:24s} bitwise-equal  {sha(a)}  {tuple(a.shape)} {a.dtype}")
    else:
        d = (a.float() - b.float()).abs()
        nz = (d > 0).sum().item()
        print(
            f"  {name:24s} DIFF max|Δ|={d.max().item():.3e} nz={nz}/{d.numel()} "
            f"{sha(a)} vs {sha(b)} {tuple(a.shape)} {a.dtype}"
        )
    return eq


def load_cond_embedder(model_root: str, device: torch.device):
    """Build the high-noise expert's WanTimeTextImageEmbedding and load its
    checkpoint weights directly from safetensors (no full model load)."""
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
    from safetensors import safe_open

    sub = os.path.join(model_root, "transformer")
    with open(os.path.join(sub, "config.json")) as f:
        cfg = json.load(f)
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    emb = WanTimeTextImageEmbedding(
        dim=dim,
        time_freq_dim=cfg["freq_dim"],
        time_proj_dim=dim * 6,
        text_embed_dim=cfg["text_dim"],
        image_embed_dim=cfg.get("image_dim"),
    )
    index_path = os.path.join(sub, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = "condition_embedder."
    state = {}
    by_file: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        if k.startswith(prefix):
            by_file.setdefault(fname, []).append(k)
    for fname, keys in by_file.items():
        with safe_open(os.path.join(sub, fname), framework="pt", device="cpu") as f:
            for k in keys:
                state[k[len(prefix):]] = f.get_tensor(k)
    emb.load_state_dict(state, strict=True)
    emb = emb.to(device=device, dtype=torch.bfloat16).eval()
    return emb, state


def run_embedder(emb, t: torch.Tensor, enc: torch.Tensor):
    with torch.no_grad():
        temb, tproj, enc_out, _ = emb(t, enc, None)
    return temb, tproj, enc_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dumps", default="/root/parity_dump")
    args = ap.parse_args()
    dev = torch.device("cuda:0")

    e_in = torch.load(os.path.join(args.dumps, "engine_in.pt"), map_location="cpu")
    t_in = torch.load(os.path.join(args.dumps, "trainer_in.pt"), map_location="cpu")
    e_cond = torch.load(os.path.join(args.dumps, "engine_cond.pt"), map_location="cpu")
    t_cond = torch.load(os.path.join(args.dumps, "trainer_cond.pt"), map_location="cpu")

    print("== forward inputs ==")
    print(f"  engine  t shape={tuple(e_in['t'].shape)} dtype={e_in['t'].dtype} vals={e_in['t'].flatten()[:4].tolist()}")
    print(f"  trainer t shape={tuple(t_in['t'].shape)} dtype={t_in['t'].dtype} vals={t_in['t'].flatten()[:4].tolist()}")
    t_bits_equal = torch.equal(
        e_in["t"].flatten()[0].view(torch.int32) if e_in["t"].dtype == torch.float32 else e_in["t"].flatten()[0],
        t_in["t"].flatten()[0].view(torch.int32) if t_in["t"].dtype == torch.float32 else t_in["t"].flatten()[0],
    )
    print(f"  t[0] bitwise-equal: {t_bits_equal}")
    cmp("enc (row0)", e_in["enc"][0], t_in["enc"][0] if t_in["enc"].shape[0] > 1 else t_in["enc"][0])
    print(f"  engine x {tuple(e_in['x'].shape)}  trainer x {tuple(t_in['x'].shape)}")

    print("== condition_embedder outputs (engine vs trainer, row 0) ==")
    names = ["temb", "timestep_proj", "encoder_hidden_states", "enc_image"]
    for i, n in enumerate(names):
        a = e_cond[i] if i < len(e_cond) else None
        b = t_cond[i] if i < len(t_cond) else None
        if a is not None and b is not None and a.shape[0] != b.shape[0]:
            a, b = a[:1], b[:1]
        cmp(n, a, b)

    print("== clean recompute (fresh process, default flags) ==")
    print(
        "  allow_bf16_reduced_precision_reduction =",
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    emb, _ = load_cond_embedder(args.model, dev)

    def one(tag, t, enc):
        temb, tproj, enc_out = run_embedder(emb, t.to(dev), enc.to(dev))
        print(f"  [{tag}] temb {sha(temb)}  tproj {sha(tproj)}  enc_out {sha(enc_out)}")
        return temb.cpu(), tproj.cpu(), enc_out.cpu()

    # ground truth at each side's native shapes
    c_e = one(f"clean@engine-shape t={tuple(e_in['t'].shape)}", e_in["t"], e_in["enc"])
    c_t = one(f"clean@trainer-shape t={tuple(t_in['t'].shape)}", t_in["t"], t_in["enc"])

    print("== verdicts ==")
    cmp("clean(e-shape) vs ENGINE temb", c_e[0][:1], e_cond[0][:1])
    cmp("clean(t-shape) vs TRAINER temb", c_t[0][:1], t_cond[0][:1])
    cmp("clean e-shape vs t-shape temb", c_e[0][:1], c_t[0][:1])

    print("== flag flip: allow_bf16_reduced_precision_reduction=False ==")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    f_e = one("flagOFF@engine-shape", e_in["t"], e_in["enc"])
    f_t = one("flagOFF@trainer-shape", t_in["t"], t_in["enc"])
    cmp("flagOFF(e) vs ENGINE temb", f_e[0][:1], e_cond[0][:1])
    cmp("flagOFF(t) vs TRAINER temb", f_t[0][:1], t_cond[0][:1])

    for side in ("engine", "trainer"):
        wpath = os.path.join(args.dumps, f"{side}_cond_w.pt")
        if os.path.exists(wpath):
            wd = torch.load(wpath, map_location="cpu")
            print(f"== in-vivo weights vs checkpoint: {side} ==")
            from safetensors import safe_open as _so

            sub = os.path.join(args.model, "transformer")
            with open(os.path.join(sub, "diffusion_pytorch_model.safetensors.index.json")) as f:
                wm = json.load(f)["weight_map"]
            for tag, key in (
                ("time_l1", "condition_embedder.time_embedder.linear_1.weight"),
                ("text_l1", "condition_embedder.text_embedder.linear_1.weight"),
            ):
                with _so(os.path.join(sub, wm[key]), framework="pt", device="cpu") as f:
                    ck_t = f.get_tensor(key)
                iv = wd[tag]
                print(f"  {tag}: in-vivo dtype={iv.dtype}")
                if iv.dtype != ck_t.dtype:
                    rt = iv.to(ck_t.dtype)
                    print(
                        f"    dtype≠ckpt({ck_t.dtype}); round-trip-equal={torch.equal(rt, ck_t)} "
                        f"(True ⇒ lossless upcast of checkpoint)"
                    )
                else:
                    cmp(f"{tag} vs ckpt", iv, ck_t)

    print("== in-vivo weight check: clean linear_1(w) vs dumps? (indirect) ==")
    # If neither clean variant matches a side, that side's WEIGHTS differed in
    # vivo (sync/FSDP effect) — report sinusoid determinism to close the loop.
    from diffusers.models.embeddings import get_timestep_embedding  # noqa: F401

    sin_e = emb.timesteps_proj(e_in["t"].to(dev).flatten())
    sin_t = emb.timesteps_proj(t_in["t"].to(dev).flatten())
    print(f"  sinusoid engine {sha(sin_e)}  trainer {sha(sin_t)}  equal_row0={torch.equal(sin_e[0], sin_t[0])}")


if __name__ == "__main__":
    main()
