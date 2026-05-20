"""Per-decoder-layer activation checkpointing experiment for HI3 replay.

The previous experiment (``profile_hi3_activation_ckpt.py``) wrapped
``step_with_logp`` — for ``--steps=1`` that's a single call that fans
out into all 32 decoder layers. Checkpointing one unit of that size
means recomputation has to materialise all 32 layers' activations
again, which is exactly what OOMs. So we measured "no ckpt" twice.

This script applies activation checkpointing at the **per-decoder-layer**
boundary by monkey-patching each ``FSDPHunyuanImage3DecoderLayer.forward``
to wrap the call in ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``.
During backward, only one layer's activations live at a time —
roughly ``activations_per_layer + model_weights`` at peak instead of
``32 * activations_per_layer + weights``.

Phases:

  P1. no_grad replay (no ckpt)    — baseline forward time.
  P2. no_grad replay (with ckpt)  — overhead check (should be ~ P1).
  P3. grad train step (with ckpt) — must succeed; record peak + dt.

We skip the no-ckpt grad OOM phase here — its result is already
documented in ``profile_hi3_replay_memory.py`` output (OOM at layer ~10).
Replaying it would just pollute the P3 measurement with carryover state.

Usage::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/profile_hi3_ckpt_per_layer.py \\
        --ckpt /dockerdata/HunyuanImage-3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, List

import torch
import torch.distributed as dist
import torch.utils.checkpoint as ckpt

from diffusionrl.models_new.hunyuan_image3.conditions import (
    HunyuanImage3DiffusionConditions,
)
from diffusionrl.models_new.hunyuan_image3.config import HunyuanImage3PipelineConfig
from diffusionrl.models_new.hunyuan_image3.diffusion import HunyuanImage3DiffusionParams
from diffusionrl.models_new.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.training_new.fsdp_policy import FSDPPolicyConfig
from diffusionrl.training_new.lora_policy import LoRAPolicyConfig
from diffusionrl.training_new.policy import Policy, compose_policy, walk_source_chain
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq

_GB = 1024.0**3


def _r(rank: int, s: str) -> None:
    if rank == 0:
        print(s, flush=True)


def _mem(label: str, rank: int) -> None:
    if rank != 0:
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / _GB
    peak = torch.cuda.max_memory_allocated() / _GB
    reserved = torch.cuda.memory_reserved() / _GB
    print(
        f"# [{label}] allocated={alloc:.3f}GB peak={peak:.3f}GB reserved={reserved:.3f}GB",
        flush=True,
    )


def _locate_decoder_layers(transformer) -> List[Any]:
    """Find the decoder ModuleList. HI3 follows HF AutoModel layout."""
    if hasattr(transformer, "model") and hasattr(transformer.model, "layers"):
        return list(transformer.model.layers)
    if hasattr(transformer, "layers"):
        return list(transformer.layers)
    raise SystemExit("Could not locate decoder layers on the transformer")


def _patch_layers_with_checkpoint(layers: List[Any]) -> List[Any]:
    """Replace each layer.forward with a checkpoint-wrapped version.

    Returns the list of original forwards so the caller can restore.
    Uses ``use_reentrant=False`` (required for FSDP2 compatibility).
    Closes over kwargs (rope_cache, attention_mask, position_ids, ...);
    HI3 decoder kwargs are all non-grad tensors / Python ints, so they
    don't need to participate in the autograd graph.
    """
    originals: List[Any] = []
    for layer in layers:
        orig = layer.forward
        originals.append(orig)

        def make_wrapped(orig_fwd):
            def wrapped(*args, **kwargs):
                # Close over kwargs so only *args are autograd-tracked.
                def fn(*a):
                    return orig_fwd(*a, **kwargs)

                return ckpt.checkpoint(fn, *args, use_reentrant=False)

            return wrapped

        layer.forward = make_wrapped(orig)
    return originals


def _restore_layers(layers: List[Any], originals: List[Any]) -> None:
    for layer, orig in zip(layers, originals):
        layer.forward = orig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument("--bot-task", default="image")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    _r(rank, f"# rank={rank}/{world_size} device={device}")

    # --- Build pipeline + LoRA + FSDP policy stack -----------------
    config = HunyuanImage3PipelineConfig(
        pretrained_model_ckpt_path=args.ckpt,
        model_precision="bf16",
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        device=device,
    )
    t0 = time.time()
    pipe = HunyuanImage3Pipeline.from_meta_config(config, strategy=FlowSDEStrategy())
    _r(rank, f"# meta-init in {time.time() - t0:.1f}s")

    configs: list = [
        LoRAPolicyConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
        FSDPPolicyConfig(
            cpu_offload=False,
            param_dtype="bf16",
            mixed_precision=True,
            fsdp_mode="full",
            reshard_after_forward=True,
        ),
    ]
    policy: Policy = compose_policy(pipe.diffusion, configs)
    chain = " → ".join(type(p).__name__ for p in walk_source_chain(policy))
    _r(rank, f"# policy chain: {chain}")

    t0 = time.time()
    pipe.bundle.materialize(device=device, with_aux=("vae",))
    _r(rank, f"# bundle.materialize in {time.time() - t0:.1f}s")
    policy.post_materialize_init()
    _mem("post-materialize", rank)

    # Locate decoder layers ONCE — used for both ckpt patch + report.
    transformer = pipe.bundle.transformer
    layers = _locate_decoder_layers(transformer)
    _r(rank, f"# located {len(layers)} decoder layers (type={type(layers[0]).__name__})")

    # --- Rollout once to get a real RolloutResp --------------------
    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={
            "task": "t2i",
            "bot_task": args.bot_task,
            "diffusion": {
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "sde_indices": list(range(args.steps)),
                "eta": args.eta,
            },
        },
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    resp = pipe.generate(req)
    _r(rank, f"# pipe.generate in {time.time() - t0:.1f}s")

    diff_conds = HunyuanImage3DiffusionConditions.from_dict(resp.conditions)
    params = HunyuanImage3DiffusionParams(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        sde_indices=list(range(args.steps)),
        eta=args.eta,
    )
    seg = resp.rollout_traces["image"]
    advantages = torch.tensor([0.5], dtype=torch.float32, device=device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    _r(rank, f"# optimizer over {sum(p.numel() for p in trainable) / 1e6:.1f}M LoRA params")

    # ====================================================================
    # P1. no_grad forward (no ckpt) — baseline forward time
    # ====================================================================
    _r(rank, "\n=== P1: no_grad replay (no ckpt) ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem("P1-pre", rank)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_p1 = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
    torch.cuda.synchronize()
    p1_dt = time.time() - t0
    _mem(f"P1-post (dt={p1_dt:.2f}s)", rank)
    p1_peak = torch.cuda.max_memory_allocated() / _GB
    del out_p1

    # ====================================================================
    # Patch every decoder layer with per-layer activation ckpt
    # ====================================================================
    originals = _patch_layers_with_checkpoint(layers)
    _r(rank, f"# patched {len(originals)} decoder layers with ckpt(use_reentrant=False)")

    # ====================================================================
    # P2. no_grad forward (with ckpt) — overhead check vs P1
    # ====================================================================
    _r(rank, "\n=== P2: no_grad replay (with per-layer ckpt) ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem("P2-pre", rank)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_p2 = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
    torch.cuda.synchronize()
    p2_dt = time.time() - t0
    _mem(f"P2-post (dt={p2_dt:.2f}s)", rank)
    p2_peak = torch.cuda.max_memory_allocated() / _GB
    del out_p2

    # ====================================================================
    # P3. grad replay + loss.backward + opt.step  (with per-layer ckpt)
    # ====================================================================
    _r(rank, "\n=== P3: grad train step (with per-layer ckpt) ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem("P3-pre", rank)
    torch.cuda.synchronize()
    t0 = time.time()
    p3_status = "?"
    p3_peak = 0.0
    p3_fwd_bwd_dt = 0.0
    p3_opt_dt = 0.0
    try:
        result = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
        log_probs = result.log_probs
        loss = -(advantages.unsqueeze(1) * log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd_done = time.time()
        p3_fwd_bwd_dt = t_bwd_done - t0
        optimizer.step()
        torch.cuda.synchronize()
        t_opt_done = time.time()
        p3_opt_dt = t_opt_done - t_bwd_done
        p3_status = f"OK loss={loss.item():.4f}"
    except torch.cuda.OutOfMemoryError as e:
        p3_status = f"OOM: {str(e).splitlines()[0]}"
        _mem("P3-OOM", rank)
    p3_dt = time.time() - t0
    p3_peak = torch.cuda.max_memory_allocated() / _GB
    _r(rank, f"# P3 status: {p3_status} dt={p3_dt:.2f}s")
    _mem("P3-end", rank)

    _restore_layers(layers, originals)

    # --- Summary report --------------------------------------------
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("EXPERIMENT SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"P1 (no_grad, no ckpt):       {p1_dt:.2f}s   peak={p1_peak:.2f}GB", flush=True)
        print(
            f"P2 (no_grad, w/ ckpt):       {p2_dt:.2f}s   peak={p2_peak:.2f}GB   "
            f"overhead vs P1: {(p2_dt / p1_dt - 1) * 100:+.1f}%",
            flush=True,
        )
        print(f"P3 (grad,    w/ ckpt):       {p3_status}", flush=True)
        if p3_status.startswith("OK"):
            print(
                f"  fwd+bwd: {p3_fwd_bwd_dt:.2f}s   opt: {p3_opt_dt:.2f}s   total: {p3_dt:.2f}s   peak={p3_peak:.2f}GB",
                flush=True,
            )
            print("\n# verdict: per-layer activation checkpointing RESOLVES the OOM.", flush=True)
            print(
                f"# perf cost of full train step = {p3_fwd_bwd_dt / p1_dt:.2f}× the no-ckpt no_grad forward", flush=True
            )
            print(
                f"# (a no-OOM no-ckpt train step would be ~3× P1 — 1 fwd + 2× fwd-equivalent bwd; we got {p3_fwd_bwd_dt / p1_dt:.2f}×, so ckpt adds 1 extra forward pass)",
                flush=True,
            )
        else:
            print(f"  peak before OOM = {p3_peak:.2f}GB", flush=True)
            print("\n# verdict: per-layer activation checkpointing did NOT resolve the OOM.", flush=True)

        print("\n# DONE", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
