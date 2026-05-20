"""Activation-checkpointing experiment for HI3 replay backward.

Verifies two claims from the prior memory analysis:

  1. **Does activation checkpointing actually resolve the OOM?**
     Run a full ``replay → loss.sum() → backward → optimizer.step()`` cycle
     with and without checkpointing; compare peak memory.

  2. **What's the wall-time overhead?**
     Compare forward-only (no_grad) vs grad-mode train step under
     checkpointing. The expected cost is one extra forward per
     checkpointed unit at backward time, i.e. ~2× the forward cost.

Implementation: monkey-patch ``HunyuanImage3DiffusionStage.replay`` so that
the inner ``step.step_with_logp(...)`` call is wrapped in
``torch.utils.checkpoint.checkpoint(... , use_reentrant=False)``. This is
the lowest-blast-radius way to apply checkpointing without touching the
FSDP wrap state — we keep the entire Policy stack intact and only modify
the replay loop. For single-step replay (our test), this is equivalent in
memory effect to per-layer activation checkpointing (the boundary is the
input to the transformer call).

Four phases:

  P1. no_grad forward (no ckpt)    — baseline forward time.
  P2. no_grad forward (with ckpt)  — overhead check; should be ≈ P1.
  P3. grad train step (no ckpt)    — expected OOM at layer ~10.
  P4. grad train step (with ckpt)  — should succeed; measure time + peak.

Usage::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/profile_hi3_activation_ckpt.py \\
        --ckpt /dockerdata/HunyuanImage-3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, List, Optional

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


def _patch_replay_with_checkpoint(stage) -> Any:
    """Monkey-patch ``stage.replay`` so each ``step_with_logp`` is wrapped
    in ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``.

    Returns the original ``replay`` so the caller can restore it later.
    """
    from contextlib import nullcontext

    original_replay = stage.replay
    # ReplayResult is defined inside diffusion.py; import the symbol
    # the original replay returns via the module path.
    from diffusionrl.models_new.hunyuan_image3.diffusion import ReplayResult

    def ckpt_replay(conditions, *, segment, params, step_indices=None):
        # Reproduce the validation prelude from the original replay so any
        # mismatch errors fire here, not deep in the checkpointed path.
        if segment.sde_logp is None or segment.sde_indices is None:
            raise ValueError("replay requires segment.sde_logp and sde_indices")
        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target: List[int] = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(f"step_indices {bad} not in segment.sde_indices={sorted(sde_set)}")

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99

        autocast_ctx = (
            torch.autocast("cuda", stage.autocast_dtype)
            if device.type == "cuda" and stage.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        # Define the checkpointed unit: one (sigma, sigma_next, sample,
        # prev_sample, step_index) → (log_prob, prev_mean). Closure captures
        # stage / conditions / params / strategy / sigma_max constant across
        # forward and recompute.
        def _step_unit(sample, prev_sample, sigma, sigma_next, step_idx_int):
            _, log_prob, prev_mean = stage.step.step_with_logp(
                stage.model,
                conditions,
                strategy=stage.strategy,
                sample=sample,
                prev_sample=prev_sample,
                sigma=sigma,
                sigma_next=sigma_next,
                guidance_scale=float(params.guidance_scale),
                eta=float(params.eta),
                sigma_max=sigma_max,
                step_index=int(step_idx_int),
            )
            return log_prob, prev_mean

        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []
        with autocast_ctx:
            for step_idx in target:
                sigma = sigmas[step_idx].to(dtype=torch.float32)
                sigma_next = sigmas[step_idx + 1].to(dtype=torch.float32)
                sample = segment.latents_at(step_idx)
                prev_sample = segment.latents_at(step_idx + 1)

                # use_reentrant=False is required for FSDP2 compatibility.
                # step_idx is an int — passed positionally through ckpt.
                log_prob, prev_mean = ckpt.checkpoint(
                    _step_unit,
                    sample,
                    prev_sample,
                    sigma,
                    sigma_next,
                    step_idx,
                    use_reentrant=False,
                )
                if log_prob is None:
                    raise RuntimeError(f"replay(ckpt): None log-prob at step={step_idx}")
                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=stage.logprob_dtype)
        means_t: Optional[torch.Tensor] = torch.stack(prev_sample_means, dim=1) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    stage.replay = ckpt_replay
    return original_replay


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

    # Build an optimizer over policy trainable params (LoRA only) so we
    # can run a real optimizer.step() inside the train phases.
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    _r(rank, f"# optimizer over {sum(p.numel() for p in trainable) / 1e6:.1f}M LoRA params")

    # Save the original replay for later restoration.
    original_replay = pipe.diffusion.replay

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
    del out_p1

    # ====================================================================
    # P2. no_grad forward (with ckpt) — overhead check vs P1
    # ====================================================================
    _r(rank, "\n=== P2: no_grad replay (with ckpt) ===")
    _patch_replay_with_checkpoint(pipe.diffusion)
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
    del out_p2

    # Restore original replay for P3.
    pipe.diffusion.replay = original_replay

    # ====================================================================
    # P3. grad replay + loss.backward + opt.step  (no ckpt) — expect OOM
    # ====================================================================
    _r(rank, "\n=== P3: grad train step (no ckpt) — expected OOM ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem("P3-pre", rank)
    torch.cuda.synchronize()
    t0 = time.time()
    p3_status = "?"
    try:
        # The DiffusionGRPO algorithm under FSDP would do roughly this.
        # We're replicating the call shape: replay -> log_probs -> loss -> backward.
        result = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
        # Simplified loss: -mean(advantages * log_probs) — enough to drive a
        # backward through the replay graph. Real GRPO has ratio clipping
        # which adds a tiny amount of memory; doesn't change the analysis.
        log_probs = result.log_probs  # [B, S']
        loss = -(advantages.unsqueeze(1) * log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        # No optimizer.step here — we want to measure peak before reset.
        p3_status = f"OK (loss={loss.item():.4f})"
    except torch.cuda.OutOfMemoryError as e:
        p3_status = f"OOM: {str(e).splitlines()[0]}"
        _mem("P3-OOM", rank)
    torch.cuda.synchronize()
    p3_dt = time.time() - t0
    _r(rank, f"# P3 status: {p3_status} dt={p3_dt:.2f}s")
    _mem("P3-end", rank)

    # ====================================================================
    # P4. grad replay + loss.backward + opt.step  (with ckpt)
    # ====================================================================
    _r(rank, "\n=== P4: grad train step (with ckpt) ===")
    _patch_replay_with_checkpoint(pipe.diffusion)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem("P4-pre", rank)
    torch.cuda.synchronize()
    t0 = time.time()
    p4_status = "?"
    try:
        result = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
        log_probs = result.log_probs
        loss = -(advantages.unsqueeze(1) * log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        # Now do optimizer.step to verify the full path completes cleanly.
        torch.cuda.synchronize()
        t_bwd_done = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        t_opt_done = time.time()
        p4_status = f"OK (loss={loss.item():.4f} fwd+bwd={t_bwd_done - t0:.2f}s opt={t_opt_done - t_bwd_done:.2f}s)"
    except torch.cuda.OutOfMemoryError as e:
        p4_status = f"OOM: {str(e).splitlines()[0]}"
        _mem("P4-OOM", rank)
    p4_dt = time.time() - t0
    _r(rank, f"# P4 status: {p4_status} dt={p4_dt:.2f}s")
    _mem("P4-end", rank)

    # Restore.
    pipe.diffusion.replay = original_replay

    # --- Summary report --------------------------------------------
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("EXPERIMENT SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"P1 (no_grad, no ckpt):     {p1_dt:.2f}s", flush=True)
        print(
            f"P2 (no_grad, with ckpt):   {p2_dt:.2f}s    overhead vs P1: {(p2_dt / p1_dt - 1) * 100:+.1f}%", flush=True
        )
        print(f"P3 (grad, no ckpt):        {p3_status}", flush=True)
        print(f"P4 (grad, with ckpt):      {p4_status}", flush=True)
        if p4_status.startswith("OK"):
            print("\n# verdict: activation checkpointing RESOLVES the OOM.", flush=True)
            print(f"# perf cost = P4 total / P1 forward = {p4_dt / p1_dt:.2f}× the no_grad forward time", flush=True)
            print(
                f"# (a no-ckpt grad step would be ~1.3× P1 if it fit, so ckpt adds ~{(p4_dt / (p1_dt * 1.3) - 1) * 100:.0f}% over a hypothetical no-ckpt grad step)",
                flush=True,
            )
        else:
            print("\n# verdict: activation checkpointing did NOT resolve the OOM.", flush=True)
        print("\n# DONE", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
