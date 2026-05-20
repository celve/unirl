"""Cross-engine rollout/replay parity smoke for HunyuanImage 3.0.

Single-process flow:

    Phase A: vllm-omni rollout
        Build VLLMOmniRolloutEngine, generate one t2i request, capture
        the LatentSegment (latents [B,T+1,C,H,W], sde_logp [B,T],
        sigmas [T+1]) and decoded image.
    Phase B: shutdown vllm-omni
        engine.shutdown() kills the worker subprocesses; CUDA contexts
        in those processes get reclaimed by the kernel, freeing GPU
        memory for Phase C.
    Phase C: load training pipe
        Wires HunyuanImage3Bundle + diffusion stage exactly like
        scripts/smoke_hunyuan_image3_t2i_replay.py — same checkpoint,
        same precision knobs, same FlowSDE strategy.
    Phase D: build conds from prompt
        bundle.build_t2i_inputs(prompts, neg_strs, ...) → fused MM
        condition. Mirrors models_new/hunyuan_image3/modes/t2i.py:63-74.
        bot_task="image" on training side ↔ "vanilla" on rollout side
        (both produce the upstream t2i_vanilla / en_vanilla template).
    Phase E: replay
        pipe.diffusion.replay(conds, segment, params) re-runs each SDE
        step with prev_sample provided, returning the closed-form
        Gaussian log-density. Same primitive validated bit-perfect
        against same-engine forward in
        docs/hunyuan_image3_diffusion_replay_smoke.md.
    Phase F: compare + write artifacts
        max_abs, mean_abs, max_rel between segment.sde_logp (rollout)
        and replay_logp (training engine). PASS if max_abs ≤ tol_abs
        (default 1e-2 fp32 logprob, matching the same-engine smoke).

Why bot_task=vanilla:
    The CoT think pass (bot_task=think) is stochastic at temperature
    0.6 — rollout and replay-side AR passes would diverge, breaking
    parity. Vanilla skips the AR think pass; prompt → conds is fully
    deterministic. Replay with think requires re-embedding rollout AR
    tokens (the "PR4" path noted in models_new/hunyuan_image3/ar.py:21-23).

Why a single tolerance not bit-perfect:
    Same-engine replay is bit-perfect (max_abs=0.0) because the same
    transformer call + same precision + same conds are reused. Cross-
    engine has two different forward paths (vllm-omni's
    HunyuanImage3Pipeline.forward vs our predict_noise) sharing the
    same upstream transformer code. Differences come from autocast
    dtype boundaries and CFG batching layout — small but nonzero in
    fp32 logprob. 1e-2 matches the existing same-engine smoke
    tolerance bound; bumping past that would mask real bugs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Tuple

import torch

from diffusionrl.models_new.hunyuan_image3.ar import HunyuanImage3ARStage
from diffusionrl.models_new.hunyuan_image3.bundle import HunyuanImage3Bundle
from diffusionrl.models_new.hunyuan_image3.conditions import (
    HunyuanImage3DiffusionConditions,
)
from diffusionrl.models_new.hunyuan_image3.diffusion import (
    HunyuanImage3DiffusionParams,
    HunyuanImage3DiffusionStage,
    HunyuanImage3DiffusionStep,
)
from diffusionrl.models_new.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from diffusionrl.models_new.hunyuan_image3.text_embed import HunyuanImage3TextEmbedStage
from diffusionrl.models_new.hunyuan_image3.vae import (
    HunyuanImage3VAEDecodeStage,
    HunyuanImage3VAEEncodeStage,
)
from diffusionrl.models_new.hunyuan_image3.vit_encode import HunyuanImage3VitEncodeStage
from diffusionrl.rollout.engine.vllm_omni import (
    VLLMOmniEngineConfig,
    VLLMOmniRolloutEngine,
)
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq

# ---------------------------------------------------------------------------
# Phase C — training pipe builder (mirrors smoke_hunyuan_image3_t2i_replay.py)
# ---------------------------------------------------------------------------


def _build_training_pipe(ckpt_path: str, max_memory_per_gpu: str = "auto") -> HunyuanImage3Pipeline:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[xengine] loading transformer from {ckpt_path} ...")
    t0 = time.time()
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if max_memory_per_gpu != "auto":
        n_gpus = torch.cuda.device_count()
        load_kwargs["max_memory"] = {i: max_memory_per_gpu for i in range(n_gpus)} | {"cpu": "300GiB"}
    transformer = AutoModelForCausalLM.from_pretrained(ckpt_path, **load_kwargs)
    transformer.eval()
    print(f"[xengine] transformer loaded in {time.time() - t0:.1f}s")

    # HunyuanImage-3.0-Instruct's load_tokenizer reads config.model_version,
    # but the released config.json doesn't define it. Inject a default so the
    # cached modeling.py at line 1793 doesn't AttributeError. The kwarg is
    # accepted-then-ignored by HunyuanImage3TokenizerFast.__init__(*args, **kwargs).
    if not hasattr(transformer.config, "model_version") or transformer.config.model_version is None:
        transformer.config.model_version = "hunyuan-image-3"

    # Two checkpoint variants (mirrors scripts/smoke_hunyuan_image3_i2t.py:64-81):
    #  - Base: load_tokenizer expects an existing AutoTokenizer, wraps as _tkwrapper.
    #  - Instruct: load_tokenizer expects a *path*, sets _tokenizer; we alias _tkwrapper.
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if getattr(transformer, "_tkwrapper", None) is not None:
        pass  # Already wired (rare, but possible after some loaders)
    elif getattr(transformer, "_tokenizer", None) is not None:
        transformer._tkwrapper = transformer._tokenizer
    else:
        try:
            transformer.load_tokenizer(ckpt_path)  # Instruct path-style
            transformer._tkwrapper = transformer._tokenizer
        except Exception:
            transformer.load_tokenizer(tokenizer)  # Base instance-style
            if getattr(transformer, "_tkwrapper", None) is None:
                transformer._tkwrapper = transformer._tokenizer

    vae = getattr(transformer, "vae", None) or getattr(transformer, "vae_model", None)
    vit = (
        getattr(transformer, "vit", None)
        or getattr(transformer, "vision_tower", None)
        or getattr(transformer, "siglip", None)
        or getattr(transformer, "vision_model", None)
    )

    embed_device = transformer.model.wte.weight.device
    bundle = HunyuanImage3Bundle(
        transformer=transformer,
        vae=vae,
        vit=vit,
        tokenizer=tokenizer,
        scheduler=None,
        dtype=torch.bfloat16,
        device=embed_device,
        pretrained_path=ckpt_path,
    )
    text_embed = HunyuanImage3TextEmbedStage(bundle)
    step = HunyuanImage3DiffusionStep()
    diffusion = HunyuanImage3DiffusionStage(
        model=bundle,
        step=step,
        strategy=FlowSDEStrategy(),
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        vae_scale_factor=16,
        latent_channels=32,
    )
    return HunyuanImage3Pipeline(
        bundle=bundle,
        text_embed=text_embed,
        diffusion=diffusion,
        vae_decode=HunyuanImage3VAEDecodeStage(bundle),
        vae_encode=HunyuanImage3VAEEncodeStage(bundle),
        ar=HunyuanImage3ARStage(model=bundle),
        vit_encode=HunyuanImage3VitEncodeStage(bundle),
    )


# ---------------------------------------------------------------------------
# Phase F — diff summary
# ---------------------------------------------------------------------------


def _summarize_mismatch(sampled: torch.Tensor, replayed: torch.Tensor) -> Tuple[float, float, float]:
    diff = (sampled - replayed).abs()
    denom = sampled.abs().clamp_min(1e-12)
    rel = diff / denom
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    max_rel = float(rel.max().item())
    print(f"[xengine] sampled[0]  = {sampled[0].tolist()}")
    print(f"[xengine] replayed[0] = {replayed[0].tolist()}")
    print(f"[xengine] abs_diff[0] = {diff[0].tolist()}")
    print(f"[xengine] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  max_rel={max_rel:.3e}")
    return max_abs, mean_abs, max_rel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--model-path",
        default="/dockerdata/HunyuanImage-3-Instruct",
        help="Local checkpoint path. Use /dockerdata/... on pod (cephfs is "
        "too slow per docs/hunyuan_image3_diffusion_replay_smoke.md).",
    )
    parser.add_argument("--prompt", default="a red apple on a wooden table")
    parser.add_argument(
        "--rollout-bot-task",
        default="vanilla",
        choices=("vanilla", "think", "recaption"),
        help="bot_task on rollout side. 'vanilla' is the only one that "
        "round-trips deterministically without AR re-embedding.",
    )
    parser.add_argument(
        "--replay-bot-task",
        default="image",
        choices=("image", "auto", "think", "recaption", "think_recaption", "img_ratio"),
        help="bot_task on training side. 'image' matches vllm-omni's t2i_vanilla preset per bundle.py:214.",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tol-abs",
        type=float,
        default=1e-2,
        help="Max absolute log-prob mismatch allowed in fp32 logprob_dtype.",
    )
    parser.add_argument("--out-dir", default="./smoke_xengine")
    parser.add_argument(
        "--max-memory-per-gpu",
        default="auto",
        help="Per-GPU weight cap for accelerate (e.g. '25GiB').",
    )
    args = parser.parse_args()

    if args.eta <= 0.0:
        raise SystemExit("[xengine] --eta must be > 0 (eta=0 disables SDE log-probs; no replay parity to check).")

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase A — vllm-omni rollout
    # -----------------------------------------------------------------------
    print(
        f"[xengine] Phase A: vllm-omni rollout — prompt={args.prompt!r}  "
        f"bot_task={args.rollout_bot_task}  steps={args.steps}  "
        f"eta={args.eta}  cfg={args.guidance_scale}  seed={args.seed}"
    )
    cfg = VLLMOmniEngineConfig(
        model_path=args.model_path,
        modality="t2i",
        default_eta=float(args.eta),
        default_num_inference_steps=int(args.steps),
        default_guidance_scale=float(args.guidance_scale),
        default_height=int(args.height),
        default_width=int(args.width),
    )
    engine = VLLMOmniRolloutEngine(cfg)
    try:
        req = RolloutReq(
            sample_ids=["s0"],
            group_ids=["g0"],
            primitives={"text": Texts(texts=[args.prompt])},
            stage_params={
                "diffusion": {
                    "height": int(args.height),
                    "width": int(args.width),
                    "num_inference_steps": int(args.steps),
                    "guidance_scale": float(args.guidance_scale),
                    "eta": float(args.eta),
                    "seed": int(args.seed),
                },
                "ar": {"max_tokens": 2048, "temperature": 0.6},
                "bot_task": args.rollout_bot_task,
            },
        )
        t0 = time.time()
        resp = engine.generate(req)
        rollout_dt = time.time() - t0
        print(f"[xengine] rollout done in {rollout_dt:.1f}s")
    finally:
        # Phase B — shutdown frees vllm-omni worker GPU memory.
        t0 = time.time()
        engine.shutdown()
        print(f"[xengine] Phase B: engine.shutdown done in {time.time() - t0:.1f}s")

    seg = resp.rollout_traces.get("image")
    if seg is None or seg.sde_logp is None or seg.latents is None:
        raise SystemExit("[xengine] rollout produced no usable image segment (missing latents or sde_logp). Aborting.")
    if seg.sigmas is None:
        raise SystemExit(
            "[xengine] segment.sigmas is None — drain_trajectory didn't "
            "surface sigmas. Did the sde_scheduler/pipeline fix land?"
        )
    # sigmas may be 1D [T+1] (preferred) or 2D [B, T+1] if a stale
    # rollout path is still emitting batched timesteps. Take the first
    # row in either case for the [0, 1] sanity check.
    sig_flat = seg.sigmas.flatten() if seg.sigmas.dim() == 1 else seg.sigmas[0]
    sig_first = float(sig_flat[0].item())
    sig_last = float(sig_flat[-1].item())
    print(
        f"[xengine] segment latents={tuple(seg.latents.shape)}  "
        f"sde_logp={tuple(seg.sde_logp.shape)}  "
        f"sigmas={tuple(seg.sigmas.shape)}  "
        f"sigmas[0]={sig_first:.4f}  "
        f"sigmas[-1]={sig_last:.4f}  "
        f"sde_indices={seg.sde_indices.tolist()}"
    )

    # Sanity: sigmas should be a [0,1] schedule, not 1000-scale timesteps.
    if sig_first > 5.0:
        raise SystemExit(
            f"[xengine] sigmas[0]={sig_first} > 5.0 — looks like "
            f"1000-scale timesteps slipped through. Check the "
            f"sde_scheduler.drain_trajectory + pipeline.py + response.py "
            f"sigmas plumbing."
        )

    # Save the decoded image as a sanity-check artifact.
    final_pixels = resp.decoded["image"].pixels[0]  # [3, H, W] in [0, 1]
    from torchvision.transforms.functional import to_pil_image

    final_path = os.path.join(args.out_dir, "final.png")
    to_pil_image(final_pixels.clamp(0, 1).float().cpu()).save(final_path)
    print(f"[xengine] saved rollout image -> {final_path}")

    # Save rollout snapshot for offline re-analysis.
    seg_state = {
        "latents": seg.latents.detach().cpu(),
        "sigmas": seg.sigmas.detach().cpu(),
        "sde_logp": seg.sde_logp.detach().cpu(),
        "sde_indices": seg.sde_indices.detach().cpu(),
        "indices": seg.indices.detach().cpu() if seg.indices is not None else None,
    }
    torch.save(
        {"prompt": args.prompt, "segment": seg_state, "args": vars(args)},
        os.path.join(args.out_dir, "rollout.pt"),
    )

    # -----------------------------------------------------------------------
    # Phase C — load training pipe on the now-freed GPUs
    # -----------------------------------------------------------------------
    print("[xengine] Phase C: loading training pipe ...")
    pipe = _build_training_pipe(args.model_path, max_memory_per_gpu=args.max_memory_per_gpu)

    # -----------------------------------------------------------------------
    # Phase D — build conds from the same prompt
    # -----------------------------------------------------------------------
    print(f"[xengine] Phase D: building conds (replay_bot_task={args.replay_bot_task})")
    cfg_on = float(args.guidance_scale) > 1.0
    neg_strs = ["" for _ in [args.prompt]] if cfg_on else None
    mm = pipe.bundle.build_t2i_inputs(
        [args.prompt],
        neg_strs,
        height=int(args.height),
        width=int(args.width),
        bot_task=args.replay_bot_task,
    )
    diff_conds = HunyuanImage3DiffusionConditions(
        fused=mm["fused"],
        tokenizer_output=mm["tokenizer_output"],
    )
    print(f"[xengine] conds.fused.input_ids.shape={tuple(diff_conds.fused.input_ids.shape)}")

    # The rollout segment came back from vllm-omni's worker IPC on CPU
    # (the orchestrator detaches tensors before shipping them to the
    # parent). Move latents/sigmas/sde_logp to the same device as
    # diff_conds.fused.input_ids so replay's forward_denoiser sees one
    # device throughout. Without this, sde/kernels.py:222's mixed
    # noise_pred (cuda) * sigma (cpu) trips a device mismatch.
    target_device = diff_conds.fused.input_ids.device
    print(f"[xengine] moving segment tensors to {target_device}")
    seg.latents = seg.latents.to(target_device)
    seg.sigmas = seg.sigmas.to(target_device)
    seg.sde_logp = seg.sde_logp.to(target_device)
    if seg.sde_indices is not None:
        seg.sde_indices = seg.sde_indices.to(target_device)
    if seg.indices is not None:
        seg.indices = seg.indices.to(target_device)

    # -----------------------------------------------------------------------
    # Phase E — replay
    # -----------------------------------------------------------------------
    params = HunyuanImage3DiffusionParams(
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed),
        sde_indices=list(range(int(args.steps))),
        eta=float(args.eta),
    )
    print(f"[xengine] Phase E: replay {args.steps} steps ...")
    t0 = time.time()
    with torch.no_grad():
        replay_result = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
    replay_logp = replay_result.log_probs
    replay_dt = time.time() - t0
    print(
        f"[xengine] replay done in {replay_dt:.1f}s  "
        f"(per-step ~{replay_dt / max(1, args.steps):.2f}s)  "
        f"replay_logp={tuple(replay_logp.shape)}"
    )

    # -----------------------------------------------------------------------
    # Phase F — compare
    # -----------------------------------------------------------------------
    sampled = seg.sde_logp.detach().to(replay_logp.dtype).to(replay_logp.device)
    if sampled.shape != replay_logp.shape:
        raise SystemExit(
            f"[xengine] shape mismatch: sampled={tuple(sampled.shape)} vs "
            f"replayed={tuple(replay_logp.shape)} — replay didn't cover the "
            f"same step set as the rollout."
        )
    if not torch.isfinite(replay_logp).all():
        raise SystemExit("[xengine] replay_logp has non-finite entries — check transformer forward / SDE math.")

    max_abs, mean_abs, max_rel = _summarize_mismatch(sampled, replay_logp)
    passed = max_abs <= args.tol_abs

    summary = {
        "prompt": args.prompt,
        "rollout_bot_task": args.rollout_bot_task,
        "replay_bot_task": args.replay_bot_task,
        "steps": int(args.steps),
        "eta": float(args.eta),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "tol_abs": float(args.tol_abs),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "passed": bool(passed),
        "rollout_seconds": float(rollout_dt),
        "replay_seconds": float(replay_dt),
        "sigmas_first": float(seg.sigmas[0]),
        "sigmas_last": float(seg.sigmas[-1]),
        "latents_shape": list(seg.latents.shape),
        "sde_logp_shape": list(seg.sde_logp.shape),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[xengine] wrote summary -> {summary_path}")

    torch.save(
        {"replay_logp": replay_logp.cpu(), "summary": summary},
        os.path.join(args.out_dir, "replay.pt"),
    )

    if passed:
        print(f"[xengine] PASS — max_abs={max_abs:.3e} ≤ tol_abs={args.tol_abs:.3e}")
        return 0
    print(f"[xengine] FAIL — max_abs={max_abs:.3e} > tol_abs={args.tol_abs:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
