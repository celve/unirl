"""Real-weight t2i diffuse + replay parity smoke for HunyuanImage 3.0.

Runs a single-prompt t2i request through ``HunyuanImage3Pipeline`` with
``eta>0`` and ``sde_indices=range(num_inference_steps)`` so every step
is stochastic and the trajectory is fully populated. Then reconstructs
:class:`HunyuanImage3DiffusionConditions` from ``resp.conditions`` and
calls ``pipe.diffusion.replay(...)`` against the stored
:class:`LatentSegment`. The returned per-step log-probs must match the
``segment.sde_logp`` captured during sampling — that's the parity claim
GRPO replay relies on.

Usage on pod (after switching the pod's diffusionrl checkout to
LIN-241/main and pip-installing if needed)::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    python scripts/smoke_hunyuan_image3_t2i_replay.py \\
        --ckpt /dockerdata/HunyuanImage-3-Instruct \\
        --prompt "A cute cat sitting on a wooden chair" \\
        --steps 8 --guidance-scale 2.5 --eta 0.3 \\
        --output ./smoke_t2i_replay.png

What "match" means: replay walks the same SDE step kernel as diffuse,
just with ``prev_sample`` provided so the strategy returns the
log-density of the *actually-drawn* sample under the predicted
Gaussian. Same noise prediction, same mean/std → same log-prob.
The only sources of divergence are:

  1. bf16 transformer non-bitwise-determinism across calls (rare; HF
     causal LM is generally deterministic for fixed input on the same
     device).
  2. Trajectory storage in ``trajectory_precision`` (bf16 by default) —
     replay reads the stored bf16 latent so both sides see the same
     bits.

We log per-step abs/rel mismatch and exit non-zero if max-abs
exceeds a tolerance (default ``1e-2`` in fp32 logprob_dtype).
"""

from __future__ import annotations

import argparse
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
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


def _build_pipeline(ckpt_path: str, max_memory_per_gpu: str = "auto") -> HunyuanImage3Pipeline:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[t2i-replay] loading transformer from {ckpt_path} ...")
    t0 = time.time()
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if max_memory_per_gpu != "auto":
        n_gpus = torch.cuda.device_count()
        load_kwargs["max_memory"] = {i: max_memory_per_gpu for i in range(n_gpus)} | {"cpu": "300GiB"}
        print(f"[t2i-replay] forcing balanced placement: max_memory_per_gpu={max_memory_per_gpu} across {n_gpus} GPUs")
    transformer = AutoModelForCausalLM.from_pretrained(ckpt_path, **load_kwargs)
    transformer.eval()
    print(f"[t2i-replay] transformer loaded in {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if getattr(transformer, "_tkwrapper", None) is None:
        transformer.load_tokenizer(tokenizer)

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


def _summarize_mismatch(label: str, sampled: torch.Tensor, replayed: torch.Tensor) -> Tuple[float, float]:
    """Print per-step abs/rel diff and return (max_abs, max_rel)."""
    diff = (sampled - replayed).abs()
    denom = sampled.abs().clamp_min(1e-12)
    rel = diff / denom
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    print(f"[t2i-replay] {label}:")
    print(f"  shape       = {tuple(sampled.shape)}")
    print(f"  sampled[0]  = {sampled[0].tolist()}")
    print(f"  replayed[0] = {replayed[0].tolist()}")
    print(f"  abs_diff[0] = {diff[0].tolist()}")
    print(f"  max_abs     = {max_abs:.3e}")
    print(f"  max_rel     = {max_rel:.3e}")
    return max_abs, max_rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3-Instruct")
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument(
        "--bot-task",
        default="image",
        choices=("auto", "image", "think", "recaption", "think_recaption", "img_ratio"),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument(
        "--eta",
        type=float,
        default=0.3,
        help="Stochastic SDE noise scale (>0). 0.3 keeps "
        "samples close to deterministic ODE while still "
        "exercising the log-prob path.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./smoke_t2i_replay.png")
    parser.add_argument("--tol-abs", type=float, default=1e-2, help="Max absolute log-prob mismatch we tolerate.")
    parser.add_argument(
        "--max-memory-per-gpu",
        default="auto",
        help="Per-GPU weight cap for accelerate (e.g. '25GiB'). 'auto' lets accelerate fill greedily.",
    )
    args = parser.parse_args()

    if args.eta <= 0.0:
        raise SystemExit("--eta must be > 0 for replay parity (eta=0 disables SDE log-probs).")

    pipe = _build_pipeline(args.ckpt, max_memory_per_gpu=args.max_memory_per_gpu)

    sde_indices = list(range(args.steps))
    diffusion_kwargs = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "sde_indices": sde_indices,
        "eta": args.eta,
    }
    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={
            "task": "t2i",
            "bot_task": args.bot_task,
            "diffusion": diffusion_kwargs,
        },
    )

    print(
        f"[t2i-replay] prompt={args.prompt!r}  bot_task={args.bot_task}  "
        f"steps={args.steps}  size={args.height}x{args.width}  "
        f"eta={args.eta}  cfg={args.guidance_scale}"
    )

    # --- 1. Diffuse (sampling) ----------------------------------------
    t0 = time.time()
    resp = pipe.generate(req)
    diffuse_dt = time.time() - t0
    seg = resp.rollout_traces["image"]
    print(
        f"[t2i-replay] diffuse+decode in {diffuse_dt:.1f}s  "
        f"latents={tuple(seg.latents.shape)}  "
        f"sde_logp={tuple(seg.sde_logp.shape)}  "
        f"sde_indices={seg.sde_indices.tolist()}  "
        f"sigmas={tuple(seg.sigmas.shape)}"
    )
    if seg.sde_logp is None or seg.sde_indices is None:
        raise SystemExit(
            "[t2i-replay] segment.sde_logp / sde_indices missing — did eta or sde_indices get overridden somewhere?"
        )

    # --- 2. Save the decoded image (sanity check on the sampling path) ----
    out_pixels = resp.decoded["image"].pixels[0]  # [3, H, W]
    from torchvision.transforms.functional import to_pil_image

    to_pil_image(out_pixels.clamp(0, 1).float().cpu()).save(args.output)
    print(f"[t2i-replay] saved sampled image -> {args.output}")

    # --- 3. Replay (training-engine recompute) ---------------------------
    diff_conds = HunyuanImage3DiffusionConditions.from_dict(resp.conditions)
    params = HunyuanImage3DiffusionParams(**diffusion_kwargs)

    t0 = time.time()
    with torch.no_grad():
        replay_result = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
    replay_logp = replay_result.log_probs
    replay_dt = time.time() - t0
    print(
        f"[t2i-replay] replay all steps in {replay_dt:.1f}s "
        f"(per-step ~{replay_dt / max(1, args.steps):.2f}s)  "
        f"replay_logp={tuple(replay_logp.shape)}"
    )

    # --- 4. Compare ---------------------------------------------------
    sampled = seg.sde_logp.detach().to(replay_logp.dtype).to(replay_logp.device)
    max_abs, max_rel = _summarize_mismatch("full SDE replay", sampled, replay_logp)

    # --- 5. Subset replay (sanity check on step_indices selection) -------
    if args.steps >= 4:
        subset = [1, args.steps - 1]
        with torch.no_grad():
            sub_result = pipe.diffusion.replay(diff_conds, segment=seg, params=params, step_indices=subset)
        sub_logp = sub_result.log_probs
        sub_sampled = sampled[:, [sde_indices.index(i) for i in subset]]
        sub_abs, _ = _summarize_mismatch(f"subset replay step_indices={subset}", sub_sampled, sub_logp)
    else:
        sub_abs = 0.0

    if max_abs > args.tol_abs:
        print(f"[t2i-replay] FAIL: full-replay max_abs={max_abs:.3e} > tol_abs={args.tol_abs:.3e}")
        sys.exit(1)
    print(f"[t2i-replay] PASS — full max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, subset max_abs={sub_abs:.3e}")


if __name__ == "__main__":
    main()
