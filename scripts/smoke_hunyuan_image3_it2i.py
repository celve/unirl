"""Real-weight it2i smoke for HunyuanImage 3.0.

Loads ``tencent/HunyuanImage-3.0`` with ``device_map="auto"``, encodes a
local image into both VAE latents (cond_vae) and ViT cond features,
threads them through the unified-MM diffusion forward, decodes the
result via the 3D-VAE, and writes the edited PNG.

Memory note: the unified MM forward at 1024² with cond+gen images
(~9k tokens) materializes a ~17 GiB MoE dispatch_mask per layer. The
default ``device_map="auto"`` greedy packer fills GPU 0 to ~85 GB of
weights, leaving <10 GB for activations -> OOM.
``--max-memory-per-gpu 15GiB`` forces accelerate to spread weights
across all 8 GPUs and offload ~12 layers to CPU; GPU 0 ends up with
~10 modules instead of ~12, freeing the activation room. Confirmed
working at 1024² CFG-off on 1×8 H20: 1 step in ~22s, 8 steps in ~63s.

Usage on pod:

    cd ~/diffusionrl && source .venv/bin/activate && \\
    python scripts/smoke_hunyuan_image3_it2i.py \\
        --ckpt /dockerdata/HunyuanImage-3 \\
        --image /root/smoke_hunyuan_image3.png \\
        --prompt "make it grayscale" \\
        --steps 8 --guidance-scale 1.0 \\
        --max-memory-per-gpu 15GiB \\
        --output /root/smoke_it2i.png
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from diffusionrl.models_new.hunyuan_image3.ar import HunyuanImage3ARStage
from diffusionrl.models_new.hunyuan_image3.bundle import HunyuanImage3Bundle
from diffusionrl.models_new.hunyuan_image3.diffusion import (
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
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.rollout_req import RolloutReq


def _build_pipeline(ckpt_path: str, max_memory_per_gpu: str = "auto") -> HunyuanImage3Pipeline:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[it2i-smoke] loading transformer from {ckpt_path} ...")
    t0 = time.time()
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if max_memory_per_gpu != "auto":
        n_gpus = torch.cuda.device_count()
        load_kwargs["max_memory"] = {i: max_memory_per_gpu for i in range(n_gpus)} | {"cpu": "300GiB"}
        print(f"[it2i-smoke] forcing balanced placement: max_memory_per_gpu={max_memory_per_gpu} across {n_gpus} GPUs")
    transformer = AutoModelForCausalLM.from_pretrained(ckpt_path, **load_kwargs)
    transformer.eval()
    print(f"[it2i-smoke] transformer loaded in {time.time() - t0:.1f}s")
    if hasattr(transformer, "hf_device_map"):
        from collections import Counter

        loc_counter: Counter = Counter()
        for v in transformer.hf_device_map.values():
            loc_counter[str(v)] += 1
        print(f"[it2i-smoke] device_map summary: {dict(loc_counter)}")

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


def _load_image_as_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    from torchvision.transforms.functional import to_tensor

    t = to_tensor(img)  # [3, H, W] in [0, 1]
    return t.unsqueeze(0)  # [1, 3, H, W]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="make it grayscale")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/root/smoke_it2i.png")
    parser.add_argument(
        "--max-memory-per-gpu",
        default="auto",
        help="Per-GPU weight cap for accelerate (e.g. '25GiB'). 'auto' lets accelerate fill greedily.",
    )
    args = parser.parse_args()

    pipe = _build_pipeline(args.ckpt, max_memory_per_gpu=args.max_memory_per_gpu)
    pixels = _load_image_as_tensor(args.image)

    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={
            "text": Texts(texts=[args.prompt]),
            "image": Images(pixels=pixels),
        },
        stage_params={
            "task": "it2i",
            "diffusion": {
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "sde_indices": [],
                "eta": 0.0,
            },
        },
    )

    print(
        f"[it2i-smoke] prompt={args.prompt!r}  image={args.image}  steps={args.steps}  size={args.height}x{args.width}"
    )
    t0 = time.time()
    resp = pipe.generate(req)
    dt = time.time() - t0
    out_pixels = resp.decoded["image"].pixels[0]  # [3, H, W]
    print(f"[it2i-smoke] diffuse+decode in {dt:.1f}s")

    from torchvision.transforms.functional import to_pil_image

    out_pil = to_pil_image(out_pixels.clamp(0, 1).float().cpu())
    out_pil.save(args.output)
    print(f"[it2i-smoke] saved -> {args.output}")


if __name__ == "__main__":
    main()
