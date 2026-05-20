"""Minimal t2i / it2i smoke against HunyuanImage-3.0-Instruct using
its own ``generate_image(prompt, image=None, ...)`` unified entrypoint.
Bypasses our diffusionrl wiring -- just verifies the upstream Instruct
model can do both modes on real weights.
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/bj/HunyuanImage-3-Instruct")
    parser.add_argument(
        "--mode",
        default="t2i",
        choices=("t2i", "it2i"),
        help="t2i: text only. it2i: text + cond image.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to cond image (required for it2i).",
    )
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bot-task", default="image", help="image / think / recaption / think_recaption / auto")
    parser.add_argument("--output", required=True, help="Where to save PNG.")
    args = parser.parse_args()

    if args.mode == "it2i" and not args.image:
        raise SystemExit("--image required for it2i mode")

    from transformers import AutoModelForCausalLM

    print(f"[img-direct] loading {args.ckpt} ...")
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print(f"[img-direct] loaded in {time.time() - t0:.1f}s")

    if not hasattr(m.config, "model_version") or m.config.model_version is None:
        m.config.model_version = "hunyuan-image-3"
    m.load_tokenizer(args.ckpt)

    cond_pil = Image.open(args.image).convert("RGB") if args.image else None

    print(
        f"[img-direct] mode={args.mode} prompt={args.prompt!r} "
        f"steps={args.steps} guidance={args.guidance_scale} "
        f"size={args.height}x{args.width} bot_task={args.bot_task}"
    )

    # Override scheduler params from CLI by patching generation_config
    # (matches what generate_image internally reads).
    m.generation_config.diff_infer_steps = args.steps
    m.generation_config.diff_guidance_scale = args.guidance_scale

    t0 = time.time()
    with torch.no_grad():
        out = m.generate_image(
            prompt=args.prompt,
            image=cond_pil,
            seed=args.seed,
            image_size=f"{args.height}x{args.width}",
            bot_task=args.bot_task,
        )
    dt = time.time() - t0
    print(f"[img-direct] generate_image returned in {dt:.1f}s, type={type(out).__name__}")

    # Unwrap (text, images) tuple → list[PIL.Image] → first PIL.Image.
    if isinstance(out, tuple) and len(out) == 2:
        text_part, img_part = out
        if text_part:
            print(f"[img-direct] text-side output: {text_part!r}")
        out = img_part
    if isinstance(out, list):
        out = out[0] if out else None
    if isinstance(out, Image.Image):
        out.save(args.output)
        print(f"[img-direct] saved -> {args.output} (size={out.size})")
    else:
        print(f"[img-direct] unexpected output type: {type(out)} value={out!r}")


if __name__ == "__main__":
    main()
