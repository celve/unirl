"""Real-weight i2t smoke for HunyuanImage 3.0.

Loads ``tencent/HunyuanImage-3.0`` with ``device_map="auto"``, encodes a
local image into ViT cond features via the canonical
``HunyuanImage3VitEncodeStage.encode_for_cond_vit`` path, runs the
unified MM transformer in gen_text mode with ``<img>`` markers spliced
into the prompt, and prints the decoded caption / answer.

Usage on pod:

    cd ~/diffusionrl && source .venv/bin/activate && \\
    python scripts/smoke_hunyuan_image3_i2t.py \\
        --ckpt /dockerdata/HunyuanImage-3 \\
        --image /path/to/cat.jpg \\
        --prompt "Describe this image briefly." \\
        --bot-task auto --max-tokens 96
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


def _build_pipeline(ckpt_path: str) -> HunyuanImage3Pipeline:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[i2t-smoke] loading transformer from {ckpt_path} ...")
    t0 = time.time()
    transformer = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    transformer.eval()
    print(f"[i2t-smoke] transformer loaded in {time.time() - t0:.1f}s")

    # HunyuanImage-3.0-Instruct's load_tokenizer reads config.model_version
    # but the released config.json doesn't define it -- inject a default.
    if not hasattr(transformer.config, "model_version") or transformer.config.model_version is None:
        transformer.config.model_version = "hunyuan-image-3"

    # Two checkpoint variants:
    #  - Base (HunyuanImage-3.0): load_tokenizer expects an existing AutoTokenizer instance,
    #    wraps it as ``_tkwrapper``.
    #  - Instruct (HunyuanImage-3.0-Instruct): load_tokenizer expects a *path*,
    #    instantiates HunyuanImage3TokenizerFast as ``_tokenizer``.
    if getattr(transformer, "_tkwrapper", None) is not None:
        # Already loaded
        tokenizer = transformer._tkwrapper
    elif getattr(transformer, "_tokenizer", None) is not None:
        tokenizer = transformer._tokenizer
    else:
        try:
            transformer.load_tokenizer(ckpt_path)  # Instruct path-style
            tokenizer = transformer._tokenizer
            transformer._tkwrapper = tokenizer  # alias for our code
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
            transformer.load_tokenizer(tokenizer)  # Base instance-style

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
    parser.add_argument("--image", required=True, help="Path to a local image.")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument(
        "--bot-task",
        default="auto",
        choices=("auto", "image", "think", "recaption", "think_recaption", "img_ratio"),
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--use-system-prompt", default="dynamic")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--stop-token-ids",
        default=None,
        help="Comma-separated token IDs to stop on (e.g. '127957,128024,128026' "
        "for HunyuanImage-3.0-Instruct's chat-mode i2t stops "
        "[<|endoftext|>,</think>,</answer>]). Overrides bot_task-derived stops.",
    )
    args = parser.parse_args()

    pipe = _build_pipeline(args.ckpt)
    pixels = _load_image_as_tensor(args.image)

    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={
            "text": Texts(texts=[args.prompt]),
            "image": Images(pixels=pixels),
        },
        stage_params={
            "task": "i2t",
            "ar": {
                "bot_task": args.bot_task,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_system_prompt": args.use_system_prompt,
                "system_prompt": args.system_prompt,
                **({"stop_token_ids": [int(t) for t in args.stop_token_ids.split(",")]} if args.stop_token_ids else {}),
            },
        },
    )

    print(
        f"[i2t-smoke] image={args.image}  prompt={args.prompt!r}  "
        f"bot_task={args.bot_task}  max_tokens={args.max_tokens}"
    )
    t0 = time.time()
    resp = pipe.generate(req)
    dt = time.time() - t0

    decoded = resp.decoded["text"]
    seg = resp.rollout_traces["text"]
    n_tok = int(seg.lengths.sum().item()) if seg.lengths is not None else 0
    print(f"[i2t-smoke] generated {n_tok} tokens in {dt:.1f}s ({n_tok / max(dt, 1e-3):.2f} tok/s)")
    print(f"[i2t-smoke] decoded: {decoded.texts[0]!r}")


if __name__ == "__main__":
    main()
