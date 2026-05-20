"""Real-weight t2t smoke for HunyuanImage 3.0.

Loads ``tencent/HunyuanImage-3.0`` from a local ckpt with
``device_map="auto"``, runs ``HunyuanImage3Pipeline.generate(task="t2t",
bot_task=...)`` against a single short prompt, and prints the decoded
text.

Usage on pod:

    cd ~/diffusionrl && source .venv/bin/activate && \\
    python scripts/smoke_hunyuan_image3_t2t.py \\
        --ckpt /dockerdata/HunyuanImage-3 \\
        --prompt "What is the capital of France?" \\
        --bot-task auto --max-tokens 64
"""

from __future__ import annotations

import argparse
import time

import torch

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
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


def _build_pipeline(ckpt_path: str) -> HunyuanImage3Pipeline:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[t2t-smoke] loading transformer from {ckpt_path} ...")
    t0 = time.time()
    transformer = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    transformer.eval()
    print(f"[t2t-smoke] transformer loaded in {time.time() - t0:.1f}s")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument(
        "--bot-task",
        default="auto",
        choices=("auto", "image", "think", "recaption", "think_recaption", "img_ratio"),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument(
        "--use-system-prompt",
        default="dynamic",
        help="One of: None | en_vanilla | en_recaption | en_think_recaption | dynamic | custom",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt text (used when --use-system-prompt=custom).",
    )
    args = parser.parse_args()

    pipe = _build_pipeline(args.ckpt)

    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={
            "task": "t2t",
            "ar": {
                "bot_task": args.bot_task,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_system_prompt": args.use_system_prompt,
                "system_prompt": args.system_prompt,
            },
        },
    )

    print(f"[t2t-smoke] prompt={args.prompt!r}  bot_task={args.bot_task}  max_tokens={args.max_tokens}")
    t0 = time.time()
    resp = pipe.generate(req)
    dt = time.time() - t0

    decoded = resp.decoded["text"]
    seg = resp.rollout_traces["text"]
    n_tok = int(seg.lengths.sum().item()) if seg.lengths is not None else 0
    print(f"[t2t-smoke] generated {n_tok} tokens in {dt:.1f}s ({n_tok / max(dt, 1e-3):.2f} tok/s)")
    print(f"[t2t-smoke] decoded: {decoded.texts[0]!r}")


if __name__ == "__main__":
    main()
