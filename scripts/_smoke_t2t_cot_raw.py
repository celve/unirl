"""Like smoke_hunyuan_image3_t2t.py but dumps raw token IDs +
``skip_special_tokens=False`` decoded output so we can see the
``<think>`` / ``<recaption>`` markers and the CoT body.
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

    print(f"[cot] loading from {ckpt_path} ...")
    transformer = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if getattr(transformer, "_tkwrapper", None) is None:
        transformer.load_tokenizer(tokenizer)
    bundle = HunyuanImage3Bundle(
        transformer=transformer,
        vae=None,
        vit=None,
        tokenizer=tokenizer,
        scheduler=None,
        dtype=torch.bfloat16,
        device=transformer.model.wte.weight.device,
        pretrained_path=ckpt_path,
    )
    return HunyuanImage3Pipeline(
        bundle=bundle,
        text_embed=HunyuanImage3TextEmbedStage(bundle),
        diffusion=HunyuanImage3DiffusionStage(
            model=bundle,
            step=HunyuanImage3DiffusionStep(),
            strategy=FlowSDEStrategy(),
            autocast_precision="bf16",
            trajectory_precision="bf16",
            logprob_precision="fp32",
            vae_scale_factor=16,
            latent_channels=32,
        ),
        vae_decode=HunyuanImage3VAEDecodeStage(bundle),
        vae_encode=HunyuanImage3VAEEncodeStage(bundle),
        ar=HunyuanImage3ARStage(model=bundle),
        vit_encode=HunyuanImage3VitEncodeStage(bundle),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument(
        "--bot-task", default="think", choices=("auto", "image", "think", "recaption", "think_recaption", "img_ratio")
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--use-system-prompt", default="dynamic")
    parser.add_argument(
        "--no-stops", action="store_true", help="Disable bot_task stop tokens so the model runs full max-tokens."
    )
    args = parser.parse_args()

    pipe = _build_pipeline(args.ckpt)
    ar_kwargs = {
        "bot_task": args.bot_task,
        "max_tokens": args.max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 1024,
        "use_system_prompt": args.use_system_prompt,
    }
    if args.no_stops:
        ar_kwargs["stop_token_ids"] = [-1]  # token id never emitted
    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={"task": "t2t", "ar": ar_kwargs},
    )

    t0 = time.time()
    resp = pipe.generate(req)
    dt = time.time() - t0
    seg = resp.rollout_traces["text"]
    n_tok = int(seg.lengths.sum().item())
    print(f"[cot] generated {n_tok} tokens in {dt:.1f}s ({n_tok / max(dt, 1e-3):.2f} tok/s)")

    a = int(seg.cu_seqlens[0].item())
    b = int(seg.cu_seqlens[1].item())
    raw_ids = seg.tokens[a:b].tolist()
    print(f"[cot] raw token IDs (first 32): {raw_ids[:32]} ...")
    print("[cot] decoded WITH special tokens:")
    print(repr(pipe.bundle.tokenizer.decode(raw_ids, skip_special_tokens=False)))
    print("[cot] decoded WITHOUT special tokens:")
    print(repr(pipe.bundle.tokenizer.decode(raw_ids, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
