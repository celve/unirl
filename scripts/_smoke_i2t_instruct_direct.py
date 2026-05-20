"""Minimal i2t test against HunyuanImage-3.0-Instruct using its own
``prepare_model_inputs`` + ``generate`` API. Bypasses our diffusionrl
wiring entirely -- just verifies the upstream Instruct model can do
chat-style i2t end-to-end.
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/bj/HunyuanImage-3-Instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--bot-task", default="auto")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    print(f"[i2t-direct] loading {args.ckpt} ...")
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print(f"[i2t-direct] loaded in {time.time() - t0:.1f}s")

    # Inject model_version so load_tokenizer doesn't crash.
    if not hasattr(m.config, "model_version") or m.config.model_version is None:
        m.config.model_version = "hunyuan-image-3"
    m.load_tokenizer(args.ckpt)

    pil_image = Image.open(args.image).convert("RGB")

    # Build the upstream model inputs. The Instruct's prepare_model_inputs
    # handles chat-template + cond-image VAE/ViT encoding + stop tokens.
    print(f"[i2t-direct] preparing inputs for prompt={args.prompt!r}")
    model_inputs = m.prepare_model_inputs(
        prompt=args.prompt,
        image=pil_image,
        mode="gen_text",
        bot_task=args.bot_task,
        max_new_tokens=args.max_tokens,
    )
    print(f"[i2t-direct] tokenizer_output.tokens.shape={model_inputs['tokenizer_output'].tokens.shape}")

    # Run the unified generate.
    print(f"[i2t-direct] generating up to {args.max_tokens} tokens ...")
    t0 = time.time()
    with torch.no_grad():
        out = m.generate(
            **model_inputs,
            do_sample=False,  # greedy for determinism
        )
    dt = time.time() - t0

    # Decode (if model didn't already detokenize).
    if hasattr(out, "tolist"):
        decoded = m._tokenizer.decode(out[0], skip_special_tokens=False)
    elif isinstance(out, list):
        decoded = m._tokenizer.decode(out[0], skip_special_tokens=False)
    else:
        decoded = repr(out)
    print(f"[i2t-direct] generate finished in {dt:.1f}s")
    print("[i2t-direct] decoded (with special tokens):")
    print(repr(decoded))
    print("[i2t-direct] decoded (without special tokens):")
    if hasattr(out, "tolist"):
        print(repr(m._tokenizer.decode(out[0], skip_special_tokens=True)))


if __name__ == "__main__":
    main()
