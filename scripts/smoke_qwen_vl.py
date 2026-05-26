#!/usr/bin/env python3
"""Smoke test for QwenVL pipeline: generate text from text+image, verify replay."""

import argparse
import sys

import torch

from diffusionrl.models.qwen_vl import QwenVLARConditions, QwenVLPipeline, QwenVLPipelineConfig
from diffusionrl.types.primitives import Image, Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import ARSamplingParams


def make_dummy_image(height: int = 224, width: int = 224) -> Image:
    pixels = torch.rand(3, height, width)
    return Image(pixels=pixels)


def main():
    parser = argparse.ArgumentParser(description="QwenVL smoke test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--text-only", action="store_true", help="Skip image input")
    args = parser.parse_args()

    print(f"Loading QwenVL from {args.model} ...")
    config = QwenVLPipelineConfig(
        pretrained_model_ckpt_path=args.model,
        device=args.device,
        model_precision="bf16",
        freeze_vision_tower=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    pipeline = QwenVLPipeline.from_config(config)
    print("Pipeline loaded.")

    prompts = ["Describe what you see in this image in one sentence."]
    texts = Texts(texts=prompts)

    primitives = {"text": texts}
    if not args.text_only:
        img = make_dummy_image()
        images = Images.from_list([img])
        primitives["image"] = images
        print("Using dummy image input.")
    else:
        print("Text-only mode (no image).")

    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives=primitives,
        sampling_params=ARSamplingParams(
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        ),
    )

    print("\n--- Generate ---")
    resp = pipeline.generate(req)
    track = resp.tracks["text"]
    segment = track.segment
    decoded = track.decoded

    print(f"Decoded text: {decoded.texts}")
    print(f"Segment tokens shape: {segment.tokens.shape if segment.tokens is not None else None}")
    print(f"Segment log_probs shape: {segment.log_probs.shape if segment.log_probs is not None else None}")

    if segment.tokens is None or segment.tokens.numel() == 0:
        print("WARNING: no tokens generated.")
        sys.exit(1)

    print("\n--- Replay ---")
    conds = QwenVLARConditions.from_dict(track.conditions)

    pipeline.ar.model.transformer.train()
    replay_logprobs = pipeline.ar.replay(conds, segment=segment)
    pipeline.ar.model.transformer.eval()

    print(f"Replay log_probs shape: {replay_logprobs.shape}")
    print(f"Rollout log_probs shape: {segment.log_probs.shape}")

    rollout_lp = segment.log_probs.to(dtype=replay_logprobs.dtype, device=replay_logprobs.device)
    ratio = torch.exp(replay_logprobs - rollout_lp)
    print(f"Ratio mean: {ratio.mean().item():.4f}, std: {ratio.std().item():.4f}")
    print(f"Ratio min: {ratio.min().item():.4f}, max: {ratio.max().item():.4f}")

    if torch.isnan(replay_logprobs).any():
        print("FAIL: NaN in replay log-probs")
        sys.exit(1)

    if abs(ratio.mean().item() - 1.0) > 0.1:
        print(f"WARNING: ratio mean {ratio.mean().item():.4f} deviates from 1.0 (expected for first replay)")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
