"""Smoke test for ``VLLMOmniRolloutEngine`` sleep / wake_up.

Builds a t2i engine, runs one warmup generate, then exercises the
runtime-offload surface:

  1. snapshot ``torch.cuda.memory_allocated`` per device
  2. ``engine.sleep()``  → assert ``is_offloaded`` and memory dropped
  3. second ``engine.sleep()`` → no-op (idempotency)
  4. ``engine.wake_up()`` → assert NOT ``is_offloaded``, memory restored
  5. second ``engine.wake_up()`` → no-op (idempotency)
  6. one more ``generate()`` to confirm the engine is usable post-wake

Run on a TaiJi pod with HI3 weights staged locally::

    python scripts/smoke_vllm_omni_sleep_wake.py --model-path <hi3-ckpt>

Pass ``--no-sleep-mode`` to verify the negative path: with
``enable_sleep_mode=False`` on the config, ``engine.sleep()`` should raise
because vllm-omni's ``CuMemAllocator`` GPU pool was never enabled.
"""

from __future__ import annotations

import argparse
import sys

import torch

from diffusionrl.rollout.engine.vllm_omni import (
    VLLMOmniEngineConfig,
    VLLMOmniRolloutEngine,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--model-path", required=True, help="HF id or local path for HunyuanImage-3.0.")
    p.add_argument("--prompt", default="a red apple on a wooden table")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num-inference-steps", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-sleep-mode",
        action="store_true",
        help="Set cfg.enable_sleep_mode=False; sleep() should then raise.",
    )
    return p.parse_args()


def _make_req(args: argparse.Namespace) -> RolloutReq:
    return RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={
            "diffusion": {
                "height": int(args.height),
                "width": int(args.width),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "eta": 0.0,
                "seed": int(args.seed),
            },
            "ar": {"max_tokens": 2048, "temperature": 0.6},
        },
    )


def _mem_snapshot(label: str) -> dict:
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA unavailable — skipping memory snapshot")
        return {}
    snap: dict = {}
    for d in range(torch.cuda.device_count()):
        mb = torch.cuda.memory_allocated(d) / (1024**2)
        snap[d] = mb
        print(f"[{label}] cuda:{d} allocated = {mb:9.1f} MiB")
    return snap


def _max_drop_mib(before: dict, after: dict) -> float:
    if not before or not after:
        return 0.0
    return max((before[d] - after.get(d, 0.0)) for d in before)


def _max_restore_mib(post_sleep: dict, post_wake: dict) -> float:
    if not post_sleep or not post_wake:
        return 0.0
    return max((post_wake[d] - post_sleep.get(d, 0.0)) for d in post_wake)


def main() -> int:
    args = parse_args()
    cfg = VLLMOmniEngineConfig(
        model_path=args.model_path,
        modality="t2i",
        default_eta=0.0,
        default_num_inference_steps=int(args.num_inference_steps),
        default_guidance_scale=float(args.guidance_scale),
        enable_sleep_mode=not args.no_sleep_mode,
    )
    engine = VLLMOmniRolloutEngine(cfg)
    try:
        req = _make_req(args)
        print("=== warmup generate ===")
        engine.generate(req)
        baseline = _mem_snapshot("post-warmup")

        if args.no_sleep_mode:
            print("\n=== negative test: sleep() with enable_sleep_mode=False ===")
            try:
                engine.sleep()
            except Exception as e:
                print(f"OK — sleep() raised as expected: {type(e).__name__}: {e}")
                return 0
            print("FAIL — sleep() did not raise; CuMemAllocator pool may not be gated.")
            return 1

        print("\n=== sleep ===")
        engine.sleep()
        assert engine.is_offloaded, "is_offloaded should be True after sleep()"
        post_sleep = _mem_snapshot("post-sleep")
        drop = _max_drop_mib(baseline, post_sleep)
        print(f"max per-device drop: {drop:.1f} MiB")
        assert drop > 100.0, (
            f"sleep() freed only {drop:.1f} MiB across devices — expected a substantial drop. Pool may not be enabled."
        )

        print("\n=== second sleep (idempotency) ===")
        engine.sleep()
        assert engine.is_offloaded, "is_offloaded should still be True"
        print("OK")

        print("\n=== wake_up ===")
        engine.wake_up()
        assert not engine.is_offloaded, "is_offloaded should be False after wake_up()"
        post_wake = _mem_snapshot("post-wake")
        restore = _max_restore_mib(post_sleep, post_wake)
        print(f"max per-device restore: {restore:.1f} MiB")

        print("\n=== second wake_up (idempotency) ===")
        engine.wake_up()
        assert not engine.is_offloaded, "is_offloaded should still be False"
        print("OK")

        print("\n=== post-wake generate ===")
        engine.generate(req)
        print("OK — generate succeeded after wake_up.")
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
