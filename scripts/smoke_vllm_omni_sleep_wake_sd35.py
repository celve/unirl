"""SD3.5-flavored smoke for VLLMOmniRolloutEngine sleep / wake_up.

Single-stage diffusion (sd35_t2i) — much cheaper than HI3 t2i. Steps:

  1. warmup generate
  2. snapshot per-device CUDA memory
  3. ``engine.sleep()`` → assert ``is_offloaded``, memory drops
  4. second ``engine.sleep()`` → no-op
  5. ``engine.wake_up()`` → memory restored
  6. second ``engine.wake_up()`` → no-op
  7. one more ``generate()`` to confirm post-wake correctness

Default model path is ``/root/models/stable-diffusion-3.5-medium`` (pod-local
mirror of the diffusers checkpoint copied by ``setup_and_launch.sh``).

Run inside an exec session::

  cd ~/diffusionrl && source .venv/bin/activate &&
  python scripts/smoke_vllm_omni_sleep_wake_sd35.py
"""

from __future__ import annotations

import argparse
import sys

from diffusionrl.rollout.engine.vllm_omni import (
    VLLMOmniEngineConfig,
    VLLMOmniRolloutEngine,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--model-path", default="/root/models/stable-diffusion-3.5-medium")
    p.add_argument("--prompt", default="a red apple on a wooden table")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=6)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
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
        },
    )


def _snap(label: str) -> dict:
    """Per-device used-memory snapshot via nvidia-smi.

    ``torch.cuda.memory_allocated`` only sees the *driver* process; the
    vllm-omni workers run as subprocesses with their own CUDA contexts, so
    we must query the device itself.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode()
    except Exception as e:
        print(f"[{label}] nvidia-smi failed: {e}")
        return {}
    snap: dict = {}
    for line in out.strip().splitlines():
        idx_s, mb_s = (x.strip() for x in line.split(","))
        idx, mb = int(idx_s), float(mb_s)
        snap[idx] = mb
        print(f"[{label}] cuda:{idx} used = {mb:9.1f} MiB")
    return snap


def main() -> int:
    args = parse_args()
    cfg = VLLMOmniEngineConfig(
        model_path=args.model_path,
        modality="sd35_t2i",
        default_eta=0.0,
        default_num_inference_steps=int(args.num_inference_steps),
        default_guidance_scale=float(args.guidance_scale),
        enable_sleep_mode=True,
    )
    engine = VLLMOmniRolloutEngine(cfg)
    try:
        req = _make_req(args)

        print("=== warmup generate ===")
        engine.generate(req)
        baseline = _snap("post-warmup")

        print("\n=== sleep ===")
        engine.sleep()
        assert engine.is_offloaded, "is_offloaded must be True"
        post_sleep = _snap("post-sleep")
        drops = [baseline[d] - post_sleep.get(d, 0.0) for d in baseline]
        max_drop = max(drops) if drops else 0.0
        print(f"max per-device drop: {max_drop:.1f} MiB")
        assert max_drop > 100.0, f"expected substantial drop; got {max_drop:.1f} MiB"

        print("\n=== second sleep (idempotency) ===")
        engine.sleep()
        assert engine.is_offloaded
        print("OK")

        print("\n=== wake_up ===")
        engine.wake_up()
        assert not engine.is_offloaded, "is_offloaded must be False"
        post_wake = _snap("post-wake")
        restores = [post_wake[d] - post_sleep.get(d, 0.0) for d in post_wake]
        max_restore = max(restores) if restores else 0.0
        print(f"max per-device restore: {max_restore:.1f} MiB")

        print("\n=== second wake_up (idempotency) ===")
        engine.wake_up()
        assert not engine.is_offloaded
        print("OK")

        print("\n=== post-wake generate ===")
        engine.generate(req)
        print("OK — generate succeeded after wake_up.")
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
