"""Real-weight smoke for the Qwen3 AR pipeline.

Loads a Qwen3 checkpoint (default: ``Qwen/Qwen3-0.6B``) via
``Qwen3Pipeline.from_config``, runs end-to-end ``generate()`` on a
single prompt, prints the decoded text, then re-runs ``replay()`` over
the rollout's stored ``TextSegment`` and asserts per-token log-prob
parity — the GRPO substitution invariant that π_old (rollout) and π_θ
(replay) agree on a frozen model.

Usage on pod:

    cd ~/diffusionrl && source .venv/bin/activate && \\
    python scripts/smoke_qwen3_ar.py \\
        --ckpt Qwen/Qwen3-0.6B \\
        --prompt "Tell me a one-line joke." \\
        --max-tokens 32 --temperature 0.0
"""

from __future__ import annotations

import argparse
import time

import torch

from diffusionrl.models.qwen3 import (
    Qwen3Pipeline,
    Qwen3PipelineConfig,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import ARSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Tell me a one-line joke.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument(
        "--system-instruction",
        default=None,
        help="Optional system message prepended via the chat template.",
    )
    args = parser.parse_args()

    print(f"[qwen3-smoke] loading pipeline from {args.ckpt} ...")
    t0 = time.time()
    cfg = Qwen3PipelineConfig(
        pretrained_model_ckpt_path=args.ckpt,
        model_precision=args.precision,
    )
    pipe = Qwen3Pipeline.from_config(cfg)
    print(f"[qwen3-smoke] pipeline loaded in {time.time() - t0:.1f}s on {pipe.bundle.device}")

    sampling_params = ARSamplingParams(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    stage_config: dict = {}
    if args.system_instruction is not None:
        stage_config["chat"] = {"system_instruction": args.system_instruction}

    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        sampling_params=sampling_params,
        stage_config=stage_config,
    )

    print(f"[qwen3-smoke] prompt={args.prompt!r}  max_tokens={args.max_tokens}  temperature={args.temperature}")
    t0 = time.time()
    resp = pipe.generate(req)
    dt = time.time() - t0

    track = resp.tracks["ar"]
    seg = track.segment
    decoded = track.decoded
    n_tok = int(seg.lengths.sum().item()) if seg.lengths is not None else 0
    print(f"[qwen3-smoke] generated {n_tok} tokens in {dt:.1f}s ({n_tok / max(dt, 1e-3):.2f} tok/s)")
    print(f"[qwen3-smoke] decoded: {decoded.texts[0]!r}")

    # Replay parity check — meaningful under greedy / deterministic sampling.
    # The threshold is precision-aware. bf16 attention accumulates differently
    # between rollout (KV-cache incremental forward) and replay (single full
    # teacher-forced forward); the ~5-10% per-token log-prob drift under bf16
    # is numerical, not an implementation bug. fp32 hits ~1e-5 cleanly.
    if args.temperature == 0.0 and n_tok > 0:
        from diffusionrl.models.qwen3.conditions import Qwen3ARConditions

        conds = Qwen3ARConditions.from_dict(resp.conditions)
        with torch.no_grad():
            replay_logps = pipe.ar.replay(conds, segment=seg)
        max_abs_diff = (replay_logps - seg.log_probs).abs().max().item()
        threshold = 1e-3 if str(args.precision).lower() in {"fp32", "float32", "float"} else 1.5e-1
        print(
            f"[qwen3-smoke] replay parity max |Δlogp| = {max_abs_diff:.2e} (threshold {threshold:.0e} @ {args.precision})"
        )
        assert max_abs_diff < threshold, f"replay log-probs diverged: max abs diff {max_abs_diff}"
        print("[qwen3-smoke] replay parity OK")


if __name__ == "__main__":
    main()
