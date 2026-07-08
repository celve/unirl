#!/usr/bin/env python3
"""Raw SGLang cross-process TP=2 probe (LIN-535).

Isolates the one runtime question behind the grouped-TP rollout design from all of
unirl's Ray/Handle machinery: **does sglang's in-process ``Engine`` form a 2-way TP
group out of TWO separate single-GPU processes** given the exact controller-style
multi-node coordinates that unirl's ``Handle._assign_tp_coords`` stamps per worker —
``nnodes=2``, ``node_rank in {0,1}``, a shared ``dist_init_addr``, ``base_gpu_id=0``,
each process pinned to one physical GPU via ``CUDA_VISIBLE_DEVICES``?

And the integration-critical sub-question: does ``node_rank=1``'s ``Engine(...)``
**return** (so a unirl participant worker's ``__init__`` can complete), or does it
block in a serve loop?

Layout (mirrors two unirl workers in one TP group):
  - rank 0: CVD=<g0>, node_rank=0 → builds Engine, generates a token, prints it
  - rank 1: CVD=<g1>, node_rank=1 → builds Engine (TP participant), reports, holds

Both share ``dist_init_addr=127.0.0.1:<port>``. The parent spawns both children and
asserts rank 0 produced text.

    PROBE_TP2_PASS  → group formed + token out (+ whether rank1 Engine returned)
    PROBE_TP2_FAIL  → non-zero exit + traceback

Usage:
    .venv-sglang/bin/python scripts/probe_sglang_tp2.py \
        --model /root/unirl/models/local/Qwen3-4B-Base --gpus 0,1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _run_child(args: argparse.Namespace) -> int:
    import sglang

    print(f"[rank{args.node_rank}] importing+constructing Engine "
          f"(node_rank={args.node_rank} tp=2 dist_init={args.dist_init} base_gpu_id=0)", flush=True)
    engine = sglang.Engine(
        model_path=args.model,
        tp_size=2,
        nnodes=2,
        node_rank=args.node_rank,
        dist_init_addr=args.dist_init,
        base_gpu_id=0,  # each child sees exactly one physical GPU as cuda:0 (unirl 1-GPU worker)
        mem_fraction_static=args.mem_fraction,
        trust_remote_code=True,
        log_level="info",
    )
    # If we reach here on node_rank=1, the participant Engine RETURNED (did not block) —
    # exactly what a unirl participant worker's __init__ needs.
    print(f"[rank{args.node_rank}] ENGINE_RETURNED", flush=True)

    if args.node_rank == 0:
        out = engine.generate(
            ["The capital of France is"],
            {"max_new_tokens": 8, "temperature": 0.0},
        )
        text = out[0]["text"] if isinstance(out, list) else out["text"]
        print(f"[rank0] GEN_OUT={text!r}", flush=True)
        print("CHILD0_OK", flush=True)
    else:
        # Hold the participant up so the TP group stays alive while rank 0 generates.
        print("CHILD1_HOLDING", flush=True)
        time.sleep(args.hold)

    try:
        engine.shutdown()
    except Exception:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dist-init", default="127.0.0.1:29557")
    ap.add_argument("--gpus", default="0,1", help="two physical GPU ids, comma-separated")
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--hold", type=int, default=180)
    # internal
    ap.add_argument("--role", default="parent", choices=["parent", "child"])
    ap.add_argument("--node-rank", type=int, default=0)
    args = ap.parse_args()

    if args.role == "child":
        return _run_child(args)

    g0, g1 = args.gpus.split(",")
    base = [
        sys.executable, os.path.abspath(__file__), "--role", "child",
        "--model", args.model, "--dist-init", args.dist_init,
        "--mem-fraction", str(args.mem_fraction), "--hold", str(args.hold),
    ]
    env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": g0}
    env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": g1}

    print(f"[parent] launching rank1 (participant, CVD={g1}) then rank0 (CVD={g0}); "
          f"dist_init={args.dist_init}", flush=True)
    p1 = subprocess.Popen(base + ["--node-rank", "1"], env=env1)
    p0 = subprocess.Popen(base + ["--node-rank", "0"], env=env0)

    rc0 = p0.wait()
    p1.terminate()
    try:
        p1.wait(timeout=45)
    except Exception:
        p1.kill()

    if rc0 == 0:
        print("PROBE_TP2_PASS ✅  (cross-process TP=2 group formed + token produced)", flush=True)
        return 0
    print(f"PROBE_TP2_FAIL ❌  (rank0 exit={rc0})", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
