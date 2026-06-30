#!/usr/bin/env python3
"""GPU multi-worker smoke for the AgenticRolloutEngine (LIN-522).

Boots a ``≥2``-worker DP slab of the agentic engine — each worker building its own
inner ``SGLangRolloutEngine`` (Qwen3-Instruct) + a ``ToolEnvironment`` + calculator —
wires the rank-0 coordinator (``set_workers``), and drives multi-turn rollout over a
batch of heterogeneous prompts. Proves the whole loop end-to-end: rank 0 fans the
batch into ``n × P`` single-trajectory tasks, workers pull + run multi-turn agent
loops (call the calculator, see the result, answer), and ``generate`` returns a flat
``List[Sample]`` of variable-depth trajectories.

    QWEN3_INSTRUCT_PATH=/root/unirl/models/local/Qwen3-4B-Instruct \
    AGENTIC_NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 .venv-sglang/bin/python scripts/agentic_engine_smoke.py

Exits 0 on PASS, non-zero on failure. GPU-only (needs ≥2 visible GPUs + a model).
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter

import torch

from unirl.distributed.group.device_pool import DevicePool
from unirl.rollout.engine.agentic import AgenticRolloutEngine, AgenticRolloutEngineConfig
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.loop import CalculatorTool, ToolEnvironment
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

# Heterogeneous prompts → heterogeneous turn counts (some need the calculator, some
# are answerable directly) so the returned trajectories are genuinely ragged.
PROMPTS = [
    ("p0", "What is 1234 multiplied by 5678? Use the calculator tool, then give the answer.", "7006652"),
    ("p1", "What is the capital of France? Answer directly.", "Paris"),
    ("p2", "Compute 99 * 99 with the calculator tool, then state the result.", "9801"),
    ("p3", "Say hello in one word.", None),
]
N = 2  # GRPO group size (samples per prompt)
MAX_TURNS = 4


def _log(msg: str) -> None:
    print(f"[agentic-smoke] {msg}", flush=True)


def main() -> int:
    model_path = os.environ.get("QWEN3_INSTRUCT_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_INSTRUCT_PATH to a local Qwen3-Instruct dir")
        return 2
    num_gpus = int(os.environ.get("AGENTIC_NUM_GPUS", "2"))  # GPUs the agentic handle claims
    pool_gpus = int(os.environ.get("AGENTIC_POOL_GPUS", "8"))  # node's total GPUs (DevicePool span)
    if num_gpus < 2:
        _log("ERROR: this smoke wants AGENTIC_NUM_GPUS>=2 (multi-worker slab)")
        return 2
    if pool_gpus % num_gpus != 0 and num_gpus % pool_gpus != 0:
        pool_gpus = num_gpus  # keep DevicePool's num_devices % devices_per_node == 0
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; gpus={num_gpus}; model={model_path}")

    import ray

    if not ray.is_initialized():
        ray.init()

    # Tools advertised to the model via the inner engine's chat template.
    env = ToolEnvironment([CalculatorTool()], max_turns=MAX_TURNS)
    inner = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",
        tp_size=1,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
        chat_template_kwargs={"tools": env.tool_schemas()},
    )
    cfg = AgenticRolloutEngineConfig(
        inner=inner,
        env=env,  # re-entrant; pickled to each worker (each gets its own copy)
        max_turns=MAX_TURNS,
        episode_sampling=ARSamplingParams(
            samples_per_prompt=N, temperature=0.7, max_new_tokens=512, top_p=0.9, top_k=20
        ),
        per_worker_concurrency=8,
    )

    # worker_max_concurrency>=3 is REQUIRED on rank 0: it runs `generate` (the routing
    # loop, blocked in ray.get) + its own `run_drain` + serves `next_task` pulls
    # concurrently on the threaded actor (design §5 / risk #2). 8 leaves headroom.
    # DevicePool spans the node's GPUs (num_devices % devices_per_node must be 0);
    # the agentic handle below claims `num_gpus` of them via create_remote(device_ids=...).
    pool = DevicePool(num_devices=pool_gpus, devices_per_node=pool_gpus, worker_max_concurrency=8)
    pool.setup()

    engine = None
    try:
        _log(f"creating a {num_gpus}-worker agentic slab (each builds its own inner SGLang) ...")
        engine = pool.create_remote(
            AgenticRolloutEngine, device_ids=list(range(num_gpus)), init_kwargs={"config": cfg}
        )
        # Wire rank 0 with its sibling handles (the NCCLWeightSync set_rollout_targets shape).
        engine.set_workers(engine.workers, engine.role_name)

        # The request is JUST the prompts — the coordinator applies the n-fanout
        # internally (no driver-side fork). stop after </tool_call> so a tool-call
        # turn ends there; final-answer turns run to EOS.
        ids = [pid for pid, _, _ in PROMPTS]
        prompts = [text for _, text, _ in PROMPTS]
        batch = Sample.request(
            Part.input(ids, primitive=Texts(texts=prompts), control={"ar": {"stop": ["</tool_call>"]}})
        )

        P = len(PROMPTS)
        _log(f"generating: P={P} prompts × n={N} = {P * N} trajectories, max_turns={MAX_TURNS} ...")
        out = engine.generate(batch)[0]  # BROADCAST+RANK_ZERO → [List[Sample]]; unwrap [0]

        # ---- contract: a flat list of n×P variable-depth trajectories ----
        assert isinstance(out, list), f"generate must return a list; got {type(out).__name__}"
        assert len(out) == P * N, f"expected {P * N} trajectories; got {len(out)}"
        assert all(isinstance(s, Sample) for s in out), "every element must be a Sample"

        # ---- group recovery by root id: n per prompt ----
        by_root = Counter(s.parts[0].sample_ids[0] for s in out)
        assert by_root == Counter({pid: N for pid in ids}), f"bucket-by-root mismatch: {dict(by_root)}"

        # ---- ragged depth + clean termination ----
        depths = sorted(len(s.gen_parts()) for s in out)
        _log(f"trajectory turn-depths (sorted): {depths}")
        assert all(1 <= d <= MAX_TURNS for d in depths), f"turn depths out of [1, {MAX_TURNS}]: {depths}"
        multi = sum(1 for d in depths if d >= 2)
        assert multi > 0, "expected at least one multi-turn (tool-using) trajectory"

        # ---- answers: the calculator prompts should yield the right number somewhere ----
        for pid, _text, expected in PROMPTS:
            if expected is None:
                continue
            trajs = [s for s in out if s.parts[0].sample_ids[0] == pid]
            rendered = [
                " ".join(txt for p in s.parts if isinstance(p.primitive, Texts) for txt in p.primitive.texts)
                for s in trajs
            ]
            hit = any(expected in r for r in rendered)
            _log(f"  {pid}: expected {expected!r} present in some sibling? {hit}")
            # informational for direct-answer prompts; required for calculator prompts
            if "calculator" in _text or "calculate" in _text.lower():
                assert hit, f"{pid}: no sibling trajectory contained {expected!r}"

        _log(f"AGENTIC SMOKE PASSED ✅  ({P * N} trajectories, ragged depths {min(depths)}–{max(depths)})")
        return 0
    except Exception:
        _log("AGENTIC SMOKE FAILED ❌")
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                _log("engine shut down")
            except Exception:
                _log("engine.shutdown() raised (ignored)")


if __name__ == "__main__":
    sys.exit(main())
