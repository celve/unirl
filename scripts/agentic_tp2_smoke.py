#!/usr/bin/env python3
"""Agentic rollout TP>1 integration smoke (LIN-535).

Boots a multi-worker agentic slab with inner ``tp_size>1`` and drives one batch of
direct-answer prompts through the WHOLE grouped-TP path:

  * ``_tp_size_from_init_kwargs`` reads ``config.inner.tp_size``;
  * ``_build_rank_infos`` lays out ``(dp, tp)`` — ``tp`` consecutive workers per group;
  * ``_assign_tp_coords`` stamps each worker's multi-node coords (``nnodes`` /
    ``node_rank`` / ``dist_init_addr`` / ``base_gpu_id``) onto its inner engine config;
  * participant workers (``node_rank>0``) boot their sglang ``Engine`` non-blocking
    (daemon thread), so ``Handle.add_remote`` returns instead of hanging;
  * ``run_drain`` (``Execute.DP_HEAD``) drains ONLY the group heads; each head's inner
    sglang ``Engine`` drives a real ``tp``-way TP group across its workers' GPUs;
  * ``_collect_dp_merge`` flattens the per-head trajectories to one flat list.

Asserts ``P×N`` trajectories carrying gen tokens, bucketed by root id, produced across
ALL ``dp`` group heads (the cross-group boundary was crossed). Deliberately
model-capability-agnostic — prompts are answered in one turn, so a small local model
suffices; no tool-use / answer-correctness assertions.

    QWEN3_PATH=/root/unirl/models/local/Qwen3-0.6B AGENTIC_NUM_GPUS=4 AGENTIC_INNER_TP=2 \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv-sglang/bin/python scripts/agentic_tp2_smoke.py

Exits 0 on PASS, non-zero on failure. GPU-only (needs ``num_gpus`` visible GPUs + a
local model).
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter

import torch

from unirl.distributed.group.device_pool import DevicePool
from unirl.distributed.tensor import TensorRef
from unirl.distributed.tensor.ref import hydrate
from unirl.distributed.utils import collect_leaves
from unirl.rollout.engine.agentic import AgenticRolloutEngine, AgenticRolloutEngineConfig
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.loop import CalculatorTool, ToolEnvironment
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

# Direct-answer prompts (one turn each) → model-capability-agnostic.
PROMPTS = [
    ("p0", "What is the capital of France? Answer in one word."),
    ("p1", "What color is a clear daytime sky? Answer in one word."),
    ("p2", "What is two plus two? Answer with a single number."),
    ("p3", "Say hello in one word."),
]
N = 2  # GRPO group size (samples per prompt)
MAX_TURNS = 2


def _log(msg: str) -> None:
    print(f"[agentic-tp2] {msg}", flush=True)


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH", "/root/unirl/models/local/Qwen3-0.6B")
    if not os.path.isdir(model_path):
        _log(f"ERROR: model dir not found: {model_path}")
        return 2
    num_gpus = int(os.environ.get("AGENTIC_NUM_GPUS", "4"))
    inner_tp = int(os.environ.get("AGENTIC_INNER_TP", "2"))
    pool_gpus = int(os.environ.get("AGENTIC_POOL_GPUS", "8"))
    # Small per-worker cap forces the P×N tasks to spread across BOTH heads (so the
    # cross-group assertion below is actually exercised).
    per_worker_conc = int(os.environ.get("AGENTIC_PER_WORKER_CONC", "2"))
    if num_gpus % inner_tp != 0:
        _log(f"ERROR: num_gpus={num_gpus} must be a multiple of inner_tp={inner_tp}")
        return 2
    dp = num_gpus // inner_tp
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; "
         f"num_gpus={num_gpus} inner_tp={inner_tp} -> dp={dp}; model={model_path}")

    import ray

    if not ray.is_initialized():
        ray.init()

    env = ToolEnvironment([CalculatorTool()], max_turns=MAX_TURNS)
    inner = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",
        tp_size=inner_tp,  # → _tp_size_from_init_kwargs reads config.inner.tp_size
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
        chat_template_kwargs={"tools": env.tool_schemas()},
    )
    cfg = AgenticRolloutEngineConfig(
        inner=inner,
        env=env,
        max_turns=MAX_TURNS,
        episode_sampling=ARSamplingParams(
            samples_per_prompt=N, temperature=0.7, max_new_tokens=64, top_p=0.9, top_k=20
        ),
        per_worker_concurrency=per_worker_conc,
    )
    pool = DevicePool(num_devices=pool_gpus, devices_per_node=pool_gpus, worker_max_concurrency=8)
    pool.setup()

    engine = None
    try:
        _log(f"creating a {num_gpus}-worker agentic slab (inner tp={inner_tp}) — "
             f"expect {dp} tp={inner_tp} groups ...")
        engine = pool.create_remote(
            AgenticRolloutEngine, device_ids=list(range(num_gpus)), init_kwargs={"config": cfg}
        )

        # The (dp, tp) layout the Handle computed from config.inner.tp_size.
        ris = engine.rank_infos
        heads = [r.rank for r in ris if r.tp_rank == 0]
        _log(f"handle layout: dp_size={ris[0].dp_size} tp_size={ris[0].tp_size}; heads={heads}; "
             f"tp_ranks={[r.tp_rank for r in ris]} dp_ranks={[r.dp_rank for r in ris]}")
        assert ris[0].tp_size == inner_tp, f"handle tp_size {ris[0].tp_size} != inner_tp {inner_tp}"
        assert len(heads) == dp, f"expected {dp} DP-group heads, got {heads}"

        ids = [pid for pid, _ in PROMPTS]
        prompts = [text for _, text in PROMPTS]
        batch = Sample.request(
            Part.input(ids, primitive=Texts(texts=prompts), control={"ar": {"stop": ["</tool_call>"]}})
        )
        P = len(PROMPTS)
        _log(f"draining: P={P} × n={N} = {P * N} trajectories over {dp} tp={inner_tp} group(s) ...")
        engine.set_batch(batch)
        out = engine.run_drain(engine.workers[0], engine.role_name)

        # ---- contract: one flat list of n×P trajectories ----
        assert isinstance(out, list), f"run_drain must return a list; got {type(out).__name__}"
        assert len(out) == P * N, f"expected {P * N} trajectories; got {len(out)}"
        assert all(isinstance(s, Sample) for s in out), "every element must be a Sample"
        by_root = Counter(s.parts[0].sample_ids[0] for s in out)
        assert by_root == Counter({pid: N for pid in ids}), f"bucket-by-root mismatch: {dict(by_root)}"

        # ---- gen tokens present + hydrate from the driver + cross-group provenance ----
        n_seg = n_tok = 0
        gen_src: set = set()
        for i, traj in enumerate(out):
            gen = traj.gen_parts()
            assert gen and gen[-1].segment is not None, f"traj {i} has no gen segment"
            n_seg += 1
            for r in collect_leaves(traj, TensorRef):
                t = hydrate(r)
                assert t is not None and hasattr(t, "shape"), f"traj {i}: TensorRef did not hydrate"
                n_tok += 1
            for p in traj.gen_parts():
                for r in collect_leaves(p.segment, TensorRef):
                    for s in r.spans:
                        gen_src.add(str(s.handle.source_id))
        _log(f"tensors: {n_seg}/{len(out)} trajectories carry a gen segment; hydrated {n_tok} refs; "
             f"gen produced on {len(gen_src)} distinct workers {sorted(gen_src)}")
        assert n_seg == len(out), "every trajectory must carry gen tokens"
        assert len(gen_src) >= dp, (
            f"expected gen produced on >= {dp} group heads (both TP groups drained); got {sorted(gen_src)}"
        )

        _log(f"AGENTIC TP={inner_tp} SMOKE PASSED ✅  "
             f"({P * N} trajectories across {dp} tp={inner_tp} group(s), heads {heads})")
        return 0
    except Exception:
        _log("AGENTIC TP2 SMOKE FAILED ❌")
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
