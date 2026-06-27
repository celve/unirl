#!/usr/bin/env python3
"""GPU smoke: drive the ``AgentLoop`` over the REAL SGLang AR engine (Qwen3, LIN-492).

Boots the real ``SGLangRolloutEngine`` (model_family "text", Qwen3-4B) and runs an
``AgentLoop`` over it — proving the agent-loop abstraction drives a real rollout engine
end-to-end (not just the ``FakeEngine`` in ``rollout_loop_smoke.py``). The single-turn loop
is the AR analogue of ``rollout_ar_smoke.py`` routed through ``AgentLoop`` (the PRIMARY,
hard-asserted contract); a 2-turn loop additionally probes multi-turn AR over the real engine
(SECONDARY — informational, since real-engine multi-turn is a later phase, not prototype scope).

Run on a GPU pod (1 free GPU), in the sglang venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_loop_ar_smoke.py

Exits 0 on PASS (single-turn AgentLoop over the real engine), non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.loop import AgentLoop
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sample_id import parent_id
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment


def _log(msg: str) -> None:
    print(f"[loop-ar-smoke] {msg}", flush=True)


def build_request(n: int) -> tuple[Sample, ARSamplingParams]:
    """A bare ``[input]`` request (2 prompts) + the AR params the loop forks with."""
    prompts = ["The capital of France is", "Two plus two equals"]
    input_part = Part.input([f"p{i}" for i in range(len(prompts))], primitive=Texts(texts=prompts), control={})
    ar_params = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part), ar_params


def _assert_ar_gen(gen, n_expect: int, input_ids: set) -> None:
    assert len(gen.sample_ids) == n_expect, f"expected {n_expect} samples; got {len(gen.sample_ids)}"
    assert all(parent_id(sid) in input_ids for sid in gen.sample_ids), "gen ids must be children of the input prompts"
    assert isinstance(gen.segment, TextSegment), f"segment must be TextSegment; got {type(gen.segment)}"
    assert isinstance(gen.primitive, Texts) and len(gen.primitive.texts) == n_expect, (
        "decoded Texts missing/wrong count"
    )
    assert gen.sampling_params is not None, "gen Part must carry sampling_params (trainable)"


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3-4B dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",  # in-process sglang Engine (no separate server)
        tp_size=1,
        max_new_tokens=48,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
    )
    n = 2  # completions per prompt → exercises sibling fan-out (n>1)
    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3) ...")
        engine = SGLangRolloutEngine(config, rank=0)

        # ---- PRIMARY: single-turn AgentLoop over the real engine (hard-asserted) ----
        request, ar_params = build_request(n)
        input_ids = set(request.parts[0].sample_ids)
        loop = AgentLoop(plan=[(n, ar_params)], environment=None)
        _log("running single-turn AgentLoop.run(engine, request) ...")
        out = loop.run(engine, request)

        assert len(out.parts) == 2, f"expected [input, ar_gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        _assert_ar_gen(gen, 2 * n, input_ids)
        gps = out.gen_parts()
        assert len(gps) == 1 and gps[0] is gen, "gen_parts must be exactly the AR gen Part"
        _log(f"PRIMARY PASS: single-turn AgentLoop produced {2 * n} Qwen completions via the real engine")
        for i, t in enumerate(gen.primitive.texts):
            _log(f"  sample[{i}] id={gen.sample_ids[i]} text={t[:80]!r}")

        # ---- SECONDARY (informational): 2-turn AR loop, multi-part history via conditioning ----
        _log("probing 2-turn AR AgentLoop (multi-part history) — informational ...")
        try:
            request2, ar_params2 = build_request(n)
            turn2 = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=32, top_p=0.9, top_k=20)
            loop2 = AgentLoop(plan=[(n, ar_params2), (1, turn2)], environment=None, max_turns=2)
            out2 = loop2.run(engine, request2)
            ok2 = len(out2.parts) == 3 and isinstance(out2.parts[-1].segment, TextSegment)
            _log(
                f"SECONDARY: 2-turn loop -> {len(out2.parts)} parts, gen_parts={len(out2.gen_parts())}, "
                f"frontier seg={type(out2.parts[-1].segment).__name__} => {'PASS ✅' if ok2 else 'unexpected shape'}"
            )
            if not ok2:
                _log("SECONDARY: multi-turn AR over the real engine is a follow-up phase (not prototype scope)")
        except Exception as e2:  # noqa: BLE001
            _log(
                f"SECONDARY: multi-turn AR raised ({type(e2).__name__}: {e2}) — expected; "
                f"real-engine multi-turn is a later phase, not in this prototype's scope"
            )

        _log("AGENT-LOOP AR SMOKE PASSED ✅  (AgentLoop drives the real Qwen SGLang engine)")
        return 0
    except Exception:
        _log("AGENT-LOOP AR SMOKE FAILED ❌")
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
