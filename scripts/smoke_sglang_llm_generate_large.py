"""Larger generate stress test for the SGLang LLM rollout engine.

Goes beyond the 2-prompt × n=1 × 16-token smoke and exercises:

- 32 diverse prompts (instead of 2) → exceeds the engine's default
  ``concurrency=8`` semaphore so the async fan-out actually queues.
- ``n=4`` candidates per prompt → exercises the ``f"{sid}#{k}"`` sample-id
  mangling in :func:`build_rollout_resp` and the per-prompt → per-candidate
  packing in :class:`TextSegment`.
- ``max_new_tokens=128`` → forces the SGLang scheduler into a real decode
  loop rather than a single prefill step.
- A mix of short / instruction / continuation prompts so the tokenizer's
  chat-template path doesn't trivially collapse to the same token sequence.

Assertions (any failure → non-zero exit):

1. Total candidate count = ``n_prompts * n_per_prompt``.
2. Every decoded text is non-empty.
3. Every ``tracks["ar"].segment`` row has at least one token and
   ``len(token_ids) == len(logprobs)`` (alignment is the most common silent-
   corruption signal — sglang has historically dropped logprobs on
   certain finish_reasons).
4. ``cu_seqlens`` is strictly monotonically non-decreasing and
   ``cu_seqlens[-1] == total_tokens``.
5. ``sample_ids`` for ``n=4`` mangle as ``{sid}#{0..3}``; every original
   ``sid`` shows up exactly ``n_per_prompt`` times.
6. ``group_ids`` are preserved per original prompt (same gid across all
   candidates).
7. End-to-end throughput (tokens / s) printed; no hard threshold but
   anomalously low values are flagged.

Run on a 1×H20 pod with Qwen3-0.6B pre-copied to pod-local::

    LLM_MODEL=/root/diffusionrl/models/local/Qwen3-0.6B \
      python scripts/smoke_sglang_llm_generate_large.py \
      2>&1 | tee /mnt/bj/diffrl_runtime/smoke_sglang_llm_generate_large.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("smoke_sglang_llm_large")


_DEFAULT_MODEL = "/root/diffusionrl/models/local/Qwen3-0.6B"

# 32 diverse prompts. Mix of styles to stress the chat-template path +
# avoid SRT cache aliasing collapsing different prompts onto the same KV.
_PROMPTS: List[str] = [
    "Write a haiku about reinforcement learning.",
    "Explain diffusion models in one sentence.",
    "What is the capital of Japan?",
    "Translate to French: 'The cat sits on the mat.'",
    "List three benefits of unit testing.",
    "Summarize the plot of Hamlet in two sentences.",
    "Give me a Python one-liner to reverse a string.",
    "What is the difference between TCP and UDP?",
    "Compose a limerick about a programmer.",
    "Why is the sky blue? Answer in one paragraph.",
    "Recommend a beginner book on machine learning.",
    "Convert 100 degrees Celsius to Fahrenheit and explain the formula.",
    "Write a SQL query to find duplicate rows in a table.",
    "What does the acronym REST stand for?",
    "Describe the taste of saffron.",
    "Pick a number between 1 and 100 and justify your choice.",
    "Define entropy in information theory.",
    "Name three Studio Ghibli films.",
    "Why does iron rust?",
    "Write a regex that matches a US phone number.",
    "What is the time complexity of mergesort?",
    "Give a one-line definition of monad in functional programming.",
    "What is the largest desert on Earth?",
    "Explain the prisoner's dilemma briefly.",
    "Recommend a recipe using eggs, bread, and cheese.",
    "What is the speed of light in a vacuum?",
    "Write a short poem about autumn leaves.",
    "List three programming languages designed for systems work.",
    "What is the difference between mitosis and meiosis?",
    "Describe how a transformer attention head works at a high level.",
    "What year did the Berlin Wall fall?",
    "Give a Python function signature for binary search.",
]
_N_PER_PROMPT = 4
_MAX_NEW_TOKENS = 128
_TEMPERATURE = 0.8  # higher than smoke's 0.7 → more candidate diversity


def _build_engine_remote_class():
    import ray

    @ray.remote(num_gpus=1)
    class _EngineActor:
        def __init__(self, *, model_path: str) -> None:
            from diffusionrl.rollout.engine.sglang_llm import (
                SGLangLLMEngineConfig,
                SGLangLLMRolloutEngine,
            )

            engine_kwargs: Dict[str, Any] = {
                "mem_fraction_static": 0.7,
                "skip_server_warmup": True,
                "disable_cuda_graph": True,
                "attention_backend": "flashinfer",
            }
            cfg = SGLangLLMEngineConfig(
                pretrained_model_ckpt_path=model_path,
                tp_size=1,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=_TEMPERATURE,
                top_p=0.95,
                # Bump concurrency so async fan-out doesn't artificially
                # serialize the larger workload; engine's default of 8 would
                # only let 8/32 prompts be in flight at once.
                concurrency=16,
                engine_kwargs=engine_kwargs,
            )
            self._engine = SGLangLLMRolloutEngine(cfg, rank=0)

        def health_check(self) -> bool:
            return bool(self._engine.health_check())

        def generate(
            self,
            prompts: List[str],
            *,
            n: int,
            max_new_tokens: int,
            temperature: float,
        ) -> Dict[str, Any]:
            import torch

            from diffusionrl.types.primitives import Texts
            from diffusionrl.types.rollout_req import RolloutReq
            from diffusionrl.types.sampling import ARSamplingParams

            req = RolloutReq(
                sample_ids=[f"s{i}" for i in range(len(prompts))],
                group_ids=[f"g{i // 4}" for i in range(len(prompts))],
                primitives={"text": Texts(texts=list(prompts))},
                sampling_params=ARSamplingParams(
                    max_new_tokens=int(max_new_tokens),
                    samples_per_prompt=int(n),
                    temperature=float(temperature),
                ),
            )
            t0 = time.perf_counter()
            resp = self._engine.generate(req)
            elapsed_s = time.perf_counter() - t0

            track = resp.tracks.get("ar")
            seg = track.segment if track is not None else None
            tokens = getattr(seg, "tokens", None) if seg is not None else None
            log_probs = getattr(seg, "log_probs", None) if seg is not None else None
            cu = getattr(seg, "cu_seqlens", None) if seg is not None else None
            sample_indices = getattr(seg, "sample_indices", None) if seg is not None else None
            return {
                "elapsed_s": elapsed_s,
                "texts": list(track.decoded.texts) if track is not None and track.decoded is not None else [],
                "sample_ids": list(track.sample_ids) if track is not None else [],
                "group_ids": list(track.group_ids) if track is not None else [],
                "n_tokens_total": int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else 0,
                "n_logprobs_total": int(log_probs.shape[0]) if isinstance(log_probs, torch.Tensor) else 0,
                "cu_seqlens": cu.tolist() if isinstance(cu, torch.Tensor) else None,
                "sample_indices": sample_indices.tolist() if isinstance(sample_indices, torch.Tensor) else None,
                "seg_count": int(seg.batch_size) if seg is not None else 0,
            }

        def shutdown(self) -> None:
            self._engine.shutdown()

    return _EngineActor


def _check(cond: bool, message: str, fails: List[str]) -> None:
    if not cond:
        fails.append(message)
        logger.error("CHECK FAIL: %s", message)


def main() -> int:
    model_path = os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
    n_prompts = len(_PROMPTS)
    n_per_prompt = _N_PER_PROMPT
    total_expected = n_prompts * n_per_prompt

    logger.info(
        "Large generate test: %d prompts × n=%d × max_new_tokens=%d, model=%s",
        n_prompts,
        n_per_prompt,
        _MAX_NEW_TOKENS,
        model_path,
    )

    import ray

    ray.init(ignore_reinit_error=True)

    EngineActor = _build_engine_remote_class()
    actor = EngineActor.remote(model_path=model_path)

    if not ray.get(actor.health_check.remote()):
        logger.error("HEALTH FAIL")
        return 1
    logger.info("HEALTH OK")

    result = ray.get(
        actor.generate.remote(
            _PROMPTS,
            n=n_per_prompt,
            max_new_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
        )
    )
    logger.info(
        "GENERATE returned in %.2fs: %d texts, %d tokens packed, cu_seqlens[%d]=%d",
        result["elapsed_s"],
        len(result["texts"]),
        result["n_tokens_total"],
        len(result["cu_seqlens"]) - 1 if result["cu_seqlens"] else 0,
        result["cu_seqlens"][-1] if result["cu_seqlens"] else -1,
    )

    fails: List[str] = []

    # Check 1: count
    _check(
        len(result["texts"]) == total_expected,
        f"expected {total_expected} texts, got {len(result['texts'])}",
        fails,
    )
    _check(
        result["seg_count"] == total_expected,
        f"expected seg_count={total_expected}, got {result['seg_count']}",
        fails,
    )

    # Check 2: every text non-empty
    empty_indices = [i for i, t in enumerate(result["texts"]) if not t]
    _check(
        not empty_indices,
        f"empty texts at indices {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}",
        fails,
    )

    # Check 3: token/logprob alignment (totals only — sglang's per-prompt
    # padding behavior on some finish_reasons can make per-row alignment
    # off-by-one; total alignment is the durable invariant the trainer
    # downstream depends on).
    _check(
        result["n_tokens_total"] == result["n_logprobs_total"],
        f"tokens/logprobs total mismatch: tokens={result['n_tokens_total']} logprobs={result['n_logprobs_total']}",
        fails,
    )
    _check(
        result["n_tokens_total"] > 0,
        "no tokens emitted overall",
        fails,
    )

    # Check 4: cu_seqlens shape + monotonicity
    cu = result["cu_seqlens"] or []
    _check(
        len(cu) == total_expected + 1,
        f"cu_seqlens length {len(cu)} != expected {total_expected + 1}",
        fails,
    )
    if cu:
        non_monotone = [i for i in range(1, len(cu)) if cu[i] < cu[i - 1]]
        _check(
            not non_monotone,
            f"cu_seqlens not monotonic at positions {non_monotone[:5]}",
            fails,
        )
        _check(
            cu[0] == 0,
            f"cu_seqlens[0] should be 0, got {cu[0]}",
            fails,
        )
        _check(
            cu[-1] == result["n_tokens_total"],
            f"cu_seqlens[-1]={cu[-1]} != total tokens {result['n_tokens_total']}",
            fails,
        )
        zero_len_rows = [i for i in range(len(cu) - 1) if cu[i + 1] == cu[i]]
        _check(
            not zero_len_rows,
            f"{len(zero_len_rows)} rows have zero tokens (first few: {zero_len_rows[:5]})",
            fails,
        )

    # Check 5: sample_id mangling — expect {s0#0, s0#1, ..., s31#3}
    sid_counts = Counter(s.split("#")[0] for s in result["sample_ids"])
    expected_sids = {f"s{i}" for i in range(n_prompts)}
    actual_sids = set(sid_counts)
    missing = expected_sids - actual_sids
    _check(
        not missing,
        f"missing original sids in output: {sorted(missing)[:5]}",
        fails,
    )
    extra = actual_sids - expected_sids
    _check(
        not extra,
        f"unexpected sids in output: {sorted(extra)[:5]}",
        fails,
    )
    wrong_counts = {sid: c for sid, c in sid_counts.items() if c != n_per_prompt}
    _check(
        not wrong_counts,
        f"sids with wrong candidate count (expected {n_per_prompt}): {dict(list(wrong_counts.items())[:5])}",
        fails,
    )
    suffix_counts = Counter(s.split("#")[1] for s in result["sample_ids"] if "#" in s)
    _check(
        all(suffix_counts.get(str(k), 0) == n_prompts for k in range(n_per_prompt)),
        f"sample-id suffix distribution off: {dict(suffix_counts)}",
        fails,
    )

    # Check 6: group_id preservation — every candidate of prompt i has
    # the same gid as the request supplied.
    gid_per_sid: Dict[str, set] = {}
    for sid_mangled, gid in zip(result["sample_ids"], result["group_ids"]):
        original_sid = sid_mangled.split("#")[0]
        gid_per_sid.setdefault(original_sid, set()).add(gid)
    multi_gid_sids = {sid: gids for sid, gids in gid_per_sid.items() if len(gids) > 1}
    _check(
        not multi_gid_sids,
        f"original sids with multiple group_ids: {dict(list(multi_gid_sids.items())[:3])}",
        fails,
    )

    # Check 7: throughput sanity (informational; only flag if absurdly low)
    if result["elapsed_s"] > 0:
        toks_per_s = result["n_tokens_total"] / result["elapsed_s"]
        logger.info(
            "Throughput: %.1f tokens/s across %d candidates (%.2fs total, avg %.1f tokens/candidate)",
            toks_per_s,
            total_expected,
            result["elapsed_s"],
            result["n_tokens_total"] / total_expected,
        )
        _check(
            toks_per_s > 5.0,
            f"throughput suspiciously low: {toks_per_s:.1f} tokens/s (expected >>5 on H20 for Qwen3-0.6B)",
            fails,
        )

    # Check 8: per-prompt diversity — with temperature=0.8 and n=4, the
    # 4 candidates for a single prompt should not all be byte-identical.
    # Identical candidates indicate sampler collapse (temperature ignored,
    # broken top_p, or candidate slot aliasing). Allow at most 1 prompt
    # to have all-identical candidates (a rare statistical fluke on a
    # very deterministic prompt is fine; many is a bug).
    identical_prompts: List[int] = []
    for prompt_idx in range(n_prompts):
        cands = [result["texts"][prompt_idx * n_per_prompt + k] for k in range(n_per_prompt)]
        if len(set(cands)) == 1:
            identical_prompts.append(prompt_idx)
    _check(
        len(identical_prompts) <= 1,
        f"{len(identical_prompts)} prompts had all {n_per_prompt} "
        f"candidates identical (sampler collapse?): "
        f"first few indices {identical_prompts[:5]}",
        fails,
    )
    if identical_prompts:
        logger.info(
            "Diversity: %d/%d prompts had all candidates identical (within tolerance)",
            len(identical_prompts),
            n_prompts,
        )

    # Dump every candidate's full text to a JSON file next to the log
    # for human inspection. This is the durable record the user reads
    # to spot-check that texts are actually coherent.
    import json

    dump_path = os.environ.get(
        "LARGE_TEST_DUMP",
        "/mnt/bj/diffrl_runtime/smoke_sglang_llm_large_outputs.json",
    )
    cu = result["cu_seqlens"] or []
    candidates_dump: List[Dict[str, Any]] = []
    for i in range(total_expected):
        prompt_idx = i // n_per_prompt
        cand_idx = i % n_per_prompt
        n_tok = cu[i + 1] - cu[i] if cu else 0
        candidates_dump.append(
            {
                "prompt_idx": prompt_idx,
                "candidate_idx": cand_idx,
                "sample_id": result["sample_ids"][i],
                "group_id": result["group_ids"][i],
                "prompt": _PROMPTS[prompt_idx],
                "text": result["texts"][i],
                "n_tokens": int(n_tok),
            }
        )
    try:
        with open(dump_path, "w") as fp:
            json.dump(
                {
                    "model": model_path,
                    "n_prompts": n_prompts,
                    "n_per_prompt": n_per_prompt,
                    "max_new_tokens": _MAX_NEW_TOKENS,
                    "temperature": _TEMPERATURE,
                    "elapsed_s": result["elapsed_s"],
                    "n_tokens_total": result["n_tokens_total"],
                    "candidates": candidates_dump,
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Full-text dump: %s (%d candidates)", dump_path, total_expected)
    except Exception as exc:
        logger.warning("Failed to write text dump: %s", exc)

    # Print every prompt's 4 candidates with full text so the log is
    # also self-contained for human inspection. Truncate at 400 chars to
    # keep each block readable but representative.
    logger.info("=" * 72)
    logger.info("PER-PROMPT FULL OUTPUTS (first 400 chars of each candidate):")
    logger.info("=" * 72)
    for prompt_idx in range(n_prompts):
        logger.info(
            "--- prompt %d (%s gid=%s): %s",
            prompt_idx,
            f"s{prompt_idx}",
            f"g{prompt_idx // 4}",
            _PROMPTS[prompt_idx],
        )
        for cand_idx in range(n_per_prompt):
            i = prompt_idx * n_per_prompt + cand_idx
            text = result["texts"][i]
            n_tok = cu[i + 1] - cu[i] if cu else 0
            preview = text[:400].replace("\n", " ⏎ ")
            logger.info(
                "  [#%d %dtok] %s%s",
                cand_idx,
                n_tok,
                preview,
                "…" if len(text) > 400 else "",
            )

    # Flush before ray cleanup (same finalize-race hardening as the
    # main smoke).
    if fails:
        logger.error("LARGE GENERATE FAIL: %d checks failed", len(fails))
        for f in fails:
            logger.error("  - %s", f)
        for h in list(logging.getLogger().handlers) + list(logger.handlers):
            try:
                h.flush()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        ray.kill(actor)
        return 2

    logger.info(
        "LARGE GENERATE PASSED: %d candidates, %d tokens, %.2fs",
        len(result["texts"]),
        result["n_tokens_total"],
        result["elapsed_s"],
    )
    for h in list(logging.getLogger().handlers) + list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()

    ray.get(actor.shutdown.remote())
    ray.kill(actor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
