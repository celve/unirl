#!/usr/bin/env python3
"""Trainside rollout+replay smoke for the Sample → Sample Qwen3 AR pipeline (LIN-479).

Loads the IN-PROCESS Qwen3 pipeline, wraps it in a ``TrainsideRolloutEngine``,
builds a request ``Sample`` by hand, runs ``generate`` (rollout) — then
RE-TOKENIZES conditions from the filled Sample and runs the AR stage's
``replay``, asserting the replayed per-token log-probs reproduce the rollout's
stored ``TextSegment.log_probs`` (ratio ≈ 1). That shared-bundle invariant —
rollout and replay over the same weights agree — is the correctness bar for the
model bundle. Conditions are NOT cached on the Part (the trainside path leaves
``Part.conditions`` empty), so this also exercises the re-tokenize path end to end.

No external inference server (the trainside engine runs the pipeline's own stages
in-process), no training loop, no reward. Run on a GPU pod (1 free GPU), torch venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_ar_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.algorithms.base import rollout_replay_logp_absdiff
from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.models.qwen3.pipeline import Qwen3Pipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment

# Rollout and replay use the SAME in-process model, so the gap is the
# autoregress-vs-teacher-forcing numeric gap (not an engine mismatch); it should
# be small. Loose pending pod calibration — the actual value is logged.
_ABSDIFF_MEAN_MAX = 0.2


def _log(msg: str) -> None:
    print(f"[trainside-ar] {msg}", flush=True)


def build_request_sample(n: int) -> Sample:
    """2 prompts, ``n`` completions each: ``[input, ar gen-shell]`` (P*n samples)."""
    prompts = ["The capital of France is", "Two plus two equals"]
    input_part = Part.input([f"p{i}" for i in range(len(prompts))], primitive=Texts(texts=prompts), control={})
    ar_params = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part).fork(n, sampling_params=ar_params)


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3-4B dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = Qwen3PipelineConfig(pretrained_model_ckpt_path=model_path, device="cuda:0")
    n = 2  # completions per prompt → exercises the sibling fan-out (n>1)
    try:
        _log("loading Qwen3Pipeline.from_config (bundle on cuda:0) ...")
        pipeline = Qwen3Pipeline.from_config(config)
        engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=("ar",))

        sample = build_request_sample(n)
        gen_in = sample.parts[-1]
        n_expect = 2 * n
        _log(f"request: {len(sample.parts)} parts; gen ids={list(gen_in.sample_ids)}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)

        # ---- rollout: the frontier gen Part is filled ----
        assert len(out.parts) == 2, f"expected [input, gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert list(gen.sample_ids) == list(gen_in.sample_ids), "gen ids changed"
        assert len(gen.sample_ids) == n_expect, f"expected {n_expect} samples; got {len(gen.sample_ids)}"
        assert isinstance(gen.segment, TextSegment), f"segment must be TextSegment; got {type(gen.segment)}"
        assert gen.segment.log_probs is not None, "TextSegment.log_probs is None (no rollout logp to compare)"
        assert isinstance(gen.primitive, Texts) and len(gen.primitive.texts) == n_expect, (
            f"expected {n_expect} decoded texts"
        )
        assert not gen.conditions, "trainside path must leave Part.conditions empty (replay re-encodes)"
        _log(f"rollout PASS: {n_expect} completions; conditions empty ✓")

        # ---- replay: re-tokenize conditions, reproduce log_probs (ratio ≈ 1) ----
        _log("re-tokenizing conditions from the filled Sample and replaying the AR stage ...")
        temperature = float(gen.sampling_params.temperature)
        texts = out.conditioning()[0]
        control = out.parts[0].control
        model = pipeline.ar.trainable_module()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                conds = pipeline._conditions_for(texts, control)
                new_logp = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
        finally:
            model.train(was_training)

        old_logp = gen.segment.log_probs.to(device=new_logp.device, dtype=new_logp.dtype)
        assert new_logp.shape == old_logp.shape, (
            f"replay logp shape {tuple(new_logp.shape)} != rollout {tuple(old_logp.shape)}"
        )
        assert torch.isfinite(new_logp).all(), "replay produced non-finite log-probs"
        m = rollout_replay_logp_absdiff(new_logp, old_logp)
        mean, mx = m["rollout_replay_logp_absdiff_mean"], m["rollout_replay_logp_absdiff_max"]
        _log(f"ratio≈1 check: mean|Δlogp|={mean:.3e} max|Δlogp|={mx:.3e} (threshold mean<{_ABSDIFF_MEAN_MAX})")
        assert mean < _ABSDIFF_MEAN_MAX, f"rollout↔replay logp drift too large: mean|Δlogp|={mean:.3e}"

        _log("TRAINSIDE AR SMOKE PASSED ✅  (rollout filled the Sample; replay re-tokenize reproduces log_probs)")
        return 0
    except Exception:
        _log("TRAINSIDE AR SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
