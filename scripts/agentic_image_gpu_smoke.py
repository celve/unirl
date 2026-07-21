#!/usr/bin/env python
"""Single-GPU rollout smoke for AgenticImageRolloutEngine with REAL models (LIN-577).

Stands up a real SGLang LLM inner (Qwen3-0.6B) + a real trainside SD3.5-medium
diffusion child + a trivial 1-turn env, drives the engine's per-worker drain
(no Ray — monkeypatch the ``_pull`` seam), and verifies each trajectory ends in a
REAL generated image on its own lineage. No trainer needed.

Env vars (default to pod-local copies):
  QWEN3_PATH=/root/unirl/models/local/Qwen3-0.6B
  SD3_PATH=/root/unirl/models/local/stable-diffusion-3.5-medium
  CUDA_VISIBLE_DEVICES=0  (set before torch import)

Run (in the pod's .venv, GPU free of the burn):
  QWEN3_PATH=... SD3_PATH=... .venv/bin/python scripts/agentic_image_gpu_smoke.py
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # must precede torch import

LLM_PATH = os.environ.get("QWEN3_PATH", "/root/unirl/models/local/Qwen3-0.6B")
SD3_PATH = os.environ.get("SD3_PATH", "/root/unirl/models/local/stable-diffusion-3.5-medium")
OUT_PNG = os.environ.get("OUT_PNG", "/root/agentic_image_out.png")
PROMPTS = ["a fluffy calico cat on a windowsill", "a red sports car at sunset", "a snowy mountain peak"]
M = 2  # images per trajectory


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_engine():
    from unirl.models.sd3.config import SD3PipelineConfig
    from unirl.models.sd3.pipeline import SD3Pipeline
    from unirl.rollout.engine.agentic.image_config import AgenticImageRolloutEngineConfig
    from unirl.rollout.engine.agentic.image_engine import AgenticImageRolloutEngine
    from unirl.rollout.engine.base import BaseEngineConfig
    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
    from unirl.sde.kernels import FlowSDEStrategy
    from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

    class TrainsideSD3Config(BaseEngineConfig):
        """Bridge: the engine calls diffusion.make_engine(strategy, device, rank, model_config).

        TrainsideRolloutEngine takes a materialized pipeline (not a config), and
        SD3Bundle.from_config auto-loads onto cuda when config.device is None.
        """

        def make_engine(self, **deps):
            strategy = deps.get("strategy") or FlowSDEStrategy()
            _log("loading SD3.5-medium (trainside) onto GPU ...")
            pipe = SD3Pipeline.from_config(
                SD3PipelineConfig(pretrained_model_ckpt_path=SD3_PATH, model_precision="bf16", shift=3.0),
                strategy=strategy,
            )
            _log("SD3.5-medium loaded")
            return TrainsideRolloutEngine(pipeline=pipe, stage_attrs=("diffusion",))

    class OneTurnEnv:
        """Trivial env: the first assistant turn is terminal."""

        max_turns = 1

        def reset(self, request):
            return request

        def step(self, sample):
            return None, True, {}

        def close(self, sample):
            pass

    cfg = AgenticImageRolloutEngineConfig(
        inner=SGLangEngineConfig(
            pretrained_model_ckpt_path=LLM_PATH,
            backend="native",
            tp_size=1,
            max_new_tokens=64,
            temperature=1.0,
            top_p=1.0,
            concurrency=2,
            chat_template_kwargs={"enable_thinking": False},
            engine_kwargs={"mem_fraction_static": 0.25, "skip_server_warmup": True, "attention_backend": "triton"},
        ),
        env=OneTurnEnv(),
        diffusion=TrainsideSD3Config(),
        diffusion_sampling=DiffusionSamplingParams(
            samples_per_prompt=M,
            num_inference_steps=10,
            guidance_scale=1.0,
            height=512,
            width=512,
            eta=0.7,
            seed=42,
            sde_indices=list(range(10)),
            autocast_precision="bf16",
        ),
        episode_sampling=ARSamplingParams(samples_per_prompt=1, temperature=1.0, top_p=1.0, max_new_tokens=64),
        max_turns=1,
        per_worker_concurrency=2,
        sleep_diffusion_on_start=False,  # trainside sleep() is a no-op; keep both resident on the big GPU
    )
    _log("booting SGLang inner (Qwen3-0.6B) + building AgenticImageRolloutEngine ...")
    engine = AgenticImageRolloutEngine(cfg, device=None, strategy=FlowSDEStrategy(), rank=0, model_config=None)
    _log("engine built (inner + diffusion child up)")
    return engine


def main() -> None:
    from unirl.types.primitives import Texts
    from unirl.types.sample import Part, Sample
    from unirl.types.sample_id import parent_id
    from unirl.types.sampling import DiffusionSamplingParams

    _log(f"LLM_PATH={LLM_PATH}")
    _log(f"SD3_PATH={SD3_PATH}")
    engine = build_engine()

    # The documented no-Ray seam: feed tasks from an in-process queue.
    queue = [
        Sample.request(
            Part.input([f"r0:{i}"], primitives={"text": Texts(texts=[p])}, control={"ar": {"stop": ["</tool_call>"]}})
        )
        for i, p in enumerate(PROMPTS)
    ]
    lock = threading.Lock()

    def _pull(coordinator, role_name):
        with lock:
            return queue.pop(0) if queue else None

    engine._pull = _pull

    _log(f"driving run_drain over {len(PROMPTS)} prompts (LLM turns + terminal SD3 diffusion) ...")
    t0 = time.time()
    engine.reset_round()
    engine.run_drain(coordinator=None, role_name="gpu-smoke")
    trajs = engine.drain_completed()
    _log(f"run_drain done in {time.time() - t0:.1f}s -> {len(trajs)} trajectories")

    ok = True
    first_image = None
    for tr in trajs:
        gens = tr.gen_parts()
        ar = [g for g in gens if not isinstance(g.sampling_params, DiffusionSamplingParams)]
        diff = [g for g in gens if isinstance(g.sampling_params, DiffusionSamplingParams)]
        imgs = diff[-1].primitives.get("image") if diff else None
        n_img = 0 if imgs is None else len(imgs)
        answer = ar[-1].primitives["text"].texts[0][:50] if ar and "text" in ar[-1].primitives else None
        # connected-lineage invariant: diffusion ids are children of the last AR turn
        lineage_ok = bool(diff and ar) and all(parent_id(sid) == ar[-1].sample_ids[0] for sid in diff[-1].sample_ids)
        _log(
            f"root={tr.parts[0].sample_ids[0]} ar_turns={len(ar)} diff_parts={len(diff)} "
            f"images={n_img} lineage_ok={lineage_ok} answer={answer!r}"
        )
        if not (len(ar) >= 1 and len(diff) == 1 and n_img == M and lineage_ok):
            ok = False
        if first_image is None and imgs is not None:
            first_image = imgs

    if first_image is not None:
        try:
            first_image.to_pils()[0].save(OUT_PNG)
            _log(f"saved first generated image -> {OUT_PNG}")
        except Exception as exc:  # noqa: BLE001 — image-save is a bonus, not the test
            _log(f"(image save skipped: {exc})")

    engine.shutdown()
    _log("engine shutdown")
    if ok and len(trajs) == len(PROMPTS):
        print("\nagentic_image_gpu_smoke: PASSED — every trajectory ended in a real SD3 image on its own lineage")
    else:
        raise SystemExit("agentic_image_gpu_smoke: FAILED — see per-trajectory lines above")


if __name__ == "__main__":
    main()
