#!/usr/bin/env python
"""Multi-turn tool trajectory for AgenticImageRolloutEngine, with a full dump (LIN-577).

Same engine as the GPU smoke, but the env is a REAL ToolEnvironment + CalculatorTool
(max_turns=4), so each trajectory is genuinely multi-turn: the agent emits a
``<tool_call>{calculator ...}</tool_call>``, the env computes and returns the result as
the next turn's observation, the agent then answers in ``<answer>...</answer>``, and the
engine finally conditions a diffusion image on that answer. Dumps the whole turn-by-turn
trajectory so you can see WHAT the agent queried and WHAT came back.

Env vars (defaults to pod-local copies):
  QWEN_PATH=/root/unirl/models/local/Qwen3-8B   (8B tool-calls reliably; 0.6B is weak)
  SD3_PATH=/root/unirl/models/local/stable-diffusion-3.5-medium
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # before torch import

QWEN_PATH = os.environ.get("QWEN_PATH", "/root/unirl/models/local/Qwen3-8B")
SD3_PATH = os.environ.get("SD3_PATH", "/root/unirl/models/local/stable-diffusion-3.5-medium")
PROMPTS = [
    "What is 1234 * 5678? Use the calculator tool for the arithmetic, then put the final numeric answer inside <answer>...</answer>.",
    "A shop sold 47 boxes with 89 items each. How many items total? Use the calculator tool, then give the answer in <answer>...</answer>.",
]
M = 1  # images per trajectory (keep small — this run is about the trajectory, not image count)


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
    from unirl.rollout.loop.tool_environment import ToolEnvironment
    from unirl.rollout.loop.tools.calculator import CalculatorTool
    from unirl.sde.kernels import FlowSDEStrategy
    from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

    class TrainsideSD3Config(BaseEngineConfig):
        def make_engine(self, **deps):
            strategy = deps.get("strategy") or FlowSDEStrategy()
            _log("loading SD3.5-medium (trainside) onto GPU ...")
            pipe = SD3Pipeline.from_config(
                SD3PipelineConfig(pretrained_model_ckpt_path=SD3_PATH, model_precision="bf16", shift=3.0),
                strategy=strategy,
            )
            _log("SD3.5-medium loaded")
            return TrainsideRolloutEngine(pipeline=pipe, stage_attrs=("diffusion",))

    cfg = AgenticImageRolloutEngineConfig(
        inner=SGLangEngineConfig(
            pretrained_model_ckpt_path=QWEN_PATH,
            backend="native",
            tp_size=1,
            max_new_tokens=384,
            temperature=0.7,
            top_p=0.9,
            concurrency=2,
            chat_template_kwargs={"enable_thinking": False},  # tools auto-injected by the engine
            engine_kwargs={"mem_fraction_static": 0.20, "skip_server_warmup": True, "attention_backend": "triton"},
        ),
        env=ToolEnvironment(tools=[CalculatorTool()], max_turns=4),
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
        episode_sampling=ARSamplingParams(samples_per_prompt=1, temperature=0.7, top_p=0.9, max_new_tokens=384),
        max_turns=4,  # must match the env's max_turns
        per_worker_concurrency=2,
        answer_marker=None,  # extract <answer>...</answer> from the terminal turn
        sleep_diffusion_on_start=False,
    )
    _log(f"booting SGLang inner ({os.path.basename(QWEN_PATH)}) + engine ...")
    engine = AgenticImageRolloutEngine(cfg, device=None, strategy=FlowSDEStrategy(), rank=0, model_config=None)
    _log("engine built (inner + diffusion child up)")
    return engine


def dump_trajectory(tr, idx: int) -> None:
    from unirl.types.primitives import Texts
    from unirl.types.sampling import DiffusionSamplingParams

    print(
        f"\n{'=' * 78}\nTRAJECTORY {idx}  (root={tr.parts[0].sample_ids[0]})  —  {len(tr.parts)} parts\n{'=' * 78}",
        flush=True,
    )
    for i, p in enumerate(tr.parts):
        if isinstance(p.sampling_params, DiffusionSamplingParams):
            kind = "DIFFUSION  (terminal image, conditioned on the <answer>)"
        elif p.is_gen:
            kind = "ASSISTANT  (LLM turn — may hold a <tool_call> or the <answer>)"
        elif p.is_root:
            kind = "USER       (prompt)"
        else:
            kind = "TOOL       (observation returned by the env)"
        txt = p.primitives.get("text")
        if isinstance(txt, Texts) and txt.texts:
            body = txt.texts[0].strip()
            body = body if len(body) <= 700 else body[:700] + " …[truncated]"
        elif "image" in p.primitives:
            body = (
                f"<{len(p.primitives['image'])} generated image(s), pixels={tuple(p.primitives['image'].pixels.shape)}>"
            )
        else:
            body = "(no text/image payload)"
        print(f"\n[part {i}] {kind}\n{'-' * 78}\n{body}", flush=True)


def main() -> None:
    from unirl.types.primitives import Texts
    from unirl.types.sample import Part, Sample

    _log(f"QWEN_PATH={QWEN_PATH}")
    engine = build_engine()

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

    _log(f"driving run_drain over {len(PROMPTS)} tool-using prompts (max_turns=4) ...")
    t0 = time.time()
    engine.reset_round()
    engine.run_drain(coordinator=None, role_name="tool-traj")
    trajs = engine.drain_completed()
    _log(f"run_drain done in {time.time() - t0:.1f}s -> {len(trajs)} trajectories")

    trajs = sorted(trajs, key=lambda t: t.parts[0].sample_ids[0])
    for idx, tr in enumerate(trajs):
        dump_trajectory(tr, idx)

    engine.shutdown()
    _log("engine shutdown")


if __name__ == "__main__":
    main()
