#!/usr/bin/env python
"""No-Ray CPU smoke for AgenticImageRolloutEngine (LIN-577).

Drives the per-worker drain DIRECTLY (monkeypatching the documented ``_pull``
seam so no Ray/GPU is needed) with stub inner / env / diffusion engines, and
asserts each finished trajectory gains a terminal diffusion image Part on its
OWN lineage (connected lineage). Also checks terminal-answer extraction
(``<answer>`` regex + marker + fallback) and the weight-sync ``track_prefix``
demux.

Run: python scripts/agentic_image_smoke.py
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

import torch

from unirl.rollout.engine.agentic.image_config import AgenticImageRolloutEngineConfig
from unirl.rollout.engine.agentic.image_engine import AgenticImageRolloutEngine
from unirl.rollout.engine.base import BaseEngineConfig, BaseSingleTurnRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sample_id import parent_id
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

ANSWER = "a fluffy calico cat"
M = 2  # images per trajectory (diffusion samples_per_prompt)


# --------------------------------------------------------------------------
# Stubs (no Ray, no GPU) — implement just the contracts the engine touches.
# --------------------------------------------------------------------------


class _StubInner(BaseSingleTurnRolloutEngine):
    def __init__(self) -> None:
        self.sync_calls: List[Tuple[str, dict]] = []
        self.slept = 0
        self.woke = 0

    def generate(self, sample: Sample) -> Sample:
        n = sample.parts[-1].batch_size
        return sample.with_filled_frontier(primitives={"text": Texts(texts=[f"<answer>{ANSWER}</answer>"] * n)})

    def shutdown(self) -> None:  # noqa: D401
        pass

    def sleep(self) -> None:
        self.slept += 1

    def wake_up(self) -> None:
        self.woke += 1

    def update_weights_from_distributed(self, **kw: Any) -> None:
        self.sync_calls.append(("distributed", kw))


class _StubDiffusion(BaseSingleTurnRolloutEngine):
    def __init__(self) -> None:
        self.sync_calls: List[Tuple[str, dict]] = []
        self.slept = 0
        self.woke = 0
        self.last_prompts: List[str] = []

    def generate(self, sample: Sample) -> Sample:
        # Record the re-rooted condition text the engine forwarded, then fill the
        # shell with dummy images + a (select-able) conditions payload.
        prompts = sample.parts[0].primitives.get("text")
        self.last_prompts = list(prompts.texts) if isinstance(prompts, Texts) else []
        n = sample.parts[-1].batch_size
        return sample.with_filled_frontier(
            primitives={"image": Images(pixels=torch.zeros(n, 3, 8, 8))},
            conditions={"text": Texts(texts=["cond"] * n)},  # stand-in Condition (Batch, select-able)
        )

    def shutdown(self) -> None:  # noqa: D401
        pass

    def sleep(self) -> None:
        self.slept += 1

    def wake_up(self) -> None:
        self.woke += 1

    def update_weights_from_distributed(self, **kw: Any) -> None:
        self.sync_calls.append(("distributed", kw))


class _StubEngineConfig(BaseEngineConfig):
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def make_engine(self, **deps: Any) -> Any:
        return self._engine


class _StubEnv:
    """1-turn env: the first assistant turn is terminal."""

    max_turns = 1

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Any, bool, Dict[str, Any]]:
        return None, True, {}

    def close(self, sample: Sample) -> None:
        pass


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _build_engine(marker: Any = None) -> Tuple[AgenticImageRolloutEngine, _StubInner, _StubDiffusion]:
    inner, diffusion = _StubInner(), _StubDiffusion()
    cfg = AgenticImageRolloutEngineConfig(
        inner=_StubEngineConfig(inner),
        env=_StubEnv(),
        diffusion=_StubEngineConfig(diffusion),
        diffusion_sampling=DiffusionSamplingParams(samples_per_prompt=M, num_inference_steps=2, height=8, width=8),
        episode_sampling=ARSamplingParams(samples_per_prompt=1),
        max_turns=1,
        per_worker_concurrency=2,
        answer_marker=marker,
        sleep_diffusion_on_start=True,
    )
    engine = AgenticImageRolloutEngine(cfg, device=None, strategy=None, rank=0, model_config=None)
    return engine, inner, diffusion


def _tasks(prompts: List[str]) -> List[Sample]:
    return [
        Sample.request(
            Part.input([f"r0:{i}"], primitives={"text": Texts(texts=[p])}, control={"ar": {"stop": ["</tool_call>"]}})
        )
        for i, p in enumerate(prompts)
    ]


def _patch_pull(engine: AgenticImageRolloutEngine, tasks: List[Sample]) -> None:
    lock, queue = threading.Lock(), list(tasks)

    def _pull(coordinator: Any, role_name: str):  # the documented no-Ray seam
        with lock:
            return queue.pop(0) if queue else None

    engine._pull = _pull  # type: ignore[assignment]


def _drive(engine: AgenticImageRolloutEngine, prompts: List[str]) -> List[Sample]:
    _patch_pull(engine, _tasks(prompts))
    engine.reset_round()
    engine.run_drain(coordinator=None, role_name="test")
    return engine.drain_completed()


def _traj(answer_text: str, prompt: str = "draw a cat") -> Sample:
    s = Sample.request(Part.input(["r0:0"], primitives={"text": Texts(texts=[prompt])}))
    s = s.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
    return s.with_filled_frontier(primitives={"text": Texts(texts=[answer_text])})


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_connected_lineage() -> None:
    prompts = ["draw a cat", "draw a dog", "draw a bird"]
    engine, inner, diffusion = _build_engine()
    out = _drive(engine, prompts)
    assert len(out) == len(prompts), f"expected {len(prompts)} trajectories, got {len(out)}"
    for tr in out:
        gens = tr.gen_parts()
        ar_gens = [g for g in gens if not isinstance(g.sampling_params, DiffusionSamplingParams)]
        diff_gens = [g for g in gens if isinstance(g.sampling_params, DiffusionSamplingParams)]
        assert len(ar_gens) >= 1, "expected >=1 AR gen Part"
        assert len(diff_gens) == 1, f"expected exactly one diffusion Part, got {len(diff_gens)}"
        dp = diff_gens[0]
        assert "image" in dp.primitives and len(dp.primitives["image"]) == M, "diffusion Part missing M images"
        assert dp.batch_size == M, f"diffusion Part batch {dp.batch_size} != M={M}"
        # lineage: the diffusion ids are children of the last AR turn's id
        last_ar_id = ar_gens[-1].sample_ids[0]
        assert all(parent_id(sid) == last_ar_id for sid in dp.sample_ids), (
            "diffusion Part not a child of the last AR turn"
        )
        # conditions provenance rode along from the diffusion child
        assert "text" in dp.conditions and len(dp.conditions["text"]) == M, "diffusion conditions not carried"
    # the extracted answer was re-rooted and forwarded as the diffusion condition
    assert diffusion.last_prompts == [ANSWER] * len(prompts), f"diffusion saw {diffusion.last_prompts!r}"
    # colocate wake/sleep dance happened (diffusion woke, inner slept)
    assert diffusion.woke >= 1 and inner.slept >= 1, "expected colocate wake/sleep dance"
    print("OK  connected-lineage terminal diffusion Part on each trajectory (extract -> re-root -> diffuse)")


def check_extraction() -> None:
    engine, _, _ = _build_engine()  # <answer> regex mode
    assert engine._extract_terminal(_traj("noise <answer>a red fox</answer> tail")) == "a red fox"
    assert engine._extract_terminal(_traj("no tags here")) == "no tags here"
    print("OK  terminal-answer extraction (<answer> regex)")

    # The marker path lazily imports unirl.models.pe.instruction, whose package
    # __init__ pulls the PE model stack (diffusers/transformers). Skip cleanly if
    # that stack isn't installed (e.g. the minimal CPU smoke env).
    marker_engine, _, _ = _build_engine(marker="Revised Prompt:")
    try:
        got = marker_engine._extract_terminal(_traj("reasoning...\nRevised Prompt:\na neon city at dusk"))
    except ImportError:
        print("SKIP  marker extraction (unirl.models.pe stack not installed)")
        return
    assert got.strip() == "a neon city at dusk", f"marker extract got {got!r}"
    # off-format output falls back to the original prompt
    fb = marker_engine._extract_terminal(_traj("no marker at all", prompt="draw a cat"))
    assert fb == "draw a cat", f"marker fallback got {fb!r}"
    print("OK  marker extraction (marker + fallback-to-original)")


def check_weight_sync_demux() -> None:
    engine, inner, diffusion = _build_engine()
    engine.update_weights_from_distributed(names=[], dtypes=[], shapes=[], group_name="g", track_prefix="diffusion")
    assert len(diffusion.sync_calls) == 1 and len(inner.sync_calls) == 0, "diffusion track misrouted"
    engine.update_weights_from_distributed(names=[], dtypes=[], shapes=[], group_name="g", track_prefix="ar")
    assert len(inner.sync_calls) == 1, "ar track misrouted"
    for bad in ({"track_prefix": "bogus"}, {}):
        try:
            engine.update_weights_from_distributed(names=[], dtypes=[], shapes=[], group_name="g", **bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass
    print("OK  weight-sync track_prefix demux (ar / diffusion routed; unknown/missing rejected)")


def main() -> None:
    check_connected_lineage()
    check_extraction()
    check_weight_sync_demux()
    print("\nagentic_image_smoke: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
