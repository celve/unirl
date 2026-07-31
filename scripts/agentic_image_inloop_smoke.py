#!/usr/bin/env python
"""No-Ray CPU smoke for IN-LOOP image turns on AgenticImageRolloutEngine (LIN-577).

Sibling of ``agentic_image_smoke.py`` (which covers the v1 *terminal* shape). Drives
the per-worker drain directly through the documented ``_pull`` seam with stub inner /
env / diffusion engines, scripting an agent that draws, inspects, and redraws.

Asserts the four things that make in-loop rendering actually work:

1. **Trainable.** Each rendered image is a diffusion *gen* Part on the trajectory's
   own lineage, carrying its ``LatentSegment`` — a policy-gradient target, not a
   mask-0 observation.
2. **Correctly rendered.** The image Parts are tagged ``role="tool"``, so
   ``build_vision_conversations`` does not fuse them into the surrounding agent text.
   The negative control asserts that WITHOUT the tag they do fuse — that is the bug
   the tag exists to prevent, and it is invisible at runtime.
3. **Edited, not redrawn.** The second draw conditions on the first image (ti2i),
   with the source image row-aligned to its prompt.
4. **Batched.** Concurrent image turns coalesce into shared ``generate`` calls
   instead of serializing one-per-trajectory behind the diffusion engine's mutex.

Run: python scripts/agentic_image_inloop_smoke.py
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.rollout.engine.agentic.image_config import AgenticImageRolloutEngineConfig
from unirl.rollout.engine.agentic.image_engine import AgenticImageRolloutEngine
from unirl.rollout.engine.base import BaseEngineConfig, BaseSingleTurnRolloutEngine
from unirl.rollout.engine.sglang.utils.conversations import build_vision_conversations
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sample_id import parent_id
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import make_image_segment

DRAW_TURNS = 2  # draws per trajectory before the agent answers
PROMPTS = ["a calico cat", "a golden retriever", "a snowy owl"]
# Generous next to the stub's instant "denoise", so the collector reliably sees every
# thread park. Real recipes leave this at the 50ms default.
WINDOW_S = 1.0


# --------------------------------------------------------------------------
# Stubs (no Ray, no GPU) — implement just the contracts the engine touches.
# --------------------------------------------------------------------------


class _StubInner(BaseSingleTurnRolloutEngine):
    """Agent that narrates each turn; the env decides which turns are draws."""

    def generate(self, sample: Sample) -> Sample:
        n = sample.parts[-1].batch_size
        turn = len([p for p in sample.gen_parts() if not isinstance(p.sampling_params, DiffusionSamplingParams)])
        return sample.with_filled_frontier(primitives={"text": Texts(texts=[f"agent turn {turn}"] * n)})

    def shutdown(self) -> None:  # noqa: D401
        pass

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass


class _StubDiffusion(BaseSingleTurnRolloutEngine):
    """Records every batch it is handed, so the test can prove coalescing."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.batch_sizes: List[int] = []
        self.saw_image_input: List[bool] = []
        self.prompts: List[List[str]] = []
        self.source_pixels: List[Optional[torch.Tensor]] = []
        self._tick = 0

    def generate(self, sample: Sample) -> Sample:
        n = sample.parts[-1].batch_size
        root = sample.parts[0]
        with self.lock:
            self.batch_sizes.append(n)
            self.saw_image_input.append(sample.has_image_input())
            texts = root.primitives.get("text")
            self.prompts.append(list(texts.texts) if isinstance(texts, Texts) else [])
            source = root.primitives.get("image")
            self.source_pixels.append(source.pixels.clone() if isinstance(source, Images) else None)
            self._tick += 1
            tick = self._tick
        # Distinct pixel values per call so the ti2i check can prove the SECOND draw
        # was conditioned on the FIRST draw's output rather than on anything else.
        return sample.with_filled_frontier(
            segment=make_image_segment(latents=torch.zeros(n, 2, 4)),
            primitives={"image": Images(pixels=torch.full((n, 3, 8, 8), float(tick)))},
        )

    def shutdown(self) -> None:  # noqa: D401
        pass

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass


class _StubEngineConfig(BaseEngineConfig):
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def make_engine(self, **deps: Any) -> Any:
        return self._engine


class _ScriptedDrawEnv:
    """Emits ``DRAW_TURNS`` draw calls, then ends the trajectory.

    Mirrors what ``ToolEnvironment.step`` hands back for a parsed tool call: a
    non-None text observation (the engine replaces it with the rendered image) plus
    the parsed call in ``info["tool_calls"]``.
    """

    max_turns = 16

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Any, bool, Dict[str, Any]]:
        drawn = len([p for p in sample.gen_parts() if isinstance(p.sampling_params, DiffusionSamplingParams)])
        if drawn >= DRAW_TURNS:
            return None, True, {"tool_calls": [None]}
        call = {"name": "draw", "arguments": {"prompt": f"{PROMPTS[0]} take {drawn + 1}"}}
        return Texts(texts=["(draw dispatched)"]), False, {"tool_calls": [call]}

    def close(self, sample: Sample) -> None:
        pass


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _build_engine() -> Tuple[AgenticImageRolloutEngine, _StubDiffusion]:
    diffusion = _StubDiffusion()
    cfg = AgenticImageRolloutEngineConfig(
        inner=_StubEngineConfig(_StubInner()),
        env=_ScriptedDrawEnv(),
        diffusion=_StubEngineConfig(diffusion),
        diffusion_sampling=DiffusionSamplingParams(samples_per_prompt=1, num_inference_steps=2, height=8, width=8),
        episode_sampling=ARSamplingParams(samples_per_prompt=1),
        max_turns=16,
        per_worker_concurrency=len(PROMPTS),
        in_loop_images=True,
        sleep_diffusion_on_start=False,
        draw_batch_window_s=WINDOW_S,
    )
    engine = AgenticImageRolloutEngine(cfg, device=None, strategy=None, rank=0, model_config=None)
    return engine, diffusion


def _drive(engine: AgenticImageRolloutEngine) -> List[Sample]:
    tasks = [
        Sample.request(Part.input([f"r0:{i}"], primitives={"text": Texts(texts=[p])})) for i, p in enumerate(PROMPTS)
    ]
    lock, pending = threading.Lock(), list(tasks)

    def _pull(coordinator: Any, role_name: str):  # the documented no-Ray seam
        with lock:
            return pending.pop(0) if pending else None

    engine._pull = _pull  # type: ignore[assignment]
    engine.reset_round()
    engine.run_drain(coordinator=None, role_name="test")
    return engine.drain_completed()


def _diff_parts(tr: Sample) -> List[Part]:
    return [p for p in tr.gen_parts() if isinstance(p.sampling_params, DiffusionSamplingParams)]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_trainable_image_parts(out: List[Sample]) -> None:
    assert len(out) == len(PROMPTS), f"expected {len(PROMPTS)} trajectories, got {len(out)}"
    for tr in out:
        diff = _diff_parts(tr)
        assert len(diff) == DRAW_TURNS, f"expected {DRAW_TURNS} diffusion Parts, got {len(diff)}"
        for dp in diff:
            assert dp.is_gen, "image Part is not a gen Part — it would never be trained"
            assert dp.segment is not None, "image Part lost its LatentSegment in the write-back"
            assert "image" in dp.primitives, "image Part carries no image primitive"
            assert dp.batch_size == 1, f"in-loop image Part must stay width 1, got {dp.batch_size}"
        # lineage: each image Part hangs off the AR turn that requested it
        for dp in diff:
            idx = tr.parts.index(dp)
            parent = tr.parts[idx - 1]
            assert all(parent_id(sid) in set(parent.sample_ids) for sid in dp.sample_ids), (
                "image Part is not a child of the preceding turn — lineage broken"
            )
    print(f"OK  {DRAW_TURNS} trainable diffusion gen Parts per trajectory, LatentSegment intact, lineage connected")


def check_role_tagging_prevents_fusion(out: List[Sample]) -> None:
    tr = out[0]
    for dp in _diff_parts(tr):
        assert dp.role == "tool", f"image Part role is {dp.role!r}; must be 'tool' or it fuses with agent text"

    conversations, images, _ = build_vision_conversations(tr)
    convo = conversations[0]
    image_msgs = [m for m in convo if any(b.get("type") == "image" for b in m["content"])]
    assert len(images[0]) == DRAW_TURNS, f"expected {DRAW_TURNS} images in the conversation, got {len(images[0])}"
    assert len(image_msgs) == DRAW_TURNS, (
        f"expected each image in its own message, got {len(image_msgs)} messages holding {DRAW_TURNS} images"
    )
    for m in image_msgs:
        assert m["role"] == "tool", f"image landed in a {m['role']!r} message, not 'tool'"
        n_img = sum(1 for b in m["content"] if b.get("type") == "image")
        assert n_img == 1, f"{n_img} images fused into one message — the agent cannot tell them apart"
    print("OK  images render as separate 'tool' messages — draw/critique/redraw order preserved")


def check_untagged_would_fuse(out: List[Sample]) -> None:
    """Negative control: strip the role tag and the conversation collapses.

    This is the failure this design exists to prevent. It is silent at runtime — the
    rollout still produces images and the trainer still steps — so it is asserted
    here rather than left to review.
    """
    from unirl.types.sample import _part_with_field

    tr = out[0]
    diff_ids = {id(p) for p in _diff_parts(tr)}  # Part is unhashable; compare by identity
    untagged = Sample(parts=[_part_with_field(p, "role", None) if id(p) in diff_ids else p for p in tr.parts])
    convo = build_vision_conversations(untagged)[0][0]
    fused = [m for m in convo if sum(1 for b in m["content"] if b.get("type") == "image") > 1]
    assert fused, "expected untagged image Parts to fuse into one assistant message (the bug); they did not"
    assert fused[0]["role"] == "assistant", f"fused message role {fused[0]['role']!r}"
    print("OK  negative control: without role='tool' the images DO fuse into one assistant message")


def check_ti2i_edits_previous_image(diffusion: _StubDiffusion) -> None:
    t2i = [i for i, saw in enumerate(diffusion.saw_image_input) if not saw]
    ti2i = [i for i, saw in enumerate(diffusion.saw_image_input) if saw]
    assert t2i, "expected at least one t2i call (the first draw)"
    assert ti2i, "expected at least one ti2i call (the redraw) — the edit path never fired"
    for i in ti2i:
        src = diffusion.source_pixels[i]
        assert src is not None, "ti2i call carried no source image"
        assert src.shape[0] == diffusion.batch_sizes[i], "source images not row-aligned to prompts"
        # The stub stamps call k's output with value k, so a source of all-zeros would
        # mean we conditioned on something other than a previously rendered image.
        assert torch.all(src > 0), "ti2i source is not a previously rendered image"
    print(f"OK  ti2i edit path: {len(t2i)} t2i call(s) then {len(ti2i)} image-conditioned redraw(s)")


def check_batching(diffusion: _StubDiffusion) -> None:
    k, t = len(PROMPTS), DRAW_TURNS
    calls, largest = len(diffusion.batch_sizes), max(diffusion.batch_sizes)
    assert largest > 1, (
        f"no coalescing: every one of {calls} calls had batch 1. Concurrent image turns would "
        "serialize behind the diffusion engine's mutex."
    )
    assert calls < k * t, f"{calls} diffusion calls for {k} trajectories x {t} draws — no batching happened"
    print(f"OK  coalescing: {calls} diffusion call(s) for {k}x{t}={k * t} image turns (largest batch {largest})")


def check_terminal_mode_unchanged() -> None:
    """v1 shape still rejects the in-loop-only config combinations."""
    for kwargs, want in (
        ({"in_loop_images": True, "sleep_diffusion_on_start": True}, "sleep_diffusion_on_start=false"),
        ({"in_loop_images": True, "sleep_diffusion_on_start": False, "m": 2}, "samples_per_prompt == 1"),
    ):
        m = kwargs.pop("m", 1)
        try:
            AgenticImageRolloutEngine(
                AgenticImageRolloutEngineConfig(
                    inner=_StubEngineConfig(_StubInner()),
                    env=_ScriptedDrawEnv(),
                    diffusion=_StubEngineConfig(_StubDiffusion()),
                    diffusion_sampling=DiffusionSamplingParams(samples_per_prompt=m, num_inference_steps=2),
                    episode_sampling=ARSamplingParams(samples_per_prompt=1),
                    max_turns=_ScriptedDrawEnv.max_turns,
                    **kwargs,
                ),
                device=None,
                strategy=None,
                rank=0,
                model_config=None,
            )
        except Exception as exc:  # noqa: BLE001 — the point is that it refuses
            assert want in str(exc), f"expected a message mentioning {want!r}, got: {exc}"
            continue
        raise AssertionError(f"expected construction to fail for {kwargs} (m={m})")
    print("OK  in-loop config guards reject M>1 and per-phase wake/sleep")


def main() -> None:
    engine, diffusion = _build_engine()
    out = _drive(engine)
    out = sorted(out, key=lambda t: t.parts[0].sample_ids[0])

    check_trainable_image_parts(out)
    check_role_tagging_prevents_fusion(out)
    check_untagged_would_fuse(out)
    check_ti2i_edits_previous_image(diffusion)
    check_batching(diffusion)
    check_terminal_mode_unchanged()
    engine.shutdown()
    print("\nALL OK — in-loop agentic image rollout")


if __name__ == "__main__":
    main()
