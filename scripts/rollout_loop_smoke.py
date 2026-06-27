#!/usr/bin/env python3
"""CPU oracle for the AgentLoop prototype (LIN-492).

Exercises the synchronous ``AgentLoop`` + ``Sample.observe`` on fabricated Samples with a
model-free ``FakeEngine`` and a ``StubEnv`` — no GPU, no backend. Mirrors
``scripts/check_sample_roundtrip.py`` (``check_*`` contracts, ``main() -> int``, non-zero
exit on the first failure). Guards the loop mechanics — fork → fill → observe → conditioning,
lineage, masking, split/concat — so a broken loop or ``observe`` surfaces here instead of
mid-rollout.

    python scripts/rollout_loop_smoke.py
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple

from unirl.rollout.loop import AgentLoop
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Primitive, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class FakeEngine:
    """Model-free engine: fills the frontier gen shell with a deterministic ``Texts``
    primitive keyed off its ``sample_ids`` (segment left None — the loop mechanics need only
    ids + primitive). Mirrors ``check_generate_fills_frontier`` in check_sample_roundtrip.py.
    """

    def generate(self, sample: Sample) -> Sample:
        shell = sample.parts[-1]
        filled = shell.fill(primitive=Texts(texts=[f"gen::{sid}" for sid in shell.sample_ids]))
        return sample.with_parts([*sample.parts[:-1], filled])


class StubEnv:
    """World-initiated stub: returns a ``Texts`` observation for ``turns`` generations, then
    signals done. Stands in for a real environment to exercise observation re-entry."""

    def __init__(self, turns: int) -> None:
        self._left = turns

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        self._left -= 1
        if self._left <= 0:
            return None, True, {}
        frontier = sample.parts[-1]
        return Texts(texts=[f"obs::{sid}" for sid in frontier.sample_ids]), False, {}


def _request(prompts: List[str]) -> Sample:
    ids = [f"p{i}" for i in range(len(prompts))]
    return Sample.request(Part.input(ids, primitive=Texts(texts=prompts)))


def check_fixed_plan_composed_shape() -> None:
    """A fixed 2-turn plan (no env) builds ``[input, ar, diffusion]`` with correct path ids —
    the Composed shape expressed purely as *config* of the generic loop."""
    loop = AgentLoop(
        plan=[
            (1, ARSamplingParams()),
            (1, DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)),
        ],
        environment=None,
    )
    out = loop.run(FakeEngine(), _request(["a cat", "a dog"]))
    _check(len(out.parts) == 3, "fixed 2-turn plan yields [input, ar, diffusion]")
    _check(list(out.parts[1].sample_ids) == ["p0/0", "p1/0"], "ar path ids preserved")
    _check(list(out.parts[2].sample_ids) == ["p0/0/0", "p1/0/0"], "diffusion path ids preserved")
    _check(isinstance(out.parts[1].sampling_params, ARSamplingParams), "ar params preserved")
    _check(isinstance(out.parts[2].sampling_params, DiffusionSamplingParams), "diffusion params preserved")
    _check(isinstance(out.parts[1].primitive, Texts), "ar gen Part was filled with a primitive")


def check_gen_parts_masking() -> None:
    """``gen_parts()`` == the generated Parts; the input Part (no sampling_params) is excluded."""
    loop = AgentLoop(plan=[(2, ARSamplingParams())], environment=None)
    out = loop.run(FakeEngine(), _request(["x"]))
    gps = out.gen_parts()
    _check(len(gps) == 1 and gps[0] is out.parts[1], "gen_parts is exactly the AR gen Part")
    _check(out.parts[0].sampling_params is None, "input Part carries no sampling_params (excluded)")
    _check(
        list(out.parts[1].sample_ids) == ["p0/0", "p0/1"],
        "branch=2 fans the single prompt (p0) out to 2 samples",
    )


def check_agentic_observe_masking_and_conditioning() -> None:
    """An agentic plan (repeat AR + StubEnv) interleaves mask-0 observe Parts: they are
    excluded from ``gen_parts()``, and ``conditioning()`` surfaces every ancestor primitive."""
    loop = AgentLoop(plan=(1, ARSamplingParams()), environment=StubEnv(turns=3), max_turns=8)
    out = loop.run(FakeEngine(), _request(["solve it"]))
    # 3 gen turns + 2 interleaved observations: [input, gen, obs, gen, obs, gen]
    _check(len(out.parts) == 6, f"expected 6 parts (input + 3 gen + 2 obs), got {len(out.parts)}")
    gps = out.gen_parts()
    _check(len(gps) == 3, "exactly the 3 AR gen Parts are trainable")
    obs_parts = [p for p in out.parts[1:] if p.sampling_params is None]
    _check(len(obs_parts) == 2, "2 mask-0 observe Parts present and excluded from gen_parts")
    cond = out.conditioning()
    _check(len(cond) == 5, f"frontier conditioning sees 5 ancestors (prompt + 2 gen + 2 obs), got {len(cond)}")
    _check(all(isinstance(c, Texts) for c in cond), "conditioning surfaces the Texts primitives")


def check_split_concat_roundtrip() -> None:
    """``split()`` -> ``concat()`` round-trips the multi-part agentic tree ids (dp-shard safety)."""
    loop = AgentLoop(plan=(1, ARSamplingParams()), environment=StubEnv(turns=2), max_turns=8)
    out = loop.run(FakeEngine(), _request(["a", "b"]))
    merged = Sample.concat(out.split())
    _check(
        [list(p.sample_ids) for p in merged.parts] == [list(p.sample_ids) for p in out.parts],
        "split -> concat round-trips the per-part sample ids in order",
    )


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_fixed_plan_composed_shape,
    check_gen_parts_masking,
    check_agentic_observe_masking_and_conditioning,
    check_split_concat_roundtrip,
)


def main() -> int:
    failures: List[str] = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("rollout-loop-smoke: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"rollout-loop-smoke: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
