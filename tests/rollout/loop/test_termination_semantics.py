"""AReaL-parity termination semantics for ``ToolEnvironment.step`` (LIN-564).

AReaL's ``react_agent`` terminates a deep-research trajectory ONLY on an explicit
``<answer>…</answer>`` tag. A generation that calls a tool continues (observe the
result), and — crucially — a generation that is *neither* a tool call nor a tagged
answer (a truncated ``<think>``, a "let me reconsider…" deliberation after a failed
search) ALSO continues: the model gets to keep going. unirl previously terminated on
the first tool-call-free generation, cutting ~16% of base-policy trajectories a turn+
short of AReaL — the dominant rollout-0 turn-count gap. These tests pin the aligned
behavior so it cannot silently regress.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from unirl.rollout.loop import ToolEnvironment  # noqa: E402
from unirl.rollout.loop.tools.calculator import CalculatorTool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

TOOLCALL = '<tool_call>{"name": "calculator", "arguments": {"expression": "6 * 7"}}</tool_call>'


def _turn(body: str, *, prior_turns: int = 0) -> Sample:
    """A single-sample trajectory whose latest (frontier) generation carries ``body``.

    ``prior_turns`` fills that many earlier tool-call turns first, so ``step`` sees a
    trajectory at depth ``prior_turns + 1`` (used to exercise the ``max_turns`` bound).
    """
    s = Sample.request(Part.input(["r0"], primitive=Texts(texts=["prompt-r0"])))
    for i in range(prior_turns):
        s = s.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
        s = s.with_filled_frontier(primitive=Texts(texts=[TOOLCALL]))
        s = s.observe(Texts(texts=[f"obs{i}"]))
    s = s.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
    return s.with_filled_frontier(primitive=Texts(texts=[body]))


def test_explicit_answer_terminates():
    """A generation carrying ``<answer>…</answer>`` (no tool call) → done, no observation."""
    env = ToolEnvironment([CalculatorTool()])
    obs, done, info = env.step(_turn("<think>done thinking</think>\n<answer>42</answer>"))
    assert done is True and obs is None
    assert info["per_sample_done"] == [True]


def test_neither_answer_nor_toolcall_continues():
    """AReaL parity (the rollout-0 fix): a generation that is neither a tool call nor a
    tagged answer (e.g. a truncated ``<think>``) does NOT terminate. It returns
    ``observation=None`` so the engine loop re-generates with nothing appended — AReaL's
    "keep going". unirl used to return ``done=True`` here."""
    env = ToolEnvironment([CalculatorTool()])
    obs, done, info = env.step(_turn("<think>let me reconsider the problem"))
    assert done is False  # would have been True before the LIN-564 fix
    assert obs is None  # nothing to observe → engine re-generates
    assert info["per_sample_done"] == [False]


def test_tool_call_continues_with_result():
    """A tool-call generation (no answer) continues, observing the tool result — unchanged."""
    env = ToolEnvironment([CalculatorTool()])
    obs, done, info = env.step(_turn(TOOLCALL))
    assert done is False and isinstance(obs, Texts) and obs.texts == ["42"]
    assert info["per_sample_done"] == [False]


def test_truncated_answer_does_not_terminate():
    """A generation with an OPEN ``<answer>`` but no closing tag (truncated at the token
    cap) is not a completed answer — like AReaL (``"</answer>" in content`` is False), it
    keeps going rather than terminating on a half-written answer."""
    env = ToolEnvironment([CalculatorTool()])
    _, done, info = env.step(_turn("<think>ok</think>\n<answer>the answer is probably"))
    assert done is False
    assert info["per_sample_done"] == [False]


def test_max_turns_still_terminates_without_answer():
    """The ``max_turns`` bound remains a hard terminal even without an ``<answer>`` — the
    engine then force-answers. Here a depth-2 trajectory meets ``max_turns=2``."""
    env = ToolEnvironment([CalculatorTool()], max_turns=2)
    obs, done, _ = env.step(_turn("<think>still deliberating", prior_turns=1))
    assert done is True and obs is None
