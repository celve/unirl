"""Agentic rollout-engine configuration (LIN-522).

Registered as a peer rollout engine (alongside ``sglang`` / ``composed`` /
``vllm_omni`` / ``trainside``) whose ``_target_`` points at
:class:`AgenticRolloutEngine`. The agentic engine wraps **one inner rollout
engine** (the single-turn generator it calls each turn) and **one environment**
(the tool/world side), and drives multi-turn rollout across a DP-replicated slab
with a rank-0 coordinator.

Like :class:`ComposedRolloutEngineConfig`, the ``inner`` and ``env`` fields are
kept ``Any``: each carries its own ``_target_`` and is built by the worker walker
(``Worker._resolve_init_kwargs``) before the engine is constructed — so each
worker gets its **own local** inner engine + environment instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class AgenticRolloutEngineConfig(BaseEngineConfig):
    """Config for the multi-turn (agentic) rollout engine."""

    #: Inner single-turn rollout engine config (e.g. ``SGLangEngineConfig``); kept
    #: ``Any`` so it is built from its own ``_target_`` via ``inner.make_engine``.
    inner: Any
    #: The environment (the world side of a turn) — an :class:`Environment`, kept
    #: ``Any`` so a ``_target_`` (e.g. ``ToolEnvironment`` + tools) is instantiated
    #: per worker by the walker. Must be re-entrant: one instance serves all of a
    #: worker's concurrent trajectories (LIN-522).
    env: Any

    #: Hard per-trajectory turn bound (the loop's ``for _ in range(max_turns)``).
    max_turns: int = 8
    #: Per-turn sampling params (``BaseSamplingParams``); its ``samples_per_prompt``
    #: is the GRPO group size ``n`` (read via ``total_samples_per_prompt``).
    episode_sampling: Any = None
    #: Max concurrent trajectories per worker — the drain thread-pool size (one
    #: trajectory per thread; a thread holds its trajectory across tool-wait).
    #: Set a small multiple of the inner backend ``concurrency`` so trajectories
    #: in tool-wait don't starve the GPU.
    per_worker_concurrency: int = 8
    #: Partial rollout (LIN-531): expose the coordinator as a ``submit``/``poll``/
    #: ``abort`` interface so the *trainer* can over-sample, commit the first
    #: ``batch_size`` complete GRPO groups, checkpoint the unfinished tail at a
    #: **turn boundary**, and carry it to the next round (resumed under new
    #: weights). ``False`` ⇒ ``abort`` never fires, so behaviour is byte-identical
    #: to the barrier ``generate``. The engine provides the mechanism; the policy
    #: (how many, when to sync) lives in the trainer.
    partial_rollout: bool = False

    #: Per-trajectory token budget (AReaL tongyi_deepresearch: 27648). When set, the
    #: agent loop forces a final answer once a trajectory's accumulated tokens reach
    #: ``force_answer_fraction`` of it — capping runaway tool loops and preventing the
    #: context overflow that would otherwise zero the reward. ``None`` disables the
    #: guard, so other agentic recipes (calculator / ALFWorld) stay byte-identical.
    max_tokens_per_trajectory: Optional[int] = None
    #: Fraction of ``max_tokens_per_trajectory`` at which to force the final answer,
    #: leaving headroom for the forced answer turn (AReaL forces at 0.8 of context).
    force_answer_fraction: float = 0.8

    #: Decoder-side repair for a generation that is neither a parseable tool call
    #: nor a complete ``<answer>...</answer>``. When enabled, the agent does not add
    #: a synthetic user turn. Instead it appends ``neither_answer_prefix`` to the
    #: exact assistant token stream and performs one raw continuation, stopping at
    #: ``neither_answer_stop``. Disabled by default so existing recipes are unchanged.
    inject_answer_after_neither: bool = False
    #: Literal assistant-side prefix inserted before the repair continuation.
    neither_answer_prefix: str = "\n<answer>"
    #: Stop boundary for the repair continuation; retained in decoded output.
    neither_answer_stop: str = "</answer>"
    #: Maximum sampled suffix length for the one-shot repair continuation.
    neither_answer_max_new_tokens: int = 1024

    #: Intervention-aware answer rescue for a true NEITHER generation. Unlike
    #: ``inject_answer_after_neither``, this appends a user-side format nudge and
    #: performs an ordinary generation, so the policy itself samples the complete
    #: ``<answer>...</answer>`` response (including the opener). The triggering
    #: NEITHER and rescued answer are marked separately for trainer-side credit.
    nudge_answer_after_neither: bool = False
    #: User observation used by the one-shot NEITHER rescue. Kept configurable so
    #: experiments can match an external controller without changing engine code.
    neither_answer_nudge: str = (
        "Your previous response did not use the required final-answer tags. Stop "
        "making tool calls and, based on all the information above, provide your "
        "most likely answer in the following format:"
        "<think>your final thinking</think>\n<answer>your answer</answer>"
    )

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

        return AgenticRolloutEngine(config=self, **deps)


__all__ = ["AgenticRolloutEngineConfig"]
