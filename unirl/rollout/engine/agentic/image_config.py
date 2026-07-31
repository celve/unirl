"""Agentic-image rollout-engine configuration (LIN-577).

Extends :class:`AgenticRolloutEngineConfig` with a diffusion child so the
multi-turn agentic trajectory's final answer conditions a **terminal** diffusion
image generation — the multi-turn analogue of the single-turn prompt-enhancement
flow driven by :class:`ComposedRolloutEngine`.

Like the base config, ``inner`` / ``env`` and the new ``diffusion`` field are
kept ``Any``: each carries its own ``_target_`` and is built per worker by the
worker walker, so every worker gets its own local inner engine + environment +
diffusion child.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig


@dataclass
class AgenticImageRolloutEngineConfig(AgenticRolloutEngineConfig):
    """Config for the agentic loop + terminal diffusion image-gen engine."""

    #: Diffusion child engine config (own ``_target_``, e.g.
    #: ``SGLangDiffusionEngineConfig``), built per worker via
    #: ``diffusion.make_engine(...)`` — mirrors ``ComposedRolloutEngineConfig``.
    #: Required; defaulted ``None`` only for dataclass field ordering (the base
    #: already has defaulted fields), validated in the engine ctor.
    diffusion: Any = None

    #: ``DiffusionSamplingParams`` for the terminal image gen; its
    #: ``samples_per_prompt`` is ``M`` (images per trajectory) plus the denoise
    #: recipe. Required (see above).
    diffusion_sampling: Any = None

    #: Terminal-answer extraction. If set (e.g. ``"Revised Prompt:"``), keep only
    #: the substring after the LAST marker (``postprocess_pe_texts``,
    #: fallback-to-original on off-format output); ``None`` uses the last
    #: ``<answer>...</answer>`` span (else the whole text).
    answer_marker: Optional[str] = None
    #: Optional char cap applied after marker extraction (marker path only).
    answer_max_chars: Optional[int] = None

    #: Colocated diffusion: sleep the diffusion child on start and wake/sleep it
    #: around the terminal diffusion phase (shared slab, like PE colocate).
    #: ``False`` keeps both children resident (no wake/sleep dance).
    #: Incompatible with ``in_loop_images`` — see below.
    sleep_diffusion_on_start: bool = True

    #: **In-loop image turns.** ``False`` (default) is the v1 shape: the agent loop is
    #: all-text and ONE diffusion generation runs after the trajectory finishes.
    #: ``True`` lets the agent render mid-trajectory by calling ``draw_tool_name``:
    #: the image lands on the trajectory as a **trainable diffusion gen Part** (it
    #: carries its ``LatentSegment``, so the denoise trajectory is a policy-gradient
    #: target), the agent sees it on the next turn, and a later draw edits it (ti2i).
    #:
    #: Requires a **VLM** inner engine — a text-only agent cannot see what it drew,
    #: and ``text_conditioning`` fails loud on the first image turn. That one is on
    #: the recipe (the engine cannot introspect a backend's modality). The engine
    #: ctor does validate:
    #:
    #: - ``diffusion_sampling.samples_per_prompt == 1`` — forking M>1 mid-trajectory
    #:   would branch the trajectory and leave later AR turns M-wide. GRPO diversity
    #:   comes from the ``n`` siblings the agentic engine already fans at submit time;
    #: - ``sleep_diffusion_on_start=False`` — per-phase wake/sleep cannot work when
    #:   diffusion turns interleave with in-flight AR turns on other threads. Both
    #:   children stay resident, as the PE recipes already do
    #:   (``examples/pe/pe_sglang_full_wise.yaml``), budgeted per engine via
    #:   ``mem_fraction_static``.
    in_loop_images: bool = False

    #: Tool name that triggers an in-loop image turn (see
    #: :class:`~unirl.rollout.loop.tools.draw.DrawTool`). The environment parses the
    #: call and reports it in ``info["tool_calls"]``; the engine renders it.
    draw_tool_name: str = "draw"

    #: Coalescing window for in-loop image turns, seconds. The drain's per-trajectory
    #: threads hit their image turns at slightly different moments; the diffusion
    #: backends *serialize* concurrent ``generate`` callers, so requests are gathered
    #: for this long (or until ``per_worker_concurrency`` have parked) and issued as
    #: one batched call. Small next to a denoise pass — a lone trajectory pays ~this
    #: much, a full drain saves a factor of K.
    draw_batch_window_s: float = 0.05

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticImageRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.image_engine import AgenticImageRolloutEngine

        return AgenticImageRolloutEngine(config=self, **deps)


__all__ = ["AgenticImageRolloutEngineConfig"]
