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
    sleep_diffusion_on_start: bool = True

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticImageRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.image_engine import AgenticImageRolloutEngine

        return AgenticImageRolloutEngine(config=self, **deps)


__all__ = ["AgenticImageRolloutEngineConfig"]
