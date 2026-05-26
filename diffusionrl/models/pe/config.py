"""Construction config for the typed PE (Prompt Enhancement) pipeline.

PE composes two registered sub-pipelines — one diffusion, one AR LLM —
into a single ``model: pe`` entry. The schema is intentionally minimal:
two nested ``Any``-typed slots that accept any registered child schema
via inline ``_target_`` recursion. Validation of each child happens
lazily when :class:`PEPipeline.from_config` dispatches that child's
construction via :func:`diffusionrl.config.instantiate.build`.

Unlike sibling configs (``SD3PipelineConfig`` / ``Qwen3PipelineConfig``)
this one is *not* constructed via the standard
:func:`diffusionrl.config.instantiate.build` path: PE needs DictConfig
access for the nested children, but ``build()`` materializes the cfg
before dispatch — see :meth:`PEPipeline.from_config` for the canonical
invocation pattern (mirroring ``RewardService.from_configs``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from diffusionrl.config.registration import register_config


@register_config(
    group="model",
    name="pe",
    target="diffusionrl.models.pe.PEPipeline.from_config",
    mutable=True,
)
@dataclass
class PEPipelineConfig:
    """Construction args for ``PEPipeline.from_config``.

    ``diffusion`` and ``llm`` are each a nested DictConfig with its own
    ``_target_`` pointing at a registered ``*Pipeline.from_config``
    classmethod (e.g. ``diffusionrl.models.sd3.SD3Pipeline.from_config``,
    ``diffusionrl.models.qwen3.Qwen3Pipeline.from_config``). Typed as
    ``Any`` so any registered child schema is accepted without further
    annotation; per-child schema validation runs inside the child's own
    ``build()`` invocation.
    """

    diffusion: Any = MISSING
    llm: Any = MISSING


__all__ = ["PEPipelineConfig"]
