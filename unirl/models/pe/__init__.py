"""PE (Prompt Enhancement) composed pipeline.

Bundles a diffusion :class:`Bundle` (e.g. :class:`SD3Bundle`) together
with an AR LLM :class:`Bundle` (e.g. :class:`Qwen3Bundle`) and runs the
two-phase flow "LLM rewrites prompt → diffusion samples image".

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing ``unirl.models.pe`` is
enough to make ``model: pe`` resolvable in the ConfigStore.
"""

from unirl.models.pe.bundle import PEBundle
from unirl.models.pe.config import PEPipelineConfig
from unirl.models.pe.pipeline import PEPipeline

__all__ = [
    "PEBundle",
    "PEPipeline",
    "PEPipelineConfig",
]
