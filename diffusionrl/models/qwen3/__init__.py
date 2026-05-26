"""Qwen3 causal LM pipeline on the typed stage/pipeline architecture.

Port of :class:`diffusionrl.models.llm.bundle.LLMModelBundle` (the
generic HF causal LM adapter that ``diffusionrl-pe`` tested with Qwen3)
into the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` / ``ARStage``
protocols. Sibling of :mod:`diffusionrl.models.qwen_image`.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing ``diffusionrl.models.qwen3`` is
enough to make ``model/qwen3_v2`` resolvable in the ConfigStore.
"""

from diffusionrl.models.qwen3.ar import (
    Qwen3ARParams,
    Qwen3ARStage,
    Qwen3ARStep,
)
from diffusionrl.models.qwen3.bundle import Qwen3Bundle
from diffusionrl.models.qwen3.chat_template import Qwen3ChatTemplateStage
from diffusionrl.models.qwen3.conditions import Qwen3ARConditions
from diffusionrl.models.qwen3.config import Qwen3PipelineConfig
from diffusionrl.models.qwen3.pipeline import Qwen3Pipeline

__all__ = [
    "Qwen3ARConditions",
    "Qwen3ARParams",
    "Qwen3ARStage",
    "Qwen3ARStep",
    "Qwen3Bundle",
    "Qwen3ChatTemplateStage",
    "Qwen3Pipeline",
    "Qwen3PipelineConfig",
]
