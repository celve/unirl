"""Qwen-Image pipeline on the new four-tier typed architecture.

Re-expression of ``diffusionrl/models/qwen_image.py`` +
``diffusionrl/samplers/fsdp/qwen_image_sampler.py`` (the PR #104 OLD
implementation) against the typed ``Bundle`` / ``Pipeline`` /
``EmbedStage`` / ``DiffusionStage`` / ``DecodeStage`` protocols. Sibling
of :mod:`diffusionrl.models_new.sd3` and :mod:`diffusionrl.models_new.wan21`.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing ``diffusionrl.models_new.qwen_image``
is enough to make ``model/qwen_image_v2`` resolvable in the ConfigStore.
"""

from diffusionrl.models_new.qwen_image.bundle import QwenImageBundle
from diffusionrl.models_new.qwen_image.conditions import QwenImageConditions
from diffusionrl.models_new.qwen_image.config import QwenImagePipelineConfig
from diffusionrl.models_new.qwen_image.diffusion import QwenImageDiffusionParams
from diffusionrl.models_new.qwen_image.pipeline import QwenImagePipeline

__all__ = [
    "QwenImageBundle",
    "QwenImageConditions",
    "QwenImageDiffusionParams",
    "QwenImagePipeline",
    "QwenImagePipelineConfig",
]
