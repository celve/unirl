"""Qwen-Image pipeline on the new four-tier typed architecture.

Re-expression of ``unirl/models/qwen_image.py`` +
``unirl/samplers/fsdp/qwen_image_sampler.py`` (the PR #104 OLD
implementation) against the typed ``Bundle`` / ``Pipeline`` /
``EmbedStage`` / ``DiffusionStage`` / ``DecodeStage`` protocols. Sibling
of :mod:`unirl.models.sd3` and :mod:`unirl.models.wan21`.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing ``unirl.models.qwen_image``
is enough to make ``model/qwen_image`` resolvable in the ConfigStore.
"""

from unirl.models.qwen_image.bundle import QwenImageBundle
from unirl.models.qwen_image.conditions import QwenImageConditions
from unirl.models.qwen_image.config import QwenImagePipelineConfig
from unirl.models.qwen_image.pipeline import QwenImagePipeline

__all__ = [
    "QwenImageBundle",
    "QwenImageConditions",
    "QwenImagePipeline",
    "QwenImagePipelineConfig",
]
