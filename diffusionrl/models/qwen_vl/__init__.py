"""Qwen2.5-VL vision-language pipeline on the typed stage/pipeline architecture.

AR-only VLM pipeline: text+images in, text out. Supports GRPO training via
ARStage.autoregress + ARStage.replay.

Imports from this package trigger @register_config in config.py.
"""

from diffusionrl.models.qwen_vl.ar import (
    QwenVLARParams,
    QwenVLARStage,
    QwenVLARStep,
)
from diffusionrl.models.qwen_vl.bundle import QwenVLBundle
from diffusionrl.models.qwen_vl.chat_template import QwenVLChatTemplateStage
from diffusionrl.models.qwen_vl.conditions import QwenVLARConditions
from diffusionrl.models.qwen_vl.config import QwenVLPipelineConfig
from diffusionrl.models.qwen_vl.pipeline import QwenVLPipeline

__all__ = [
    "QwenVLARConditions",
    "QwenVLARParams",
    "QwenVLARStage",
    "QwenVLARStep",
    "QwenVLBundle",
    "QwenVLChatTemplateStage",
    "QwenVLPipeline",
    "QwenVLPipelineConfig",
]
