"""SGLang LLM (SRT) rollout engine — new ``BaseRolloutEngine`` protocol.

Text-generation companion to :mod:`diffusionrl.rollout.engine.sglang` (which
hosts the diffusion rollout engine). Recipes opt in via
``cfg.rollout.engine=sglang_llm``.
"""

from diffusionrl.rollout.engine.sglang_llm.config import SGLangLLMEngineConfig
from diffusionrl.rollout.engine.sglang_llm.engine import (
    SGLangLLMRolloutEngine,
    build_rollout_resp,
)

__all__ = [
    "SGLangLLMRolloutEngine",
    "SGLangLLMEngineConfig",
    "build_rollout_resp",
]
