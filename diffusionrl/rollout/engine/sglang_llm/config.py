"""SGLang LLM rollout-engine configuration (new ``BaseRolloutEngine`` protocol).

Registered under ``rollout/engine: sglang_llm`` with ``_target_`` pointing at
:class:`SGLangLLMRolloutEngine`. Materialized via
``build(cfg.rollout.engine, device=..., strategy=..., rank=..., model_config=...)``
from :class:`diffusionrl.ray.new_rollout_actor.NewRolloutActor`.

Unlike :class:`diffusionrl.rollout.engine.sglang.config.SGLangEngineConfig`
(which delegates the model path to the per-actor ``model_config``), this
LLM engine carries its own ``pretrained_model_ckpt_path`` field — PE-style
LLM rollouts don't surface a diffusion ``ModelBundleConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.rollout.engine.base import BaseEngineConfig


@register_config(
    group="rollout/engine",
    name="sglang_llm",
    target="diffusionrl.rollout.engine.sglang_llm.engine.SGLangLLMRolloutEngine",
)
@dataclass
class SGLangLLMEngineConfig(BaseEngineConfig):
    """Configuration for the SGLang LLM (SRT) rollout engine."""

    # --- Model ---
    pretrained_model_ckpt_path: str = ""

    # --- Parallelism & GPU ---
    tp_size: Optional[int] = None

    # --- SGLang network ---
    host: Optional[str] = None
    port: Optional[int] = None

    # --- Concurrency / async ---
    concurrency: int = 8

    # --- VLM multimodal ---
    # Image token placeholder injected into the chat template at image
    # positions.  Model-specific: e.g. "<|vision_start|><|image_pad|><|vision_end|>"
    # for Qwen2.5-VL.  None (default) = text-only mode.
    image_token: Optional[str] = None

    # --- LLM sampling (forwarded to SGLang /generate sampling_params) ---
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # --- Escape hatch for advanced ServerArgs / engine knobs ---
    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        require(
            bool(self.pretrained_model_ckpt_path),
            "SGLangLLMEngineConfig.pretrained_model_ckpt_path must be set",
        )
        require(
            self.tp_size is None or self.tp_size >= 1,
            f"SGLangLLMEngineConfig.tp_size must be >= 1 when set; got {self.tp_size!r}",
        )
        require(
            self.concurrency >= 1,
            f"SGLangLLMEngineConfig.concurrency must be >= 1; got {self.concurrency!r}",
        )
        require(
            self.max_new_tokens >= 1,
            f"SGLangLLMEngineConfig.max_new_tokens must be >= 1; got {self.max_new_tokens!r}",
        )
        require(
            self.temperature > 0,
            f"SGLangLLMEngineConfig.temperature must be > 0; got {self.temperature!r}",
        )
        require(
            0.0 < self.top_p <= 1.0,
            f"SGLangLLMEngineConfig.top_p must be in (0, 1]; got {self.top_p!r}",
        )


__all__ = ["SGLangLLMEngineConfig"]
