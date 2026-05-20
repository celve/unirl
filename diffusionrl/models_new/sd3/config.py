"""Construction config for the new typed SD3 pipeline.

Mirrors :class:`diffusionrl.models.config.ModelBundleConfig` minus the
LoRA / training‑lifecycle knobs. The new bundle is weights+params only;
LoRA injection, FSDP wrapping, adapter switching, gradient checkpointing,
and offload control all live outside the bundle (in legacy world today,
in the training/rollout actors after the consumer migration lands).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.validation import validate_precision_type


@register_config(
    group="model",
    name="sd3_v2",
    target="diffusionrl.models_new.sd3.SD3Pipeline.from_config",
    mutable=True,
)
@dataclass
class SD3PipelineConfig:
    """Construction args for ``SD3Pipeline.from_config``.

    ``device`` may be runtime‑injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    # Stage-level precision / numerical policy. Lives here (not on
    # SD3DiffusionParams) because these are operator/runtime knobs,
    # not per-request shape. Defaults match legacy SD3Sampler.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    # Diffusion schedule policy. ``shift`` is the FlowMatch time-shift used
    # by ``sde.runtime.get_sigma_schedule`` (static branch); defaults to
    # 3.0 to match legacy.
    shift: float = 3.0

    # Trainer-side policy wraps the bare DiT, while vLLM-Omni loads it under
    # the pipeline's ``transformer.*`` namespace.
    weight_sync_param_name_prefix: str = "transformer."

    # ------------------------------------------------------------------
    # Optional LoRA hints for rollout-side engines (sglang_new in particular).
    #
    # The new design moves *trainer-side* LoRA into ``cfg.training.policies``
    # (LoRAPolicy → PEFT injection on the FSDP-wrapped module). However the
    # SGLang rollout server still needs to know at construction time
    # whether it should boot in LoRA mode and which target modules to wrap
    # — those flags travel through ``model_config`` (typed
    # :class:`ModelBundleConfig` in legacy world, :class:`SD3PipelineConfig`
    # here). To keep ``sglang_new`` engine code paths uniform across the
    # legacy and new pipelines, mirror the two LoRA fields here.
    #
    # ``None`` / ``False`` keep backward-compatible defaults: a recipe that
    # doesn't set them gets the same behavior as before.
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="SD3PipelineConfig.model_precision")


__all__ = ["SD3PipelineConfig"]
