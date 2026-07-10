"""Construction config for the typed SD3 pipeline.

Weights + params only — LoRA injection, FSDP wrapping, adapter switching,
gradient checkpointing, and offload control live outside the bundle (in
the training / rollout actors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


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
    # not per-request shape.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"
    # See SD3DiffusionStage.batch_replay_steps; exposed here so non-trainside
    # recipes can opt in via SD3Pipeline.from_config.
    batch_replay_steps: bool = False

    # Diffusion schedule policy. ``shift`` is the FlowMatch time-shift used
    # by ``sde.runtime.get_sigma_schedule`` (static branch); defaults to
    # 3.0 to match legacy.
    shift: float = 3.0

    # Trainer-side policy wraps the bare DiT, while vLLM-Omni loads it under
    # the pipeline's ``transformer.*`` namespace.
    weight_sync_param_name_prefix: str = "transformer."

    # ------------------------------------------------------------------
    # Optional LoRA hints for rollout-side engines (sglang in particular).
    #
    # Trainer-side LoRA lives in ``cfg.training.policies`` (LoRAPolicy →
    # PEFT injection on the FSDP-wrapped module). The SGLang rollout server
    # still needs to know at construction time whether to boot in LoRA mode
    # and which target modules to wrap — those flags travel through this
    # model_config.  ``None`` / ``False`` are the default (no LoRA).
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # VeOmniBackend lifecycle: build the transformer on the meta device
    # (architecture only, no weight allocation). VeOmni's parallelize
    # asserts meta init, materializes storage via ``to_empty``, and the
    # backend loads real weights from ``<pretrained>/transformer`` after
    # sharding. FSDPBackend recipes leave this False (eager load).
    meta_init_transformer: bool = False

    # ------------------------------------------------------------------
    # Shared-kernel parity mode (rollout↔train numerics unification).
    #
    # ``shared_kernels`` installs the engine-parity attention processor +
    # canonical fp32 pos_embed on the trainer transformer (see
    # ``models/sd3/parity.py``); the vLLM-Omni worker installs the SAME
    # kernels via ``patch_sd3_shared_kernels`` when the engine config sets
    # ``parity_mode``, making the two forwards bitwise-identical on the
    # same GPU. Eager load path only (asserted in the bundle).
    #
    # ``timestep_in_model_dtype`` casts the DiT timestep input
    # (``sigma*1000``) to the transformer's param dtype before the
    # sinusoidal embedding — matching vLLM-Omni's
    # ``t.expand(B).to(dtype=latents.dtype)`` (pipeline_sd3.py). Gated
    # separately because it perturbs numerics for every existing recipe.
    shared_kernels: bool = False
    timestep_in_model_dtype: bool = False
    # Grouped-parity replay: the engine's grouped forward runs [G, ...] latents
    # against ONE [1, L, D] text row (AdaLN broadcasts it). When true, replay
    # micros whose condition rows are all identical (a noise group; requires
    # micro_batch_size == samples_per_prompt) are forwarded with the single
    # de-duplicated text row so the trainer reproduces the engine's broadcast
    # geometry bit-for-bit at full grouped throughput. Pair with the engine's
    # ``parity_ungrouped: false``.
    group_broadcast_conditions: bool = False

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="SD3PipelineConfig.model_precision")


__all__ = ["SD3PipelineConfig"]
