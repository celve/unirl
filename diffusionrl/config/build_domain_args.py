"""Config builders — convert TrainingArguments into domain-specific dicts for actors.

Each build_*() function extracts a domain-specific slice from args and returns a
plain dict that the corresponding actor/service consumes via its init() method.
This decouples actors from the TrainingArguments structure.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from diffusionrl.config.resolution import (
    ResolvedModelSpec,
    ResolvedTrainingPlan,
    ResolvedTrainTopology,
    normalize_rollout_service_engine,
    normalize_lora_target_modules,
    derive_model_spec,
)
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.training.backends import (
    ResolvedTrainBackendConfig,
    resolve_train_backend_config_from_args,
)
from diffusionrl.types.sampling import ResolvedSamplingSpec

# ---------------------------------------------------------------------------
# Model domain
# ---------------------------------------------------------------------------

def build_model_config(
    args,
    *,
    model_spec: Optional[ResolvedModelSpec] = None,
) -> Dict[str, Any]:
    """Build model config consumed by training and rollout actors.

    Pulls from: ModelConfig (identity/paths) + TrainingConfig (LoRA/checkpointing)
    + PrecisionConfig (model load dtype).
    """
    resolved_model = model_spec if model_spec is not None else derive_model_spec(args)
    mc = args.model    # ModelConfig
    tc = args.training  # TrainingConfig
    pc = args.precision  # PrecisionConfig
    return {
        "model_path": resolved_model.model_path,
        "pretrained_model_saved_path": mc.pretrained_model_saved_path,
        "vae_saved_path": mc.vae_saved_path,
        "text_encoder_path": mc.text_encoder_path,
        "use_lora": tc.use_lora,
        "lora_rank": tc.lora_rank,
        "lora_alpha": tc.lora_alpha,
        "lora_target_modules": normalize_lora_target_modules(tc.lora_target_modules),
        "use_gradient_checkpointing": tc.use_gradient_checkpointing,
        "model_precision": pc.model_precision,
    }


# ---------------------------------------------------------------------------
# Reward domain
# ---------------------------------------------------------------------------


def build_reward_config(args) -> Dict[str, Any]:
    """Build reward config consumed by actors and services."""
    return asdict(RewardSchema.from_args(args))


# ---------------------------------------------------------------------------
# Sampling / rollout domain
# ---------------------------------------------------------------------------

def _build_shared_sampling_payload(sampling_spec: ResolvedSamplingSpec) -> Dict[str, Any]:
    return {
        "sampler_path": sampling_spec.sampler_path,
        "num_inference_steps": int(sampling_spec.num_inference_steps),
        "sde_config": sampling_spec.sde_config.to_dict(),
        "guidance_scale": float(sampling_spec.guidance_scale),
        "height": int(sampling_spec.height),
        "width": int(sampling_spec.width),
        "num_frames": int(sampling_spec.num_frames),
    }


def build_training_sampling_config(
    args,
    *,
    sampling_spec: ResolvedSamplingSpec,
    sampler_engine_type: str,
) -> Dict[str, Any]:
    """Build training-actor sampling config from the canonical resolved sampling spec."""
    pc = args.precision  # PrecisionConfig
    payload = _build_shared_sampling_payload(sampling_spec)
    sampler_kwargs = dict(sampling_spec.sampler_kwargs)
    sampler_kwargs["autocast_precision"] = pc.autocast_precision
    sampler_kwargs["trajectory_precision"] = pc.trajectory_precision
    sampler_kwargs["logprob_precision"] = pc.logprob_precision
    payload.update({
        "sampler_engine_type": normalize_rollout_service_engine(sampler_engine_type)
        or str(sampler_engine_type).strip().lower(),
        "replay_sampler_path": sampling_spec.replay_sampler_path,
        "sde_schedule_config": sampling_spec.sde_schedule_config.to_dict(),
        "seed": int(sampling_spec.seed),
        "sampling_adapter": sampling_spec.sampling_adapter,
        "init_same_noise": bool(sampling_spec.init_same_noise),
        "sampler_kwargs": sampler_kwargs,
    })
    return payload


def _build_rollout_engine_base_kwargs(args) -> Dict[str, Any]:
    """Build dedicated rollout-engine kwargs from canonical rollout fields."""
    resolved: Dict[str, Any] = {}
    rollout_topology_config = args.rollout.topology

    field_map = {
        "service_num_gpus": "num_gpus",
        "engine_tp_size": "tp_size",
        "engine_sp_size": "sp_size",
        "service_transport_dtype": "transport_dtype",
        "service_transport_drop_decoded_videos": "transport_drop_decoded_videos",
        "service_transport_log_payload_bytes": "transport_log_payload_bytes",
        "service_require_memory_api": "require_memory_api",
    }
    for attr_name, engine_key in field_map.items():
        value = getattr(rollout_topology_config, attr_name, None)
        if value is not None:
            resolved[engine_key] = value

    sglang_field_map = {
        "sglang_local_mode": "local_mode",
        "sglang_verify_weight_checksum": "verify_weight_checksum",
        "sglang_prompt_encoder_device": "prompt_encoder_device",
        "sglang_prompt_encoder_dtype": "prompt_encoder_dtype",
        "sglang_prompt_encoder_max_length": "prompt_encoder_max_length",
    }
    for attr_name, engine_key in sglang_field_map.items():
        value = getattr(rollout_topology_config, attr_name, None)
        if value is not None:
            resolved[engine_key] = value

    sglang_kwargs = getattr(rollout_topology_config, "sglang_kwargs", None)
    if sglang_kwargs:
        if not isinstance(sglang_kwargs, dict):
            raise ValueError(
                "rollout.topology.sglang_kwargs must be a dict after normalization."
            )
        resolved["server_kwargs"] = dict(sglang_kwargs)

    return resolved


def build_rollout_engine_config(
    *,
    args,
    sampler_engine_type: str,
    model_config: Dict[str, Any],
    sampling_spec: ResolvedSamplingSpec,
    offload_rollout: bool = False,
) -> Dict[str, Any]:
    """Build final dedicated rollout engine runtime config."""
    pc = args.precision  # PrecisionConfig

    merged_engine_kwargs = _build_rollout_engine_base_kwargs(args)
    merged_engine_kwargs.setdefault("use_lora", model_config["use_lora"])
    merged_engine_kwargs.setdefault("lora_rank", model_config["lora_rank"])
    merged_engine_kwargs.setdefault("lora_alpha", model_config["lora_alpha"])
    merged_engine_kwargs.setdefault("lora_target_modules", model_config["lora_target_modules"])
    if model_config.get("vae_saved_path"):
        merged_engine_kwargs.setdefault("vae_saved_path", model_config["vae_saved_path"])
    if model_config.get("text_encoder_path"):
        merged_engine_kwargs.setdefault("text_encoder_path", model_config["text_encoder_path"])
    merged_engine_kwargs["model_precision"] = pc.model_precision
    merged_engine_kwargs["fsdp_precision"] = pc.fsdp_precision
    merged_engine_kwargs.setdefault("prompt_encoder_dtype", pc.model_precision)
    # Wire top-level fps into engine_kwargs so SGLang engine can consume it
    # without requiring users to duplicate rollout-topology config.
    merged_engine_kwargs.setdefault("fps", args.fps)
    merged_engine_kwargs.setdefault("weight_sync_dir", args.sync.dir)
    if args.sync.target_modules is not None:
        merged_engine_kwargs.setdefault("target_modules", list(args.sync.target_modules))
    if sampler_engine_type == "sglang":
        merged_engine_kwargs["logprob_source"] = args.sampling.logprob_source
        if offload_rollout:
            merged_engine_kwargs["require_memory_api"] = True

    return {
        "sampler_engine_type": sampler_engine_type,
        "model_path": model_config["model_path"],
        "pretrained_model_saved_path": model_config["pretrained_model_saved_path"],
        "sampler_path": sampling_spec.sampler_path,
        "num_inference_steps": int(sampling_spec.num_inference_steps),
        "eta": float(sampling_spec.sde_config.eta),
        "sde_type": str(sampling_spec.sde_config.sde_type),
        "shift": float(sampling_spec.sde_config.shift),
        "guidance_scale": float(sampling_spec.guidance_scale),
        "height": int(sampling_spec.height),
        "width": int(sampling_spec.width),
        "num_frames": int(sampling_spec.num_frames),
        "engine_kwargs": merged_engine_kwargs,
    }


def build_rollout_actor_init_config(
    *,
    engine_runtime_config: Dict[str, Any],
    reward_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build dedicated rollout actor init config from resolved runtime payload."""
    return {
        "engine_runtime_config": dict(engine_runtime_config),
        "reward_config": dict(reward_config),
    }

# ---------------------------------------------------------------------------
# Training domain
# ---------------------------------------------------------------------------

def _build_optimizer_config(args) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    return {
        "learning_rate": tc.learning_rate,
        "adam_beta1": tc.adam_beta1,
        "adam_beta2": tc.adam_beta2,
        "adam_epsilon": tc.adam_epsilon,
        "weight_decay": tc.weight_decay,
    }


def _build_scheduler_config(args, *, total_steps: int) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    return {
        "type": tc.lr_scheduler_type,
        "warmup_steps": tc.warmup_steps,
        "total_steps": total_steps,
    }


def _build_training_execution_config(
    args,
) -> Dict[str, Any]:
    tc = args.training    # TrainingConfig
    return {
        "max_grad_norm": tc.max_grad_norm,
        "replay_log_probs": args.sampling.replay_log_probs,
    }


def _build_training_topology_config(topology: ResolvedTrainTopology) -> Dict[str, Any]:
    return topology.as_dict()


def _build_training_plan_config(training_plan: ResolvedTrainingPlan) -> Dict[str, Any]:
    return training_plan.as_dict()


def build_train_backend_config(
    args,
    *,
    resolved_backend_config: Optional[ResolvedTrainBackendConfig] = None,
) -> Dict[str, Any]:
    """Build canonical backend config payload for TrainingActor.init."""
    backend_config = (
        resolved_backend_config
        if resolved_backend_config is not None
        else resolve_train_backend_config_from_args(args)
    )
    return backend_config.as_dict()


def build_training_actor_init_config(
    *,
    args,
    topology: ResolvedTrainTopology,
    training_plan: ResolvedTrainingPlan,
    algorithm_config: Dict[str, Any],
    model_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    sampling_config: Dict[str, Any],
    train_backend_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build full TrainingActor.init config directly from args.

    Pulls from: ModelConfig + TrainingConfig + AlgorithmConfig + SamplingConfig.
    """
    if not isinstance(algorithm_config, dict):
        raise ValueError(
            "build_training_actor_init_config requires algorithm_config to be provided by the driver."
        )

    rollout_control = args.rollout.control
    return {
        "model_config": dict(model_config),
        "reward_config": dict(reward_config),
        "optimizer_config": _build_optimizer_config(args),
        "scheduler_config": _build_scheduler_config(
            args,
            total_steps=rollout_control.num_rollout,
        ),
        "algorithm_config": dict(algorithm_config),
        "training_config": _build_training_execution_config(args),
        "topology_config": _build_training_topology_config(topology),
        "training_plan_config": _build_training_plan_config(training_plan),
        "sampling_config": dict(sampling_config),
        "train_backend_config": dict(train_backend_config),
    }


__all__ = [
    "build_model_config",
    "build_reward_config",
    "build_training_sampling_config",
    "build_rollout_engine_config",
    "build_rollout_actor_init_config",
    "build_train_backend_config",
    "build_training_actor_init_config",
]
