"""Projection helpers from resolved config slices to actor/service payload dicts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict
from diffusionrl.config.resolution import (
    ModelSpec,
    TrainingPlan,
    TrainTopology,
    normalize_rollout_service_engine,
    normalize_lora_target_modules,
)
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.training.backends import TrainBackendConfig
from diffusionrl.types.sampling import SamplingSpec

# ---------------------------------------------------------------------------
# Model domain
# ---------------------------------------------------------------------------

def build_model_config(
    *,
    model_spec: ModelSpec,
    model_settings,
    training_settings,
    precision_settings,
) -> Dict[str, Any]:
    """Build model config consumed by training and rollout actors.

    Pulls from: ModelConfig (identity/paths) + TrainingConfig (LoRA/checkpointing)
    + precision.training (model load dtype).
    """
    training_precision = precision_settings.training
    return {
        "model_dotpath": model_spec.model_dotpath,
        "pretrained_model_ckpt_path": model_settings.pretrained_model_ckpt_path,
        "vae_ckpt_path": model_settings.vae_ckpt_path,
        "text_encoder_ckpt_path": model_settings.text_encoder_ckpt_path,
        "use_lora": training_settings.use_lora,
        "lora_rank": training_settings.lora_rank,
        "lora_alpha": training_settings.lora_alpha,
        "lora_target_modules": normalize_lora_target_modules(training_settings.lora_target_modules),
        "use_gradient_checkpointing": training_settings.use_gradient_checkpointing,
        "model_precision": training_precision.model_precision,
    }


# ---------------------------------------------------------------------------
# Reward domain
# ---------------------------------------------------------------------------


def build_reward_config(
    *,
    reward_schema: RewardSchema,
) -> Dict[str, Any]:
    """Build reward config consumed by actors and services."""
    return asdict(reward_schema)


# ---------------------------------------------------------------------------
# Sampling / rollout domain
# ---------------------------------------------------------------------------

def _build_shared_sampling_payload(sampling_spec: SamplingSpec) -> Dict[str, Any]:
    return {
        "sampler_dotpath": sampling_spec.sampler_dotpath,
        "num_inference_steps": int(sampling_spec.num_inference_steps),
        "sde_config": sampling_spec.sde_config.to_dict(),
        "guidance_scale": float(sampling_spec.guidance_scale),
        "height": int(sampling_spec.height),
        "width": int(sampling_spec.width),
        "num_frames": int(sampling_spec.num_frames),
    }


def build_training_sampling_config(
    *,
    precision_settings,
    sampling_spec: SamplingSpec,
    sampler_engine_type: str,
) -> Dict[str, Any]:
    """Build training-actor sampling config from the canonical resolved sampling spec."""
    payload = _build_shared_sampling_payload(sampling_spec)
    rollout_precision = precision_settings.rollout
    payload.update({
        "sampler_engine_type": normalize_rollout_service_engine(sampler_engine_type)
        or str(sampler_engine_type).strip().lower(),
        "replay_sampler_dotpath": sampling_spec.replay_sampler_dotpath,
        "seed": int(sampling_spec.seed),
        "sampling_adapter": sampling_spec.sampling_adapter,
        "init_same_noise": bool(sampling_spec.init_same_noise),
        "sampler_kwargs": dict(sampling_spec.sampler_kwargs),
        # Rollout precision lives at sampling_config top-level, not inside
        # sampler_kwargs, because it is a framework contract — not a sampler
        # constructor parameter.
        "autocast_precision": rollout_precision.autocast_precision,
        "trajectory_precision": rollout_precision.trajectory_precision,
        "logprob_precision": rollout_precision.logprob_precision,
    })
    return payload


def _build_rollout_engine_base_kwargs(
    *,
    rollout_topology_settings,
) -> Dict[str, Any]:
    """Build dedicated rollout-engine kwargs from canonical rollout fields."""
    resolved: Dict[str, Any] = {}

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
        value = getattr(rollout_topology_settings, attr_name)
        if value is not None:
            resolved[engine_key] = value

    sglang_field_map = {
        "sglang_local_mode": "local_mode",
        "sglang_verify_weight_checksum": "verify_weight_checksum",
        "sglang_prompt_encoder_device": "prompt_encoder_device",
        "sglang_prompt_encoder_max_length": "prompt_encoder_max_length",
        "sglang_disable_autocast": "disable_autocast",
    }
    for attr_name, engine_key in sglang_field_map.items():
        value = getattr(rollout_topology_settings, attr_name)
        if value is not None:
            resolved[engine_key] = value

    sglang_kwargs = rollout_topology_settings.sglang_kwargs
    if sglang_kwargs:
        if not isinstance(sglang_kwargs, dict):
            raise ValueError(
                "rollout.topology.sglang_kwargs must be a dict after normalization."
            )
        resolved["server_kwargs"] = dict(sglang_kwargs)

    return resolved


def build_rollout_engine_config(
    *,
    rollout_topology_settings,
    precision_settings,
    sync_settings,
    fps: int,
    logprob_source: str,
    sampler_engine_type: str,
    model_config: Dict[str, Any],
    sampling_spec: SamplingSpec,
    offload_rollout: bool = False,
) -> Dict[str, Any]:
    """Build final dedicated rollout engine runtime config."""
    merged_engine_kwargs = _build_rollout_engine_base_kwargs(
        rollout_topology_settings=rollout_topology_settings,
    )
    rollout_precision = precision_settings.rollout
    merged_engine_kwargs.setdefault("use_lora", model_config["use_lora"])
    merged_engine_kwargs.setdefault("lora_rank", model_config["lora_rank"])
    merged_engine_kwargs.setdefault("lora_alpha", model_config["lora_alpha"])
    merged_engine_kwargs.setdefault("lora_target_modules", model_config["lora_target_modules"])
    if model_config.get("vae_ckpt_path"):
        merged_engine_kwargs.setdefault("vae_ckpt_path", model_config["vae_ckpt_path"])
    if model_config.get("text_encoder_ckpt_path"):
        merged_engine_kwargs.setdefault("text_encoder_ckpt_path", model_config["text_encoder_ckpt_path"])
    if model_config.get("use_lora"):
        merged_engine_kwargs.setdefault("lora_merge_mode", "online")
    # Keep SGLang prompt-encoder precision on the canonical rollout precision
    # surface so rollout compute settings do not split across config namespaces.
    merged_engine_kwargs["prompt_encoder_dtype"] = rollout_precision.autocast_precision
    # Wire top-level fps into engine_kwargs so SGLang engine can consume it
    # without requiring users to duplicate rollout-topology config.
    merged_engine_kwargs.setdefault("fps", fps)
    merged_engine_kwargs.setdefault("weight_sync_dir", sync_settings.dir)
    if sync_settings.target_modules is not None:
        merged_engine_kwargs.setdefault("target_modules", list(sync_settings.target_modules))
    if sampler_engine_type == "sglang":
        merged_engine_kwargs["logprob_source"] = str(logprob_source)
        if offload_rollout:
            merged_engine_kwargs["require_memory_api"] = True
    rollout_batch_size = getattr(rollout_topology_settings, "rollout_batch_size", None)

    return {
        "sampler_engine_type": sampler_engine_type,
        "model_dotpath": model_config["model_dotpath"],
        "pretrained_model_ckpt_path": model_config["pretrained_model_ckpt_path"],
        "sampler_dotpath": sampling_spec.sampler_dotpath,
        "num_inference_steps": int(sampling_spec.num_inference_steps),
        "eta": float(sampling_spec.sde_config.eta),
        "sde_type": str(sampling_spec.sde_config.sde_type),
        "shift": float(sampling_spec.sde_config.shift),
        "guidance_scale": float(sampling_spec.guidance_scale),
        "height": int(sampling_spec.height),
        "width": int(sampling_spec.width),
        "num_frames": int(sampling_spec.num_frames),
        "engine_kwargs": merged_engine_kwargs,
        "rollout_batch_size": int(rollout_batch_size) if rollout_batch_size is not None else None,
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

def _build_optimizer_config(training_settings) -> Dict[str, Any]:
    return {
        "learning_rate": training_settings.learning_rate,
        "adam_beta1": training_settings.adam_beta1,
        "adam_beta2": training_settings.adam_beta2,
        "adam_epsilon": training_settings.adam_epsilon,
        "weight_decay": training_settings.weight_decay,
    }


def _build_scheduler_config(training_settings, *, total_steps: int) -> Dict[str, Any]:
    return {
        "type": training_settings.lr_scheduler_type,
        "warmup_steps": training_settings.warmup_steps,
        "total_steps": total_steps,
    }


def _build_training_execution_config(
    *,
    training_settings,
    replay_enabled: bool,
) -> Dict[str, Any]:
    return {
        "max_grad_norm": training_settings.max_grad_norm,
        "replay_enabled": bool(replay_enabled),
    }


def _build_training_topology_config(topology: TrainTopology) -> Dict[str, Any]:
    return topology.as_dict()


def _build_training_plan_config(training_plan: TrainingPlan) -> Dict[str, Any]:
    return training_plan.as_dict()


def build_train_backend_config(
    *,
    resolved_backend_config: TrainBackendConfig,
) -> Dict[str, Any]:
    """Build canonical backend config payload for TrainingActor.init."""
    return resolved_backend_config.as_dict()


def build_training_actor_init_config(
    *,
    training_settings,
    rollout_control_settings,
    replay_enabled: bool,
    topology: TrainTopology,
    training_plan: TrainingPlan,
    algorithm_config: Dict[str, Any],
    model_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    sampling_config: Dict[str, Any],
    train_backend_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build full TrainingActor.init config from resolved/runtime config slices.

    Pulls from: ModelConfig + TrainingConfig + AlgorithmConfig + SamplingConfig.
    """
    if not isinstance(algorithm_config, dict):
        raise ValueError(
            "build_training_actor_init_config requires algorithm_config to be provided by the driver."
        )

    return {
        "model_config": dict(model_config),
        "reward_config": dict(reward_config),
        "optimizer_config": _build_optimizer_config(training_settings),
        "scheduler_config": _build_scheduler_config(
            training_settings,
            total_steps=rollout_control_settings.num_rollout,
        ),
        "algorithm_config": dict(algorithm_config),
        "training_config": _build_training_execution_config(
            training_settings=training_settings,
            replay_enabled=replay_enabled,
        ),
        "topology_config": _build_training_topology_config(topology),
        "training_plan_config": _build_training_plan_config(training_plan),
        "sampling_config": dict(sampling_config),
        "train_backend_config": dict(train_backend_config),
    }


# ---------------------------------------------------------------------------
# SDE config resolution helpers
# ---------------------------------------------------------------------------

def resolve_sde_config(payload=None, *, default=None):
    """Resolve canonical nested ``sde_config`` from a domain-config dict."""
    from diffusionrl.types.sde import SDEConfig
    fallback = default or SDEConfig()
    if payload is None:
        return fallback
    raw = payload.get("sde_config") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return fallback
    return SDEConfig.from_mapping(raw, **fallback.to_dict())

__all__ = [
    "build_model_config",
    "build_reward_config",
    "build_training_sampling_config",
    "build_rollout_engine_config",
    "build_rollout_actor_init_config",
    "build_train_backend_config",
    "build_training_actor_init_config",
    "resolve_sde_config",
]
