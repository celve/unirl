"""Config builders — convert flat TrainingArguments into domain-specific dicts for actors.

Each build_*() function extracts a domain-specific slice from args and returns a
plain dict that the corresponding actor/service consumes via its init() method.
This decouples actors from the TrainingArguments structure.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from diffusionrl.config.resolution import (
    DEFAULT_SAMPLER_PATH,
    ResolvedTrainingPlan,
    ResolvedTrainTopology,
    resolve_algorithm_kwargs,
    resolve_algorithm_path,
    resolve_lora_target_modules,
    resolve_model_runtime,
    resolve_train_backend_kwargs,
    resolve_train_backend_name,
    resolve_training_plan,
)
from diffusionrl.config.rollout_topology import (
    normalize_rollout_service_engine,
    resolve_rollout_service_kwargs,
)
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.types.sde import SDEConfig, SDEScheduleConfig

# ---------------------------------------------------------------------------
# Model domain
# ---------------------------------------------------------------------------

def build_model_config(args) -> Dict[str, Any]:
    """Build model config consumed by training and rollout actors.

    Pulls from: ModelConfig (identity/paths) + TrainingConfig (LoRA/checkpointing)
    + PrecisionConfig (model load dtype).
    """
    resolved_model = resolve_model_runtime(
        args,
        explicit_sampler_path=(getattr(args.sampling, "sampler_path", None) != DEFAULT_SAMPLER_PATH),
    )
    mc = args.model    # ModelConfig
    tc = args.training  # TrainingConfig
    pc = args.precision  # PrecisionConfig
    return {
        "model_path": resolved_model.model_path,
        "pretrained_model_saved_path": mc.pretrained_model_saved_path,
        "vae_saved_path": mc.vae_saved_path,
        "text_encoder_path": mc.text_encoder_path,
        "use_lora": bool(tc.use_lora),
        "lora_rank": int(tc.lora_rank),
        "lora_alpha": int(tc.lora_alpha),
        "lora_target_modules": resolve_lora_target_modules(tc.lora_target_modules),
        "use_gradient_checkpointing": bool(tc.use_gradient_checkpointing),
        "model_precision": str(pc.model_precision),
    }


# ---------------------------------------------------------------------------
# Reward domain
# ---------------------------------------------------------------------------


def build_reward_config(args) -> Dict[str, Any]:
    """Build reward config consumed by actors and services."""
    return asdict(RewardSchema.from_args(args))


# ---------------------------------------------------------------------------
# Sampling / runtime domain
# ---------------------------------------------------------------------------

def _build_sde_config(args) -> SDEConfig:
    """Build the stable SDE math contract from sampling args."""
    sc = args.sampling  # SamplingConfig
    return SDEConfig(
        eta=float(sc.eta),
        sde_type=normalize_sde_type(sc.sde_type),
        shift=float(sc.time_shift),
    )


def _build_sde_schedule_config(args) -> SDEScheduleConfig:
    """Build the rollout-time SDE scheduling policy from sampling args."""
    sc = args.sampling  # SamplingConfig
    return SDEScheduleConfig(
        sde_ratio=float(sc.sde_ratio),
        timestep_fraction=getattr(sc, "timestep_fraction", 1.0),
    )


def build_sampling_config(args) -> Dict[str, Any]:
    """Build sampling config consumed by TrainingActor sampling path.

    Pulls from: SamplingConfig + TrainingArguments top-level (height/width/num_frames).
    Rollout collection geometry stays in rollout/training runtime config rather than
    sampler config so engines only receive knobs they actually consume.

    This is intentionally the superset config: direct actor sampling needs a few
    train-side extras such as replay sampler wiring, scheduling policy, seed, and
    adapter knobs that dedicated rollout engines do not consume.
    """
    resolved_model = resolve_model_runtime(
        args,
        explicit_sampler_path=(getattr(args.sampling, "sampler_path", None) != DEFAULT_SAMPLER_PATH),
    )
    sc = args.sampling  # SamplingConfig
    pc = args.precision  # PrecisionConfig
    sde_config = _build_sde_config(args)
    sde_schedule_config = _build_sde_schedule_config(args)
    sampler_kwargs = dict(sc.sampler_kwargs or {})
    sampler_kwargs["autocast_precision"] = str(pc.autocast_precision)
    sampler_kwargs["trajectory_precision"] = str(pc.trajectory_precision)
    sampler_kwargs["logprob_precision"] = str(pc.logprob_precision)

    return {
        "sampler_path": resolved_model.sampler_path,
        "sampler_engine_type": normalize_rollout_service_engine(
            getattr(args.rollout, "service_engine", None)
        ),
        "replay_sampler_path": sc.replay_sampler_path,
        "num_inference_steps": int(sc.num_inference_steps),
        "sde_config": sde_config.to_dict(),
        "guidance_scale": float(sc.guidance_scale),
        "sde_schedule_config": sde_schedule_config.to_dict(),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "sampling_adapter": sc.sampling_adapter,
        "init_same_noise": bool(sc.init_same_noise),
        "sampler_kwargs": sampler_kwargs,
    }


def build_rollout_sampling_config(args) -> Dict[str, Any]:
    """Build sampling defaults consumed by dedicated rollout actors/engines.

    Keep this as the subset of ``build_sampling_config()`` that the dedicated
    rollout path actually consumes. Direct actor sampling remains a transitional
    mode, so the split is intentional even though the two payloads overlap.
    """
    resolved_model = resolve_model_runtime(
        args,
        explicit_sampler_path=(getattr(args.sampling, "sampler_path", None) != DEFAULT_SAMPLER_PATH),
    )
    sc = args.sampling  # SamplingConfig
    sde_config = _build_sde_config(args)

    return {
        "sampler_path": resolved_model.sampler_path,
        "num_inference_steps": int(sc.num_inference_steps),
        "sde_config": sde_config.to_dict(),
        "guidance_scale": float(sc.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
    }


def build_rollout_engine_config(
    *,
    args,
    sampler_engine_type: str,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build dedicated rollout engine bootstrap config directly from args.

    Pulls from: build_model_config + caller-provided engine_kwargs.
    """
    model_config = build_model_config(args)
    pc = args.precision  # PrecisionConfig

    merged_engine_kwargs = resolve_rollout_service_kwargs(args)
    if engine_kwargs:
        merged_engine_kwargs.update(dict(engine_kwargs))
    merged_engine_kwargs.setdefault("use_lora", model_config["use_lora"])
    merged_engine_kwargs.setdefault("lora_rank", model_config["lora_rank"])
    merged_engine_kwargs.setdefault("lora_alpha", model_config["lora_alpha"])
    merged_engine_kwargs.setdefault("lora_target_modules", model_config["lora_target_modules"])
    if model_config.get("vae_saved_path"):
        merged_engine_kwargs.setdefault("vae_saved_path", model_config["vae_saved_path"])
    if model_config.get("text_encoder_path"):
        merged_engine_kwargs.setdefault("text_encoder_path", model_config["text_encoder_path"])
    merged_engine_kwargs["model_precision"] = str(pc.model_precision)
    merged_engine_kwargs["fsdp_precision"] = str(pc.fsdp_precision)
    merged_engine_kwargs.setdefault("prompt_encoder_dtype", str(pc.model_precision))
    # Wire top-level fps into engine_kwargs so SGLang engine can consume it
    # without requiring users to duplicate rollout-runtime config.
    merged_engine_kwargs.setdefault("fps", int(args.fps))
    merged_engine_kwargs.setdefault("weight_sync_dir", str(args.sync.dir))
    if getattr(args.sync, "target_modules", None) is not None:
        merged_engine_kwargs.setdefault("target_modules", list(args.sync.target_modules))
    if sampler_engine_type == "sglang":
        merged_engine_kwargs["logprob_source"] = str(args.sampling.logprob_source)

    return {
        "sampler_engine_type": sampler_engine_type,
        "model_path": model_config["model_path"],
        "pretrained_model_saved_path": model_config["pretrained_model_saved_path"],
        "engine_kwargs": merged_engine_kwargs,
    }


def build_rollout_actor_init_config(
    *,
    args,
    sampler_engine_type: str,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build dedicated rollout actor init config with split engine/sampling sections."""
    return {
        "engine_config": build_rollout_engine_config(
            args=args,
            sampler_engine_type=sampler_engine_type,
            engine_kwargs=engine_kwargs,
        ),
        "sampling_config": build_rollout_sampling_config(args),
        "reward_config": build_reward_config(args),
    }

# ---------------------------------------------------------------------------
# Training domain
# ---------------------------------------------------------------------------

def _build_optimizer_config(args) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    return {
        "learning_rate": float(tc.learning_rate),
        "adam_beta1": float(tc.adam_beta1),
        "adam_beta2": float(tc.adam_beta2),
        "adam_epsilon": float(tc.adam_epsilon),
        "weight_decay": float(tc.weight_decay),
    }


def _build_scheduler_config(args, *, total_steps: int) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    return {
        "type": str(tc.lr_scheduler_type),
        "warmup_steps": int(tc.warmup_steps),
        "total_steps": int(total_steps),
    }


def _build_algorithm_kwargs(args) -> Dict[str, Any]:
    ac = args.algorithm  # AlgorithmConfig
    algorithm_kwargs = resolve_algorithm_kwargs(args)

    # Single source of truth for algorithm construction.
    # Both rollout-side and train-side algorithms consume this same payload.
    resolved_kwargs = dict(algorithm_kwargs)
    algorithm_type = str(ac.algorithm_type or "").strip().lower()
    if algorithm_type in {"grpo", "mix_grpo"}:
        resolved_kwargs.setdefault("clip_range", float(ac.clip_range))
        resolved_kwargs.setdefault("clip_schedule", str(ac.clip_schedule))
        resolved_kwargs.setdefault("use_kl_penalty", bool(ac.use_kl_penalty))
    resolved_kwargs.setdefault("kl_coef", float(ac.kl_coef))
    resolved_kwargs.setdefault("component_mix_stage", str(ac.component_mix_stage))
    resolved_kwargs.setdefault("adv_normalization", str(ac.adv_normalization))
    resolved_kwargs.setdefault("samples_per_prompt", int(ac.samples_per_prompt))
    resolved_kwargs.setdefault("adv_norm_eps", float(ac.adv_norm_eps))
    resolved_kwargs.setdefault("adv_clip_abs", ac.adv_clip_abs)
    resolved_kwargs.setdefault("trimmed_ratio", float(ac.trimmed_ratio))
    resolved_kwargs.setdefault("use_global_std", bool(ac.use_global_std))
    resolved_kwargs.setdefault("skip_last_timestep", bool(ac.skip_last_timestep))
    resolved_kwargs.setdefault("skip_initial_timesteps", int(ac.skip_initial_timesteps))
    resolved_kwargs.setdefault("eval_ema_decay", float(ac.eval_ema_decay))
    resolved_kwargs.setdefault("eval_ema_update_interval", int(ac.eval_ema_update_interval))
    resolved_kwargs.setdefault("window_training", bool(ac.window.window_training))
    return resolved_kwargs


def build_algorithm_config(args) -> Dict[str, Any]:
    ac = args.algorithm  # AlgorithmConfig
    sc = args.sampling   # SamplingConfig
    sde_config = _build_sde_config(args)
    sde_schedule_config = _build_sde_schedule_config(args)

    algorithm_path = str(ac.algorithm_path or "").strip()
    if not algorithm_path:
        algorithm_path = resolve_algorithm_path(args)

    return {
        "algorithm_type": str(ac.algorithm_type),
        "algorithm_path": algorithm_path,
        "algorithm_kwargs": _build_algorithm_kwargs(args),
        "sde_config": sde_config.to_dict(),
        "sde_schedule_config": sde_schedule_config.to_dict(),
        "guidance_scale": float(sc.guidance_scale),
        "debug_output_dir": getattr(args.debug, "debug_output_dir", None),
    }

def _build_training_runtime_config(
    args,
    *,
    dp_size: int,
    training_plan: ResolvedTrainingPlan,
) -> Dict[str, Any]:
    tc = args.training    # TrainingConfig
    return {
        "max_grad_norm": float(tc.max_grad_norm),
        "local_micro_batch_size": int(training_plan.local_micro_batch_size),
        "local_update_batch_size": int(training_plan.local_update_batch_size),
        "num_updates_per_local_batch": int(training_plan.num_updates_per_local_batch),
        "dp_size": int(dp_size),
        "replay_log_probs": bool(args.sampling.replay_log_probs),
    }


def _build_training_topology_config(topology: ResolvedTrainTopology) -> Dict[str, Any]:
    return topology.as_dict()


def _build_training_plan_config(training_plan: ResolvedTrainingPlan) -> Dict[str, Any]:
    return training_plan.as_dict()


def _build_fsdp_config(args) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    pc = args.precision  # PrecisionConfig
    return {
        "cpu_offload": bool(tc.fsdp_cpu_offload),
        "param_dtype": str(pc.fsdp_precision),
    }


def _build_veomni_config(args, *, dp_size: Optional[int] = None) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    # Keep VeOmni dp topology aligned with the actual training actor group size.
    world_size = int(dp_size) if dp_size is not None else int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node)
    enable_mixed_precision = bool(world_size > 1)
    return {
        # VeOmni path is FSDP2-focused. Keep defaults explicit and let users
        # override with train_backend_kwargs.
        "data_parallel_mode": "fsdp2",
        "dp_size": int(world_size),
        "dp_replicate_size": 1,
        "dp_shard_size": int(world_size),
        "tp_size": 1,
        "pp_size": 1,
        "sp_size": 1,
        "ep_size": 1,
        "enable_full_shard": True,
        "enable_reshard_after_forward": True,
        "enable_mixed_precision": enable_mixed_precision,
        "enable_gradient_checkpointing": bool(tc.use_gradient_checkpointing),
        "init_device": "meta",
        "broadcast_model_weights_from_rank0": True,
        "weights_path_mode": "transformer_subdir",
    }


def _build_train_backend_config(args, *, dp_size: Optional[int] = None) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    backend_name = resolve_train_backend_name(args)
    if not backend_name:
        raise ValueError(
            "training.train_backend is empty after validation; "
            "expected a non-empty backend name."
        )

    backend_kwargs = resolve_train_backend_kwargs(args)

    if backend_name == "fsdp":
        merged = dict(backend_kwargs)
        merged.update(_build_fsdp_config(args))
        backend_kwargs = merged
    elif backend_name == "veomni":
        merged = _build_veomni_config(args, dp_size=dp_size)
        merged.update(backend_kwargs)
        backend_kwargs = merged

    return {
        "name": backend_name,
        "backend_path": tc.train_backend_path,
        "kwargs": backend_kwargs,
    }


def build_training_actor_init_config(
    *,
    args,
    topology: ResolvedTrainTopology,
    algorithm_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build full TrainingActor.init config directly from args.

    Pulls from: ModelConfig + TrainingConfig + AlgorithmConfig + SamplingConfig.
    """
    if not isinstance(algorithm_config, dict):
        raise ValueError(
            "build_training_actor_init_config requires algorithm_config to be provided by the driver."
        )

    training_plan = resolve_training_plan(args)
    backend_config = _build_train_backend_config(args, dp_size=topology.dp_size)
    return {
        "model_config": build_model_config(args),
        "reward_config": build_reward_config(args),
        "optimizer_config": _build_optimizer_config(args),
        "scheduler_config": _build_scheduler_config(args, total_steps=args.rollout.num_rollout),
        "algorithm_config": dict(algorithm_config),
        "training_config": _build_training_runtime_config(
            args,
            dp_size=topology.dp_size,
            training_plan=training_plan,
        ),
        "topology_config": _build_training_topology_config(topology),
        "training_plan_config": _build_training_plan_config(training_plan),
        "sampling_config": build_sampling_config(args),
        "train_backend_config": backend_config,
    }


def _require_dict_section(config: Dict[str, Any], *, name: str) -> Dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got: {type(value).__name__}")
    return value


def validate_rollout_engine_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for dedicated rollout engine config."""
    if not isinstance(config, dict):
        raise ValueError(f"rollout_engine_config must be a dict, got: {type(config).__name__}")
    if not isinstance(config.get("engine_kwargs"), dict):
        raise ValueError(
            "rollout_engine_config.engine_kwargs must be a dict, "
            f"got: {type(config.get('engine_kwargs')).__name__}"
        )


def validate_rollout_actor_init_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for rollout actor init config."""
    if not isinstance(config, dict):
        raise ValueError(f"rollout_actor_init_config must be a dict, got: {type(config).__name__}")

    engine_config = _require_dict_section(config, name="engine_config")
    sampling_config = _require_dict_section(config, name="sampling_config")
    reward_config = _require_dict_section(config, name="reward_config")

    validate_rollout_engine_config(engine_config)

    sampler_path = str(sampling_config.get("sampler_path", "") or "").strip()
    if not sampler_path:
        raise ValueError(
            "rollout_actor_init_config.sampling_config.sampler_path is required."
        )
    if not isinstance(sampling_config.get("sde_config"), dict):
        raise ValueError(
            "rollout_actor_init_config.sampling_config.sde_config must be a dict, "
            f"got: {type(sampling_config.get('sde_config')).__name__}"
        )
    if not isinstance(reward_config, dict):
        raise ValueError(
            "rollout_actor_init_config.reward_config must be a dict, "
            f"got: {type(reward_config).__name__}"
        )


def validate_training_actor_init_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for training actor config."""
    if not isinstance(config, dict):
        raise ValueError(f"training_actor_init_config must be a dict, got: {type(config).__name__}")

    for section in (
        "model_config",
        "reward_config",
        "optimizer_config",
        "scheduler_config",
        "algorithm_config",
        "training_config",
        "topology_config",
        "training_plan_config",
        "sampling_config",
        "train_backend_config",
    ):
        _require_dict_section(config, name=section)

    algorithm_config = config["algorithm_config"]
    if not isinstance(algorithm_config.get("algorithm_kwargs"), dict):
        raise ValueError(
            "algorithm_config.algorithm_kwargs must be a dict, "
            f"got: {type(algorithm_config.get('algorithm_kwargs')).__name__}"
        )

    backend_config = config["train_backend_config"]
    if not isinstance(backend_config.get("kwargs"), dict):
        raise ValueError(
            "train_backend_config.kwargs must be a dict, "
            f"got: {type(backend_config.get('kwargs')).__name__}"
        )
    backend_name = str(backend_config.get("name", "") or "").strip().lower()
    if not backend_name:
        raise ValueError("train_backend_config.name is required.")

    topology_config = config["topology_config"]
    for required_key in ("actor_count", "world_size", "dp_size"):
        value = topology_config.get(required_key)
        if value is None:
            raise ValueError(f"topology_config.{required_key} is required.")
        if int(value) < 1:
            raise ValueError(
                f"topology_config.{required_key} must be >= 1, got: {value!r}"
            )

    training_plan_config = config["training_plan_config"]
    for required_key in (
        "global_batch_size",
        "local_batch_size",
        "local_update_batch_size",
        "local_micro_batch_size",
        "num_updates_per_local_batch",
    ):
        value = training_plan_config.get(required_key)
        if value is None:
            raise ValueError(f"training_plan_config.{required_key} is required.")
        if int(value) < 1:
            raise ValueError(
                f"training_plan_config.{required_key} must be >= 1, got: {value!r}"
            )

    update_slices = training_plan_config.get("update_slices")
    if not isinstance(update_slices, list) or not update_slices:
        raise ValueError("training_plan_config.update_slices must be a non-empty list.")
    for index, item in enumerate(update_slices):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(
                f"training_plan_config.update_slices[{index}] must be a length-2 list."
            )
        start, end = int(item[0]), int(item[1])
        if start < 0 or end <= start:
            raise ValueError(
                f"training_plan_config.update_slices[{index}] must satisfy 0 <= start < end."
            )

    mini_batch_slices = training_plan_config.get("mini_batch_slices_per_update")
    if not isinstance(mini_batch_slices, list) or not mini_batch_slices:
        raise ValueError(
            "training_plan_config.mini_batch_slices_per_update must be a non-empty list."
        )
    if len(mini_batch_slices) != len(update_slices):
        raise ValueError(
            "training_plan_config.mini_batch_slices_per_update must align with update_slices. "
            f"Got {len(mini_batch_slices)} vs {len(update_slices)}."
        )
    for update_index, per_update in enumerate(mini_batch_slices):
        if not isinstance(per_update, list) or not per_update:
            raise ValueError(
                "training_plan_config.mini_batch_slices_per_update entries must be non-empty lists. "
                f"Got index={update_index}."
            )
        update_size = int(update_slices[update_index][1]) - int(update_slices[update_index][0])
        for mini_index, mini_slice in enumerate(per_update):
            if not isinstance(mini_slice, list) or len(mini_slice) != 2:
                raise ValueError(
                    "training_plan_config.mini_batch_slices_per_update"
                    f"[{update_index}][{mini_index}] must be a length-2 list."
                )
            start, end = int(mini_slice[0]), int(mini_slice[1])
            if start < 0 or end <= start or end > update_size:
                raise ValueError(
                    "training_plan_config.mini_batch_slices_per_update"
                    f"[{update_index}][{mini_index}] must satisfy 0 <= start < end <= update_size."
                )


__all__ = [
    "build_model_config",
    "build_reward_config",
    "build_sampling_config",
    "build_rollout_sampling_config",
    "build_rollout_engine_config",
    "build_rollout_actor_init_config",
    "build_training_actor_init_config",
    "validate_rollout_engine_config",
    "validate_rollout_actor_init_config",
    "validate_training_actor_init_config",
]
