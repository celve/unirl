"""Config builders — convert flat TrainingArguments into domain-specific dicts for actors.

Each build_*() function extracts a domain-specific slice from args and returns a
plain dict that the corresponding actor/service consumes via its init() method.
This decouples actors from the TrainingArguments structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Model domain
# ---------------------------------------------------------------------------

def build_model_config(args) -> Dict[str, Any]:
    """Build model config consumed by training and rollout actors.

    Pulls from: ModelConfig (identity/paths) + TrainingConfig (LoRA/checkpointing).
    """
    mc = args.model    # ModelConfig
    tc = args.training  # TrainingConfig
    return {
        "model_path": mc.model_path,
        "pretrained_model_saved_path": mc.pretrained_model_saved_path,
        "vae_saved_path": mc.vae_saved_path,
        "text_encoder_path": mc.text_encoder_path,
        "use_lora": bool(tc.use_lora),
        "lora_rank": int(tc.lora_rank),
        "lora_alpha": int(tc.lora_alpha),
        "lora_target_modules": tc.lora_target_modules,
        "use_gradient_checkpointing": bool(tc.use_gradient_checkpointing),
    }


# ---------------------------------------------------------------------------
# Reward domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardSchema:
    """Typed view of reward-related CLI/config options.

    Mirrors RewardConfig fields. Use ``from_args()`` to construct from
    TrainingArguments — it delegates to the ``args.reward`` group internally.
    """

    reward_path: Optional[str]
    reward_model_saved_path: Optional[str]
    reward_model_name: str
    reward_batch_size: int
    reward_timeout: float
    local_reward_device: str
    use_http_reward: bool
    reward_service_url: Optional[str]
    reward_service_urls: Optional[List[str]]
    reward_models: Optional[List[str]]
    reward_weights: Optional[List[float]]
    reward_aggregation: str
    reward_mix_mode: str
    reward_dedicated_gpus_per_actor: int
    reward_dedicated_num_gpus: int
    reward_dedicated_num_nodes: int
    reward_dedicated_num_gpus_per_node: int
    reward_execution_mode: str

    @classmethod
    def from_args(cls, args) -> "RewardSchema":
        """Construct from TrainingArguments, delegating to the RewardConfig group."""
        rc = args.reward  # RewardConfig
        return cls(
            reward_path=rc.reward_path,
            reward_model_saved_path=rc.reward_model_saved_path,
            reward_model_name=rc.reward_model_name,
            reward_batch_size=int(rc.reward_batch_size),
            reward_timeout=float(rc.reward_timeout),
            local_reward_device=str(getattr(rc, "local_reward_device", "cpu")),
            use_http_reward=bool(rc.use_http_reward),
            reward_service_url=rc.reward_service_url,
            reward_service_urls=getattr(rc, "reward_service_urls", None),
            reward_models=rc.reward_models,
            reward_weights=rc.reward_weights,
            reward_aggregation=rc.reward_aggregation,
            reward_mix_mode=rc.reward_mix_mode,
            reward_dedicated_gpus_per_actor=int(rc.reward_dedicated_gpus_per_actor),
            reward_dedicated_num_gpus=int(rc.reward_dedicated_num_gpus),
            reward_dedicated_num_nodes=int(rc.reward_dedicated_num_nodes),
            reward_dedicated_num_gpus_per_node=int(rc.reward_dedicated_num_gpus_per_node),
            reward_execution_mode=str(getattr(rc, "reward_execution_mode", "manager")),
        )

    @property
    def uses_rollout_execution(self) -> bool:
        return str(self.reward_execution_mode or "manager").strip().lower() == "rollout"

    def component_weights(self) -> Dict[str, float]:
        if self.reward_models:
            weights = self.reward_weights or []
            return {
                str(model): float(weights[idx]) if idx < len(weights) else 1.0
                for idx, model in enumerate(self.reward_models)
            }
        return {str(self.reward_model_name): 1.0}


def build_reward_config(args) -> Dict[str, Any]:
    """Build reward config consumed by actors and services."""
    return asdict(RewardSchema.from_args(args))


# ---------------------------------------------------------------------------
# Sampling / runtime domain
# ---------------------------------------------------------------------------

def build_sampling_config(args) -> Dict[str, Any]:
    """Build sampling config consumed by TrainingActor sampling path.

    Pulls from: SamplingConfig + TrainingArguments top-level (height/width/num_frames).
    Rollout collection geometry stays in rollout/training runtime config rather than
    sampler config so engines only receive knobs they actually consume.
    """
    sc = args.sampling  # SamplingConfig
    engine_kwargs = sc.engine_kwargs
    if not isinstance(engine_kwargs, dict):
        raise ValueError(
            "sampling.engine_kwargs must be a dict after normalization, "
            f"got: {type(engine_kwargs).__name__}"
        )
    engine_kwargs = dict(engine_kwargs)
    return {
        "sampler_path": sc.sampler_path,
        "sampler_engine_type": sc.sampler_engine_type,
        "replay_sampler_path": sc.replay_sampler_path,
        "num_inference_steps": int(sc.num_inference_steps),
        "eta": float(sc.eta),
        "sde_type": str(sc.sde_type),
        "shift": float(sc.shift),
        "guidance_scale": float(sc.guidance_scale),
        "timestep_fraction": getattr(sc, "timestep_fraction", 1.0),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "sampling_adapter": sc.sampling_adapter,
        "init_same_noise": bool(sc.init_same_noise),
        "num_samples_per_prompt": int(args.algorithm.num_samples_per_prompt),
        "sampler_kwargs": engine_kwargs.get("sampler_kwargs", {}),
    }


def build_rollout_engine_config(
    *,
    args,
    sampler_engine_type: str,
    engine_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build rollout actor engine config directly from args.

    Pulls from: build_model_config + build_sampling_config + caller-provided engine_kwargs.
    """
    model_config = build_model_config(args)
    sampling_config = build_sampling_config(args)

    merged_engine_kwargs = dict(engine_kwargs)
    merged_engine_kwargs.setdefault("use_lora", model_config["use_lora"])
    merged_engine_kwargs.setdefault("lora_rank", model_config["lora_rank"])
    merged_engine_kwargs.setdefault("lora_alpha", model_config["lora_alpha"])
    merged_engine_kwargs.setdefault("lora_target_modules", model_config["lora_target_modules"])
    if model_config.get("vae_saved_path"):
        merged_engine_kwargs.setdefault("vae_saved_path", model_config["vae_saved_path"])
    if model_config.get("text_encoder_path"):
        merged_engine_kwargs.setdefault("text_encoder_path", model_config["text_encoder_path"])
    # Wire top-level fps into engine_kwargs so SGLang engine can consume it
    # without requiring users to pass it through --engine-kwargs JSON.
    merged_engine_kwargs.setdefault("fps", int(args.fps))

    return {
        "sampler_engine_type": sampler_engine_type,
        "sampler_path": sampling_config["sampler_path"],
        "model_path": model_config["model_path"],
        "pretrained_model_saved_path": model_config["pretrained_model_saved_path"],
        "lora_rank": model_config["lora_rank"],
        "lora_alpha": model_config["lora_alpha"],
        "lora_target_modules": model_config["lora_target_modules"],
        "num_inference_steps": sampling_config["num_inference_steps"],
        "eta": sampling_config["eta"],
        "sde_type": sampling_config["sde_type"],
        "shift": sampling_config["shift"],
        "guidance_scale": sampling_config["guidance_scale"],
        "height": sampling_config["height"],
        "width": sampling_config["width"],
        "num_frames": sampling_config["num_frames"],
        "engine_kwargs": merged_engine_kwargs,
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


def _build_loss_config(args) -> Dict[str, Any]:
    ac = args.algorithm  # AlgorithmConfig
    sc = args.sampling   # SamplingConfig
    loss_kwargs = _require_normalized_kwargs_dict(
        ac.loss_kwargs,
        field_name="algorithm.loss_kwargs",
    )
    algorithm_kwargs = _require_normalized_kwargs_dict(
        ac.algorithm_kwargs,
        field_name="algorithm.algorithm_kwargs",
    )

    loss_path = str(ac.loss_path or "").strip()
    if not loss_path:
        raise ValueError(
            "algorithm.loss_path must be resolved before build_domain_args(). "
            "validate_args() should set this from loss_type or explicit config."
        )

    return {
        "algorithm_path": str(ac.algorithm_path),
        "algorithm_kwargs": algorithm_kwargs,
        "loss_type": str(ac.loss_type),
        "loss_path": loss_path,
        "loss_kwargs": loss_kwargs,
        "clip_range": float(ac.clip_range),
        "clip_range_mode": str(ac.clip_range_mode),
        "use_kl_penalty": bool(ac.use_kl_penalty),
        "kl_coef": float(ac.kl_coef),
        "eta": float(sc.eta),
        "sde_type": str(sc.sde_type),
        "guidance_scale": float(sc.guidance_scale),
        "ignore_last": bool(ac.ignore_last),
        "frozen_init_timesteps": int(ac.frozen_init_timesteps),
        "shift": float(sc.shift),
        "debug_output_dir": getattr(args.debug, "debug_output_dir", None),
    }

def _build_training_runtime_config(args, *, dp_size: int) -> Dict[str, Any]:
    tc = args.training    # TrainingConfig
    if tc.gradient_accumulation_batch_size is None:
        raise ValueError(
            "training.gradient_accumulation_batch_size must be normalized before "
            "build_domain_args()."
        )
    return {
        "max_grad_norm": float(tc.max_grad_norm),
        "gradient_accumulation_batch_size": int(tc.gradient_accumulation_batch_size),
        "multi_update_batch_size": (
            int(tc.multi_update_batch_size)
            if tc.multi_update_batch_size is not None
            else None
        ),
        "update_mode": str(tc.update_mode),
        "dp_size": int(dp_size),
        "replay_log_probs": bool(args.sampling.replay_log_probs),
    }


def _build_fsdp_config(args) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    return {
        "cpu_offload": bool(tc.fsdp_cpu_offload),
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
        "tp_size": int(args.sampling.tp_size),
        "pp_size": 1,
        "sp_size": int(args.sampling.sp_size),
        "ep_size": 1,
        "enable_full_shard": True,
        "enable_reshard_after_forward": True,
        "enable_mixed_precision": enable_mixed_precision,
        "enable_gradient_checkpointing": bool(tc.use_gradient_checkpointing),
        "init_device": "meta",
        "broadcast_model_weights_from_rank0": True,
        "weights_path_mode": "transformer_subdir",
    }


def _require_normalized_kwargs_dict(raw: Any, *, field_name: str) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError(
        f"{field_name} must be a dict after normalize/validate, "
        f"got: {type(raw).__name__}"
    )


def _build_train_backend_config(args, *, dp_size: Optional[int] = None) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    backend_name = str(tc.train_backend or "").strip().lower()
    if not backend_name:
        raise ValueError(
            "training.train_backend is empty after normalization; "
            "expected a non-empty backend name."
        )

    backend_kwargs = _require_normalized_kwargs_dict(
        tc.train_backend_kwargs,
        field_name="training.train_backend_kwargs",
    )

    if backend_name == "fsdp":
        merged = _build_fsdp_config(args)
        merged.update(backend_kwargs)
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


def build_training_actor_init_config(*, args, dp_size: int) -> Dict[str, Any]:
    """Build full TrainingActor.init config directly from args.

    Pulls from: ModelConfig + TrainingConfig + AlgorithmConfig + SamplingConfig.
    """
    backend_config = _build_train_backend_config(args, dp_size=dp_size)
    return {
        "model_config": build_model_config(args),
        "reward_config": build_reward_config(args),
        "optimizer_config": _build_optimizer_config(args),
        "scheduler_config": _build_scheduler_config(args, total_steps=args.rollout.num_rollout),
        "loss_config": _build_loss_config(args),
        "training_config": _build_training_runtime_config(args, dp_size=dp_size),
        "sampling_config": build_sampling_config(args),
        "train_backend_config": backend_config,
    }


def _require_dict_section(config: Dict[str, Any], *, name: str) -> Dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got: {type(value).__name__}")
    return value


def validate_rollout_engine_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for rollout actor config."""
    if not isinstance(config, dict):
        raise ValueError(f"rollout_engine_config must be a dict, got: {type(config).__name__}")
    if not isinstance(config.get("engine_kwargs"), dict):
        raise ValueError(
            "rollout_engine_config.engine_kwargs must be a dict, "
            f"got: {type(config.get('engine_kwargs')).__name__}"
        )
    if not isinstance(config.get("reward_config"), dict):
        raise ValueError(
            "rollout_engine_config.reward_config must be a dict, "
            f"got: {type(config.get('reward_config')).__name__}"
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
        "loss_config",
        "training_config",
        "sampling_config",
        "train_backend_config",
    ):
        _require_dict_section(config, name=section)

    loss_config = config["loss_config"]
    if not isinstance(loss_config.get("loss_kwargs"), dict):
        raise ValueError(
            "loss_config.loss_kwargs must be a dict, "
            f"got: {type(loss_config.get('loss_kwargs')).__name__}"
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


__all__ = [
    "RewardSchema",
    "build_model_config",
    "build_reward_config",
    "build_sampling_config",
    "build_rollout_engine_config",
    "build_training_actor_init_config",
    "validate_rollout_engine_config",
    "validate_training_actor_init_config",
]
