"""Training-domain config builders for TrainingActor initialization."""

from __future__ import annotations

from typing import Any, Dict

from diffusionrl.config.schema.model import build_model_config
from diffusionrl.config.schema.runtime import build_sampling_config


def _build_optimizer_config(args) -> Dict[str, Any]:
    return {
        "learning_rate": float(args.learning_rate),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "adam_epsilon": float(args.adam_epsilon),
        "weight_decay": float(args.weight_decay),
    }


def _build_scheduler_config(args, *, total_steps: int) -> Dict[str, Any]:
    return {
        "type": str(args.lr_scheduler_type),
        "warmup_steps": int(args.warmup_steps),
        "total_steps": int(total_steps),
    }


def _build_loss_config(args) -> Dict[str, Any]:
    return {
        "loss_type": str(getattr(args, "loss_type", "grpo")),
        "clip_range": float(args.clip_range),
        "clip_range_mode": str(args.clip_range_mode),
        "use_kl_penalty": bool(args.use_kl_penalty),
        "kl_coef": float(args.kl_coef),
        "eta": float(args.eta),
        "sde_type": str(args.sde_type),
        "guidance_scale": float(args.guidance_scale),
        "ignore_last": bool(getattr(args, "ignore_last", False)),
        "frozen_init_timesteps": int(getattr(args, "frozen_init_timesteps", 0)),
        "beta": float(getattr(args, "nft_beta", 0.1)),
        "adv_clip_max": float(getattr(args, "nft_adv_clip_max", 5.0)),
        "adv_mode": str(getattr(args, "nft_adv_mode", "raw")),
        "use_adaptive_weight": bool(getattr(args, "nft_use_adaptive_weight", True)),
        "shift": float(getattr(args, "shift", 3.0)),
        "nft_timestep_mode": str(getattr(args, "nft_timestep_mode", "random")),
        "nft_shuffle_timesteps": bool(getattr(args, "nft_shuffle_timesteps", True)),
        "nft_apply_shift": bool(getattr(args, "nft_apply_shift", False)),
        "use_ema": bool(getattr(args, "use_ema", False)),
        "ema_decay": float(getattr(args, "ema_decay", 0.001)),
        "decay_type": str(getattr(args, "ema_decay_type", "constant")),
        "ema_flat_steps": int(getattr(args, "ema_flat_steps", 0)),
        "ema_uprate": float(getattr(args, "ema_uprate", 0.001)),
        "ema_uphold": float(getattr(args, "ema_uphold", 0.5)),
    }


def _build_training_runtime_config(args, *, world_size: int) -> Dict[str, Any]:
    return {
        "max_grad_norm": float(args.max_grad_norm),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
        "prompts_per_batch": int(getattr(args, "prompts_per_batch", 1)),
        "num_samples_per_prompt": int(getattr(args, "num_samples_per_prompt", 1)),
        "batch_size": int(args.batch_size),
        "gradient_steps_per_epoch": int(getattr(args, "gradient_steps_per_epoch", 1)),
        "num_inner_epochs": int(getattr(args, "num_inner_epochs", 1)),
        "world_size": int(world_size),
        "replay_log_probs": bool(
            getattr(args, "replay_log_probs", False)
            or getattr(args, "fastvideo_replay_log_probs", False)
        ),
        "fastvideo_replay_log_probs": bool(
            getattr(args, "replay_log_probs", False)
            or getattr(args, "fastvideo_replay_log_probs", False)
        ),
    }


def _build_fsdp_config(args) -> Dict[str, Any]:
    return {
        "sharding_strategy": str(args.fsdp_sharding_strategy),
        "cpu_offload": bool(args.fsdp_cpu_offload),
        "backward_prefetch": str(args.fsdp_backward_prefetch),
    }


def build_training_actor_init_config(*, args, world_size: int) -> Dict[str, Any]:
    """Build full TrainingActor.init config directly from args."""
    return {
        "model_config": build_model_config(args),
        "optimizer_config": _build_optimizer_config(args),
        "scheduler_config": _build_scheduler_config(args, total_steps=args.num_rollout),
        "loss_config": _build_loss_config(args),
        "training_config": _build_training_runtime_config(args, world_size=world_size),
        "sampling_config": build_sampling_config(args),
        "use_fsdp": bool(args.use_fsdp),
        "fsdp_config": _build_fsdp_config(args),
    }
