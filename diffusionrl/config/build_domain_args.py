"""Config builders — convert flat TrainingArguments into domain-specific dicts for actors.

Each build_*() function extracts a domain-specific slice from args and returns a
plain dict that the corresponding actor/service consumes via its init() method.
This decouples actors from the TrainingArguments structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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

    reward_path: str
    reward_model_saved_path: Optional[str]
    reward_model_name: str
    reward_batch_size: int
    reward_timeout: float
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
        )


# ---------------------------------------------------------------------------
# Sampling / runtime domain
# ---------------------------------------------------------------------------

def build_sampling_config(args) -> Dict[str, Any]:
    """Build sampling config consumed by TrainingActor sampling path.

    Pulls from: SamplingConfig + TrainingArguments top-level (height/width/num_frames).
    """
    sc = args.sampling  # SamplingConfig
    engine_kwargs = _resolve_engine_kwargs(sc)
    return {
        "sampler_path": sc.sampler_path,
        "replay_sampler_path": sc.replay_sampler_path,
        "num_inference_steps": int(sc.num_inference_steps),
        "eta": float(sc.eta),
        "sde_type": str(sc.sde_type),
        "shift": float(sc.shift),
        "guidance_scale": float(sc.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "sampling_adapter": sc.sampling_adapter,
        "init_same_noise": bool(sc.init_same_noise),
        "num_samples_per_prompt": int(args.algorithm.num_samples_per_prompt),
        "sampler_kwargs": engine_kwargs.get("sampler_kwargs", {}),
    }


def _resolve_engine_kwargs(sc) -> Dict[str, Any]:
    engine_kwargs = getattr(sc, "engine_kwargs", {})
    if not isinstance(engine_kwargs, dict):
        return {}
    return dict(engine_kwargs)


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
    loss_kwargs: Dict[str, Any] = {}
    raw_loss_kwargs = getattr(ac, "loss_kwargs_json", "")
    if isinstance(raw_loss_kwargs, str) and raw_loss_kwargs.strip():
        try:
            parsed = json.loads(raw_loss_kwargs)
            if isinstance(parsed, dict):
                loss_kwargs = parsed
        except Exception:
            # Validation should already catch malformed JSON.
            loss_kwargs = {}
    return {
        "loss_type": str(ac.loss_type),
        "loss_path": str(ac.loss_path) if ac.loss_path else None,
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
        "beta": float(ac.nft.nft_beta),
        "adv_clip_max": float(ac.nft.nft_adv_clip_max),
        "adv_mode": str(ac.nft.nft_adv_mode),
        "use_adaptive_weight": bool(ac.nft.nft_use_adaptive_weight),
        "shift": float(sc.shift),
        "nft_timestep_mode": str(ac.nft.nft_timestep_mode),
        "nft_shuffle_timesteps": bool(ac.nft.nft_shuffle_timesteps),
        "nft_apply_shift": bool(ac.nft.nft_apply_shift),
        "use_ema": bool(ac.ema.use_ema),
        "ema_decay": float(ac.ema.ema_decay),
        "decay_type": str(ac.ema.ema_decay_type),
        "ema_flat_steps": int(ac.ema.ema_flat_steps),
        "ema_uprate": float(ac.ema.ema_uprate),
        "ema_uphold": float(ac.ema.ema_uphold),
    }


def _build_training_runtime_config(args, *, dp_size: int) -> Dict[str, Any]:
    tc = args.training    # TrainingConfig
    ac = args.algorithm   # AlgorithmConfig
    sc = args.sampling    # SamplingConfig
    return {
        "max_grad_norm": float(tc.max_grad_norm),
        "gradient_accumulation_steps": tc.gradient_accumulation_steps,
        "prompts_per_batch": int(ac.prompts_per_batch),
        "num_samples_per_prompt": int(ac.num_samples_per_prompt),
        "batch_size": int(tc.batch_size),
        "gradient_steps_per_epoch": int(tc.gradient_steps_per_epoch),
        "num_inner_epochs": int(tc.num_inner_epochs),
        "dp_size": int(dp_size),
        "replay_log_probs": bool(sc.replay_log_probs),
    }


def _build_fsdp_config(args) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    return {
        "sharding_strategy": str(tc.fsdp_sharding_strategy),
        "cpu_offload": bool(tc.fsdp_cpu_offload),
        "backward_prefetch": str(tc.fsdp_backward_prefetch),
    }


def _parse_json_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _build_train_backend_config(args) -> Dict[str, Any]:
    tc = args.training  # TrainingConfig
    backend_name = str(getattr(tc, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_kwargs = _parse_json_dict(getattr(tc, "train_backend_kwargs_json", ""))

    if backend_name == "fsdp":
        merged = _build_fsdp_config(args)
        merged["use_fsdp"] = bool(tc.use_fsdp)
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
    backend_config = _build_train_backend_config(args)
    return {
        "model_config": build_model_config(args),
        "optimizer_config": _build_optimizer_config(args),
        "scheduler_config": _build_scheduler_config(args, total_steps=args.rollout.num_rollout),
        "loss_config": _build_loss_config(args),
        "training_config": _build_training_runtime_config(args, dp_size=dp_size),
        "sampling_config": build_sampling_config(args),
        "train_backend": backend_config["name"],
        "train_backend_path": backend_config["backend_path"],
        "train_backend_kwargs": backend_config["kwargs"],
        "train_backend_config": backend_config,
        # Legacy compatibility keys for old actor code paths.
        "use_fsdp": bool(args.training.use_fsdp),
        "fsdp_config": _build_fsdp_config(args),
    }
