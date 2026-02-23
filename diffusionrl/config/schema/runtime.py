"""Runtime-domain config builders for inference/sampling configuration."""

from __future__ import annotations

from typing import Any, Dict

from diffusionrl.config.schema.model import build_model_config


def build_sampling_config(args) -> Dict[str, Any]:
    """Build sampling config consumed by TrainingActor sampling path."""
    engine_kwargs = _resolve_engine_kwargs(args)
    return {
        "sampler_path": args.sampler_path,
        "replay_sampler_path": getattr(args, "replay_sampler_path", None),
        "num_inference_steps": int(args.num_inference_steps),
        "eta": float(args.eta),
        "sde_type": str(args.sde_type),
        "shift": float(args.shift),
        "guidance_scale": float(args.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(getattr(args, "num_frames", 16)),
        "sampling_adapter": getattr(args, "sampling_adapter", None),
        "init_same_noise": bool(getattr(args, "init_same_noise", False)),
        "num_samples_per_prompt": int(getattr(args, "num_samples_per_prompt", 1)),
        "sampler_kwargs": engine_kwargs.get("sampler_kwargs", {}),
    }


def _resolve_engine_kwargs(args) -> Dict[str, Any]:
    engine_kwargs = getattr(args, "engine_kwargs", {})
    if not isinstance(engine_kwargs, dict):
        return {}
    return dict(engine_kwargs)


def build_inference_engine_config(
    *,
    args,
    sampler_engine_type: str,
    engine_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build inference actor engine config directly from args."""
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
