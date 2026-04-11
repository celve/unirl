"""Built-in rollout-engine cmdline adaptation helpers."""

from __future__ import annotations

from typing import Any, Dict

from diffusionrl.cmdline.construction import build_component_init_payload_from_args
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.samplers.registry import ROLLOUT_ENGINE_COMPONENT_FAMILY
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.sampling import SamplingSpec


@register_cmdline_config_parser(EngineConfig)
def build_rollout_engine_config_from_args(
    args: Any,
    *,
    model_init_payload: ComponentInitPayload,
    sampling_spec: SamplingSpec,
    rollout_mode_info: Any,
) -> EngineConfig:
    """Build typed rollout-engine config from framework args and resolved slices."""
    model_config = model_init_payload.component_config
    if not isinstance(model_config, ModelBundleConfig):
        raise ValueError(
            "model_init_payload.component_config must be a ModelBundleConfig, "
            f"got: {type(model_config).__name__}"
        )

    rollout_engine = str(rollout_mode_info.rollout_topology.rollout_engine or "")
    logprob_source = str(rollout_mode_info.logprob_source)

    merged_engine_kwargs: Dict[str, Any] = {}

    if args.rollout.num_gpus_per_actor is not None:
        merged_engine_kwargs["num_gpus"] = args.rollout.num_gpus_per_actor

    for attr_name in (
        "tp_size",
        "sp_size",
        "transport_dtype",
        "transport_drop_decoded_videos",
    ):
        value = getattr(args.rollout, attr_name)
        if value is not None:
            merged_engine_kwargs[attr_name] = value

    log_bytes = getattr(args.logging, "transport_log_payload_bytes", None)
    if log_bytes is not None:
        merged_engine_kwargs["transport_log_payload_bytes"] = log_bytes

    sglang_field_map = {
        "sglang_local_mode": "local_mode",
        "sglang_verify_weight_checksum": "verify_weight_checksum",
        "sglang_disable_autocast": "disable_autocast",
    }
    for attr_name, engine_key in sglang_field_map.items():
        value = getattr(args.rollout, attr_name)
        if value is not None:
            merged_engine_kwargs[engine_key] = value

    sglang_kwargs = args.rollout.sglang_kwargs
    if sglang_kwargs:
        if not isinstance(sglang_kwargs, dict):
            raise ValueError(
                "rollout.sglang_kwargs must be a dict after normalization."
            )
        merged_engine_kwargs["server_kwargs"] = dict(sglang_kwargs)

    merged_engine_kwargs.setdefault("use_lora", model_config.use_lora)
    merged_engine_kwargs.setdefault("lora_rank", model_config.lora_rank)
    merged_engine_kwargs.setdefault("lora_alpha", model_config.lora_alpha)
    merged_engine_kwargs.setdefault(
        "lora_target_modules", model_config.lora_target_modules
    )
    if model_config.vae_ckpt_path:
        merged_engine_kwargs.setdefault("vae_ckpt_path", model_config.vae_ckpt_path)
    if model_config.text_encoder_ckpt_path:
        merged_engine_kwargs.setdefault(
            "text_encoder_ckpt_path", model_config.text_encoder_ckpt_path
        )
    if model_config.use_lora:
        merged_engine_kwargs.setdefault("lora_merge_mode", "online")
    merged_engine_kwargs["prompt_encoder_dtype"] = (
        args.precision.rollout_autocast_precision
    )
    merged_engine_kwargs.setdefault("fps", int(args.sampling.fps))
    merged_engine_kwargs.setdefault("weight_sync_dir", args.sync.dir)
    if args.sync.target_modules is not None:
        merged_engine_kwargs.setdefault(
            "target_modules", list(args.sync.target_modules)
        )
    if rollout_engine == "sglang":
        merged_engine_kwargs["logprob_source"] = logprob_source
        if bool(args.ray.offload_rollout):
            merged_engine_kwargs["require_memory_api"] = True

    return EngineConfig(
        model_dotpath=model_init_payload.component_dotpath,
        pretrained_model_ckpt_path=model_config.pretrained_model_ckpt_path,
        sampler_dotpath=sampling_spec.sampler_dotpath,
        num_inference_steps=int(sampling_spec.num_inference_steps),
        eta=float(sampling_spec.sde_config.eta),
        sde_type=str(sampling_spec.sde_config.sde_type),
        shift=float(sampling_spec.sde_config.shift),
        guidance_scale=float(sampling_spec.guidance_scale),
        height=int(sampling_spec.height),
        width=int(sampling_spec.width),
        num_frames=int(sampling_spec.num_frames),
        engine_kwargs=merged_engine_kwargs,
    )


def build_rollout_engine_init_payload_from_args(
    args: Any,
    *,
    model_init_payload: ComponentInitPayload,
    sampling_spec: SamplingSpec,
    rollout_mode_info: Any,
) -> ComponentInitPayload:
    rollout_engine = str(rollout_mode_info.rollout_topology.rollout_engine or "")
    return build_component_init_payload_from_args(
        component_family=ROLLOUT_ENGINE_COMPONENT_FAMILY,
        identifier=rollout_engine,
        args=args,
        parser_kwargs={
            "model_init_payload": model_init_payload,
            "sampling_spec": sampling_spec,
            "rollout_mode_info": rollout_mode_info,
        },
    )


__all__ = [
    "build_rollout_engine_init_payload_from_args",
]
