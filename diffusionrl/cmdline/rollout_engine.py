"""Built-in rollout-engine cmdline adaptation helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from diffusionrl.cmdline.construction import build_component_init_payload_from_args
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.config.spec import RolloutInfo
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.samplers.registry import ROLLOUT_ENGINE_COMPONENT_FAMILY
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.sampling import SamplingParams


_ENDPOINT_POOL_KEYS = ("remote_scheduler_endpoints", "scheduler_endpoints", "sglang_scheduler_endpoints")
_ENDPOINT_SINGLE_KEYS = ("remote_scheduler_endpoint", "scheduler_endpoint", "sglang_scheduler_endpoint")
_SGLANG_NETWORK_KEYS = _ENDPOINT_POOL_KEYS + _ENDPOINT_SINGLE_KEYS + (
    "sglang_port_base", "sglang_port_stride",
)


def _resolve_sglang_network_config(
    engine_kwargs: Dict[str, Any],
) -> Tuple[Optional[str], Optional[int], int, int]:
    """Extract and resolve SGLang network settings from engine_kwargs.

    Consumes recognised keys from engine_kwargs (mutates in place) and returns
    ``(host, scheduler_port, port_base, port_stride)``.
    """
    from diffusionrl.utils.sglang_endpoint import (
        parse_scheduler_endpoint,
        parse_scheduler_endpoint_pool,
    )

    host: Optional[str] = None
    scheduler_port: Optional[int] = None

    # Resolve remote endpoint from pool or single key.
    for key in _ENDPOINT_POOL_KEYS:
        raw = engine_kwargs.pop(key, None)
        if raw is not None:
            pool = parse_scheduler_endpoint_pool(raw)
            if pool:
                # Store first endpoint; actor selects by rank at init time.
                # Full pool is not needed — the config layer picks host:port
                # from the first entry; multi-endpoint round-robin is not
                # supported in the typed config model.
                host, scheduler_port = pool[0]
            break

    if host is None:
        for key in _ENDPOINT_SINGLE_KEYS:
            raw = engine_kwargs.pop(key, None)
            if raw is not None:
                result = parse_scheduler_endpoint(raw)
                if result is not None:
                    host, scheduler_port = result
                break

    # Fallback: explicit host + scheduler_port already in kwargs
    if host is None and "host" in engine_kwargs:
        host = str(engine_kwargs.pop("host"))
    if scheduler_port is None and "scheduler_port" in engine_kwargs:
        scheduler_port = int(engine_kwargs.pop("scheduler_port"))

    # Port base / stride
    port_base = int(engine_kwargs.pop("sglang_port_base", os.getenv("DIFFUSIONRL_SGLANG_PORT_BASE", 33000)))
    port_stride = int(engine_kwargs.pop("sglang_port_stride", os.getenv("DIFFUSIONRL_SGLANG_PORT_STRIDE", 100)))

    return host, scheduler_port, port_base, port_stride


@register_cmdline_config_parser(EngineConfig)
def build_rollout_engine_config_from_args(
    args: Any,
    *,
    model_init_payload: ComponentInitPayload,
    sampling_spec: SamplingParams,
    rollout_info: RolloutInfo,
    sampler_dotpath: str = "",
) -> EngineConfig:
    """Build typed rollout-engine config from framework args and resolved slices."""
    model_config = model_init_payload.component_config
    if not isinstance(model_config, ModelBundleConfig):
        raise ValueError(
            "model_init_payload.component_config must be a ModelBundleConfig, "
            f"got: {type(model_config).__name__}"
        )

    rollout_engine = str(rollout_info.rollout_engine or "")
    logprob_source = str(rollout_info.logprob_source)

    # sglang_kwargs → engine_kwargs escape hatch for freestyle ServerArgs
    engine_kwargs: Dict[str, Any] = {}
    sglang_kwargs = args.rollout.sglang_kwargs
    if sglang_kwargs:
        if not isinstance(sglang_kwargs, dict):
            raise ValueError(
                "rollout.sglang_kwargs must be a dict after normalization."
            )
        engine_kwargs.update(sglang_kwargs)

    # Resolve SGLang endpoint / port config from engine_kwargs into typed fields.
    host, scheduler_port, sglang_port_base, sglang_port_stride = (
        _resolve_sglang_network_config(engine_kwargs)
    )

    # --- Typed fields ---
    num_gpus = int(args.rollout.num_gpus_per_actor) if args.rollout.num_gpus_per_actor is not None else 1
    tp_size = int(args.rollout.tp_size) if args.rollout.tp_size is not None else None
    sp_degree = int(args.rollout.sp_size) if args.rollout.sp_size is not None else None

    local_mode_raw = getattr(args.rollout, "sglang_local_mode", None)
    local_mode = bool(local_mode_raw) if local_mode_raw is not None else True

    verify_raw = getattr(args.rollout, "sglang_verify_weight_checksum", None)
    verify_weight_checksum = bool(verify_raw) if verify_raw is not None else True

    disable_autocast_raw = getattr(args.rollout, "sglang_disable_autocast", None)
    disable_autocast = bool(disable_autocast_raw) if disable_autocast_raw is not None else False

    require_memory_api = bool(args.ray.offload_rollout) if rollout_engine == "sglang" else False

    use_lora = bool(model_config.use_lora)
    lora_merge_mode = "online" if use_lora else None

    lora_target_modules_raw = model_config.lora_target_modules
    lora_target_modules = (
        tuple(lora_target_modules_raw)
        if lora_target_modules_raw
        else None
    )

    target_modules_raw = getattr(args.sync, "target_modules", None)
    target_modules = (
        tuple(target_modules_raw)
        if target_modules_raw is not None
        else None
    )

    return EngineConfig(
        model_dotpath=model_init_payload.component_dotpath,
        pretrained_model_ckpt_path=model_config.pretrained_model_ckpt_path,
        sampler_dotpath=sampler_dotpath,
        num_inference_steps=int(sampling_spec.num_inference_steps),
        eta=float(sampling_spec.sde_config.eta),
        sde_type=str(sampling_spec.sde_config.sde_type),
        shift=float(sampling_spec.sde_config.shift),
        guidance_scale=float(sampling_spec.guidance_scale),
        height=int(sampling_spec.height),
        width=int(sampling_spec.width),
        num_frames=int(sampling_spec.num_frames),
        fps=int(args.sampling.fps),
        # Parallelism & GPU
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        # SGLang engine behaviour
        local_mode=local_mode,
        logprob_source=logprob_source if rollout_engine == "sglang" else "replay",
        verify_weight_checksum=verify_weight_checksum,
        require_memory_api=require_memory_api,
        disable_autocast=disable_autocast,
        # Weight sync
        target_modules=target_modules,
        weight_sync_dir=getattr(args.sync, "dir", None),
        # LoRA
        use_lora=use_lora,
        lora_rank=int(model_config.lora_rank),
        lora_alpha=int(model_config.lora_alpha),
        lora_target_modules=lora_target_modules,
        lora_merge_mode=lora_merge_mode,
        # Auxiliary models
        vae_ckpt_path=model_config.vae_ckpt_path or None,
        text_encoder_ckpt_path=model_config.text_encoder_ckpt_path or None,
        prompt_encoder_dtype=args.precision.rollout_autocast_precision,
        # SGLang network (resolved from sglang_kwargs)
        host=host,
        scheduler_port=scheduler_port,
        sglang_port_base=sglang_port_base,
        sglang_port_stride=sglang_port_stride,
        # Escape hatch (remaining keys after resolution)
        engine_kwargs=engine_kwargs,
    )


def build_rollout_engine_init_payload_from_args(
    args: Any,
    *,
    model_init_payload: ComponentInitPayload,
    sampling_spec: SamplingParams,
    rollout_info: RolloutInfo,
    sampler_dotpath: str = "",
) -> ComponentInitPayload:
    rollout_engine = str(rollout_info.rollout_engine or "")
    return build_component_init_payload_from_args(
        component_family=ROLLOUT_ENGINE_COMPONENT_FAMILY,
        identifier=rollout_engine,
        args=args,
        parser_kwargs={
            "model_init_payload": model_init_payload,
            "sampling_spec": sampling_spec,
            "rollout_info": rollout_info,
            "sampler_dotpath": sampler_dotpath,
        },
    )


__all__ = [
    "build_rollout_engine_init_payload_from_args",
]
