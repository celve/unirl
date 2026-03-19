"""Canonical rollout topology and rollout-engine helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from diffusionrl.types.engine import normalize_engine_type

DIRECT_ROLLOUT_MODE = "direct_rollout"
SEPARATE_ROLLOUT_MODE = "separate_rollout"
COLOCATE_ROLLOUT_MODE = "colocate_rollout"

ROLLOUT_MODES = {
    DIRECT_ROLLOUT_MODE,
    SEPARATE_ROLLOUT_MODE,
    COLOCATE_ROLLOUT_MODE,
}
ROLLOUT_ENGINE_TYPES = {
    "fsdp",
    "sglang",
}


def normalize_rollout_mode(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_rollout_service_engine(value: Any) -> Optional[str]:
    normalized = normalize_engine_type(value)
    if normalized and normalized not in ROLLOUT_ENGINE_TYPES:
        raise ValueError(
            "rollout.service_engine must be one of "
            f"{sorted(ROLLOUT_ENGINE_TYPES)}, got: {value!r}"
        )
    return normalized or None


def rollout_mode_uses_service(mode: Any) -> bool:
    return normalize_rollout_mode(mode) in {
        SEPARATE_ROLLOUT_MODE,
        COLOCATE_ROLLOUT_MODE,
    }


def rollout_mode_is_colocated(mode: Any) -> bool:
    return normalize_rollout_mode(mode) == COLOCATE_ROLLOUT_MODE


def runtime_mode_label_for_rollout_mode(mode: Any) -> str:
    normalized = normalize_rollout_mode(mode)
    if normalized == DIRECT_ROLLOUT_MODE:
        return "direct_sampling"
    if normalized == COLOCATE_ROLLOUT_MODE:
        return "colocate"
    return "separate"


def resolve_rollout_service_kwargs(args: Any) -> Dict[str, Any]:
    """Resolve dedicated rollout-service kwargs from canonical rollout fields."""
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
    rollout_cfg = getattr(args, "rollout", None)
    for attr_name, engine_key in field_map.items():
        value = getattr(rollout_cfg, attr_name, None)
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
        value = getattr(rollout_cfg, attr_name, None)
        if value is not None:
            resolved[engine_key] = value

    sglang_kwargs = getattr(rollout_cfg, "sglang_kwargs", None)
    if sglang_kwargs:
        if not isinstance(sglang_kwargs, dict):
            raise ValueError(
                "rollout.sglang_kwargs must be a dict after normalization."
            )
        resolved["server_kwargs"] = dict(sglang_kwargs)

    return resolved


def resolve_rollout_service_num_gpus(args: Any) -> int:
    """Resolve dedicated rollout-service GPU ownership without implicit fallback."""
    if not rollout_mode_uses_service(getattr(args.rollout, "mode", None)):
        return 0

    raw_num_gpus = getattr(args.rollout, "service_num_gpus", None)
    if raw_num_gpus is None:
        raise ValueError(
            "Dedicated rollout services require rollout.service_num_gpus to be set explicitly. "
            "Do not infer actor GPU ownership from tp/sp parallel hints."
        )
    try:
        resolved = int(raw_num_gpus)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"rollout.service_num_gpus must be an integer >= 1, got: {raw_num_gpus!r}"
        ) from exc
    if resolved < 1:
        raise ValueError(
            f"rollout.service_num_gpus must be >= 1 for dedicated rollout services, got: {resolved}"
        )
    return resolved
