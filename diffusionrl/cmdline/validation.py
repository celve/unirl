"""Validation helpers that operate on cmdline args and arg-derived state."""

from __future__ import annotations

import logging
from dataclasses import MISSING, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Optional

from diffusionrl.buffer.buffer_plugins import normalize_plugin_dotpaths
from diffusionrl.cmdline.argument_parsing import resolve_dataclass_field_default
from diffusionrl.cmdline.resolution import (
    derive_algorithm_dotpath,
    derive_model_spec,
)
from diffusionrl.config.resolution import DIRECT_ROLLOUT_MODE, rollout_mode_is_colocated
from diffusionrl.config.spec import RolloutInfo
from diffusionrl.config.validation import (
    validate_colocate_fractions,
    validate_direct_sampling_batch_geometry,
    validate_dotpath,
    validate_training_actor_sampling_mode,
    validate_weight_sync,
)
from diffusionrl.sde.rules import is_deterministic_sde_type
from diffusionrl.types.engine import uses_dedicated_rollout_engine

logger = logging.getLogger(__name__)


# ============================================================================
# Grouped cmdline config validation
# ============================================================================


def _validate_metadata_choices(
    instance: Any,
    *,
    config_name: str = "",
) -> None:
    """Validate fields against ``metadata["choices"]`` and normalize to canonical form."""
    for field_info in dataclass_fields(type(instance)):
        choices = (field_info.metadata or {}).get("choices")
        if not choices:
            child = getattr(instance, field_info.name)
            if is_dataclass(child):
                sub_name = f"{config_name}.{field_info.name}" if config_name else field_info.name
                _validate_metadata_choices(child, config_name=sub_name)
            continue

        value = getattr(instance, field_info.name)
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if not normalized:
            continue
        canonical_map = {str(c).strip().lower(): c for c in choices}
        if normalized not in canonical_map:
            prefix = f"{config_name}." if config_name else ""
            raise ValueError(f"{prefix}{field_info.name} must be one of {sorted(choices)}, got: {value!r}")
        canonical = canonical_map[normalized]
        if value != canonical:
            setattr(instance, field_info.name, canonical)


def validate_grouped_configs(args: Any) -> None:
    """Run per-group config dataclass validators."""
    normalized_plugins = normalize_plugin_dotpaths(args.rollout.plugin_dotpaths)
    if not (isinstance(args.rollout.plugin_dotpaths, list) and args.rollout.plugin_dotpaths == normalized_plugins):
        args.rollout.plugin_dotpaths = normalized_plugins
    for field_info in dataclass_fields(type(args)):
        child = getattr(args, field_info.name)
        if is_dataclass(child):
            _validate_metadata_choices(child, config_name=field_info.name)
    args.model.validate()
    args.sampling.validate()
    args.reward.validate()
    args.ray.validate()
    args.sync.validate()
    args.algorithm.validate()
    args.training.validate()
    args.precision.validate()
    args.rollout.validate()
    args.evaluation.validate()
    args.logging.validate()
    args.debug.validate()


# ============================================================================
# Extension path validation
# ============================================================================


def validate_dynamic_dotpaths(
    args: Any,
    *,
    resolved_model: Optional[Any] = None,
    algorithm_dotpath: Optional[str] = None,
    include_data_source: bool = True,
    include_rollout_buffer_plugins: bool = True,
) -> None:
    """Validate configured dynamic extension dotpaths."""
    if resolved_model is None:
        resolved_model = derive_model_spec(args)
    if algorithm_dotpath is None:
        algorithm_dotpath = derive_algorithm_dotpath(args)
    validate_dotpath(resolved_model.sampler_dotpath, label="sampler")
    validate_dotpath(algorithm_dotpath, label="algorithm")
    if include_data_source:
        validate_dotpath(args.data_source_dotpath, label="data_source")
    if args.sampling.replay_sampler_dotpath:
        validate_dotpath(args.sampling.replay_sampler_dotpath, label="replay_sampler")
    if include_rollout_buffer_plugins:
        for plugin_dotpath in normalize_plugin_dotpaths(args.rollout.plugin_dotpaths):
            validate_dotpath(plugin_dotpath, label="rollout_buffer_plugin")


def validate_nft_sampling_contract(args: Any) -> None:
    """Validate NFT-specific rollout sampling contract."""
    if args.algorithm.algorithm_type == "nft":
        old_adapter_name = "old"
        if args.algorithm.algorithm_kwargs is None:
            parsed: Dict[str, Any] = {}
        elif not isinstance(args.algorithm.algorithm_kwargs, dict):
            raise ValueError(
                "algorithm.algorithm_kwargs must already be a dict after config parsing. "
                f"Got: {type(args.algorithm.algorithm_kwargs).__name__}."
            )
        else:
            parsed = dict(args.algorithm.algorithm_kwargs)
        if parsed:
            old_adapter_name = str(parsed.get("old_adapter_name", old_adapter_name) or old_adapter_name)

        if not args.sampling.sampling_adapter:
            raise ValueError(
                "algorithm_type='nft' requires --sampling.sampling-adapter to be set "
                f"(must match old_adapter_name={old_adapter_name!r})."
            )
        if str(args.sampling.sampling_adapter) != old_adapter_name:
            raise ValueError(
                "algorithm_type='nft' requires rollout sampling from the old adapter. "
                f"Set --sampling.sampling-adapter {old_adapter_name!r}, "
                f"got {args.sampling.sampling_adapter!r}."
            )
        sde_type = str(args.sampling.sde_type)
        eta = float(args.sampling.eta)
        if not is_deterministic_sde_type(sde_type, eta):
            raise ValueError(
                "algorithm_type='nft' targets DiffusionNFT deterministic sampling. "
                "Set --sampling.sde-type dpm2, or use another transition rule "
                f"with --sampling.eta 0.0 (ODE mode). "
                f"Got sde_type={sde_type!r}, eta={eta}."
            )


def validate_rollout_mode_constraints(
    *,
    rollout_info: RolloutInfo,
    model_cls: Any,
) -> None:
    """Validate rollout-mode constraints that depend on resolved rollout/model info."""
    model_label = f"{model_cls.__module__}.{model_cls.__qualname__}"
    if not rollout_info.training_actor_sampling_mode and rollout_info.rollout_engine == "sglang":
        supports_sglang = getattr(model_cls, "supports_sglang_prompt_mode", None)
        if not callable(supports_sglang):
            raise ValueError(
                f"rollout.rollout_engine='sglang' requires model {model_label!r} "
                "to define classmethod supports_sglang_prompt_mode()."
            )
        if not supports_sglang():
            raise ValueError(
                f"rollout.rollout_engine='sglang' is not supported by model {model_label!r}. "
                "The model must implement classmethod supports_sglang_prompt_mode() returning True."
            )


# ============================================================================
# Cmdline rollout mode checks
# ============================================================================


_ROLLOUT_TOPOLOGY_FIELD_NAMES = frozenset(
    {
        "mode",
        "rollout_engine",
        "rollout_batch_size",
        "num_gpus_per_actor",
        "tp_size",
        "sp_size",
        "sglang_local_mode",
        "sglang_verify_weight_checksum",
        "sglang_disable_autocast",
        "sglang_kwargs",
    }
)


def _collect_direct_sampling_incompatible_fields(rollout_config: Any) -> list[str]:
    incompatible: list[str] = []
    for field_info in dataclass_fields(type(rollout_config)):
        if field_info.name not in _ROLLOUT_TOPOLOGY_FIELD_NAMES:
            continue
        if field_info.name in {"mode", "rollout_engine"}:
            continue
        field_value = getattr(rollout_config, field_info.name)
        field_default = resolve_dataclass_field_default(field_info, missing=MISSING)
        if field_default is not MISSING and field_value == field_default:
            continue
        if field_value in (None, "", False):
            continue
        if isinstance(field_value, dict) and not field_value:
            continue
        incompatible.append(f"rollout.{field_info.name}")
    return incompatible


def _format_rollout_mode_state(
    args: Any,
    *,
    rollout_info: RolloutInfo,
) -> str:
    return "\n".join(
        [
            "Resolved rollout mode:",
            f"  rollout.mode = {rollout_info.mode!r}",
            f"  rollout.rollout_engine = {rollout_info.rollout_engine!r}",
            f"  training_actor_sampling_mode = {rollout_info.training_actor_sampling_mode}",
            f"  is_sglang_engine = {rollout_info.is_sglang_engine}",
            f"  sampling.logprob_source = {rollout_info.logprob_source!r}",
            f"  derived.replay_enabled = {rollout_info.replay_enabled}",
            f"  sampling.max_samples_per_request = {args.sampling.max_samples_per_request!r}",
            f"  sync.protocol = {args.sync.protocol!r}",
            "  offload flags = "
            f"(ray.offload_train={bool(args.ray.offload_train)}, "
            f"ray.offload_rollout={bool(args.ray.offload_rollout)})",
        ]
    )


def validate_rollout_topology_contract(
    args: Any,
    *,
    rollout_info: RolloutInfo,
) -> RolloutInfo:
    """Validate rollout topology contract after strict topology resolution."""
    topology = rollout_info
    rollout_config = args.rollout

    if topology.mode != DIRECT_ROLLOUT_MODE:
        if topology.rollout_engine is None:
            raise ValueError("Dedicated rollout modes require rollout.rollout_engine to be set explicitly.")
        if not uses_dedicated_rollout_engine(topology.rollout_engine):
            raise ValueError(
                "rollout.mode in {separate,colocate} requires a dedicated rollout "
                f"engine. Got rollout.rollout_engine={topology.rollout_engine!r}."
            )
        if rollout_config.num_gpus_per_actor is None:
            raise ValueError("Dedicated rollout modes require rollout.num_gpus_per_actor to be set explicitly.")
        return topology

    configured_direct_incompatible_fields = _collect_direct_sampling_incompatible_fields(rollout_config)
    if configured_direct_incompatible_fields:
        raise ValueError(
            "direct_sampling runs sampling on training actors, so dedicated rollout-service fields "
            f"must be unset. Remove: {', '.join(sorted(configured_direct_incompatible_fields))}."
        )

    return topology


def validate_algorithm_kwargs_payload(args: Any) -> None:
    """Validate algorithm_kwargs payload without mutating args."""
    if args.algorithm.algorithm_kwargs is None:
        parsed = {}
    elif not isinstance(args.algorithm.algorithm_kwargs, dict):
        raise ValueError(
            "algorithm.algorithm_kwargs must already be a dict after config parsing. "
            f"Got: {type(args.algorithm.algorithm_kwargs).__name__}."
        )
    else:
        parsed = dict(args.algorithm.algorithm_kwargs)
    reserved_paths = {
        "samples_per_prompt": "algorithm.samples_per_prompt",
        "prompts_per_rollout": "algorithm.prompts_per_rollout",
        "component_mix_stage": "algorithm.component_mix_stage",
        "adv_normalization_scope": "algorithm.adv_normalization_scope",
        "adv_norm_eps": "algorithm.adv_norm_eps",
        "clip_max": "algorithm.clip_max",
        "use_global_std": "algorithm.use_global_std",
        "trim_outliers_ratio": "algorithm.trim_outliers_ratio",
        "eval_ema_decay": "algorithm.eval_ema_decay",
        "eval_ema_update_interval": "algorithm.eval_ema_update_interval",
        "shuffle_samples": "algorithm.shuffle_samples",
        "shuffle_seed": "algorithm.shuffle_seed",
        "autocast_precision": "precision.training.autocast_precision",
    }
    collisions = [
        f"algorithm.algorithm_kwargs.{key} (use {reserved_paths[key]} instead)"
        for key in sorted(reserved_paths)
        if key in parsed
    ]
    if collisions:
        raise ValueError(
            "algorithm.algorithm_kwargs contains framework-owned keys that must be configured "
            "through the public config surface. Remove: "
            f"{', '.join(collisions)}."
        )


def validate_rollout_mode(
    args: Any,
    *,
    rollout_info: RolloutInfo,
    backend_capabilities: Dict[str, Any],
    backend_name: str,
) -> None:
    """Validate rollout mode against a pre-resolved rollout info."""
    check_name = "rollout topology"
    try:
        validate_rollout_topology_contract(
            args,
            rollout_info=rollout_info,
        )

        check_name = "training-actor sampling compatibility"
        validate_training_actor_sampling_mode(
            rollout_info=rollout_info,
            backend_capabilities=backend_capabilities,
            backend_name=backend_name,
        )

        check_name = "offload/colocate settings"
        validate_offload_and_colocate_config(
            args,
            rollout_info=rollout_info,
        )

        check_name = "direct-sampling request sizing"
        validate_direct_sampling_batch_geometry(
            rollout_info=rollout_info,
            max_samples_per_request=args.sampling.max_samples_per_request,
        )

        check_name = "weight-sync settings"
        validate_weight_sync(
            rollout_info=rollout_info,
            sync_protocol=args.sync.protocol,
            sync_dir=args.sync.dir,
            rollout_num_nodes=args.ray.rollout_num_nodes,
            training_num_nodes=args.ray.training_num_nodes,
        )
    except ValueError as exc:
        raise ValueError(
            "\n".join(
                [
                    f"Invalid rollout mode while validating {check_name}.",
                    _format_rollout_mode_state(
                        args,
                        rollout_info=rollout_info,
                    ),
                    f"Cause: {exc}",
                ]
            )
        ) from exc


def validate_offload_and_colocate_config(
    args: Any,
    *,
    rollout_info: RolloutInfo,
) -> None:
    """Validate offload switches and colocate fractions."""
    if rollout_info.training_actor_sampling_mode:
        if bool(args.ray.offload_train) or bool(args.ray.offload_rollout):
            raise ValueError(
                "direct_sampling uses training actors for sampling and cannot be combined with "
                "ray.offload_train / ray.offload_rollout."
            )
        return

    if rollout_mode_is_colocated(rollout_info.mode):
        validate_colocate_fractions(
            colocate_training_gpu_fraction=float(args.ray.colocate_training_gpu_fraction),
            colocate_rollout_gpu_fraction=float(args.ray.colocate_rollout_gpu_fraction),
        )


__all__ = [
    "validate_algorithm_kwargs_payload",
    "validate_dynamic_dotpaths",
    "validate_grouped_configs",
    "validate_nft_sampling_contract",
    "validate_offload_and_colocate_config",
    "validate_rollout_mode",
    "validate_rollout_mode_constraints",
    "validate_rollout_topology_contract",
]
