"""Built-in algorithm cmdline adaptation helpers."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Dict

from diffusionrl.algorithms.grpo import GRPOAlgorithmConfig
from diffusionrl.algorithms.nft import NFTAlgorithmConfig
from diffusionrl.algorithms.registry import ALGORITHM_COMPONENT_FAMILY
from diffusionrl.cmdline.construction import build_component_init_payload_from_args
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.types.sampling import SamplingParams


def build_algorithm_init_payload_from_args(
    args: Any,
    *,
    sampling_spec: SamplingParams | None = None,
) -> ComponentInitPayload:
    # Lazy import to break cmdline.algorithms <-> cmdline.resolution cycle:
    # resolution.py imports actors.py, which top-imports this module.
    from diffusionrl.cmdline.resolution import derive_algorithm_identifier

    resolved_sampling_spec = _require_sampling_spec(args, sampling_spec=sampling_spec)
    return build_component_init_payload_from_args(
        component_family=ALGORITHM_COMPONENT_FAMILY,
        identifier=derive_algorithm_identifier(args),
        args=args,
        parser_kwargs={"sampling_spec": resolved_sampling_spec},
    )


def validate_algorithm_kwargs(
    *,
    config_class: type,
    algorithm_kwargs: Dict[str, Any],
) -> None:
    """Helper function to validate the field names of algorithm kwargs."""
    framework_owned_runtime_fields = {
        "component_mix_stage",
        "adv_normalization_scope",
        "samples_per_prompt",
        "num_inference_steps",
        "eval_ema_decay",
        "eval_ema_update_interval",
        "epsilon",
        "clip_max",
        "use_global_std",
        "trim_outliers_ratio",
        "training_share_rollout_indices",
        "rollout_scheduler_config",
        "training_scheduler_config",
        "sde_config",
    }
    config_field_names = {field.name for field in dataclass_fields(config_class)}
    allowed_extension_fields = config_field_names - framework_owned_runtime_fields
    unknown = sorted(
        key for key in algorithm_kwargs.keys() if key not in allowed_extension_fields
    )
    if unknown:
        raise ValueError(
            "algorithm.algorithm_kwargs contains unsupported keys for "
            f"{config_class.__name__}: {unknown}. "
            f"Allowed algorithm-specific keys: {sorted(allowed_extension_fields)}."
        )


## -------- Component-specific cmdline parser for config --------
@register_cmdline_config_parser(GRPOAlgorithmConfig)
def build_grpo_algorithm_config_from_args(
    args: Any,
    *,
    sampling_spec: SamplingParams | None = None,
) -> GRPOAlgorithmConfig:
    sampling_spec = _require_sampling_spec(args, sampling_spec=sampling_spec)
    extra = dict(getattr(args.algorithm, "algorithm_kwargs", {}) or {})
    validate_algorithm_kwargs(
        config_class=GRPOAlgorithmConfig,
        algorithm_kwargs=extra,
    )
    extra = _coerce_dict_field_type(
        GRPOAlgorithmConfig,
        extra,
    )
    return GRPOAlgorithmConfig(
        **_build_base_algorithm_kwargs(args, sampling_spec=sampling_spec),
        eta=float(sampling_spec.sde_config.eta),
        sde_type=str(sampling_spec.sde_config.sde_type),
        shift=float(sampling_spec.sde_config.shift),
        training_share_rollout_indices=bool(
            args.algorithm.training_share_rollout_indices
        ),
        rollout_scheduler_config=_scheduler_payload(args.algorithm.rollout_scheduler),
        training_scheduler_config=_scheduler_payload(args.algorithm.training_scheduler),
        **extra,
    )


@register_cmdline_config_parser(NFTAlgorithmConfig)
def build_nft_algorithm_config_from_args(
    args: Any,
    *,
    sampling_spec: SamplingParams | None = None,
) -> NFTAlgorithmConfig:
    sampling_spec = _require_sampling_spec(args, sampling_spec=sampling_spec)
    extra = dict(getattr(args.algorithm, "algorithm_kwargs", {}) or {})
    validate_algorithm_kwargs(
        config_class=NFTAlgorithmConfig,
        algorithm_kwargs=extra,
    )
    extra = _coerce_dict_field_type(
        NFTAlgorithmConfig,
        extra,
    )
    return NFTAlgorithmConfig(
        **_build_base_algorithm_kwargs(args, sampling_spec=sampling_spec),
        eta=float(sampling_spec.sde_config.eta),
        sde_type=str(sampling_spec.sde_config.sde_type),
        shift=float(sampling_spec.sde_config.shift),
        training_scheduler_config=_scheduler_payload(args.algorithm.training_scheduler),
        **extra,
    )


## -------- Helper functions --------
def _scheduler_payload(raw_cfg: Any) -> Dict[str, Any]:
    if is_dataclass(raw_cfg):
        return asdict(raw_cfg)
    return dict(raw_cfg or {})


def _require_sampling_spec(
    args: Any, *, sampling_spec: SamplingParams | None
) -> SamplingParams:
    if isinstance(sampling_spec, SamplingParams):
        return sampling_spec
    cached_resolved = getattr(args, "_diffusionrl_resolved_config", None)
    cached_sampling_spec = getattr(cached_resolved, "sampling_spec", None)
    if isinstance(cached_sampling_spec, SamplingParams):
        return cached_sampling_spec
    raise ValueError(
        "Algorithm cmdline config parsing requires SamplingParams. "
        "Pass sampling_spec explicitly or cache resolved config on args."
    )


def _build_base_algorithm_kwargs(
    args: Any, *, sampling_spec: SamplingParams
) -> Dict[str, Any]:
    """Helper function to build the base algorithm kwargs."""
    ac = args.algorithm
    return {
        "component_mix_stage": str(ac.component_mix_stage),
        "adv_normalization_scope": str(ac.adv_normalization_scope),
        "samples_per_prompt": int(ac.samples_per_prompt),
        "num_inference_steps": int(sampling_spec.num_inference_steps),
        "eval_ema_decay": float(ac.eval_ema_decay),
        "eval_ema_update_interval": int(ac.eval_ema_update_interval),
        "epsilon": float(ac.adv_norm_eps),
        "clip_max": ac.clip_max,
        "use_global_std": bool(ac.use_global_std),
        "trim_outliers_ratio": float(ac.trim_outliers_ratio),
    }


def _coerce_dict_field_type(
    config_class: type,
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper function to convert the type of algorithm kwargs."""
    # TODO: If algorithm_kwargs later needs list-typed fields, handle JSON-string
    # parsing here in the cmdline layer instead of growing a generic
    # complex-type coercion system.
    import typing

    simple_types = {int, float, bool, str}
    hints = typing.get_type_hints(config_class)
    field_name_to_type = {
        name: hint for name, hint in hints.items()
        if hint in simple_types
    }
    coerced: Dict[str, Any] = {}
    for key, value in raw.items():
        field_type = field_name_to_type.get(key)
        if field_type is None:
            raise ValueError(
                f"Dictionary contains unsupported key for {config_class.__name__}: {key}. "
                f"Allowed algorithm-specific keys: {sorted(field_name_to_type.keys())}"
            )
        coerced[key] = field_type(value)
    return coerced


__all__ = [
    # General build_ xxx _from_args call
    "build_algorithm_init_payload_from_args",
    # Kwargs validation helper
    "validate_algorithm_kwargs",
]
