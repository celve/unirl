"""Configuration validation: shared primitives, model, rollout, training, reward, and payload checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields as dataclass_fields
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from diffusionrl.algorithms.construction import build_algorithm_kwargs, resolve_algorithm_path
from diffusionrl.config.resolution import (
    ModelSpec,
    RolloutModeInfo,
    RolloutTopology,
    TrainTopology,
    derive_global_rollout_batch_size,
    derive_model_spec,
    derive_rollout_actor_gpu_count,
    derive_rollout_gpu_pool_size,
    derive_rollout_topology,
    derive_num_updates_per_local_batch,
    derive_training_topology,
    normalize_lora_target_modules,
    require_prompts_per_rollout,
    rollout_mode_is_colocated,
    rollout_mode_uses_service,
)
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.sde.rules import (
    SUPPORTED_USER_SDE_TYPES,
    is_deterministic_sde_type,
    supported_sde_type_text,
)
from diffusionrl.training.backends.factory import supported_train_backends
from diffusionrl.training.update_schedule import coerce_training_execution_plan
from diffusionrl.types.engine import uses_dedicated_rollout_engine
from diffusionrl.types.sampling import SamplingRequirements
from diffusionrl.utils.misc import load_function

from .argument_parsing import resolve_dataclass_field_default

logger = logging.getLogger(__name__)


# ============================================================================
# Common primitives
# ============================================================================

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"


class PrecisionName(str, Enum):
    BF16 = "bf16"
    BFLOAT16 = "bfloat16"
    FP16 = "fp16"
    FLOAT16 = "float16"
    HALF = "half"
    FP32 = "fp32"
    FLOAT32 = "float32"
    FLOAT = "float"


def repo_root(*, env_repo_root: str) -> str:
    """Resolve repository root from environment override or package-relative path."""
    env_root = os.getenv(env_repo_root)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def is_probably_local_weight_sync_dir(path: str, *, root: str) -> bool:
    """Best-effort guard for local-only paths in multi-node checkpoint sync."""
    if not path:
        return True
    real = os.path.realpath(path)
    for prefix in ("/tmp", "/var/tmp", "/dev/shm"):
        if real == prefix or real.startswith(prefix + os.sep):
            return True
    if real == root or real.startswith(root + os.sep):
        return True
    return False


def validate_dotpath(path: str, *, label: str) -> None:
    """Fail fast when a configured dotpath is not importable."""
    try:
        load_function(path)
    except Exception as exc:
        raise ValueError(
            f"Invalid {label} path: {path!r}. Import failed: {exc}. "
            f"Check that the module is installed and the dotpath is correct "
            f"(e.g. 'diffusionrl.algorithms.grpo.GRPOAlgorithm')."
        ) from exc


def validate_precision_name(value: Any, *, field_name: str) -> None:
    """Validate precision aliases used by config-facing precision fields."""
    key = str(value or "").strip().lower()
    try:
        PrecisionName(key)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of bf16/fp16/fp32, got: {value!r}"
        ) from exc


def validate_grouped_configs(args: Any) -> None:
    """Run per-group config dataclass validators."""
    args.model.validate()
    args.sampling.validate()
    args.reward.validate()
    args.ray.validate()
    args.sync.validate()
    args.algorithm.validate()
    args.training.validate()
    args.precision.validate()
    args.rollout.validate()
    args.debug.validate()


def validate_dynamic_dotpaths(
    args: Any,
    *,
    resolved_model: Optional[ModelSpec] = None,
    include_data_source: bool = True,
    include_rollout_buffer_plugins: bool = True,
) -> None:
    """Validate configured dynamic extension dotpaths."""
    if resolved_model is None:
        resolved_model = derive_model_spec(args)
    validate_dotpath(resolved_model.model_path, label="model")
    validate_dotpath(resolved_model.sampler_path, label="sampler")
    validate_dotpath(
        resolve_algorithm_path(
            algorithm_type=args.algorithm.algorithm_type,
            algorithm_path=args.algorithm.algorithm_path,
        ),
        label="algorithm",
    )
    if include_data_source:
        validate_dotpath(args.data_source_path, label="data_source")
    if getattr(args, "rollout_function_path", None):
        validate_dotpath(args.rollout_function_path, label="rollout_function")
    if getattr(args, "eval_function_path", None):
        validate_dotpath(args.eval_function_path, label="eval_function")
    if getattr(args, "reward_hook_path", None):
        validate_dotpath(args.reward_hook_path, label="reward_hook")
    if args.training.train_backend_path:
        validate_dotpath(args.training.train_backend_path, label="train_backend")
    if args.sampling.replay_sampler_path:
        validate_dotpath(args.sampling.replay_sampler_path, label="replay_sampler")
    if include_rollout_buffer_plugins:
        rollout_buffer_plugin_paths = args.rollout.buffer.plugin_paths or ""
        if isinstance(rollout_buffer_plugin_paths, str):
            for plugin_path in [part.strip() for part in rollout_buffer_plugin_paths.split(",") if part.strip()]:
                validate_dotpath(plugin_path, label="rollout_buffer_plugin")


def validate_colocate_fractions(args: Any) -> None:
    """Validate colocate GPU fraction bounds."""
    if args.ray.colocate_training_gpu_fraction <= 0 or args.ray.colocate_rollout_gpu_fraction <= 0:
        raise ValueError(
            "colocate_training_gpu_fraction and colocate_rollout_gpu_fraction must be > 0"
        )
    if args.ray.colocate_training_gpu_fraction + args.ray.colocate_rollout_gpu_fraction > 1.0:
        raise ValueError(
            "colocate_training_gpu_fraction + colocate_rollout_gpu_fraction must be <= 1.0"
        )


# ============================================================================
# Model, algorithm, and engine capability validation
# ============================================================================


def validate_model_sampling_contract(args: Any) -> None:
    """Validate model/sampling combinations that should never reach model hooks."""
    raw_sde_type = str(args.sampling.sde_type or "").strip().lower()
    if raw_sde_type not in SUPPORTED_USER_SDE_TYPES:
        raise ValueError(
            f"Unknown sampling.sde_type={args.sampling.sde_type!r}. "
            f"Supported values: {supported_sde_type_text()}."
        )


def apply_model_config_hook(args: Any, *, model_cls: Any) -> None:
    """Run model-provided config hook without allowing it to mutate args."""
    model_validate_fn = getattr(model_cls, "validate_config", None)
    if callable(model_validate_fn):
        before_dotted = args.to_dotted_dict() if hasattr(args, "to_dotted_dict") else None
        model_validate_fn(args)
        if isinstance(before_dotted, dict) and hasattr(args, "to_dotted_dict"):
            after_dotted = args.to_dotted_dict()
            if isinstance(after_dotted, dict):
                changed = []
                for key in sorted(set(before_dotted) | set(after_dotted)):
                    before = before_dotted.get(key)
                    after = after_dotted.get(key)
                    if before != after:
                        changed.append(
                            f"{key}: {before!r} -> {after!r}"
                        )
                if changed:
                    raise ValueError(
                        "model.validate_config() must not mutate TrainingArguments. "
                        f"Observed changes: {', '.join(changed[:5])}"
                    )


def validate_nft_sampling_contract(args: Any) -> None:
    """Validate NFT-specific rollout sampling contract."""
    if args.algorithm.algorithm_type == "nft":
        old_adapter_name = "old"
        parsed: Dict[str, Any] = build_algorithm_kwargs(args)
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


def validate_resolved_engine_algorithm_contract(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
    sampling_requirements: SamplingRequirements,
) -> None:
    """Validate engine/algorithm compatibility using pre-resolved capabilities."""
    if rollout_mode_info.training_actor_sampling_mode:
        return

    rollout_topology = rollout_mode_info.rollout_topology
    service_engine = rollout_topology.service_engine
    if not service_engine:
        raise ValueError(
            "Dedicated rollout validation requires rollout.topology.service_engine to be set explicitly. "
            "Run validate_args() before resolving dedicated rollout engine capabilities."
        )

    engine_caps = rollout_mode_info.effective_engine_capabilities
    if engine_caps is None:
        raise ValueError(
            "Dedicated rollout validation requires resolved engine capabilities. "
            "Run resolve_config() before validate_args()."
        )

    required = sampling_requirements

    required_dict = required.to_dict()
    missing = [
        key
        for key, needed in required_dict.items()
        if bool(needed) and not bool(engine_caps.get(key, False))
    ]
    if missing:
        raise ValueError(
            f"Engine capability mismatch for algorithm_type={args.algorithm.algorithm_type}: "
            f"rollout.topology.service_engine={service_engine} lacks {missing}. "
            f"engine_capabilities={engine_caps}, required={required_dict}. "
            "Use a compatible dedicated rollout engine, or fall back to direct "
            "training-actor sampling for trajectory/log-prob-heavy algorithms."
        )


def validate_rollout_mode_constraints(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
    model_cls: Any,
    rollout_topology: RolloutTopology,
) -> None:
    """Validate rollout-mode constraints and mutually-exclusive switches."""
    model_label = f"{model_cls.__module__}.{model_cls.__qualname__}"
    if (
        not training_actor_sampling_mode
        and rollout_topology.service_engine == "sglang"
    ):
        supports_sglang = getattr(model_cls, "supports_sglang_prompt_mode", None)
        if not callable(supports_sglang):
            raise ValueError(
                f"rollout.topology.service_engine='sglang' requires model {model_label!r} "
                "to define classmethod supports_sglang_prompt_mode()."
            )
        if not supports_sglang():
            raise ValueError(
                f"rollout.topology.service_engine='sglang' is not supported by model {model_label!r}. "
                "The model must implement classmethod supports_sglang_prompt_mode() returning True."
            )


# ============================================================================
# Rollout topology, backend, and dedicated-rollout validation
# ============================================================================


def _collect_direct_sampling_incompatible_fields(rollout_topology_config: Any) -> list[str]:
    incompatible: list[str] = []
    for field_info in dataclass_fields(type(rollout_topology_config)):
        if field_info.name in {"mode", "service_engine"}:
            continue
        field_value = getattr(rollout_topology_config, field_info.name)
        field_default = resolve_dataclass_field_default(field_info, missing=MISSING)
        if field_default is not MISSING and field_value == field_default:
            continue
        if field_value in (None, "", False):
            continue
        if isinstance(field_value, dict) and not field_value:
            continue
        incompatible.append(f"rollout.topology.{field_info.name}")
    return incompatible


def _format_rollout_mode_state(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
) -> str:
    rollout_topology = rollout_mode_info.rollout_topology
    training_actor_sampling_mode = bool(rollout_mode_info.training_actor_sampling_mode)
    replay_enabled = bool(rollout_mode_info.replay_enabled)
    sync_protocol = str(rollout_mode_info.sync_protocol)
    max_samples_per_request = rollout_mode_info.max_samples_per_request
    logprob_source = str(rollout_mode_info.logprob_source)
    is_sglang_engine = rollout_mode_info.is_sglang_engine
    replay_guard = rollout_mode_info.replay_guard

    service_engine = rollout_topology.service_engine
    return "\n".join(
        [
            "Resolved rollout mode:",
            f"  rollout.topology.mode = {rollout_topology.mode!r}",
            f"  rollout.topology.service_engine = {service_engine!r}",
            f"  training_actor_sampling_mode = {training_actor_sampling_mode}",
            f"  is_sglang_engine = {bool(is_sglang_engine)}",
            f"  sampling.logprob_source = {logprob_source!r}",
            f"  sampling.replay_log_probs = {replay_enabled}",
            f"  replay_guard = {bool(replay_guard)}",
            f"  sampling.max_samples_per_request = {max_samples_per_request!r}",
            f"  sync.protocol = {sync_protocol!r}",
            "  offload flags = "
            f"(ray.offload_train={bool(args.ray.offload_train)}, "
            f"ray.offload_rollout={bool(args.ray.offload_rollout)})",
        ]
    )


def validate_async_training_runner(args: Any) -> None:
    """Validate constraints that apply only to the async entrypoint."""
    debug_mode = args.debug.debug_mode
    if debug_mode == "train_only":
        raise ValueError(
            "train_async.py does not support debug_mode=train_only. "
            "Use python -m diffusionrl.train for train_only debug runs."
        )
    if bool(args.debug.debug_save_intermediates):
        raise ValueError(
            "train_async.py does not support debug_save_intermediates=true. "
            "Use python -m diffusionrl.train for rollout debug artifact capture."
        )

    rollout_topology = derive_rollout_topology(args)
    if rollout_mode_is_colocated(rollout_topology.mode):
        raise ValueError("train_async.py requires rollout.topology.mode='separate'.")
    if rollout_topology.training_actor_sampling_mode:
        raise ValueError(
            "train_async.py requires a dedicated rollout engine "
            "(for example rollout.topology.service_engine='sglang')."
        )
    if args.ray.offload_train or args.ray.offload_rollout:
        raise ValueError(
            "train_async.py is incompatible with offload_train/offload_rollout. "
            "Set --ray.offload-train=false --ray.offload-rollout=false."
        )


def validate_rollout_topology_contract(
    args: Any,
    *,
    rollout_topology: RolloutTopology,
) -> RolloutTopology:
    """Validate rollout topology contract after strict topology resolution."""
    topology = rollout_topology
    rollout_topology_config = args.rollout.topology

    if rollout_mode_uses_service(topology.mode):
        if topology.service_engine is None:
            raise ValueError(
                "Dedicated rollout modes require rollout.topology.service_engine to be set explicitly."
            )
        if not uses_dedicated_rollout_engine(topology.service_engine):
            raise ValueError(
                "rollout.topology.mode in {separate,colocate} requires a dedicated rollout "
                f"service engine. Got rollout.topology.service_engine={topology.service_engine!r}."
            )
        if rollout_topology_config.service_num_gpus is None:
            raise ValueError(
                "Dedicated rollout services require rollout.topology.service_num_gpus to be set explicitly."
            )
        return topology

    configured_direct_incompatible_fields = _collect_direct_sampling_incompatible_fields(
        rollout_topology_config
    )
    if configured_direct_incompatible_fields:
        raise ValueError(
            "direct_sampling runs sampling on training actors, so dedicated rollout-service fields "
            f"must be unset. Remove: {', '.join(sorted(configured_direct_incompatible_fields))}."
        )

    return topology


def validate_algorithm_kwargs_payload(args: Any) -> None:
    """Validate algorithm_kwargs payload without mutating args."""
    parsed = build_algorithm_kwargs(args)
    reserved_paths = {
        "samples_per_prompt": "algorithm.samples_per_prompt",
        "prompts_per_rollout": "algorithm.prompts_per_rollout",
        "component_mix_stage": "algorithm.component_mix_stage",
        "adv_normalization": "algorithm.adv_normalization",
        "adv_norm_eps": "algorithm.adv_norm_eps",
        "adv_clip_abs": "algorithm.adv_clip_abs",
        "use_global_std": "algorithm.use_global_std",
        "trimmed_ratio": "algorithm.trimmed_ratio",
        "eval_ema_decay": "algorithm.eval_ema_decay",
        "eval_ema_update_interval": "algorithm.eval_ema_update_interval",
        "shuffle_samples": "algorithm.shuffle_samples",
        "shuffle_seed": "algorithm.shuffle_seed",
        "window_training": "algorithm.window.window_training",
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


def validate_algorithm_path(args: Any) -> None:
    """Validate algorithm path resolution without mutating args."""
    resolve_algorithm_path(
        algorithm_type=args.algorithm.algorithm_type,
        algorithm_path=args.algorithm.algorithm_path,
    )


def validate_training_actor_sampling_mode(
    *,
    rollout_mode_info: RolloutModeInfo,
    backend_capabilities: Mapping[str, Any],
    backend_name: str,
) -> None:
    """Validate direct-sampling topology compatibility."""
    if not rollout_mode_info.training_actor_sampling_mode:
        return

    resolved_topology = rollout_mode_info.rollout_topology
    if rollout_mode_uses_service(resolved_topology.mode) or uses_dedicated_rollout_engine(resolved_topology.service_engine):
        raise ValueError(
            "Dedicated rollout service engines cannot use direct_sampling mode. "
            f"Got rollout.topology.mode={resolved_topology.mode!r}, "
            f"rollout.topology.service_engine={resolved_topology.service_engine!r}."
        )

    if not bool(backend_capabilities.get("supports_training_actor_sampling", False)):
        raise ValueError(
            "rollout.topology.mode=%r resolves to direct training-actor sampling, "
            "but train_backend=%r does not declare supports_training_actor_sampling=true."
            % (resolved_topology.mode, backend_name)
        )


def validate_replay_mode(*, rollout_mode_info: RolloutModeInfo) -> None:
    """Validate replay/log-prob flags against the resolved rollout-mode context."""
    if rollout_mode_info.is_sglang_engine and rollout_mode_info.replay_guard:
        if rollout_mode_info.logprob_source == "replay" and not rollout_mode_info.replay_enabled:
            raise ValueError(
                "rollout.topology.service_engine='sglang' with logprob_source='replay' requires "
                "--sampling.replay-log-probs=true. Set it explicitly."
            )
        if rollout_mode_info.logprob_source == "native" and rollout_mode_info.replay_enabled:
            raise ValueError(
                "logprob_source='native' is incompatible with replay_log_probs=true. "
                "Set --sampling.replay-log-probs=false when using native log_prob mode."
            )

    if rollout_mode_info.replay_enabled and not rollout_mode_info.replay_guard:
        raise ValueError(
            "replay_log_probs=true is only valid for "
            "dedicated rollout services + algorithm_type='grpo'. "
            "Either disable replay_log_probs or adjust your config."
        )


def validate_offload_and_colocate_config(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
) -> None:
    """Validate offload switches and colocate fractions."""
    if rollout_mode_info.training_actor_sampling_mode:
        if bool(args.ray.offload_train) or bool(args.ray.offload_rollout):
            raise ValueError(
                "direct_sampling uses training actors for sampling and cannot be combined with "
                "ray.offload_train / ray.offload_rollout."
            )
        return

    if rollout_mode_is_colocated(rollout_mode_info.rollout_topology.mode):
        validate_colocate_fractions(args)


def validate_rollout_mode(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
    backend_capabilities: Mapping[str, Any],
    backend_name: str,
) -> None:
    """Validate rollout mode against a pre-resolved rollout-mode context."""
    check_name = "rollout topology"
    try:
        validate_rollout_topology_contract(
            args,
            rollout_topology=rollout_mode_info.rollout_topology,
        )

        check_name = "training-actor sampling compatibility"
        validate_training_actor_sampling_mode(
            rollout_mode_info=rollout_mode_info,
            backend_capabilities=backend_capabilities,
            backend_name=backend_name,
        )

        check_name = "replay/log-prob configuration"
        validate_replay_mode(
            rollout_mode_info=rollout_mode_info,
        )

        check_name = "offload/colocate settings"
        validate_offload_and_colocate_config(
            args,
            rollout_mode_info=rollout_mode_info,
        )

        check_name = "direct-sampling request sizing"
        validate_direct_sampling_batch_geometry(
            args,
            rollout_mode_info=rollout_mode_info,
        )

        check_name = "weight-sync settings"
        validate_weight_sync(
            args,
            rollout_mode_info=rollout_mode_info,
        )
    except ValueError as exc:
        raise ValueError(
            "\n".join(
                [
                    f"Invalid rollout mode while validating {check_name}.",
                    _format_rollout_mode_state(
                        args,
                        rollout_mode_info=rollout_mode_info,
                    ),
                    f"Cause: {exc}",
                ]
            )
        ) from exc


def validate_direct_sampling_batch_geometry(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
) -> None:
    """Validate prompt-batch splitting for training-actor direct sampling."""
    max_samples_per_request = rollout_mode_info.max_samples_per_request
    if max_samples_per_request is None:
        return

    if not rollout_mode_info.training_actor_sampling_mode:
        raise ValueError(
            "sampling.max_samples_per_request is only valid when "
            "sampling runs directly on training actors."
        )

    max_samples_per_request = int(max_samples_per_request)
    if max_samples_per_request < 1:
        raise ValueError("sampling.max_samples_per_request must be >= 1.")


def validate_weight_sync(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
) -> None:
    """Validate explicit weight-sync protocol against rollout topology."""
    rollout_service_engine = rollout_mode_info.rollout_topology.service_engine
    resolved_mode = rollout_mode_info.sync_protocol
    if rollout_mode_info.training_actor_sampling_mode:
        if resolved_mode != "disabled":
            raise ValueError(
                "direct training-actor sampling requires sync.protocol='disabled'. "
                f"Got sync.protocol={resolved_mode!r}."
            )
        return
    if resolved_mode == "disabled":
        raise ValueError(
            "sync.protocol='disabled' is only valid when sampling runs directly on training actors. "
            f"Got rollout.topology.service_engine={rollout_service_engine!r}."
        )
    is_multi_node = (
        int(args.ray.rollout_num_nodes) > 1
        or int(args.ray.training_num_nodes) > 1
        or int(args.reward.reward_dedicated_num_nodes) > 1
    )
    if resolved_mode in {"tensor_payload", "nccl_broadcast"} and rollout_service_engine != "sglang":
        raise ValueError(
            "sync.protocol in {tensor_payload,nccl_broadcast} currently requires "
            "rollout.topology.service_engine='sglang'. "
            f"Got rollout.topology.service_engine={rollout_service_engine!r}."
        )

    if (
        resolved_mode == "checkpoint_path"
        and is_multi_node
        and is_probably_local_weight_sync_dir(
            args.sync.dir,
            root=repo_root(env_repo_root=ENV_REPO_ROOT),
        )
    ):
        raise ValueError(
            "sync.protocol=checkpoint_path in multi-node mode requires a shared filesystem path. "
            f"Got local-only sync.dir={args.sync.dir}. "
            "Use a shared mount (e.g. /mnt/shared/... or NFS path)."
        )


def validate_rollout_layout(
    args: Any,
    *,
    rollout_mode_info: RolloutModeInfo,
) -> None:
    """Validate rollout actor GPU layout and colocate constraints."""
    if rollout_mode_info.training_actor_sampling_mode:
        return

    rollout_topology = rollout_mode_info.rollout_topology
    rollout_gpus = derive_rollout_actor_gpu_count(args, topology=rollout_topology)
    rollout_gpu_pool_size = derive_rollout_gpu_pool_size(args, topology=rollout_topology)
    if not rollout_mode_uses_service(rollout_topology.mode):
        raise ValueError(
            "Dedicated rollout actor layout validation only applies to dedicated rollout engines. "
            f"Got rollout.topology.mode={rollout_topology.mode!r}."
        )
    if rollout_gpu_pool_size < 1:
        raise ValueError(
            "Dedicated rollout services require a positive rollout GPU pool from placement config. "
            f"Got rollout_num_nodes={args.ray.rollout_num_nodes}, "
            f"rollout_num_gpus_per_node={args.ray.rollout_num_gpus_per_node}, "
            f"training_num_nodes={args.ray.training_num_nodes}, "
            f"training_num_gpus_per_node={args.ray.training_num_gpus_per_node}."
        )
    if rollout_gpu_pool_size < rollout_gpus:
        raise ValueError(
            "Dedicated rollout placement does not have enough GPUs for one rollout actor. "
            f"Available rollout GPU pool={rollout_gpu_pool_size}, "
            f"rollout.topology.service_num_gpus={rollout_gpus}."
        )
    if rollout_gpus > 1 and rollout_gpu_pool_size % rollout_gpus != 0:
        raise ValueError(
            "Dedicated rollout GPU pool must be divisible by rollout.topology.service_num_gpus "
            "for multi-GPU rollout actors. "
            f"Available rollout GPU pool={rollout_gpu_pool_size}, "
            f"rollout.topology.service_num_gpus={rollout_gpus}."
        )
    is_sglang_engine = rollout_mode_info.is_sglang_engine
    if rollout_gpus > 1 and rollout_mode_is_colocated(rollout_topology.mode) and not is_sglang_engine:
        raise ValueError(
            "colocate with multi-GPU rollout actors is only supported "
            "for rollout.topology.service_engine='sglang'."
        )
    if rollout_gpus > 1:
        allow_noset_multi_gpu_inference = bool(args.ray.allow_noset_multi_gpu_inference)
        if not allow_noset_multi_gpu_inference:
            if rollout_mode_is_colocated(rollout_topology.mode) and is_sglang_engine:
                raise ValueError(
                    "sglang colocate with multi-GPU rollout requires NOSET actor layout. "
                    "Set --ray.allow-noset-multi-gpu-inference=true."
                )
            raise ValueError(
                "multi-GPU rollout actor layout requires "
                "--ray.allow-noset-multi-gpu-inference=true. "
                "Default layout keeps integer single-GPU actors."
            )
        logger.warning(
            "allow_noset_multi_gpu_inference=true enabled. "
            "This is an experimental actor layout and is not part of the default path."
        )


# ============================================================================
# Reward and rollout-buffer validation
# ============================================================================


def validate_reward_config(args: Any) -> None:
    """Validate reward pool/source configuration consistency."""
    reward_config = args.reward

    if reward_config.reward_dedicated_gpus_per_actor > 1 and reward_config.reward_dedicated_num_gpus > 0:
        if reward_config.reward_dedicated_num_gpus < reward_config.reward_dedicated_gpus_per_actor:
            raise ValueError(
                f"reward_dedicated_num_gpus ({reward_config.reward_dedicated_num_gpus}) must be >= "
                f"reward_dedicated_gpus_per_actor ({reward_config.reward_dedicated_gpus_per_actor})"
            )
        if reward_config.reward_dedicated_num_gpus % reward_config.reward_dedicated_gpus_per_actor != 0:
            raise ValueError(
                f"reward_dedicated_num_gpus ({reward_config.reward_dedicated_num_gpus}) must be divisible by "
                f"reward_dedicated_gpus_per_actor ({reward_config.reward_dedicated_gpus_per_actor})"
            )

    if reward_config.reward_dedicated_num_nodes > 0 and reward_config.reward_dedicated_num_gpus_per_node <= 0:
        raise ValueError(
            "reward_dedicated_num_gpus_per_node must be > 0 when reward_dedicated_num_nodes > 0"
        )

    if reward_config.reward_dedicated_num_gpus > 0 and reward_config.reward_dedicated_num_nodes > 0:
        raise ValueError(
            "reward_dedicated_num_gpus and reward_dedicated_num_nodes are mutually exclusive. "
            "Use either total dedicated GPUs, or nodes * gpus_per_node."
        )

    reward_schema = RewardSchema.from_args(args)
    execution_plan = reward_schema.to_execution_plan()
    has_dedicated_reward_pool = reward_config.has_dedicated_reward_pool
    has_http_reward_urls = reward_config.has_http_reward_urls
    has_http_reward = reward_config.has_http_reward
    local_reward_device = str(reward_config.local_reward_device or "cpu").strip().lower()
    requested_reward_location = str(reward_config.reward_location or "auto").strip().lower()
    reward_location = str(execution_plan.location or "driver").strip().lower()
    allow_local_reward_cuda_contention = bool(reward_config.allow_local_reward_cuda_contention)

    if reward_config.use_http_reward and not has_http_reward_urls:
        raise ValueError(
            "use_http_reward=true requires reward_service_url or reward_service_urls."
        )

    if reward_location == "sampling_actor":
        if has_dedicated_reward_pool:
            raise ValueError(
                "reward_location='sampling_actor' cannot be combined with dedicated reward actors. "
                "Use reward_location='driver' for reward_dedicated_* modes."
            )

    uses_local_same_process_reward = (
        reward_location == "driver"
        and not has_http_reward
        and not has_dedicated_reward_pool
    )
    if (
        uses_local_same_process_reward
        and local_reward_device == "cuda"
        and not allow_local_reward_cuda_contention
    ):
        raise ValueError(
            "local_reward_device='cuda' in driver-local reward mode can contend with "
            "rollout/training GPUs. Use dedicated reward actors (reward_dedicated_*), "
            "use_http_reward, or set allow_local_reward_cuda_contention=true to force."
        )

    if requested_reward_location == "auto":
        logger.info(
            "Resolved reward_location='auto' -> %s (backend=%s)",
            reward_location,
            execution_plan.backend,
        )

    if reward_location == "sampling_actor":
        if has_http_reward:
            logger.info("Reward mode: sampling-actor HTTP (external service)")
        else:
            logger.info(
                "Reward mode: sampling-actor-local worker (local_reward_device=%s)",
                local_reward_device,
            )
    elif has_http_reward:
        logger.info("Reward mode: driver HTTP (external service)")
    elif has_dedicated_reward_pool:
        total_gpus = reward_config.reward_dedicated_num_gpus
        if reward_config.reward_dedicated_num_nodes > 0:
            total_gpus = reward_config.reward_dedicated_num_nodes * reward_config.reward_dedicated_num_gpus_per_node
        num_actors = total_gpus // reward_config.reward_dedicated_gpus_per_actor
        logger.info(
            "Reward mode: Independent GPU (%s GPUs, %s actors, %s GPUs/actor)",
            total_gpus,
            num_actors,
            reward_config.reward_dedicated_gpus_per_actor,
        )
    else:
        logger.info(
            "Reward mode: driver-local same-process worker (local_reward_device=%s)",
            local_reward_device,
        )


def validate_reward_and_rollout_buffer_config(args: Any) -> None:
    """Validate reward pool config and rollout-buffer controls."""
    validate_reward_config(args)
    rollout_buffer = args.rollout.buffer

    if bool(rollout_buffer.reassemble_by_group) and rollout_buffer.group_size is not None:
        if bool(rollout_buffer.drop_invalid):
            raise ValueError(
                "rollout.buffer.reassemble_by_group is incompatible with "
                "rollout.buffer.drop_invalid=true. Sample-dropping finite-value "
                "filtering can leave incomplete groups pending forever. Set "
                "rollout.buffer.drop_invalid=false so invalid batches fail fast."
            )
        if rollout_buffer.reward_min is not None or rollout_buffer.reward_max is not None:
            raise ValueError(
                "rollout.buffer.reassemble_by_group is incompatible with "
                "rollout.buffer.reward_min/max. Reward-range filtering drops "
                "samples and breaks the complete-group producer contract."
            )
        target_batch_size = int(derive_global_rollout_batch_size(args))
        if int(rollout_buffer.group_size) > target_batch_size:
            raise ValueError(
                "rollout.buffer.group_size cannot exceed the resolved training batch size. "
                f"Got group_size={rollout_buffer.group_size}, target_batch_size={target_batch_size}."
            )
        if target_batch_size % int(rollout_buffer.group_size) != 0:
            raise ValueError(
                "rollout.buffer.reassemble_by_group requires the resolved training batch size "
                "to be divisible by rollout.buffer.group_size. "
                f"Got target_batch_size={target_batch_size}, group_size={rollout_buffer.group_size}."
            )


# ============================================================================
# Training geometry, backend, and optimizer validation
# ============================================================================


def validate_train_backend_config(
    *,
    backend_name: str,
    backend_kwargs: Mapping[str, Any],
    backend_path: Optional[str],
) -> None:
    """Validate cross-domain backend constraints after canonicalization."""
    backend = backend_name
    supported = supported_train_backends()
    if backend not in supported and not backend_path:
        raise ValueError(
            f"Unsupported train_backend={backend!r}. "
            f"Expected one of {sorted(supported)} or provide --training.train-backend-path."
        )
    if backend == "megatron" and not backend_path:
        logger.warning(
            "train_backend=%s is currently a scaffold backend: launch/topology interfaces are wired, "
            "but the training execution path is not fully implemented yet. "
            "Use train_backend_kwargs.actor_class_path to provide a Megatron-dedicated actor.",
            backend,
        )

    if backend == "megatron" and not backend_path and not str(
        dict(backend_kwargs).get("actor_class_path", "") or ""
    ).strip():
        logger.warning(
            "train_backend=%s requires actor_class_path in train_backend_kwargs "
            "to launch a Megatron-specific training actor.",
            backend,
        )


def validate_training_batch_geometry(
    args: Any,
    *,
    topology: Optional[TrainTopology] = None,
) -> None:
    """Validate batch-geometry invariants using resolved training geometry."""
    prompts_per_rollout = int(require_prompts_per_rollout(args))
    samples_per_prompt = int(args.algorithm.samples_per_prompt)
    if samples_per_prompt < 1:
        raise ValueError(f"algorithm.samples_per_prompt must be >= 1. Got {samples_per_prompt}.")

    num_updates_per_local_batch = derive_num_updates_per_local_batch(args)
    global_batch_size = derive_global_rollout_batch_size(args)
    topology = topology if topology is not None else derive_training_topology(args)
    dp_size = int(topology.dp_size)
    dp_replicate_size = int(topology.dp_replicate_size)
    raw_micro_batch_size = args.training.local_micro_batch_size

    def _format_geometry(
        *,
        local_batch_size: Optional[int],
        update_batch_size: Optional[int],
        micro_batch_size: Optional[int],
    ) -> str:
        local_text = str(local_batch_size) if local_batch_size is not None else "<not divisible by dp_size>"
        update_text = (
            str(update_batch_size)
            if update_batch_size is not None
            else "<not divisible by num_updates_per_local_batch>"
        )
        if raw_micro_batch_size is None:
            micro_text = "auto"
            if micro_batch_size is not None:
                micro_text = f"auto (= {micro_batch_size})"
        else:
            micro_text = str(micro_batch_size if micro_batch_size is not None else raw_micro_batch_size)
        return "\n".join(
            [
                "Resolved training batch geometry:",
                f"  global_batch_size = prompts_per_rollout({prompts_per_rollout}) * "
                f"samples_per_prompt({samples_per_prompt}) = {global_batch_size}",
                f"  local_batch_size = global_batch_size / dp_size({dp_size}) = {local_text}",
                f"  local_update_batch_size = local_batch_size / "
                f"num_updates_per_local_batch({num_updates_per_local_batch}) = {update_text}",
                f"  local_micro_batch_size = {micro_text}",
                f"  dp_replicate_size = {dp_replicate_size}",
            ]
        )

    def _raise_geometry_error(
        *,
        reason: str,
        fix_hint: str,
        local_batch_size: Optional[int],
        update_batch_size: Optional[int],
        micro_batch_size: Optional[int],
    ) -> None:
        raise ValueError(
            "\n".join(
                [
                    f"Invalid training batch geometry: {reason}",
                    _format_geometry(
                        local_batch_size=local_batch_size,
                        update_batch_size=update_batch_size,
                        micro_batch_size=micro_batch_size,
                    ),
                    f"Fix: {fix_hint}",
                ]
            )
        )

    if global_batch_size % dp_size != 0:
        _raise_geometry_error(
            reason="global rollout batch cannot be split evenly across training DP ranks "
            "(global_batch_size % dp_size != 0).",
            fix_hint="Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend dp_size.",
            local_batch_size=None,
            update_batch_size=None,
            micro_batch_size=None,
        )
    local_batch_size = int(global_batch_size // dp_size)

    if global_batch_size % dp_replicate_size != 0:
        _raise_geometry_error(
            reason="global rollout batch must also be divisible by dp_replicate_size "
            "(global_batch_size % dp_replicate_size != 0).",
            fix_hint="Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the backend replicate topology.",
            local_batch_size=local_batch_size,
            update_batch_size=None,
            micro_batch_size=None,
        )

    if local_batch_size % num_updates_per_local_batch != 0:
        _raise_geometry_error(
            reason="local batch cannot be split evenly into optimizer updates "
            "(local_batch_size % num_updates_per_local_batch != 0).",
            fix_hint="Choose a training.num_updates_per_local_batch that evenly divides "
            "the resolved local_batch_size.",
            local_batch_size=local_batch_size,
            update_batch_size=None,
            micro_batch_size=None,
        )
    update_batch_size = int(local_batch_size // num_updates_per_local_batch)

    if raw_micro_batch_size is None:
        micro_batch_size = int(update_batch_size)
    else:
        micro_batch_size = int(raw_micro_batch_size)
        if micro_batch_size < 1:
            _raise_geometry_error(
                reason="training.local_micro_batch_size must be >= 1.",
                fix_hint="Set training.local_micro_batch_size to a positive integer, "
                "or omit it to use the resolved local_update_batch_size.",
                local_batch_size=local_batch_size,
                update_batch_size=update_batch_size,
                micro_batch_size=micro_batch_size,
            )

    if update_batch_size % micro_batch_size != 0:
        _raise_geometry_error(
            reason="local update batch cannot be split evenly into micro-batches "
            "(local_update_batch_size % local_micro_batch_size != 0).",
            fix_hint="Choose a training.local_micro_batch_size that evenly divides "
            "the resolved local_update_batch_size.",
            local_batch_size=local_batch_size,
            update_batch_size=update_batch_size,
            micro_batch_size=micro_batch_size,
        )


def validate_training_misc(args: Any) -> None:
    """Validate non-batch training knobs that affect downstream components."""
    normalize_lora_target_modules(args.training.lora_target_modules)


# ============================================================================
# Payload-shape validation for actor/service init configs
# ============================================================================


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

    engine_runtime_config = _require_dict_section(config, name="engine_runtime_config")
    reward_config = _require_dict_section(config, name="reward_config")

    validate_rollout_engine_config(engine_runtime_config)

    sampler_path = str(engine_runtime_config.get("sampler_path", "") or "").strip()
    if not sampler_path:
        raise ValueError(
            "rollout_actor_init_config.engine_runtime_config.sampler_path is required."
        )
    sampler_engine_type = str(
        engine_runtime_config.get("sampler_engine_type", "") or ""
    ).strip()
    if not sampler_engine_type:
        raise ValueError(
            "rollout_actor_init_config.engine_runtime_config.sampler_engine_type is required."
        )
    for required_key in (
        "num_inference_steps",
        "eta",
        "sde_type",
        "shift",
        "guidance_scale",
        "height",
        "width",
        "num_frames",
    ):
        if engine_runtime_config.get(required_key) is None:
            raise ValueError(
                f"rollout_actor_init_config.engine_runtime_config.{required_key} is required."
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
    try:
        coerce_training_execution_plan(training_plan_config)
    except ValueError as exc:
        raise ValueError(
            "Invalid training_plan_config in training actor init payload. "
            f"{exc}"
        ) from exc


# ============================================================================
# __all__
# ============================================================================

__all__ = [
    "ENV_REPO_ROOT",
    "apply_model_config_hook",
    "is_probably_local_weight_sync_dir",
    "repo_root",
    "validate_algorithm_kwargs_payload",
    "validate_algorithm_path",
    "validate_async_training_runner",
    "validate_colocate_fractions",
    "validate_direct_sampling_batch_geometry",
    "validate_dotpath",
    "validate_dynamic_dotpaths",
    "validate_grouped_configs",
    "validate_model_sampling_contract",
    "validate_nft_sampling_contract",
    "validate_offload_and_colocate_config",
    "validate_precision_name",
    "validate_replay_mode",
    "validate_resolved_engine_algorithm_contract",
    "validate_reward_and_rollout_buffer_config",
    "validate_reward_config",
    "validate_rollout_actor_init_config",
    "validate_rollout_engine_config",
    "validate_rollout_layout",
    "validate_rollout_mode",
    "validate_rollout_mode_constraints",
    "validate_rollout_topology_contract",
    "validate_train_backend_config",
    "validate_training_actor_init_config",
    "validate_training_actor_sampling_mode",
    "validate_training_batch_geometry",
    "validate_training_misc",
    "validate_weight_sync",
]
