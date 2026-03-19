"""Configuration validation helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from diffusionrl.config.rollout_topology import (
    normalize_rollout_service_engine,
    resolve_rollout_service_num_gpus,
    rollout_mode_is_colocated,
    rollout_mode_uses_service,
)
from diffusionrl.config.resolution import (
    DEFAULT_SAMPLER_PATH,
    resolve_algorithm_kwargs,
    resolve_algorithm_path,
    resolve_global_rollout_batch_size,
    resolve_model_runtime,
)
from diffusionrl.sde.rules import (
    SUPPORTED_USER_SDE_TYPES,
    is_deterministic_sde_type,
    supported_sde_type_text,
)
from diffusionrl.runtime.contracts import (
    resolve_engine_capabilities,
    resolve_sampling_requirements,
)
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)


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

def validate_reward_config(args) -> None:
    """Validate reward pool/source configuration consistency."""
    if args.reward.reward_dedicated_gpus_per_actor > 1 and args.reward.reward_dedicated_num_gpus > 0:
        if args.reward.reward_dedicated_num_gpus < args.reward.reward_dedicated_gpus_per_actor:
            raise ValueError(
                f"reward_dedicated_num_gpus ({args.reward.reward_dedicated_num_gpus}) must be >= "
                f"reward_dedicated_gpus_per_actor ({args.reward.reward_dedicated_gpus_per_actor})"
            )
        if args.reward.reward_dedicated_num_gpus % args.reward.reward_dedicated_gpus_per_actor != 0:
            raise ValueError(
                f"reward_dedicated_num_gpus ({args.reward.reward_dedicated_num_gpus}) must be divisible by "
                f"reward_dedicated_gpus_per_actor ({args.reward.reward_dedicated_gpus_per_actor})"
            )

    if args.reward.reward_dedicated_num_nodes > 0 and args.reward.reward_dedicated_num_gpus_per_node <= 0:
        raise ValueError(
            "reward_dedicated_num_gpus_per_node must be > 0 when reward_dedicated_num_nodes > 0"
        )

    if args.reward.reward_dedicated_num_gpus > 0 and args.reward.reward_dedicated_num_nodes > 0:
        raise ValueError(
            "reward_dedicated_num_gpus and reward_dedicated_num_nodes are mutually exclusive. "
            "Use either total dedicated GPUs, or nodes * gpus_per_node."
        )

    has_dedicated_reward_pool = (
        args.reward.reward_dedicated_num_gpus > 0 or args.reward.reward_dedicated_num_nodes > 0
    )
    has_http_reward = bool(
        getattr(args.reward, "use_http_reward", False)
        or getattr(args.reward, "reward_service_url", None)
        or getattr(args.reward, "reward_service_urls", None)
    )
    local_reward_device = str(
        getattr(args.reward, "local_reward_device", "cpu") or "cpu"
    ).strip().lower()
    reward_location = str(
        getattr(args.reward, "reward_location", "manager") or "manager"
    ).strip().lower()
    allow_local_reward_cuda_contention = bool(
        getattr(args.reward, "allow_local_reward_cuda_contention", False)
    )

    if args.reward.use_http_reward and not (
        getattr(args.reward, "reward_service_url", None)
        or getattr(args.reward, "reward_service_urls", None)
    ):
        raise ValueError(
            "use_http_reward=true requires reward_service_url or reward_service_urls."
        )

    if reward_location == "sampling_actor":
        if has_http_reward:
            raise ValueError(
                "reward_location='sampling_actor' cannot be combined with HTTP reward service. "
                "Use reward_location='manager' for HTTP reward."
            )
        if has_dedicated_reward_pool:
            raise ValueError(
                "reward_location='sampling_actor' cannot be combined with dedicated reward actors. "
                "Use reward_location='manager' for reward_dedicated_* modes."
            )

    uses_local_same_process_reward = (
        reward_location == "manager"
        and not has_http_reward
        and not has_dedicated_reward_pool
    )
    if (
        uses_local_same_process_reward
        and local_reward_device == "cuda"
        and not allow_local_reward_cuda_contention
    ):
        raise ValueError(
            "local_reward_device='cuda' in local same-process reward mode can contend with "
            "rollout/training GPUs. Use dedicated reward actors (reward_dedicated_*), "
            "use_http_reward, or set allow_local_reward_cuda_contention=true to force."
        )

    if reward_location == "sampling_actor":
        logger.info(
            "Reward mode: sampling-actor-local worker (local_reward_device=%s)",
            local_reward_device,
        )
    elif has_http_reward:
        logger.info("Reward mode: HTTP (external service)")
    elif has_dedicated_reward_pool:
        total_gpus = args.reward.reward_dedicated_num_gpus
        if args.reward.reward_dedicated_num_nodes > 0:
            total_gpus = args.reward.reward_dedicated_num_nodes * args.reward.reward_dedicated_num_gpus_per_node
        num_actors = total_gpus // args.reward.reward_dedicated_gpus_per_actor
        logger.info(
            f"Reward mode: Independent GPU ({total_gpus} GPUs, "
            f"{num_actors} actors, {args.reward.reward_dedicated_gpus_per_actor} GPUs/actor)"
        )
    else:
        logger.info(
            "Reward mode: Local same-process worker (local_reward_device=%s)",
            local_reward_device,
        )


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


def validate_dynamic_dotpaths(args: Any) -> None:
    """Validate configured runtime extension dotpaths."""
    resolved_model = resolve_model_runtime(
        args,
        explicit_sampler_path=(getattr(args.sampling, "sampler_path", None) != DEFAULT_SAMPLER_PATH),
    )
    validate_dotpath(resolved_model.model_path, label="model")
    validate_dotpath(resolved_model.sampler_path, label="sampler")
    validate_dotpath(resolve_algorithm_path(args), label="algorithm")
    validate_dotpath(args.data_source_path, label="data_source")
    if getattr(args.training, "train_backend_path", None):
        validate_dotpath(args.training.train_backend_path, label="train_backend")
    if getattr(args.sampling, "replay_sampler_path", None):
        validate_dotpath(args.sampling.replay_sampler_path, label="replay_sampler")
    rollout_buffer_plugin_paths = getattr(args.rollout, "rollout_buffer_plugin_paths", "") or ""
    if isinstance(rollout_buffer_plugin_paths, str):
        for plugin_path in [part.strip() for part in rollout_buffer_plugin_paths.split(",") if part.strip()]:
            validate_dotpath(plugin_path, label="rollout_buffer_plugin")


def validate_colocate_fractions(args) -> None:
    """Validate colocate GPU fraction bounds."""
    if args.ray.colocate_training_gpu_fraction <= 0 or args.ray.colocate_rollout_gpu_fraction <= 0:
        raise ValueError(
            "colocate_training_gpu_fraction and colocate_rollout_gpu_fraction must be > 0"
        )
    if args.ray.colocate_training_gpu_fraction + args.ray.colocate_rollout_gpu_fraction > 1.0:
        raise ValueError(
            "colocate_training_gpu_fraction + colocate_rollout_gpu_fraction must be <= 1.0"
        )


def get_rollout_gpus_per_actor(args) -> int:
    """Resolve GPUs per rollout actor based on sampler engine and engine config."""
    service_engine = normalize_rollout_service_engine(getattr(args.rollout, "service_engine", None))
    if not rollout_mode_uses_service(getattr(args.rollout, "mode", None)):
        return 0
    if not service_engine:
        raise ValueError(
            "rollout.mode requires a dedicated rollout service, but rollout.service_engine is unset."
        )
    if service_engine != "sglang":
        raise ValueError(
            f"Unsupported dedicated rollout engine: {service_engine!r}. "
            "Expected: sglang."
        )
    return resolve_rollout_service_num_gpus(args)


def _resolve_rollout_gpu_pool_size(args: Any) -> int:
    """Resolve bundle capacity available to rollout actors from placement topology."""
    rollout_total_gpus = int(args.ray.rollout_num_nodes) * int(args.ray.rollout_num_gpus_per_node)
    if not rollout_mode_is_colocated(args.rollout.mode):
        return rollout_total_gpus
    training_total_gpus = int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node)
    return max(rollout_total_gpus, training_total_gpus)


def validate_reward_and_rollout_buffer_config(args: Any) -> None:
    """Validate reward pool config and rollout-buffer controls."""
    validate_reward_config(args)

    if int(getattr(args.rollout, "rollout_buffer_max_queue_size", 0)) < 0:
        raise ValueError(
            f"rollout_buffer_max_queue_size must be >= 0, got: {args.rollout.rollout_buffer_max_queue_size}"
        )
    if int(getattr(args.rollout, "rollout_buffer_min_samples", 1)) < 1:
        raise ValueError(
            f"rollout_buffer_min_samples must be >= 1, got: {args.rollout.rollout_buffer_min_samples}"
        )
    reward_min = getattr(args.rollout, "rollout_buffer_reward_min", None)
    reward_max = getattr(args.rollout, "rollout_buffer_reward_max", None)
    if reward_min is not None and reward_max is not None and float(reward_min) > float(reward_max):
        raise ValueError(
            "rollout_buffer_reward_min must be <= rollout_buffer_reward_max, "
            f"got min={reward_min}, max={reward_max}"
        )
    group_size = getattr(args.rollout, "rollout_buffer_group_size", None)
    if group_size is not None and int(group_size) < 1:
        raise ValueError(
            f"rollout_buffer_group_size must be >= 1 when provided, got: {group_size}"
        )
    if float(getattr(args.rollout, "rollout_buffer_group_ttl_seconds", 0.0)) < 0:
        raise ValueError(
            "rollout_buffer_group_ttl_seconds must be >= 0, "
            f"got: {args.rollout.rollout_buffer_group_ttl_seconds}"
        )
    if int(getattr(args.rollout, "rollout_buffer_max_pending_samples", 0)) < 0:
        raise ValueError(
            "rollout_buffer_max_pending_samples must be >= 0, "
            f"got: {args.rollout.rollout_buffer_max_pending_samples}"
        )
    if bool(getattr(args.rollout, "rollout_buffer_reassemble_by_group", False)) and group_size is not None:
        if bool(getattr(args.rollout, "rollout_buffer_drop_invalid", True)):
            raise ValueError(
                "rollout_buffer_reassemble_by_group is incompatible with "
                "rollout_buffer_drop_invalid=true. Sample-dropping finite-value "
                "filtering can leave incomplete groups pending forever. Set "
                "rollout.rollout_buffer_drop_invalid=false so invalid batches fail fast."
            )
        if reward_min is not None or reward_max is not None:
            raise ValueError(
                "rollout_buffer_reassemble_by_group is incompatible with "
                "rollout_buffer_reward_min/max. Reward-range filtering drops "
                "samples and breaks the complete-group producer contract."
            )
        target_batch_size = int(resolve_global_rollout_batch_size(args))
        if int(group_size) > target_batch_size:
            raise ValueError(
                "rollout_buffer_group_size cannot exceed the resolved training batch size. "
                f"Got group_size={group_size}, target_batch_size={target_batch_size}."
            )
        if target_batch_size % int(group_size) != 0:
            raise ValueError(
                "rollout_buffer_reassemble_by_group requires the resolved training batch size "
                "to be divisible by rollout_buffer_group_size. "
                f"Got target_batch_size={target_batch_size}, group_size={group_size}."
            )


def validate_rollout_layout(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate rollout actor GPU layout and colocate constraints."""
    if training_actor_sampling_mode:
        return

    rollout_gpus = get_rollout_gpus_per_actor(args)
    rollout_gpu_pool_size = _resolve_rollout_gpu_pool_size(args)
    service_engine = normalize_rollout_service_engine(getattr(args.rollout, "service_engine", None))
    if not rollout_mode_uses_service(getattr(args.rollout, "mode", None)):
        raise ValueError(
            "Dedicated rollout actor layout validation only applies to dedicated rollout engines. "
            f"Got rollout.mode={getattr(args.rollout, 'mode', None)!r}."
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
            f"rollout.service_num_gpus={rollout_gpus}."
        )
    if rollout_gpus > 1 and rollout_gpu_pool_size % rollout_gpus != 0:
        raise ValueError(
            "Dedicated rollout GPU pool must be divisible by rollout.service_num_gpus "
            "for multi-GPU rollout actors. "
            f"Available rollout GPU pool={rollout_gpu_pool_size}, "
            f"rollout.service_num_gpus={rollout_gpus}."
        )
    is_sglang_engine = service_engine == "sglang"
    if rollout_gpus > 1 and rollout_mode_is_colocated(args.rollout.mode) and not is_sglang_engine:
        raise ValueError(
            "colocate_rollout with multi-GPU rollout actors is only supported "
            "for rollout.service_engine='sglang'."
        )
    if (
        rollout_gpus > 1
        and rollout_mode_is_colocated(args.rollout.mode)
        and is_sglang_engine
        and not bool(getattr(args.ray, "allow_noset_multi_gpu_inference", False))
    ):
        raise ValueError(
            "sglang colocate with multi-GPU rollout requires NOSET actor layout. "
            "Set --ray.allow-noset-multi-gpu-inference=true."
        )
    if rollout_gpus > 1 and not bool(getattr(args.ray, "allow_noset_multi_gpu_inference", False)):
        raise ValueError(
            "multi-GPU rollout actor layout requires "
            "--ray.allow-noset-multi-gpu-inference=true. "
            "Default layout keeps integer single-GPU actors."
        )
    if rollout_gpus > 1 and bool(getattr(args.ray, "allow_noset_multi_gpu_inference", False)):
        logger.warning(
            "allow_noset_multi_gpu_inference=true enabled. "
            "This is an experimental actor layout and is not part of the default path."
        )


def validate_model_runtime_contract(args: Any) -> None:
    """Validate model/runtime combinations that should never reach model hooks."""
    raw_sde_type = str(getattr(args.sampling, "sde_type", "") or "").strip().lower()
    if raw_sde_type not in SUPPORTED_USER_SDE_TYPES:
        raise ValueError(
            f"Unknown sampling.sde_type={args.sampling.sde_type!r}. "
            f"Supported values: {supported_sde_type_text()}."
        )


def apply_model_config_hook(args: Any, *, model_cls: Any) -> None:
    """Run model-provided config hook without allowing it to mutate args."""
    model_validate_fn = getattr(model_cls, "validate_config", None)
    if callable(model_validate_fn):
        before_flat = args.to_flat_dict() if hasattr(args, "to_flat_dict") else None
        model_validate_fn(args)
        if isinstance(before_flat, dict) and hasattr(args, "to_flat_dict"):
            after_flat = args.to_flat_dict()
            if isinstance(after_flat, dict):
                changed = []
                for key in sorted(set(before_flat) | set(after_flat)):
                    before = before_flat.get(key)
                    after = after_flat.get(key)
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
        # DiffusionNFT reproduction contract:
        # - rollout samples from old adapter
        # - deterministic solver (dpm2)
        old_adapter_name = "old"
        parsed: Dict[str, Any] = resolve_algorithm_kwargs(args)
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
        eta = float(getattr(args.sampling, "eta", 1.0))
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
    training_actor_sampling_mode: bool,
    is_sglang_engine: bool,
    replay_guard: bool,
    logprob_source: str,
) -> None:
    """Validate engine/algorithm compatibility using resolved capabilities."""
    if training_actor_sampling_mode:
        return

    service_engine = normalize_rollout_service_engine(getattr(args.rollout, "service_engine", None))
    if not service_engine:
        raise ValueError(
            "Dedicated rollout validation requires rollout.service_engine to be set explicitly. "
            "Run validate_args() before resolving dedicated rollout engine capabilities."
        )
    engine_caps = resolve_engine_capabilities(engine_type=service_engine)

    allow_replay = (
        bool(getattr(args.sampling, "replay_log_probs", False))
        and getattr(args.algorithm, "algorithm_type", "grpo") == "grpo"
    )
    if allow_replay:
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)
        logger.warning(
            "replay_log_probs=true enabled: allowing %s+GRPO with "
            "training-side old-log-prob replay (experimental path).",
            service_engine,
        )

    if is_sglang_engine and replay_guard and logprob_source == "native":
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)

    required = resolve_sampling_requirements(args)
    if is_sglang_engine and str(getattr(args.model, "model_type", "") or "").strip().lower() == "sd3":
        # Current sglang-diffusion SD3 path may return only final samples without
        # trajectory latents. Fail fast for trajectory-dependent losses.
        if bool(required.requires_trajectory):
            raise ValueError(
                "rollout.service_engine='sglang' with model_type='sd3' currently does not "
                "provide trajectory_latents required by trajectory-based algorithms "
                "(e.g. GRPO/MixGRPO). Use a direct-sampling engine path "
                "(the non-sglang default, which runs on training actors), or use "
                "algorithm_type='nft' when running SD3 with sglang."
            )
        engine_caps = dict(engine_caps, requires_trajectory=False)

    required_dict = required.to_dict()
    missing = [
        key
        for key, needed in required_dict.items()
        if bool(needed) and not bool(engine_caps.get(key, False))
    ]
    if missing:
        raise ValueError(
            f"Engine capability mismatch for algorithm_type={args.algorithm.algorithm_type}: "
            f"rollout.service_engine={service_engine} lacks {missing}. "
            f"engine_capabilities={engine_caps}, required={required_dict}. "
            "Use a compatible dedicated rollout engine, or fall back to direct "
            "training-actor sampling for trajectory/log-prob-heavy algorithms."
        )


def validate_runtime_mode_constraints(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
    model_cls: Any,
) -> None:
    """Validate runtime mode constraints and mutually-exclusive switches."""
    model_label = f"{model_cls.__module__}.{model_cls.__qualname__}"
    if (
        not training_actor_sampling_mode
        and normalize_rollout_service_engine(getattr(args.rollout, "service_engine", None)) == "sglang"
    ):
        supports_sglang = getattr(model_cls, "supports_sglang_prompt_mode", None)
        if not callable(supports_sglang):
            raise ValueError(
                f"rollout.service_engine='sglang' requires model {model_label!r} "
                "to define classmethod supports_sglang_prompt_mode()."
            )
        if not supports_sglang():
            raise ValueError(
                f"rollout.service_engine='sglang' is not supported by model {model_label!r}. "
                "The model must implement classmethod supports_sglang_prompt_mode() returning True."
            )

    if getattr(args.rollout, "async_pipeline", False):
        if rollout_mode_is_colocated(args.rollout.mode):
            raise ValueError("async_pipeline requires rollout.mode='separate_rollout'.")
        if training_actor_sampling_mode:
            raise ValueError(
                "async_pipeline currently requires a dedicated rollout engine "
                "(for example rollout.service_engine='sglang')."
            )
        if int(getattr(args.rollout, "async_max_inflight", 1)) < 1:
            raise ValueError("async_max_inflight must be >= 1.")
        if args.rollout.update_weights_interval <= 0:
            raise ValueError("update_weights_interval must be > 0.")
        if args.ray.offload_train or args.ray.offload_rollout:
            raise ValueError(
                "async_pipeline is incompatible with offload_train/offload_rollout. "
                "Set --ray.offload-train=false --ray.offload-rollout=false "
                "when using --rollout.async-pipeline."
            )

__all__ = [
    "apply_model_config_hook",
    "validate_colocate_fractions",
    "get_rollout_gpus_per_actor",
    "validate_dotpath",
    "validate_grouped_configs",
    "validate_dynamic_dotpaths",
    "validate_model_runtime_contract",
    "validate_reward_config",
    "validate_reward_and_rollout_buffer_config",
    "validate_rollout_layout",
    "validate_nft_sampling_contract",
    "validate_resolved_engine_algorithm_contract",
    "validate_runtime_mode_constraints",
]
