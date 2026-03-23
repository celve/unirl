"""Configuration validation helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

from diffusionrl.config.rollout_topology import (
    rollout_mode_is_colocated,
    rollout_mode_uses_service,
)
from diffusionrl.config.resolution import (
    DEFAULT_SAMPLER_PATH,
    ResolvedRolloutTopology,
    resolve_algorithm_kwargs,
    resolve_algorithm_path,
    resolve_debug_mode,
    resolve_engine_capabilities,
    resolve_global_rollout_batch_size,
    resolve_local_micro_batch_size,
    resolve_local_update_batch_size,
    resolve_logprob_source,
    resolve_lora_target_modules,
    resolve_model_runtime,
    resolve_nominal_local_training_batch_size,
    resolve_num_updates_per_local_batch,
    resolve_prompts_per_rollout,
    resolve_rollout_gpu_pool_size,
    resolve_rollout_gpus_per_actor,
    resolve_rollout_topology,
    resolve_sampling_requirements,
    resolve_train_backend_kwargs,
    resolve_train_backend_name,
    resolve_training_dp_size,
)
from diffusionrl.sde.rules import (
    SUPPORTED_USER_SDE_TYPES,
    is_deterministic_sde_type,
    supported_sde_type_text,
)
from diffusionrl.types.engine import uses_dedicated_rollout_engine
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"


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


def validate_debug_config(args: Any) -> str:
    """Validate debug mode and mode-specific runtime constraints."""
    debug_mode = resolve_debug_mode(args)
    return debug_mode


def validate_async_training_runner(args: Any) -> None:
    """Validate constraints that apply only to the async entrypoint."""
    debug_mode = resolve_debug_mode(args)
    if debug_mode == "train_only":
        raise ValueError(
            "train_async.py does not support debug_mode=train_only. "
            "Use python -m diffusionrl.train for train_only debug runs."
        )
    if bool(getattr(args.debug, "debug_save_intermediates", False)):
        raise ValueError(
            "train_async.py does not support debug_save_intermediates=true. "
            "Use python -m diffusionrl.train for rollout debug artifact capture."
        )

    rollout_topology = resolve_rollout_topology(args)
    if rollout_mode_is_colocated(rollout_topology.mode):
        raise ValueError("train_async.py requires rollout.mode='separate_rollout'.")
    if rollout_topology.training_actor_sampling_mode:
        raise ValueError(
            "train_async.py requires a dedicated rollout engine "
            "(for example rollout.service_engine='sglang')."
        )
    if int(getattr(args.rollout, "async_max_inflight", 1)) < 1:
        raise ValueError("async_max_inflight must be >= 1.")
    if args.rollout.update_weights_interval <= 0:
        raise ValueError("update_weights_interval must be > 0.")
    if args.ray.offload_train or args.ray.offload_rollout:
        raise ValueError(
            "train_async.py is incompatible with offload_train/offload_rollout. "
            "Set --ray.offload-train=false --ray.offload-rollout=false."
        )


def validate_rollout_topology_contract(args: Any) -> ResolvedRolloutTopology:
    """Validate rollout topology contract after strict topology resolution."""
    topology = resolve_rollout_topology(args)

    if rollout_mode_uses_service(topology.mode):
        if topology.service_engine is None:
            raise ValueError(
                "Dedicated rollout modes require rollout.service_engine to be set explicitly."
            )
        if not uses_dedicated_rollout_engine(topology.service_engine):
            raise ValueError(
                "rollout.mode in {separate_rollout,colocate_rollout} requires a dedicated rollout "
                f"service engine. Got rollout.service_engine={topology.service_engine!r}."
            )
        if getattr(args.rollout, "service_num_gpus", None) is None:
            raise ValueError(
                "Dedicated rollout services require rollout.service_num_gpus to be set explicitly."
            )
        return topology

    if topology.service_engine is not None and uses_dedicated_rollout_engine(topology.service_engine):
        raise ValueError(
            "rollout.mode='direct_rollout' cannot be combined with a dedicated rollout service engine. "
            f"Got rollout.service_engine={topology.service_engine!r}."
        )

    configured_direct_incompatible_fields = []
    direct_incompatible_values = {
        "rollout.service_num_gpus": getattr(args.rollout, "service_num_gpus", None),
        "rollout.engine_tp_size": getattr(args.rollout, "engine_tp_size", None),
        "rollout.engine_sp_size": getattr(args.rollout, "engine_sp_size", None),
        "rollout.service_require_memory_api": getattr(args.rollout, "service_require_memory_api", None),
        "rollout.service_transport_dtype": getattr(args.rollout, "service_transport_dtype", None),
        "rollout.service_transport_drop_decoded_videos": getattr(
            args.rollout, "service_transport_drop_decoded_videos", None
        ),
        "rollout.service_transport_log_payload_bytes": getattr(
            args.rollout, "service_transport_log_payload_bytes", None
        ),
        "rollout.sglang_local_mode": getattr(args.rollout, "sglang_local_mode", None),
        "rollout.sglang_verify_weight_checksum": getattr(
            args.rollout, "sglang_verify_weight_checksum", None
        ),
        "rollout.sglang_prompt_encoder_device": getattr(
            args.rollout, "sglang_prompt_encoder_device", None
        ),
        "rollout.sglang_prompt_encoder_dtype": getattr(
            args.rollout, "sglang_prompt_encoder_dtype", None
        ),
        "rollout.sglang_prompt_encoder_max_length": getattr(
            args.rollout, "sglang_prompt_encoder_max_length", None
        ),
        "rollout.sglang_kwargs": getattr(args.rollout, "sglang_kwargs", None),
    }
    for field_name, field_value in direct_incompatible_values.items():
        if field_value in (None, {}, "", False):
            continue
        configured_direct_incompatible_fields.append(field_name)
    if configured_direct_incompatible_fields:
        raise ValueError(
            "direct_rollout runs sampling on training actors, so dedicated rollout-service fields "
            f"must be unset. Remove: {', '.join(sorted(configured_direct_incompatible_fields))}."
        )

    return topology


def validate_sampling_basics(
    args: Any,
    *,
    rollout_service_engine: Optional[str],
) -> tuple[bool, str]:
    """Validate sampler kwargs payload and return validated sampling context."""
    if not isinstance(args.sampling.sampler_kwargs, dict):
        raise ValueError("sampling.sampler_kwargs must be a dict.")

    if args.sampling.sampler_kwargs:
        logger.info("sampling.sampler_kwargs configured with explicit sampler constructor overrides.")

    is_sglang_engine = rollout_service_engine == "sglang"
    logprob_source = resolve_logprob_source(args)
    return is_sglang_engine, logprob_source


def validate_algorithm_kwargs_payload(args: Any) -> None:
    """Validate algorithm_kwargs payload without mutating args."""
    resolve_algorithm_kwargs(args)


def validate_algorithm_path(args: Any) -> None:
    """Validate algorithm path resolution without mutating args."""
    resolve_algorithm_path(args)


def validate_train_backend_config(args: Any) -> None:
    """Validate train-backend selection and kwargs payload."""
    backend = resolve_train_backend_name(args)
    backend_path = getattr(args.training, "train_backend_path", None)
    supported = {"fsdp", "megatron", "veomni"}
    if backend not in supported and not backend_path:
        raise ValueError(
            f"Unsupported train_backend={backend!r}. "
            f"Expected one of {sorted(supported)} or provide --training.train-backend-path."
        )
    if backend == "megatron" and not backend_path:
        logger.warning(
            "train_backend=%s is currently a scaffold backend: launch/topology interfaces are wired, "
            "but runtime training flow is not fully implemented yet. "
            "Use train_backend_kwargs.actor_class_path to provide a Megatron-dedicated actor.",
            backend,
        )

    parsed = resolve_train_backend_kwargs(args)

    if backend == "veomni":
        mode = str(parsed.get("data_parallel_mode", "fsdp2") or "fsdp2").strip().lower()
        if mode == "ddp":
            raise ValueError(
                "train_backend=veomni in diffusionRL does not support data_parallel_mode='ddp'. "
                "Use data_parallel_mode='fsdp2'."
            )
        if mode != "fsdp2":
            raise ValueError(
                "train_backend=veomni in diffusionRL now targets FSDP2 only. "
                "Set data_parallel_mode='fsdp2' or omit this field."
            )

    if "num_actors" in parsed:
        raise ValueError(
            "training.train_backend_kwargs.num_actors is not supported. "
            "Training actor count is owned by ray.training_num_nodes × "
            "ray.training_num_gpus_per_node."
        )

    if backend == "megatron" and not backend_path and not str(parsed.get("actor_class_path", "")).strip():
        logger.warning(
            "train_backend=%s requires actor_class_path in train_backend_kwargs "
            "to launch a Megatron-specific training actor.",
            backend,
        )


def validate_training_actor_sampling_mode(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
    backend_capabilities: Optional[Mapping[str, Any]] = None,
) -> None:
    """Validate direct-sampling runtime compatibility."""
    if not training_actor_sampling_mode:
        return

    rollout_topology = resolve_rollout_topology(args)
    if rollout_mode_uses_service(rollout_topology.mode) or uses_dedicated_rollout_engine(rollout_topology.service_engine):
        raise ValueError(
            "Dedicated rollout service engines cannot use direct_rollout mode. "
            f"Got rollout.mode={rollout_topology.mode!r}, rollout.service_engine={rollout_topology.service_engine!r}."
        )

    if backend_capabilities is None:
        return

    backend_name = resolve_train_backend_name(args)
    if not bool(backend_capabilities.get("supports_training_actor_sampling", False)):
        raise ValueError(
            "rollout.mode=%r resolves to direct training-actor sampling, "
            "but train_backend=%r does not declare supports_training_actor_sampling=true."
            % (rollout_topology.mode, backend_name)
        )


def validate_replay_mode(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
    is_sglang_engine: bool,
    logprob_source: str,
) -> tuple[bool, bool, str]:
    """Validate replay flags for sglang and non-sglang engines."""
    replay_enabled = bool(getattr(args.sampling, "replay_log_probs", False))
    replay_guard = (
        (not training_actor_sampling_mode)
        and getattr(args.algorithm, "algorithm_type", "grpo") == "grpo"
    )

    if is_sglang_engine and replay_guard:
        if logprob_source == "replay" and not replay_enabled:
            raise ValueError(
                "rollout.service_engine='sglang' with logprob_source='replay' requires "
                "--sampling.replay-log-probs=true. Set it explicitly."
            )
        if logprob_source == "native" and replay_enabled:
            raise ValueError(
                "logprob_source='native' is incompatible with replay_log_probs=true. "
                "Set --sampling.replay-log-probs=false when using native log_prob mode."
            )

    if replay_enabled and not replay_guard:
        raise ValueError(
            "replay_log_probs=true is only valid for "
            "dedicated rollout services + algorithm_type='grpo'. "
            "Either disable replay_log_probs or adjust your config."
        )

    return replay_guard, replay_enabled, logprob_source


def validate_offload_and_colocate_config(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate offload switches and colocate fractions."""
    if training_actor_sampling_mode:
        if bool(args.ray.offload) or bool(args.ray.offload_train) or bool(args.ray.offload_rollout):
            raise ValueError(
                "direct_rollout uses training actors for sampling and cannot be combined with "
                "ray.offload / ray.offload_train / ray.offload_rollout."
            )
        return

    if rollout_mode_is_colocated(resolve_rollout_topology(args).mode):
        validate_colocate_fractions(args)


def validate_training_batch_geometry(args: Any) -> None:
    """Validate batch-geometry invariants using resolved training geometry."""
    update_batch_size = resolve_local_update_batch_size(args)
    micro_batch_size = resolve_local_micro_batch_size(args)
    num_updates_per_local_batch = resolve_num_updates_per_local_batch(args)
    local_batch_size = resolve_nominal_local_training_batch_size(args)
    global_batch_size = resolve_global_rollout_batch_size(args)
    dp_size = resolve_training_dp_size(args)
    resolved_prompts_per_rollout = resolve_prompts_per_rollout(args)

    if micro_batch_size > update_batch_size:
        raise ValueError(
            "training.local_micro_batch_size must be <= "
            "training.local_update_batch_size. "
            f"Got micro_batch_size={micro_batch_size}, "
            f"update_batch_size={update_batch_size}."
        )
    if local_batch_size != update_batch_size * num_updates_per_local_batch:
        raise ValueError(
            "Resolved local training batch size does not match the training geometry. "
            f"Got local_batch_size={local_batch_size}, "
            f"local_update_batch_size={update_batch_size}, "
            f"num_updates_per_local_batch={num_updates_per_local_batch}."
        )
    if global_batch_size != local_batch_size * dp_size:
        raise ValueError(
            "Resolved global rollout batch size does not match local_batch_size * dp_size. "
            f"Got global_batch_size={global_batch_size}, local_batch_size={local_batch_size}, "
            f"dp_size={dp_size}."
        )
    if resolved_prompts_per_rollout < 1:
        raise ValueError(
            "Resolved prompts_per_rollout must be >= 1. "
            f"Got: {resolved_prompts_per_rollout}."
        )


def validate_training_misc(args: Any) -> None:
    """Validate misc training knobs that affect downstream components."""
    if args.algorithm.adv_normalization not in {"global", "group"}:
        raise ValueError(
            "algorithm.adv_normalization must be 'global' or 'group'. "
            f"Got: {args.algorithm.adv_normalization!r}."
        )

    resolve_lora_target_modules(args.training.lora_target_modules)

    if bool(getattr(args.rollout, "rollout_buffer_reassemble_by_group", False)):
        configured_group_size = getattr(args.rollout, "rollout_buffer_group_size", None)
        if configured_group_size is None:
            raise ValueError(
                "rollout.rollout_buffer_reassemble_by_group=true requires rollout.rollout_buffer_group_size "
                "to be set explicitly. Implicit binding to algorithm.samples_per_prompt was removed."
            )

    validate_training_batch_geometry(args)


def validate_direct_sampling_batch_geometry(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate prompt-batch splitting for training-actor direct sampling."""
    max_samples_per_request = getattr(args.sampling, "max_samples_per_request", None)
    if max_samples_per_request is None:
        return

    if not training_actor_sampling_mode:
        raise ValueError(
            "sampling.max_samples_per_request is only valid when "
            "sampling runs directly on training actors."
        )

    max_samples_per_request = int(max_samples_per_request)
    prompts_per_rollout = int(resolve_prompts_per_rollout(args))
    if max_samples_per_request < 1:
        raise ValueError("sampling.max_samples_per_request must be >= 1.")
    if prompts_per_rollout < 1:
        raise ValueError("Resolved prompts_per_rollout must be >= 1.")


def validate_weight_sync(
    args: Any,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate explicit weight-sync protocol against runtime topology."""
    rollout_service_engine = resolve_rollout_topology(args).service_engine
    resolved_mode = str(getattr(args.sync, "protocol", "") or "").strip().lower()
    if not resolved_mode:
        raise ValueError(
            "sync.protocol must be set explicitly before validate_weight_sync()."
        )
    if training_actor_sampling_mode:
        if resolved_mode != "disabled":
            raise ValueError(
                "direct training-actor sampling requires sync.protocol='disabled'. "
                f"Got sync.protocol={resolved_mode!r}."
            )
        return
    if resolved_mode == "disabled":
        raise ValueError(
            "sync.protocol='disabled' is only valid when sampling runs directly on training actors. "
            f"Got rollout.service_engine={rollout_service_engine!r}."
        )
    is_multi_node = (
        int(getattr(args.ray, "rollout_num_nodes", 1)) > 1
        or int(getattr(args.ray, "training_num_nodes", 1)) > 1
        or int(getattr(args.reward, "reward_dedicated_num_nodes", 0)) > 1
    )
    if resolved_mode in {"tensor_payload", "nccl_broadcast"} and rollout_service_engine != "sglang":
        raise ValueError(
            "sync.protocol in {tensor_payload,nccl_broadcast} currently requires "
            "rollout.service_engine='sglang'. "
            f"Got rollout.service_engine={rollout_service_engine!r}."
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

    rollout_topology = resolve_rollout_topology(args)
    rollout_gpus = resolve_rollout_gpus_per_actor(args)
    rollout_gpu_pool_size = resolve_rollout_gpu_pool_size(args)
    if not rollout_mode_uses_service(rollout_topology.mode):
        raise ValueError(
            "Dedicated rollout actor layout validation only applies to dedicated rollout engines. "
            f"Got rollout.mode={rollout_topology.mode!r}."
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
    is_sglang_engine = rollout_topology.service_engine == "sglang"
    if rollout_gpus > 1 and rollout_mode_is_colocated(rollout_topology.mode) and not is_sglang_engine:
        raise ValueError(
            "colocate_rollout with multi-GPU rollout actors is only supported "
            "for rollout.service_engine='sglang'."
        )
    if (
        rollout_gpus > 1
        and rollout_mode_is_colocated(rollout_topology.mode)
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

    service_engine = resolve_rollout_topology(args).service_engine
    if not service_engine:
        raise ValueError(
            "Dedicated rollout validation requires rollout.service_engine to be set explicitly. "
            "Run validate_args() before resolving dedicated rollout engine capabilities."
        )
    engine_caps = resolve_engine_capabilities(service_engine)

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
    rollout_topology = resolve_rollout_topology(args)
    if (
        not training_actor_sampling_mode
        and rollout_topology.service_engine == "sglang"
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

__all__ = [
    "apply_model_config_hook",
    "validate_async_training_runner",
    "validate_algorithm_kwargs_payload",
    "validate_algorithm_path",
    "validate_colocate_fractions",
    "validate_debug_config",
    "validate_direct_sampling_batch_geometry",
    "validate_dotpath",
    "validate_grouped_configs",
    "validate_dynamic_dotpaths",
    "validate_model_runtime_contract",
    "validate_offload_and_colocate_config",
    "validate_replay_mode",
    "validate_reward_config",
    "validate_reward_and_rollout_buffer_config",
    "validate_rollout_layout",
    "validate_rollout_topology_contract",
    "validate_sampling_basics",
    "validate_train_backend_config",
    "validate_training_actor_sampling_mode",
    "validate_training_batch_geometry",
    "validate_training_misc",
    "validate_nft_sampling_contract",
    "validate_resolved_engine_algorithm_contract",
    "validate_runtime_mode_constraints",
    "validate_weight_sync",
]
