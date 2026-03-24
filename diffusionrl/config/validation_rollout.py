"""Rollout topology, backend, and dedicated-rollout validation helpers."""

from __future__ import annotations

from dataclasses import MISSING, fields as dataclass_fields
import logging
from typing import Any, Mapping, Optional

from diffusionrl.algorithms.construction import build_algorithm_kwargs, resolve_algorithm_path
from diffusionrl.config.resolution import (
    ResolvedRolloutModeInfo,
    ResolvedRolloutTopology,
    rollout_mode_is_colocated,
    rollout_mode_uses_service,
    derive_rollout_gpu_pool_size,
    derive_rollout_actor_gpu_count,
    derive_rollout_topology,
)
from diffusionrl.training.backends.factory import supported_train_backends
from diffusionrl.types.engine import uses_dedicated_rollout_engine

from .validation_common import (
    ENV_REPO_ROOT,
    is_probably_local_weight_sync_dir,
    repo_root,
    validate_colocate_fractions,
)

logger = logging.getLogger(__name__)


def _resolve_dataclass_field_default(field_info: Any) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return MISSING


def _collect_direct_rollout_incompatible_fields(rollout_topology_config: Any) -> list[str]:
    incompatible: list[str] = []
    for field_info in dataclass_fields(type(rollout_topology_config)):
        if field_info.name in {"mode", "service_engine"}:
            continue
        field_value = getattr(rollout_topology_config, field_info.name)
        field_default = _resolve_dataclass_field_default(field_info)
        if field_default is not MISSING and field_value == field_default:
            continue
        if field_value in (None, "", False):
            continue
        if isinstance(field_value, dict) and not field_value:
            continue
        incompatible.append(f"rollout.topology.{field_info.name}")
    return incompatible


def _format_rollout_mode_state(rollout_state: Mapping[str, Any]) -> str:
    args = rollout_state["args"]
    rollout_mode_info: ResolvedRolloutModeInfo = rollout_state["rollout_mode_info"]
    rollout_topology = rollout_mode_info.rollout_topology
    training_actor_sampling_mode = bool(rollout_mode_info.training_actor_sampling_mode)
    replay_enabled = bool(rollout_mode_info.replay_enabled)
    sync_protocol = str(rollout_mode_info.sync_protocol)
    max_samples_per_request = rollout_mode_info.max_samples_per_request
    algorithm_type = str(rollout_mode_info.algorithm_type)
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
        raise ValueError("train_async.py requires rollout.topology.mode='separate_rollout'.")
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
    rollout_topology: ResolvedRolloutTopology,
) -> ResolvedRolloutTopology:
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
                "rollout.topology.mode in {separate_rollout,colocate_rollout} requires a dedicated rollout "
                f"service engine. Got rollout.topology.service_engine={topology.service_engine!r}."
            )
        if rollout_topology_config.service_num_gpus is None:
            raise ValueError(
                "Dedicated rollout services require rollout.topology.service_num_gpus to be set explicitly."
            )
        return topology

    if topology.service_engine is not None and uses_dedicated_rollout_engine(topology.service_engine):
        raise ValueError(
            "rollout.topology.mode='direct_rollout' cannot be combined with a dedicated rollout service engine. "
            f"Got rollout.topology.service_engine={topology.service_engine!r}."
        )
    if topology.service_engine is not None:
        raise ValueError(
            "direct_rollout is the only public direct-sampling selector. "
            "Leave rollout.topology.service_engine unset in direct_rollout mode."
        )

    configured_direct_incompatible_fields = _collect_direct_rollout_incompatible_fields(
        rollout_topology_config
    )
    if configured_direct_incompatible_fields:
        raise ValueError(
            "direct_rollout runs sampling on training actors, so dedicated rollout-service fields "
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


def validate_training_actor_sampling_mode(
    *,
    rollout_mode_info: ResolvedRolloutModeInfo,
    backend_capabilities: Mapping[str, Any],
    backend_name: str,
) -> None:
    """Validate direct-sampling topology compatibility."""
    if not rollout_mode_info.training_actor_sampling_mode:
        return

    resolved_topology = rollout_mode_info.rollout_topology
    if rollout_mode_uses_service(resolved_topology.mode) or uses_dedicated_rollout_engine(resolved_topology.service_engine):
        raise ValueError(
            "Dedicated rollout service engines cannot use direct_rollout mode. "
            f"Got rollout.topology.mode={resolved_topology.mode!r}, "
            f"rollout.topology.service_engine={resolved_topology.service_engine!r}."
        )

    if not bool(backend_capabilities.get("supports_training_actor_sampling", False)):
        raise ValueError(
            "rollout.topology.mode=%r resolves to direct training-actor sampling, "
            "but train_backend=%r does not declare supports_training_actor_sampling=true."
            % (resolved_topology.mode, backend_name)
        )


def validate_replay_mode(*, rollout_mode_info: ResolvedRolloutModeInfo) -> None:
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
    rollout_mode_info: ResolvedRolloutModeInfo,
) -> None:
    """Validate offload switches and colocate fractions."""
    if rollout_mode_info.training_actor_sampling_mode:
        if bool(args.ray.offload_train) or bool(args.ray.offload_rollout):
            raise ValueError(
                "direct_rollout uses training actors for sampling and cannot be combined with "
                "ray.offload_train / ray.offload_rollout."
            )
        return

    if rollout_mode_is_colocated(rollout_mode_info.rollout_topology.mode):
        validate_colocate_fractions(args)


def validate_rollout_mode(
    args: Any,
    *,
    rollout_mode_info: ResolvedRolloutModeInfo,
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
                        {
                            "args": args,
                            "rollout_mode_info": rollout_mode_info,
                        }
                    ),
                    f"Cause: {exc}",
                ]
            )
        ) from exc


def validate_direct_sampling_batch_geometry(
    args: Any,
    *,
    rollout_mode_info: ResolvedRolloutModeInfo,
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
    rollout_mode_info: ResolvedRolloutModeInfo,
) -> None:
    """Validate explicit weight-sync protocol against rollout topology."""
    rollout_service_engine = rollout_mode_info.rollout_topology.service_engine
    resolved_mode = rollout_mode_info.sync_protocol
    if not resolved_mode:
        raise ValueError(
            "sync.protocol must be set explicitly before validate_weight_sync()."
        )
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
    rollout_mode_info: ResolvedRolloutModeInfo,
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
    if rollout_topology.service_engine != "sglang":
        raise ValueError(
            f"Unsupported dedicated rollout engine for actor layout: {rollout_topology.service_engine!r}. "
            "Expected: sglang."
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
            "colocate_rollout with multi-GPU rollout actors is only supported "
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


__all__ = [
    "validate_algorithm_kwargs_payload",
    "validate_algorithm_path",
    "validate_async_training_runner",
    "validate_direct_sampling_batch_geometry",
    "validate_offload_and_colocate_config",
    "validate_replay_mode",
    "validate_rollout_layout",
    "validate_rollout_mode",
    "validate_rollout_topology_contract",
    "validate_train_backend_config",
    "validate_training_actor_sampling_mode",
    "validate_weight_sync",
]
