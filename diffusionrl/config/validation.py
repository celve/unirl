"""Configuration validation helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from diffusionrl.runtime.contracts import (
    resolve_engine_capabilities,
    resolve_sampling_requirements,
)
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)


def _trace_normalize_change(
    args: Any,
    key: str,
    before: Any,
    after: Any,
    *,
    source: str,
) -> None:
    """Emit normalize trace through callback set by arguments.validate_args()."""
    if before == after:
        return
    callback = getattr(args, "_normalize_trace_callback", None)
    if callable(callback):
        callback(key, before, after, source=source)


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
    reward_execution_mode = str(
        getattr(args.reward, "reward_execution_mode", "manager") or "manager"
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

    if reward_execution_mode == "rollout":
        if has_http_reward:
            raise ValueError(
                "reward_execution_mode='rollout' cannot be combined with HTTP reward service. "
                "Use reward_execution_mode='manager' for HTTP reward."
            )
        if has_dedicated_reward_pool:
            raise ValueError(
                "reward_execution_mode='rollout' cannot be combined with dedicated reward actors. "
                "Use reward_execution_mode='manager' for reward_dedicated_* modes."
            )

    uses_local_same_process_reward = (
        reward_execution_mode == "manager"
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

    if reward_execution_mode == "rollout":
        logger.info(
            "Reward mode: rollout-local worker (local_reward_device=%s)",
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
    args.algorithm.validate()
    args.training.validate()
    args.rollout.validate()
    args.debug.validate()


def validate_dynamic_dotpaths(args: Any) -> None:
    """Validate configured runtime extension dotpaths."""
    validate_dotpath(args.model.model_path, label="model")
    validate_dotpath(args.sampling.sampler_path, label="sampler")
    validate_dotpath(args.algorithm.algorithm_path, label="algorithm")
    validate_dotpath(args.data_source_path, label="data_source")
    if getattr(args.training, "train_backend_path", None):
        validate_dotpath(args.training.train_backend_path, label="train_backend")
    if getattr(args.sampling, "replay_sampler_path", None):
        validate_dotpath(args.sampling.replay_sampler_path, label="replay_sampler")
    if getattr(args.rollout, "rollout_pipeline_path", None):
        validate_dotpath(args.rollout.rollout_pipeline_path, label="rollout_pipeline")
    if getattr(args.algorithm, "loss_path", None):
        validate_dotpath(args.algorithm.loss_path, label="loss")
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
    sampler_engine_type = str(getattr(args.sampling, "sampler_engine_type", "") or "fsdp").lower()
    if sampler_engine_type == "fsdp":
        return args.sampling.fsdp_num_gpus
    if sampler_engine_type == "sglang":
        engine_kwargs = getattr(args.sampling, "engine_kwargs", {})
        if not isinstance(engine_kwargs, dict):
            engine_kwargs = {}
        num_gpus = engine_kwargs.get("num_gpus")
        if num_gpus is None:
            # Keep this consistent with RolloutActorGroup factory:
            # tp_size is treated as the per-engine GPU count when num_gpus
            # is not explicitly provided.
            num_gpus = engine_kwargs.get("tp_size", getattr(args.sampling, "tp_size", 1))
        if num_gpus is None:
            # Keep behavior explicit: default single-GPU engine unless user opts in.
            return 1
        try:
            resolved = int(num_gpus)
        except (TypeError, ValueError):
            return 1
        return max(1, resolved)
    return 1


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
    if int(getattr(args.rollout, "rollout_buffer_dispatch_groups", 0)) < 0:
        raise ValueError(
            "rollout_buffer_dispatch_groups must be >= 0 "
            f"(0 means prompts_per_batch), got: {args.rollout.rollout_buffer_dispatch_groups}"
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


def validate_rollout_layout(
    args: Any,
    *,
    training_actor_direct_sampling: bool,
) -> None:
    """Validate rollout actor GPU layout and colocate constraints."""
    if training_actor_direct_sampling:
        return

    rollout_gpus = get_rollout_gpus_per_actor(args)
    is_sglang_engine = str(getattr(args.sampling, "sampler_engine_type", "")).lower() == "sglang"
    if rollout_gpus > 1 and args.ray.colocate_rollout_training and not is_sglang_engine:
        raise ValueError(
            "colocate_rollout_training=True with multi-GPU rollout actors is only supported "
            "for sampler_engine_type='sglang'."
        )
    if (
        rollout_gpus > 1
        and args.ray.colocate_rollout_training
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


def validate_model_specific_logic(args: Any, *, model_cls: Any) -> None:
    """Run model-specific runtime validation."""
    if args.model.model_type != "flux" and args.sampling.sde_type.startswith("flux_"):
        raise ValueError(
            f"sde_type '{args.sampling.sde_type}' is only valid for model_type='flux'"
        )

    model_validate_fn = getattr(model_cls, "validate_config", None)
    if callable(model_validate_fn):
        before_flat = args.to_flat_dict() if hasattr(args, "to_flat_dict") else None
        model_validate_fn(args)
        if isinstance(before_flat, dict) and hasattr(args, "to_flat_dict"):
            after_flat = args.to_flat_dict()
            if isinstance(after_flat, dict):
                for key in sorted(set(before_flat) | set(after_flat)):
                    _trace_normalize_change(
                        args,
                        key,
                        before_flat.get(key),
                        after_flat.get(key),
                        source="model_validate_config",
                    )

    if args.algorithm.loss_type == "nft":
        # DiffusionNFT reproduction contract:
        # - rollout samples from old adapter
        # - deterministic solver (dpm2)
        old_adapter_name = "old"
        loss_kwargs = getattr(args.algorithm, "loss_kwargs", {})
        parsed: Dict[str, Any] = {}
        if loss_kwargs is None:
            parsed = {}
        elif isinstance(loss_kwargs, dict):
            parsed = dict(loss_kwargs)
        else:
            raise ValueError(
                "algorithm.loss_kwargs must be a dict after validate_loss_kwargs(), "
                f"got: {type(loss_kwargs).__name__}"
            )
        if parsed:
            old_adapter_name = str(parsed.get("old_adapter_name", old_adapter_name) or old_adapter_name)

        if not args.sampling.sampling_adapter:
            raise ValueError(
                "loss_type='nft' requires --sampling.sampling-adapter to be set "
                f"(must match old_adapter_name={old_adapter_name!r})."
            )
        if str(args.sampling.sampling_adapter) != old_adapter_name:
            raise ValueError(
                "loss_type='nft' requires rollout sampling from the old adapter. "
                f"Set --sampling.sampling-adapter {old_adapter_name!r}, "
                f"got {args.sampling.sampling_adapter!r}."
            )
        sde_type = str(args.sampling.sde_type)
        eta = float(getattr(args.sampling, "eta", 1.0))
        if sde_type != "dpm2" and not (sde_type == "sde" and eta == 0.0):
            raise ValueError(
                "loss_type='nft' targets DiffusionNFT deterministic sampling. "
                "Set --sampling.sde-type dpm2, or use --sampling.sde-type sde "
                f"with --sampling.eta 0.0 (ODE mode). "
                f"Got sde_type={sde_type!r}, eta={eta}."
            )


def validate_loss_kwargs(args: Any) -> None:
    """Validate and normalize loss_kwargs into a dict."""
    raw = getattr(args.algorithm, "loss_kwargs", "")
    if raw is None:
        args.algorithm.loss_kwargs = {}
        return
    if isinstance(raw, dict):
        args.algorithm.loss_kwargs = dict(raw)
        return
    if not isinstance(raw, str):
        raise ValueError(
            "loss_kwargs must be a JSON object (YAML mapping) "
            f"or JSON object string, got: {type(raw).__name__}"
        )

    text = raw.strip()
    if not text:
        args.algorithm.loss_kwargs = {}
        return

    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid loss_kwargs: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("loss_kwargs must decode to a JSON object.")
    args.algorithm.loss_kwargs = dict(parsed)


def validate_resolved_engine_loss_contract(
    args: Any,
    *,
    training_actor_direct_sampling: bool,
    is_sglang_engine: bool,
    replay_guard: bool,
    sglang_logprob_mode: str,
) -> None:
    """Validate engine/loss compatibility using resolved capabilities."""
    if training_actor_direct_sampling:
        return

    sampler_engine_type = str(getattr(args.sampling, "sampler_engine_type", "") or "")
    engine_caps = resolve_engine_capabilities(engine_type=args.sampling.sampler_engine_type)

    allow_replay = (
        bool(getattr(args.sampling, "replay_log_probs", False))
        and getattr(args.algorithm, "loss_type", "grpo") == "grpo"
    )
    if allow_replay:
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)
        logger.warning(
            "replay_log_probs=true enabled: allowing %s+GRPO with "
            "training-side old-log-prob replay (experimental path).",
            sampler_engine_type,
        )

    if is_sglang_engine and replay_guard and sglang_logprob_mode == "native":
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)

    required = resolve_sampling_requirements(args)
    if is_sglang_engine and str(getattr(args.model, "model_type", "") or "").strip().lower() == "sd3":
        # Current sglang-diffusion SD3 path may return only final samples without
        # trajectory latents. Fail fast for trajectory-dependent losses.
        if bool(required.requires_trajectory):
            raise ValueError(
                "sampler_engine_type='sglang' with model_type='sd3' currently does not "
                "provide trajectory_latents required by trajectory-based losses "
                "(e.g. GRPO/MixGRPO). Use sampler_engine_type='fsdp', or use loss_type='nft' "
                "when running SD3 with sglang."
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
            f"Engine capability mismatch for loss_type={args.algorithm.loss_type}: "
            f"sampler_engine_type={sampler_engine_type} lacks {missing}. "
            f"engine_capabilities={engine_caps}, required={required_dict}. "
            "Use a compatible engine/loss pair (for example: fsdp+grpo or fsdp+nft)."
        )


def validate_runtime_mode_constraints(
    args: Any,
    *,
    training_actor_direct_sampling: bool,
    model_cls: Any,
) -> None:
    """Validate runtime mode constraints and mutually-exclusive switches."""
    if (
        not training_actor_direct_sampling
        and str(getattr(args.sampling, "sampler_engine_type", "")).lower() == "sglang"
    ):
        supports_sglang = getattr(model_cls, "supports_sglang_prompt_mode", None)
        if not callable(supports_sglang):
            raise ValueError(
                f"sampler_engine_type='sglang' requires model {args.model.model_path!r} "
                "to define classmethod supports_sglang_prompt_mode()."
            )
        if not supports_sglang():
            raise ValueError(
                f"sampler_engine_type='sglang' is not supported by model {args.model.model_path!r}. "
                "The model must implement classmethod supports_sglang_prompt_mode() returning True."
            )

    if getattr(args.rollout, "async_pipeline", False):
        if args.ray.colocate_rollout_training:
            raise ValueError("async_pipeline requires separate mode (colocate_rollout_training=False).")
        if training_actor_direct_sampling:
            raise ValueError("async_pipeline currently requires training_actor_direct_sampling=false.")
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
    "validate_colocate_fractions",
    "get_rollout_gpus_per_actor",
    "validate_dotpath",
    "validate_grouped_configs",
    "validate_dynamic_dotpaths",
    "validate_reward_config",
    "validate_reward_and_rollout_buffer_config",
    "validate_rollout_layout",
    "validate_model_specific_logic",
    "validate_loss_kwargs",
    "validate_resolved_engine_loss_contract",
    "validate_runtime_mode_constraints",
]
