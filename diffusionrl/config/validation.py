"""Configuration validation helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from diffusionrl.samplers.engine import get_engine_class_path
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)

DEFAULT_LOSS_TYPE_REQUIREMENTS: Dict[str, Dict[str, bool]] = {
    "grpo": {
        "requires_trajectory": True,
        "requires_log_prob": True,
        "requires_embeddings": True,
    },
    "nft": {
        "requires_trajectory": False,
        "requires_log_prob": False,
        "requires_embeddings": True,
    },
}


def resolve_loss_type_requirements(
    *,
    loss_type: str,
    loss_type_requirements: Dict[str, Dict[str, bool]],
) -> Dict[str, bool]:
    """Resolve required sampling capabilities for the selected loss type."""
    requirements = loss_type_requirements.get(loss_type)
    if requirements is None:
        raise ValueError(
            f"Unsupported loss_type={loss_type}. "
            f"Expected one of: {sorted(loss_type_requirements.keys())}. "
            "Use --loss-type grpo for standard GRPO training or --loss-type nft for Negative Fine-Tuning."
        )
    return dict(requirements)


def validate_engine_loss_contract(
    *,
    args,
    training_actor_direct_sampling: bool,
    engine_capabilities: Dict[str, bool],
    loss_type_requirements: Dict[str, Dict[str, bool]],
) -> None:
    """Fail-fast guard for engine/loss capability contract violations."""
    sampler_engine_type = str(getattr(args, "sampler_engine_type", "") or "")
    capabilities = dict(engine_capabilities)

    if training_actor_direct_sampling:
        return

    allow_replay = (
        bool(getattr(args, "replay_log_probs", False))
        and getattr(args, "loss_type", "grpo") == "grpo"
    )
    if allow_replay:
        capabilities["requires_log_prob"] = True
        capabilities["requires_embeddings"] = True
        logger.warning(
            "replay_log_probs=true enabled: allowing %s+GRPO with "
            "training-side old-log-prob replay (experimental path).",
            sampler_engine_type,
        )

    required = resolve_loss_type_requirements(
        loss_type=getattr(args, "loss_type", "grpo"),
        loss_type_requirements=loss_type_requirements,
    )
    missing = [
        key
        for key, needed in required.items()
        if bool(needed) and not bool(capabilities.get(key, False))
    ]
    if missing:
        raise ValueError(
            f"Engine capability mismatch for loss_type={args.loss_type}: "
            f"sampler_engine_type={sampler_engine_type} lacks {missing}. "
            f"engine_capabilities={capabilities}, required={required}. "
            "Use a compatible engine/loss pair (for example: fsdp+grpo or fsdp+nft)."
        )


def resolve_runtime_loss_type_requirements(
    args: Any,
    *,
    default_loss_type_requirements: Dict[str, Dict[str, bool]] = DEFAULT_LOSS_TYPE_REQUIREMENTS,
) -> Dict[str, Dict[str, bool]]:
    """Resolve loss requirements with classmethod declaration taking precedence."""
    requirements = dict(default_loss_type_requirements)
    loss_type = str(getattr(args, "loss_type", "grpo"))
    loss_cls = None

    try:
        from diffusionrl.losses import LOSS_REGISTRY

        loss_cls = LOSS_REGISTRY.get(loss_type)
    except Exception:
        loss_cls = None

    if loss_cls is None and getattr(args, "loss_path", None):
        loss_cls = load_function(args.loss_path)

    declared = getattr(loss_cls, "declared_requirements", None) if loss_cls is not None else None
    if callable(declared):
        requirements[loss_type] = dict(declared())
    elif loss_cls is not None:
        # Fallback for custom loss classes that do not declare requirements.
        # Use conservative GRPO-like contract to avoid under-validating engines.
        requirements[loss_type] = dict(default_loss_type_requirements["grpo"])
    return requirements


def resolve_engine_capabilities(*, engine_type: str) -> Dict[str, bool]:
    """Resolve engine capabilities from engine class declaration."""
    engine_path = get_engine_class_path(engine_type)
    engine_cls = load_function(engine_path)
    declared = getattr(engine_cls, "declared_capabilities", None)
    if not callable(declared):
        raise ValueError(
            f"Engine class {engine_path} must define classmethod declared_capabilities()."
        )
    return dict(declared())


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


def repo_root(*, env_repo_root: str) -> str:
    """Resolve repository root from environment override or package-relative path."""
    env_root = os.getenv(env_repo_root)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    # validation.py lives at diffusionrl/config/validation.py.
    # Three levels up resolves to workspace repo root.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def resolve_repo_relative_path(path: str, root: str) -> str:
    """Resolve path relative to repository root unless absolute."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(root, expanded))


def looks_like_local_path(path: str, root: str) -> bool:
    """Best-effort check whether a value should be interpreted as local path."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return True
    if any(
        path.startswith(prefix)
        for prefix in ("./", "../", "~", "data/", "models/", "outputs/", "shared_models/")
    ):
        return True
    if os.path.exists(expanded):
        return True
    if os.path.exists(os.path.join(root, expanded)):
        return True
    if path.count("/") >= 2:
        return True
    if path.endswith((".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".json", ".txt")):
        return True
    return False


def normalize_repo_relative_paths(
    args,
    *,
    env_repo_root: str,
    env_data_root: str,
    env_model_root: str,
    local_to_hf_fallback: Dict[str, str],
) -> None:
    """Normalize configured paths and apply local-model fallback to HF ids."""
    root = repo_root(env_repo_root=env_repo_root)
    data_root_env = os.getenv(env_data_root)
    model_root_env = os.getenv(env_model_root)

    for field_name in ("output_dir", "logging_dir", "weight_sync_dir", "resume_from_checkpoint"):
        value = getattr(args, field_name, None)
        if isinstance(value, str) and value:
            setattr(args, field_name, resolve_repo_relative_path(value, root))

    data_path = getattr(args, "data_path", None)
    if isinstance(data_path, str) and data_path:
        if data_root_env and not os.path.isabs(os.path.expanduser(data_path)):
            trimmed = data_path[5:] if data_path.startswith("data/") else data_path
            args.data_path = os.path.abspath(
                os.path.join(os.path.expanduser(data_root_env), trimmed)
            )
        else:
            args.data_path = resolve_repo_relative_path(data_path, root)

    for field_name in ("pretrained_model_saved_path", "vae_saved_path", "text_encoder_path", "reward_model_saved_path"):
        value = getattr(args, field_name, None)
        if not isinstance(value, str) or not value:
            continue
        if not looks_like_local_path(value, root):
            continue
        if model_root_env and not os.path.isabs(os.path.expanduser(value)):
            trimmed = value[7:] if value.startswith("models/") else value
            setattr(
                args,
                field_name,
                os.path.abspath(os.path.join(os.path.expanduser(model_root_env), trimmed)),
            )
        else:
            setattr(args, field_name, resolve_repo_relative_path(value, root))

    resolved = getattr(args, "pretrained_model_saved_path", "")
    if resolved and not os.path.exists(resolved):
        for local_prefix, hf_id in local_to_hf_fallback.items():
            abs_local = os.path.join(root, local_prefix)
            if resolved == abs_local or resolved.endswith("/" + local_prefix):
                logger.info(
                    "Local model not found at %s, falling back to HF: %s",
                    resolved,
                    hf_id,
                )
                args.pretrained_model_saved_path = hf_id
                break


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


def validate_reward_config(args) -> None:
    """Validate reward pool/source configuration consistency."""
    if args.reward_dedicated_gpus_per_actor > 1 and args.reward_dedicated_num_gpus > 0:
        if args.reward_dedicated_num_gpus < args.reward_dedicated_gpus_per_actor:
            raise ValueError(
                f"reward_dedicated_num_gpus ({args.reward_dedicated_num_gpus}) must be >= "
                f"reward_dedicated_gpus_per_actor ({args.reward_dedicated_gpus_per_actor})"
            )
        if args.reward_dedicated_num_gpus % args.reward_dedicated_gpus_per_actor != 0:
            raise ValueError(
                f"reward_dedicated_num_gpus ({args.reward_dedicated_num_gpus}) must be divisible by "
                f"reward_dedicated_gpus_per_actor ({args.reward_dedicated_gpus_per_actor})"
            )

    if args.reward_dedicated_num_nodes > 0 and args.reward_dedicated_num_gpus_per_node <= 0:
        raise ValueError(
            "reward_dedicated_num_gpus_per_node must be > 0 when reward_dedicated_num_nodes > 0"
        )

    if args.reward_dedicated_num_gpus > 0 and args.reward_dedicated_num_nodes > 0:
        raise ValueError(
            "reward_dedicated_num_gpus and reward_dedicated_num_nodes are mutually exclusive. "
            "Use either total dedicated GPUs, or nodes * gpus_per_node."
        )

    has_dedicated_reward_pool = (
        args.reward_dedicated_num_gpus > 0 or args.reward_dedicated_num_nodes > 0
    )
    has_http_reward = bool(
        getattr(args, "use_http_reward", False)
        or getattr(args, "reward_service_url", None)
        or getattr(args, "reward_service_urls", None)
    )
    local_reward_device = str(
        getattr(args, "local_reward_device", "cpu") or "cpu"
    ).strip().lower()
    allow_local_reward_cuda_contention = bool(
        getattr(args, "allow_local_reward_cuda_contention", False)
    )

    if args.use_http_reward and not (
        getattr(args, "reward_service_url", None)
        or getattr(args, "reward_service_urls", None)
    ):
        raise ValueError(
            "use_http_reward=true requires reward_service_url or reward_service_urls."
        )

    uses_local_same_process_reward = not has_http_reward and not has_dedicated_reward_pool
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

    if has_http_reward:
        logger.info("Reward mode: HTTP (external service)")
    elif has_dedicated_reward_pool:
        total_gpus = args.reward_dedicated_num_gpus
        if args.reward_dedicated_num_nodes > 0:
            total_gpus = args.reward_dedicated_num_nodes * args.reward_dedicated_num_gpus_per_node
        num_actors = total_gpus // args.reward_dedicated_gpus_per_actor
        logger.info(
            f"Reward mode: Independent GPU ({total_gpus} GPUs, "
            f"{num_actors} actors, {args.reward_dedicated_gpus_per_actor} GPUs/actor)"
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


def validate_dynamic_dotpaths(args: Any) -> None:
    """Validate configured runtime extension dotpaths."""
    validate_dotpath(args.model_path, label="model")
    validate_dotpath(args.sampler_path, label="sampler")
    validate_dotpath(args.algorithm_path, label="algorithm")
    validate_dotpath(args.data_source_path, label="data_source")
    if getattr(args, "train_backend_path", None):
        validate_dotpath(args.train_backend_path, label="train_backend")
    if getattr(args, "replay_sampler_path", None):
        validate_dotpath(args.replay_sampler_path, label="replay_sampler")
    if getattr(args, "rollout_pipeline_path", None):
        validate_dotpath(args.rollout_pipeline_path, label="rollout_pipeline")
    if getattr(args, "loss_path", None):
        validate_dotpath(args.loss_path, label="loss")
    rollout_buffer_plugin_paths = getattr(args, "rollout_buffer_plugin_paths", "") or ""
    if isinstance(rollout_buffer_plugin_paths, str):
        for plugin_path in [part.strip() for part in rollout_buffer_plugin_paths.split(",") if part.strip()]:
            validate_dotpath(plugin_path, label="rollout_buffer_plugin")


def validate_colocate_fractions(args) -> None:
    """Validate colocate GPU fraction bounds."""
    if args.colocate_training_gpu_fraction <= 0 or args.colocate_rollout_gpu_fraction <= 0:
        raise ValueError(
            "colocate_training_gpu_fraction and colocate_rollout_gpu_fraction must be > 0"
        )
    if args.colocate_training_gpu_fraction + args.colocate_rollout_gpu_fraction > 1.0:
        raise ValueError(
            "colocate_training_gpu_fraction + colocate_rollout_gpu_fraction must be <= 1.0"
        )


def get_rollout_gpus_per_actor(args) -> int:
    """Resolve GPUs per rollout actor based on sampler engine and engine config."""
    sampler_engine_type = str(getattr(args, "sampler_engine_type", "") or "fsdp").lower()
    if sampler_engine_type == "fsdp":
        return args.fsdp_num_gpus
    if sampler_engine_type == "sglang":
        engine_kwargs = getattr(args, "engine_kwargs", {})
        if not isinstance(engine_kwargs, dict):
            engine_kwargs = {}
        num_gpus = engine_kwargs.get("num_gpus")
        if num_gpus is None:
            # Keep this consistent with RolloutActorGroup factory:
            # tp_size is treated as the per-engine GPU count when num_gpus
            # is not explicitly provided.
            num_gpus = engine_kwargs.get("tp_size", getattr(args, "tp_size", 1))
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

    if int(getattr(args, "rollout_buffer_max_queue_size", 0)) < 0:
        raise ValueError(
            f"rollout_buffer_max_queue_size must be >= 0, got: {args.rollout_buffer_max_queue_size}"
        )
    if int(getattr(args, "rollout_buffer_min_samples", 1)) < 1:
        raise ValueError(
            f"rollout_buffer_min_samples must be >= 1, got: {args.rollout_buffer_min_samples}"
        )
    reward_min = getattr(args, "rollout_buffer_reward_min", None)
    reward_max = getattr(args, "rollout_buffer_reward_max", None)
    if reward_min is not None and reward_max is not None and float(reward_min) > float(reward_max):
        raise ValueError(
            "rollout_buffer_reward_min must be <= rollout_buffer_reward_max, "
            f"got min={reward_min}, max={reward_max}"
        )
    group_size = getattr(args, "rollout_buffer_group_size", None)
    if group_size is not None and int(group_size) < 1:
        raise ValueError(
            f"rollout_buffer_group_size must be >= 1 when provided, got: {group_size}"
        )
    if int(getattr(args, "rollout_buffer_dispatch_groups", 0)) < 0:
        raise ValueError(
            "rollout_buffer_dispatch_groups must be >= 0 "
            f"(0 means prompts_per_batch), got: {args.rollout_buffer_dispatch_groups}"
        )
    if float(getattr(args, "rollout_buffer_group_ttl_seconds", 0.0)) < 0:
        raise ValueError(
            "rollout_buffer_group_ttl_seconds must be >= 0, "
            f"got: {args.rollout_buffer_group_ttl_seconds}"
        )
    if int(getattr(args, "rollout_buffer_max_pending_samples", 0)) < 0:
        raise ValueError(
            "rollout_buffer_max_pending_samples must be >= 0, "
            f"got: {args.rollout_buffer_max_pending_samples}"
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
    is_sglang_engine = str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    if rollout_gpus > 1 and args.colocate_rollout_training and not is_sglang_engine:
        raise ValueError(
            "colocate_rollout_training=True with multi-GPU rollout actors is only supported "
            "for sampler_engine_type='sglang'."
        )
    if (
        rollout_gpus > 1
        and args.colocate_rollout_training
        and is_sglang_engine
        and not bool(getattr(args, "allow_noset_multi_gpu_inference", False))
    ):
        logger.warning(
            "sglang colocate with multi-GPU rollout requires NOSET actor layout. "
            "Auto-enabling allow_noset_multi_gpu_inference=true."
        )
        args.allow_noset_multi_gpu_inference = True
    if rollout_gpus > 1 and not bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
        raise ValueError(
            "multi-GPU rollout actor layout requires --allow-noset-multi-gpu-inference=true. "
            "Default layout keeps integer single-GPU actors."
        )
    if rollout_gpus > 1 and bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
        logger.warning(
            "allow_noset_multi_gpu_inference=true enabled. "
            "This is an experimental actor layout and is not part of the default path."
        )


def validate_model_specific_logic(args: Any, *, model_cls: Any) -> None:
    """Run model-specific runtime validation and loss defaults."""
    if args.model_type != "flux" and args.sde_type.startswith("flux_"):
        raise ValueError(
            f"sde_type '{args.sde_type}' is only valid for model_type='flux'"
        )

    model_validate_fn = getattr(model_cls, "validate_config", None)
    if callable(model_validate_fn):
        model_validate_fn(args)

    if args.loss_type == "nft" and not args.sampling_adapter:
        args.sampling_adapter = "old"
        logger.info("NFT: default sampling_adapter set to 'old'")

    if args.loss_type == "nft" and args.sde_type == "sde":
        args.sde_type = "dpm2"
        logger.info("NFT: default sde_type set to 'dpm2' for deterministic sampling")


def validate_algorithm_loss_consistency(args: Any) -> None:
    """Validate built-in algorithm/loss consistency."""
    if getattr(args, "loss_path", None):
        logger.info("Custom loss_path is set; skipping built-in algorithm/loss consistency mapping.")
        return

    mapping = {
        "diffusionrl.algorithms.grpo.GRPOAlgorithm": "grpo",
        "diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm": "grpo",
        "diffusionrl.algorithms.nft.NFTAlgorithm": "nft",
    }
    expected_loss_type = mapping.get(args.algorithm_path)
    if expected_loss_type and args.loss_type != expected_loss_type:
        raise ValueError(
            f"algorithm_path={args.algorithm_path} requires loss_type={expected_loss_type}, "
            f"but got loss_type={args.loss_type}. "
            f"Please either change --loss-type to {expected_loss_type} or use a compatible algorithm."
        )


def validate_loss_kwargs_json(args: Any) -> None:
    """Validate and normalize loss_kwargs_json into canonical JSON text."""
    raw = getattr(args, "loss_kwargs_json", "")
    if raw is None:
        args.loss_kwargs_json = ""
        return
    if not isinstance(raw, str):
        raise ValueError(f"loss_kwargs_json must be a JSON object string, got: {type(raw).__name__}")

    text = raw.strip()
    if not text:
        args.loss_kwargs_json = ""
        return

    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid loss_kwargs_json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("loss_kwargs_json must decode to a JSON object.")
    args.loss_kwargs_json = json.dumps(parsed)


def validate_algorithm_kwargs_json(args: Any) -> None:
    """Validate and normalize algorithm_kwargs_json into canonical JSON text."""
    raw = getattr(args, "algorithm_kwargs_json", "")
    if raw is None:
        args.algorithm_kwargs_json = ""
        return
    if not isinstance(raw, str):
        raise ValueError(
            f"algorithm_kwargs_json must be a JSON object string, got: {type(raw).__name__}"
        )

    text = raw.strip()
    if not text:
        args.algorithm_kwargs_json = ""
        return

    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid algorithm_kwargs_json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("algorithm_kwargs_json must decode to a JSON object.")
    args.algorithm_kwargs_json = json.dumps(parsed)


def validate_resolved_engine_loss_contract(
    args: Any,
    *,
    training_actor_direct_sampling: bool,
    is_sglang_engine: bool,
    replay_guard: bool,
    sglang_logprob_mode: str,
) -> None:
    """Validate engine/loss compatibility using resolved capabilities."""
    effective_loss_requirements = resolve_runtime_loss_type_requirements(args)
    effective_engine_caps = resolve_engine_capabilities(engine_type=args.sampler_engine_type)
    if is_sglang_engine and replay_guard and sglang_logprob_mode == "native":
        effective_engine_caps = dict(
            effective_engine_caps,
            requires_log_prob=True,
            requires_embeddings=True,
        )
    validate_engine_loss_contract(
        args=args,
        training_actor_direct_sampling=training_actor_direct_sampling,
        engine_capabilities=effective_engine_caps,
        loss_type_requirements=effective_loss_requirements,
    )


def validate_runtime_mode_constraints(
    args: Any,
    *,
    training_actor_direct_sampling: bool,
) -> None:
    """Validate runtime mode constraints and mutually-exclusive switches."""
    if (
        not training_actor_direct_sampling
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    ):
        supported_sglang_prompt_models = {"hunyuan", "flux", "mochi", "sd3"}
        model_type = str(getattr(args, "model_type", "")).lower()
        if model_type not in supported_sglang_prompt_models:
            raise ValueError(
                "sampler_engine_type='sglang' now uses prompt-only rollout input mode "
                "(no prompt-embedding input path). "
                f"Unsupported model_type={args.model_type!r}. "
                f"Supported model types: {sorted(supported_sglang_prompt_models)}."
            )

    if getattr(args, "async_pipeline", False):
        if args.colocate_rollout_training:
            raise ValueError("async_pipeline requires separate mode (colocate_rollout_training=False).")
        if training_actor_direct_sampling:
            raise ValueError("async_pipeline currently requires training_actor_direct_sampling=false.")
        if int(getattr(args, "async_max_inflight", 1)) < 1:
            raise ValueError("async_max_inflight must be >= 1.")
        if args.update_weights_interval <= 0:
            raise ValueError("update_weights_interval must be > 0.")
        if args.offload_train or args.offload_rollout:
            logger.warning("async_pipeline: disabling offload_train/offload_rollout for stable overlap.")
            args.offload_train = False
            args.offload_rollout = False

__all__ = [
    "DEFAULT_LOSS_TYPE_REQUIREMENTS",
    "repo_root",
    "normalize_repo_relative_paths",
    "resolve_repo_relative_path",
    "looks_like_local_path",
    "is_probably_local_weight_sync_dir",
    "validate_colocate_fractions",
    "get_rollout_gpus_per_actor",
    "resolve_loss_type_requirements",
    "resolve_runtime_loss_type_requirements",
    "resolve_engine_capabilities",
    "validate_dotpath",
    "validate_grouped_configs",
    "validate_dynamic_dotpaths",
    "validate_engine_loss_contract",
    "validate_reward_config",
    "validate_reward_and_rollout_buffer_config",
    "validate_rollout_layout",
    "validate_model_specific_logic",
    "validate_algorithm_loss_consistency",
    "validate_algorithm_kwargs_json",
    "validate_loss_kwargs_json",
    "validate_resolved_engine_loss_contract",
    "validate_runtime_mode_constraints",
]
