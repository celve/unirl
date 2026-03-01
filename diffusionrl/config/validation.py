"""Configuration validation helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from diffusionrl.samplers.engine import get_engine_class_path
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)


def _resolve_loss_class(args: Any):
    """Load loss class from registry or loss_path."""
    loss_type = str(getattr(args, "loss_type", "grpo"))
    loss_cls = None

    try:
        from diffusionrl.losses import LOSS_REGISTRY
        loss_cls = LOSS_REGISTRY.get(loss_type)
    except Exception:
        loss_cls = None

    if loss_cls is None and getattr(args, "loss_path", None):
        loss_cls = load_function(args.loss_path)

    return loss_cls


def _get_loss_requirements(args: Any) -> Dict[str, bool]:
    """Get loss requirements from loss class's declared_requirements().

    Loss classes MUST implement declared_requirements() classmethod.
    """
    loss_cls = _resolve_loss_class(args)
    if loss_cls is None:
        raise ValueError(
            f"Cannot resolve loss class for loss_type={getattr(args, 'loss_type', None)!r}, "
            f"loss_path={getattr(args, 'loss_path', None)!r}. "
            "Ensure loss_type is registered or loss_path is importable."
        )
    declared = getattr(loss_cls, "declared_requirements", None)
    if not callable(declared):
        raise ValueError(
            f"Loss class {loss_cls.__name__} must define classmethod declared_requirements() "
            "returning a dict like {'requires_trajectory': True, 'requires_log_prob': True, ...}."
        )
    return dict(declared())


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
) -> None:
    """Normalize configured paths relative to repository root."""
    root = repo_root(env_repo_root=env_repo_root)
    data_root_env = os.getenv(env_data_root)
    model_root_env = os.getenv(env_model_root)

    for field_name in (
        "output_dir",
        "logging_dir",
        "weight_sync_dir",
        "resume_from_checkpoint",
        "debug_save_dir",
        "debug_load_path",
    ):
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
    args.debug_cfg.validate()


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
        raise ValueError(
            "sglang colocate with multi-GPU rollout requires NOSET actor layout. "
            "Set --allow-noset-multi-gpu-inference=true."
        )
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
    """Run model-specific runtime validation."""
    if args.model_type != "flux" and args.sde_type.startswith("flux_"):
        raise ValueError(
            f"sde_type '{args.sde_type}' is only valid for model_type='flux'"
        )

    model_validate_fn = getattr(model_cls, "validate_config", None)
    if callable(model_validate_fn):
        model_validate_fn(args)

    if args.loss_type == "nft":
        # DiffusionNFT reproduction contract:
        # - rollout samples from old adapter
        # - deterministic solver (dpm2)
        old_adapter_name = "old"
        loss_kwargs_json = getattr(args, "loss_kwargs_json", "")
        if isinstance(loss_kwargs_json, str) and loss_kwargs_json.strip():
            try:
                parsed = json.loads(loss_kwargs_json)
                if isinstance(parsed, dict):
                    old_adapter_name = str(parsed.get("old_adapter_name", old_adapter_name) or old_adapter_name)
            except Exception:
                # validate_loss_kwargs_json() should already have rejected malformed JSON.
                pass

        if not args.sampling_adapter:
            raise ValueError(
                "loss_type='nft' requires --sampling-adapter to be set "
                f"(must match old_adapter_name={old_adapter_name!r})."
            )
        if str(args.sampling_adapter) != old_adapter_name:
            raise ValueError(
                "loss_type='nft' requires rollout sampling from the old adapter. "
                f"Set --sampling-adapter {old_adapter_name!r}, got {args.sampling_adapter!r}."
            )
        if str(args.sde_type) != "dpm2":
            raise ValueError(
                "loss_type='nft' targets DiffusionNFT deterministic sampling. "
                f"Set --sde-type dpm2, got sde_type={args.sde_type!r}."
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


def validate_algorithm_loss_contract(args: Any) -> None:
    """Validate algorithm sampling requirements against selected loss contract."""
    algorithm_cls = load_function(args.algorithm_path)
    from_args_fn = getattr(algorithm_cls, "from_args", None)
    if not callable(from_args_fn):
        raise ValueError(
            f"Algorithm class {args.algorithm_path!r} must implement classmethod from_args(args)."
        )

    algorithm = from_args_fn(args)
    get_requirements_fn = getattr(algorithm, "get_sampling_requirements", None)
    if not callable(get_requirements_fn):
        raise ValueError(
            f"Algorithm class {args.algorithm_path!r} must implement get_sampling_requirements()."
        )

    requirements = get_requirements_fn()
    algorithm_caps = {
        "requires_trajectory": bool(getattr(requirements, "requires_trajectory", False)),
        "requires_log_prob": bool(getattr(requirements, "requires_log_prob", False)),
        "requires_embeddings": bool(getattr(requirements, "requires_embeddings", False)),
    }
    loss_requirements = _get_loss_requirements(args)

    missing = [
        key
        for key, needed in loss_requirements.items()
        if bool(needed) and not bool(algorithm_caps.get(key, False))
    ]
    if missing:
        raise ValueError(
            "Algorithm/loss contract mismatch: "
            f"algorithm_path={args.algorithm_path} cannot satisfy loss_type={args.loss_type}. "
            f"Missing requirements={missing}. "
            f"algorithm_caps={algorithm_caps}, loss_requirements={loss_requirements}."
        )


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

    sampler_engine_type = str(getattr(args, "sampler_engine_type", "") or "")
    engine_caps = resolve_engine_capabilities(engine_type=args.sampler_engine_type)

    allow_replay = (
        bool(getattr(args, "replay_log_probs", False))
        and getattr(args, "loss_type", "grpo") == "grpo"
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

    required = _get_loss_requirements(args)
    missing = [
        key
        for key, needed in required.items()
        if bool(needed) and not bool(engine_caps.get(key, False))
    ]
    if missing:
        raise ValueError(
            f"Engine capability mismatch for loss_type={args.loss_type}: "
            f"sampler_engine_type={sampler_engine_type} lacks {missing}. "
            f"engine_capabilities={engine_caps}, required={required}. "
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
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    ):
        supports_sglang = getattr(model_cls, "supports_sglang_prompt_mode", None)
        if not callable(supports_sglang):
            raise ValueError(
                f"sampler_engine_type='sglang' requires model {args.model_path!r} "
                "to define classmethod supports_sglang_prompt_mode()."
            )
        if not supports_sglang():
            raise ValueError(
                f"sampler_engine_type='sglang' is not supported by model {args.model_path!r}. "
                "The model must implement classmethod supports_sglang_prompt_mode() returning True."
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
            raise ValueError(
                "async_pipeline is incompatible with offload_train/offload_rollout. "
                "Set --offload-train=false --offload-rollout=false when using --async-pipeline."
            )

__all__ = [
    "repo_root",
    "normalize_repo_relative_paths",
    "resolve_repo_relative_path",
    "looks_like_local_path",
    "is_probably_local_weight_sync_dir",
    "validate_colocate_fractions",
    "get_rollout_gpus_per_actor",
    "resolve_engine_capabilities",
    "validate_dotpath",
    "validate_grouped_configs",
    "validate_dynamic_dotpaths",
    "validate_reward_config",
    "validate_reward_and_rollout_buffer_config",
    "validate_rollout_layout",
    "validate_model_specific_logic",
    "validate_algorithm_kwargs_json",
    "validate_loss_kwargs_json",
    "validate_algorithm_loss_contract",
    "validate_resolved_engine_loss_contract",
    "validate_runtime_mode_constraints",
]
