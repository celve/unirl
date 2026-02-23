"""Configuration validation helpers."""

from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


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
            f"Expected one of: {sorted(loss_type_requirements.keys())}."
        )
    return dict(requirements)


def validate_engine_loss_compatibility(
    *,
    args,
    sampling_backend: str,
    model_type_to_sampler_engine: Dict[str, str],
    engine_capability_requirements: Dict[str, Dict[str, bool]],
    loss_type_requirements: Dict[str, Dict[str, bool]],
) -> None:
    """Fail-fast guard for incompatible sampler engine and loss capability pairs."""
    sampler_engine_type = args.sampler_engine_type or model_type_to_sampler_engine.get(
        args.model_type, "fsdp"
    )
    capabilities = engine_capability_requirements.get(sampler_engine_type)
    if capabilities is None:
        raise ValueError(
            f"Unknown sampler_engine_type={sampler_engine_type}. "
            f"Supported: {sorted(engine_capability_requirements.keys())}."
        )
    capabilities = dict(capabilities)

    if sampling_backend == "training":
        return

    allow_replay = (
        bool(
            getattr(args, "replay_log_probs", False)
            or getattr(args, "fastvideo_replay_log_probs", False)
        )
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

    if args.colocate_reward:
        raise ValueError(
            "colocate_reward=True is no longer supported. "
            "InferenceActor is restricted to prompts->SamplerOutput. "
            "Use RewardService (CPU/HTTP/independent-GPU reward pools)."
        )

    if args.use_http_reward:
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
        logger.info("Reward mode: CPU (LocalRewardWorker)")


def validate_colocate_fractions(args) -> None:
    """Validate colocate GPU fraction bounds."""
    if args.colocate_training_gpu_fraction <= 0 or args.colocate_inference_gpu_fraction <= 0:
        raise ValueError(
            "colocate_training_gpu_fraction and colocate_inference_gpu_fraction must be > 0"
        )
    if args.colocate_training_gpu_fraction + args.colocate_inference_gpu_fraction > 1.0:
        raise ValueError(
            "colocate_training_gpu_fraction + colocate_inference_gpu_fraction must be <= 1.0"
        )


def get_inference_gpus_per_actor(args, *, model_type_to_sampler_engine: Dict[str, str]) -> int:
    """Resolve GPUs per inference actor based on sampler engine and engine config."""
    sampler_engine_type = args.sampler_engine_type or model_type_to_sampler_engine.get(
        args.model_type, "fsdp"
    )
    if sampler_engine_type == "fastvideo":
        return args.fastvideo_num_gpus if args.fastvideo_num_gpus else args.sp_size
    if sampler_engine_type == "fsdp":
        return args.fsdp_num_gpus
    if sampler_engine_type == "sglang":
        engine_kwargs = getattr(args, "engine_kwargs", {})
        if not isinstance(engine_kwargs, dict):
            engine_kwargs = {}
        num_gpus = engine_kwargs.get("num_gpus")
        if num_gpus is None:
            # Keep behavior explicit: default single-GPU engine unless user opts in.
            return 1
        try:
            resolved = int(num_gpus)
        except (TypeError, ValueError):
            return 1
        return max(1, resolved)
    return 1


__all__ = [
    "repo_root",
    "normalize_repo_relative_paths",
    "resolve_repo_relative_path",
    "looks_like_local_path",
    "is_probably_local_weight_sync_dir",
    "validate_colocate_fractions",
    "get_inference_gpus_per_actor",
    "resolve_loss_type_requirements",
    "validate_engine_loss_compatibility",
    "validate_reward_config",
]
