"""Model, algorithm, and engine capability validation helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict

from diffusionrl.algorithms.construction import build_algorithm_kwargs
from diffusionrl.config.resolution import (
    ResolvedRolloutTopology,
    load_engine_capabilities,
)
from diffusionrl.sde.rules import (
    SUPPORTED_USER_SDE_TYPES,
    is_deterministic_sde_type,
    supported_sde_type_text,
)
from diffusionrl.types.sampling import SamplingRequirements

logger = logging.getLogger(__name__)


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
    training_actor_sampling_mode: bool,
    is_sglang_engine: bool,
    replay_guard: bool,
    logprob_source: str,
    sampling_requirements: SamplingRequirements,
    rollout_topology: ResolvedRolloutTopology,
) -> None:
    """Validate engine/algorithm compatibility using resolved capabilities."""
    if training_actor_sampling_mode:
        return

    service_engine = rollout_topology.service_engine
    if not service_engine:
        raise ValueError(
            "Dedicated rollout validation requires rollout.topology.service_engine to be set explicitly. "
            "Run validate_args() before resolving dedicated rollout engine capabilities."
        )
    engine_caps = load_engine_capabilities(service_engine)

    allow_replay = (
        bool(args.sampling.replay_log_probs)
        and args.algorithm.algorithm_type == "grpo"
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

    required = sampling_requirements
    if is_sglang_engine and str(args.model.model_type or "").strip().lower() == "sd3":
        if bool(required.requires_trajectory):
            raise ValueError(
                "rollout.topology.service_engine='sglang' with model_type='sd3' currently does not "
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
    rollout_topology: ResolvedRolloutTopology,
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


__all__ = [
    "apply_model_config_hook",
    "validate_model_sampling_contract",
    "validate_nft_sampling_contract",
    "validate_resolved_engine_algorithm_contract",
    "validate_rollout_mode_constraints",
]
