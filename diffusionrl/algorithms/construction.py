"""Algorithm-construction helpers shared by config/runtime entrypoints.

This module owns the canonical algorithm_config surface consumed by
``algorithm_cls.from_config(config)``. The payload contains:

- algorithm selection (`algorithm_type`, `algorithm_dotpath`)
- canonical `algorithm_kwargs` (user-provided values plus framework-injected defaults)
- a small set of framework-owned shared fields algorithms may read
  (`samples_per_prompt`, scheduler configs, SDE math config)

Keeping that construction logic close to the algorithms package avoids
hard-coding built-in algorithm details under the generic config module.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from diffusionrl.algorithms.registry import DEFAULT_ALGORITHM_PATHS
from diffusionrl.sde.rules import normalize_sde_type

from diffusionrl.types.sampling import SamplingSpec
from diffusionrl.types.sde import SDEConfig
from diffusionrl.utils.misc import load_function


def resolve_algorithm_dotpath(
    *,
    algorithm_type: Any,
    algorithm_dotpath: Any,
) -> str:
    """Resolve built-in algorithm type or explicit algorithm class path."""
    if isinstance(algorithm_dotpath, str) and algorithm_dotpath.strip():
        return algorithm_dotpath.strip()

    normalized_type = str(algorithm_type or "").strip().lower()
    resolved = DEFAULT_ALGORITHM_PATHS.get(normalized_type)
    if not resolved:
        raise ValueError(
            "Cannot resolve algorithm_dotpath for "
            f"algorithm_type={normalized_type!r}. "
            "Provide algorithm.algorithm_dotpath explicitly or register this algorithm_type."
        )
    return resolved


def resolve_algorithm_kwargs(raw: Any) -> Dict[str, Any]:
    """Return canonical algorithm_kwargs without reparsing user input."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "algorithm.algorithm_kwargs must already be a dict after config parsing. "
            f"Got: {type(raw).__name__}."
        )
    return dict(raw)


def resolve_sampling_spec(
    *,
    sampling: Any,
    sampler_dotpath: Any,
    height: Any,
    width: Any,
    num_frames: Any,
    seed: Any,
) -> SamplingSpec:
    """Build the canonical resolved sampling spec from a sampling-config-like object."""
    return SamplingSpec(
        sampler_dotpath=str(sampler_dotpath or ""),
        num_inference_steps=int(getattr(sampling, "num_inference_steps")),
        guidance_scale=float(getattr(sampling, "guidance_scale")),
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        seed=int(seed),
        replay_sampler_dotpath=getattr(sampling, "replay_sampler_dotpath", None),
        sampling_adapter=getattr(sampling, "sampling_adapter", None),
        init_same_noise=bool(getattr(sampling, "init_same_noise", False)),
        sampler_kwargs=dict(getattr(sampling, "sampler_kwargs", {}) or {}),
        sde_config=SDEConfig(
            eta=float(getattr(sampling, "eta")),
            sde_type=normalize_sde_type(getattr(sampling, "sde_type")),
            shift=float(getattr(sampling, "shift")),
        ),
    )


def build_algorithm_kwargs(args: Any) -> Dict[str, Any]:
    """Return the exact user-provided algorithm_kwargs payload."""
    return resolve_algorithm_kwargs(args.algorithm.algorithm_kwargs)


def build_algorithm_config(
    args: Any,
    *,
    sampling_spec: Optional[SamplingSpec] = None,
) -> Dict[str, Any]:
    """Build the canonical algorithm_config passed to algorithm.from_config()."""
    ac = args.algorithm
    pc = args.precision
    cached_resolved = getattr(args, "_diffusionrl_resolved_config", None)
    cached_sampling_spec = getattr(cached_resolved, "sampling_spec", None)
    resolved_sampling_spec = (
        sampling_spec
        if sampling_spec is not None
        else (
            cached_sampling_spec
            if isinstance(cached_sampling_spec, SamplingSpec)
            else resolve_sampling_spec(
                sampling=args.sampling,
                sampler_dotpath=args.sampling.sampler_dotpath,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                seed=args.seed,
            )
        )
    )
    prompts_per_rollout = ac.prompts_per_rollout
    if prompts_per_rollout is None:
        raise ValueError(
            "algorithm.prompts_per_rollout must be set before building algorithm_config."
        )

    algorithm_kwargs = build_algorithm_kwargs(args)

    def _scheduler_payload(raw_cfg: Any) -> Dict[str, Any]:
        if is_dataclass(raw_cfg):
            return asdict(raw_cfg)
        return dict(raw_cfg or {})

    return {
        "algorithm_type": str(ac.algorithm_type),
        "algorithm_dotpath": resolve_algorithm_dotpath(
            algorithm_type=ac.algorithm_type,
            algorithm_dotpath=ac.algorithm_dotpath,
        ),
        "algorithm_kwargs": algorithm_kwargs,
        "samples_per_prompt": int(ac.samples_per_prompt),
        "prompts_per_rollout": int(prompts_per_rollout),
        "component_mix_stage": str(ac.component_mix_stage),
        "adv_normalization": str(ac.adv_normalization),
        "adv_norm_eps": float(ac.adv_norm_eps),
        "adv_clip_abs": ac.adv_clip_abs,
        "use_global_std": bool(ac.use_global_std),
        "trimmed_ratio": float(ac.trimmed_ratio),
        "eval_ema_decay": float(ac.eval_ema_decay),
        "eval_ema_update_interval": int(ac.eval_ema_update_interval),
        "shuffle_samples": bool(ac.shuffle_samples),
        "shuffle_seed": ac.shuffle_seed,
        "training_share_rollout_indices": bool(ac.training_share_rollout_indices),
        "rollout_scheduler": _scheduler_payload(ac.rollout_scheduler),
        "training_scheduler": _scheduler_payload(ac.training_scheduler),
        # Training compute precision is framework-owned and must not travel
        # through algorithm_kwargs, otherwise researchers get two competing
        # config entrypoints for the same runtime contract.
        "training_autocast_precision": str(pc.training.autocast_precision),
        "sde_config": resolved_sampling_spec.sde_config.to_dict(),
        "num_inference_steps": int(resolved_sampling_spec.num_inference_steps),
        "guidance_scale": float(resolved_sampling_spec.guidance_scale),
        "debug_output_dir": args.debug.debug_output_dir,
    }


def instantiate_algorithm_from_config(algorithm_config: Dict[str, Any]) -> Any:
    """Instantiate an algorithm from the canonical algorithm_config payload."""
    algorithm_dotpath = algorithm_config.get("algorithm_dotpath")
    if not isinstance(algorithm_dotpath, str) or not algorithm_dotpath.strip():
        raise ValueError("algorithm_config.algorithm_dotpath must be a non-empty string.")

    algorithm_cls = load_function(algorithm_dotpath.strip())
    from_config = getattr(algorithm_cls, "from_config", None)
    if not callable(from_config):
        raise TypeError(
            f"Algorithm {algorithm_dotpath!r} must implement classmethod from_config(config)."
        )
    return from_config(dict(algorithm_config))


__all__ = [
    "build_algorithm_config",
    "build_algorithm_kwargs",
    "instantiate_algorithm_from_config",
    "resolve_algorithm_kwargs",
    "resolve_algorithm_dotpath",
    "resolve_sampling_spec",
]
