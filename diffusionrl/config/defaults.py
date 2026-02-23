"""Model-specific validation hooks used by argument normalization."""

from __future__ import annotations

from typing import Any, Callable, Dict


def validate_hunyuan_args(args: Any) -> None:
    """Validate/normalize Hunyuan-specific runtime constraints."""
    fixed_guidance = 6018.0
    if getattr(args, "sampler_engine_type", None) != "fsdp":
        return

    if abs(float(args.guidance_scale) - 7.5) <= 1e-6:
        args.guidance_scale = fixed_guidance
    if abs(float(args.guidance_scale) - fixed_guidance) > 1e-6:
        raise ValueError(
            f"FSDP Hunyuan sampler uses fixed guidance_scale={fixed_guidance}. "
            f"Got guidance_scale={args.guidance_scale}."
        )


def validate_flux_args(args: Any) -> None:
    """Validate/normalize FLUX-specific runtime constraints."""
    if args.sde_type in ("dance", "sde"):
        args.sde_type = "flux_dance"
    elif args.sde_type == "flow":
        args.sde_type = "flux_flow"
    elif args.sde_type in ("flux_dance", "flux_flow"):
        pass
    elif args.sde_type.startswith("flux_"):
        raise ValueError(f"Unknown FLUX sde_type: {args.sde_type}")


MODEL_VALIDATORS: Dict[str, Callable[[Any], None]] = {
    "flux": validate_flux_args,
    "hunyuan": validate_hunyuan_args,
}

__all__ = ["MODEL_VALIDATORS", "validate_flux_args", "validate_hunyuan_args"]

