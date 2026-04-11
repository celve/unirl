"""Cmdline adapters for sampling-related shared specs."""

from __future__ import annotations

from typing import Any

from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.types.sampling import SamplingSpec
from diffusionrl.types.sde import SDEConfig


def build_sampling_spec_from_args(
    args: Any,
    *,
    sampler_dotpath: Any | None = None,
) -> SamplingSpec:
    sampling = args.sampling
    return SamplingSpec(
        sampler_dotpath=str(
            sampler_dotpath
            if sampler_dotpath is not None
            else sampling.sampler_dotpath or ""
        ),
        num_inference_steps=int(getattr(sampling, "num_inference_steps")),
        guidance_scale=float(getattr(sampling, "guidance_scale")),
        height=int(getattr(sampling, "height")),
        width=int(getattr(sampling, "width")),
        num_frames=int(getattr(sampling, "num_frames")),
        seed=int(getattr(args, "seed")),
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


__all__ = ["build_sampling_spec_from_args"]
