"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from diffusionrl.sde.rules import normalize_sde_type


@dataclass(frozen=True)
class SDEConfig:
    """Stable SDE math contract shared by rollout and training."""

    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sde_type", normalize_sde_type(self.sde_type))

    @classmethod
    def from_mapping(
        cls,
        raw: Optional[Mapping[str, Any]] = None,
        *,
        eta: float = 1.0,
        sde_type: str = "flow",
        shift: float = 3.0,
    ) -> "SDEConfig":
        payload = dict(raw or {})
        return cls(
            eta=float(payload.get("eta", eta)),
            sde_type=str(payload.get("sde_type", sde_type)),
            shift=float(payload.get("shift", shift)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingParams:
    """Canonical resolved sampling view built once from SamplingConfig."""

    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    seed: int
    num_samples_per_prompt: int = 1
    init_same_noise: bool = False
    sde_config: SDEConfig = field(default_factory=SDEConfig)
    sde_indices: Optional[List[int]] = None
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Numerical policy (construction-time ride-along; SGLang ignores these)
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"
    # Maximum samples per chunkable rollout-side operation. When set,
    # applied to:
    #   1. engine.generate prompt batches (via chunked_engine_generate)
    #   2. engine.decode_latents reward-decode batches in attach_reward
    #      (via chunked_decode_latents).
    # Engine-agnostic on the FSDP / TrainActor path. SGLang's engine-side
    # decode is controlled by SGLang server args, not by this knob (the
    # SGLang rollout path skips diffusionrl's decode_latents because the
    # SGLang server returns decoded media directly).
    sampling_forward_batch: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingRequirements:
    """Algorithm-declared sampling contract shared with runtime."""

    requires_trajectory: bool = True
    requires_log_prob: bool = True
    requires_embeddings: bool = True
    requires_clean_latents: bool = False

    @property
    def is_forward_process(self) -> bool:
        """Whether this is a forward process algorithm (NFT)."""
        return self.requires_clean_latents and not self.requires_trajectory

    def to_dict(self) -> Dict[str, bool]:
        """Convert core boolean requirements to a plain dictionary."""
        return {
            "requires_trajectory": bool(self.requires_trajectory),
            "requires_log_prob": bool(self.requires_log_prob),
            "requires_embeddings": bool(self.requires_embeddings),
        }
