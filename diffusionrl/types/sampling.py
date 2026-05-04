"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require


@dataclass
class SDEConfig:
    """Stable SDE math contract shared by rollout and training.

    Strategy choice (flow/cps/dance/dpm2) is now a separate Hydra group at
    ``cfg.sampling.sde_strategy`` (registered via the ``sampling/sde_strategy``
    group in :mod:`diffusionrl.sde.kernels`); ``SDEConfig`` only carries
    the per-strategy math params.
    """

    eta: float = 1.0
    shift: float = 3.0

    @classmethod
    def from_mapping(
        cls,
        raw: Optional[Mapping[str, Any]] = None,
        *,
        eta: float = 1.0,
        shift: float = 3.0,
    ) -> "SDEConfig":
        payload = dict(raw or {})
        return cls(
            eta=float(payload.get("eta", eta)),
            shift=float(payload.get("shift", shift)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@register_config(group="sampling", name="default")
@dataclass
class SamplingParams:
    """Canonical resolved sampling view built once from SamplingConfig."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 256
    width: int = 256
    num_frames: int = 16
    seed: int = 42
    num_samples_per_prompt: int = 1
    init_same_noise: bool = False
    sde_config: SDEConfig = field(default_factory=SDEConfig)
    # SDE step strategy chosen via the Hydra group ``sampling/sde_strategy``
    # (e.g. ``defaults: [- sampling/sde_strategy: dpm2]``). Holds the
    # registered Spec dataclass (FlowSpec / CPSSpec / DanceSpec / DPM2Spec)
    # whose ``_target_`` resolves to the matching strategy class. Built into
    # an instance via ``build(cfg.sampling.sde_strategy)`` at the boundary.
    # Defaults to ``None`` so nested ``SamplingParams`` copies (e.g.
    # ``cfg.rollout.engine.sampling``) compose without requiring their own
    # group selection — actors look up the strategy from ``cfg.sampling``.
    sde_strategy: Any = None
    sde_indices: Optional[List[int]] = None
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Numerical policy (construction-time ride-along; SGLang ignores these)
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    def __post_init__(self) -> None:
        # Every typed field (except sampler_kwargs itself) is part of the engine
        # contract and cannot be shadowed via sampler_kwargs — engine-pinned keys
        # like ``num_inference_steps``/``guidance_scale``/``height``/``width`` and
        # the precision knobs would otherwise be silently overridden by the
        # typed-field copy at the use site, masking user typos as no-ops.
        reserved = {f.name for f in fields(self) if f.name != "sampler_kwargs"}
        shadowed = reserved & set(self.sampler_kwargs)
        require(
            not shadowed,
            f"SamplingParams.sampler_kwargs cannot contain reserved keys {sorted(shadowed)}; set them as fields instead",
        )

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
