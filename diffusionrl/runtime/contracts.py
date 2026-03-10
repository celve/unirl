"""Runtime capability and loss-contract resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from diffusionrl.samplers.engine import get_engine_class_path
from diffusionrl.utils.misc import load_function


@dataclass(frozen=True)
class ResolvedSamplingRequirements:
    """Final sampling contract: loss-required fields + algorithm extras."""

    requires_trajectory: bool = True
    requires_log_prob: bool = True
    requires_embeddings: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def sde_ratio(self) -> float:
        return float(self.extras.get("sde_ratio", 1.0))

    @property
    def requires_clean_latents(self) -> bool:
        return bool(self.extras.get("requires_clean_latents", False))

    @property
    def forward_diffusion_in_loss(self) -> bool:
        return bool(self.extras.get("forward_diffusion_in_loss", False))

    def to_dict(self) -> Dict[str, bool]:
        return {
            "requires_trajectory": bool(self.requires_trajectory),
            "requires_log_prob": bool(self.requires_log_prob),
            "requires_embeddings": bool(self.requires_embeddings),
        }

def get_loss_requirements(args: Any) -> Dict[str, bool]:
    """Get loss requirements from loss class's declared_requirements()."""
    loss_path = getattr(args.algorithm, "loss_path", None)
    if not isinstance(loss_path, str) or not loss_path.strip():
        raise ValueError(
            f"Cannot resolve loss class for loss_type={getattr(args.algorithm, 'loss_type', None)!r}, "
            f"loss_path={loss_path!r}. Ensure loss_type is registered or loss_path is importable."
        )
    try:
        loss_cls = load_function(loss_path.strip())
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve loss class for loss_type={getattr(args.algorithm, 'loss_type', None)!r}, "
            f"loss_path={loss_path!r}. Ensure loss_type is registered or loss_path is importable."
        ) from exc
    declared = getattr(loss_cls, "declared_requirements", None)
    if not callable(declared):
        raise ValueError(
            f"Loss class {loss_cls.__name__} must define classmethod declared_requirements() "
            "returning a dict like {'requires_trajectory': True, 'requires_log_prob': True, ...}."
        )
    return dict(declared())


def resolve_sampling_requirements(
    args: Any,
    *,
    algorithm_requirements: Optional[Any] = None,
) -> ResolvedSamplingRequirements:
    """Resolve final sampling contract with loss as single source of requires_*."""
    required = get_loss_requirements(args)
    extras: Dict[str, Any] = {}
    if algorithm_requirements is not None:
        raw_extras = getattr(algorithm_requirements, "extras", None)
        if isinstance(raw_extras, dict):
            extras.update(dict(raw_extras))
        for key in ("sde_ratio", "requires_clean_latents", "forward_diffusion_in_loss"):
            if key in extras:
                continue
            if hasattr(algorithm_requirements, key):
                try:
                    extras[key] = getattr(algorithm_requirements, key)
                except Exception:
                    continue

    return ResolvedSamplingRequirements(
        requires_trajectory=bool(required.get("requires_trajectory", True)),
        requires_log_prob=bool(required.get("requires_log_prob", True)),
        requires_embeddings=bool(required.get("requires_embeddings", True)),
        extras=extras,
    )


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


__all__ = [
    "ResolvedSamplingRequirements",
    "get_loss_requirements",
    "resolve_sampling_requirements",
    "resolve_engine_capabilities",
]
