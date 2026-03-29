"""Shared SDE data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

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
