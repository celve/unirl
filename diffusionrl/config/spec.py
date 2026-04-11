"""Framework-level resolved spec objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model/sampler selection without mutating args."""

    model_dotpath: str
    model_cls: Any
    model_type: str
    sampler_dotpath: str
    model_default_engine_type: Optional[str]


__all__ = ["ModelSpec"]
