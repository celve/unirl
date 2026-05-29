"""Typed per-run config registered under ``cfg.run``.

Holds the scalars that define a single training run: RNG seed, loop
budget, data pipeline dotpath and data paths. The ``run`` group name
intentionally avoids ``experiment`` so ``conf/experiment/`` can host
Hydra ``@package _global_`` recipes without a defaults-list collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require


@register_config(group="run", name="default")
@dataclass
class RunConfig:
    """Per-run driver inputs: seed, loop budget, data pipeline."""

    seed: Optional[int] = 42
    num_rollouts: int = 1
    data_source_dotpath: str = "diffusionrl.data.DefaultDataSource"
    data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    weight_sync_interval: int = 1

    def __post_init__(self) -> None:
        require(self.num_rollouts >= 1, f"RunConfig.num_rollouts must be >= 1; got {self.num_rollouts!r}")
        require(
            bool(str(self.data_source_dotpath or "").strip()),
            f"RunConfig.data_source_dotpath must be a non-empty dotpath; got {self.data_source_dotpath!r}",
        )
        require(
            self.weight_sync_interval >= 1,
            f"RunConfig.weight_sync_interval must be >= 1; got {self.weight_sync_interval!r}",
        )


__all__ = ["RunConfig"]
