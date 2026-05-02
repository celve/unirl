"""Typed checkpoint save/load config registered under ``cfg.resume``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require


@register_config(group="resume", name="default", mutable=True)
@dataclass
class ResumeConfig:
    """Checkpoint save/load state for the training loop.

    Collects fields that previously lived on ``cfg.training`` (load path) and
    ``cfg.rollout`` (save cadence + output directory + loop start id). Keeping
    them in a single typed section makes resume behavior inspectable and
    validates the combination on the driver before Ray work starts.

    Marked ``mutable=True``: ``start_rollout_id`` is rewritten post-compose by
    ``utils.train_utils._resolve_start_rollout_id_from_checkpoint`` when a
    resume directory is detected. Every other field is set at compose time.
    """

    resume_from_checkpoint: Optional[str] = None
    start_rollout_id: int = 0
    output_dir: str = "/tmp/diffusionrl/output"
    save_steps: int = 0

    def __post_init__(self) -> None:
        require(
            self.start_rollout_id >= 0, f"ResumeConfig.start_rollout_id must be >= 0; got {self.start_rollout_id!r}"
        )
        require(self.save_steps >= 0, f"ResumeConfig.save_steps must be >= 0; got {self.save_steps!r}")
        require(
            bool(str(self.output_dir or "").strip()),
            f"ResumeConfig.output_dir must be a non-empty string; got {self.output_dir!r}",
        )


__all__ = ["ResumeConfig"]
