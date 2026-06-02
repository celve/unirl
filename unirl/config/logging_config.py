"""Typed logging/wandb config registered under ``cfg.logging``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from unirl.config.registration import register_config
from unirl.config.require import require


@register_config(group="logging", name="default")
@dataclass
class LoggingConfig:
    """Per-run logging + wandb settings read by ``train.py``."""

    logging_steps: int = 1
    logging_dir: str = "/tmp/unirl/logs"
    report_to_wandb: bool = False
    project_name: str = "unirl"
    run_name: Optional[str] = None
    log_media: bool = False
    media_max_items: int = 0
    tags: Optional[List[str]] = None
    entity: Optional[str] = None

    def __post_init__(self) -> None:
        require(self.logging_steps >= 0, f"LoggingConfig.logging_steps must be >= 0; got {self.logging_steps!r}")
        require(self.media_max_items >= 0, f"LoggingConfig.media_max_items must be >= 0; got {self.media_max_items!r}")


__all__ = ["LoggingConfig"]
