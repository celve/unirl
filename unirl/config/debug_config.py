"""Typed debug config registered under ``cfg.debug``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from unirl.config.registration import register_config


@register_config(group="debug", name="default")
@dataclass
class DebugConfig:
    """Debug / intermediate-dump knobs.

    ``mode`` values other than ``"none"`` trigger NotImplementedError at
    ``train.py:24-29`` on the cfg-native entry; the typed schema keeps the
    accepted shape explicit without duplicating the feature-gate message.
    """

    mode: str = "none"
    save_dir: Optional[str] = None
    save_intermediates: bool = False


__all__ = ["DebugConfig"]
