"""Per-track metadata for multi-track weight synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class TrackSyncSpec:
    """Per-track metadata that :class:`WeightSync` needs to extract and ship tensors.

    Each track can differ in every field — different model architecture,
    different LoRA rank, different parameter key layout.
    """

    model: nn.Module
    use_lora: bool
    param_name_prefix: str = ""
    packed_modules: dict = field(default_factory=dict)
    base_sync_done: bool = True


__all__ = ["TrackSyncSpec"]
