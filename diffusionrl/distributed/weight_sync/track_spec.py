"""Per-track metadata for multi-track weight synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn

LoRAMaterialization = Literal["adapter_tensor", "merged_dense"]


@dataclass
class TrackSyncSpec:
    """Per-track metadata that :class:`WeightSync` needs to extract and ship tensors.

    Each track can differ in every field — different model architecture,
    different LoRA rank, different parameter key layout.

    ``lora_materialization`` selects how LoRA deltas reach the rollout engine:

    * ``"adapter_tensor"`` (default): base weights sent once, then LoRA
      deltas shipped separately via ``set_lora_from_tensors``. Requires a
      LoRA pool on the engine (SGLang diffusion, vllm-omni).
    * ``"merged_dense"``: every sync ships ``base + α·B·A`` via the normal
      bucket path; bypasses the engine's LoRA pool entirely. Required for
      ``sglang_llm`` whose LoRA pool deadlocks after sleep/wake in composed
      PE mode.
    """

    model: nn.Module
    use_lora: bool
    param_name_prefix: str = ""
    packed_modules: dict = field(default_factory=dict)
    base_sync_done: bool = True
    lora_materialization: LoRAMaterialization = "adapter_tensor"


__all__ = ["TrackSyncSpec", "LoRAMaterialization"]
