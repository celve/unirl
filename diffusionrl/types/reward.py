"""Shared reward data types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image


class RewardType(Enum):
    """Types of reward computation."""

    IMAGE_TEXT_ALIGNMENT = "image_text_alignment"
    AESTHETIC = "aesthetic"
    VIDEO_QUALITY = "video_quality"
    SAFETY = "safety"
    CUSTOM = "custom"


@dataclass
class RewardRequest:
    """
    Request for reward computation.

    Supports both image and video inputs.
    """

    images: Optional[List[Union[Image.Image, torch.Tensor]]] = None
    videos: Optional[List[torch.Tensor]] = None  # [B, T, C, H, W] or [B, C, T, H, W]
    prompts: List[str] = field(default_factory=list)
    metadata: Optional[List[Dict[str, Any]]] = None
    reward_types: List[RewardType] = field(
        default_factory=lambda: [RewardType.IMAGE_TEXT_ALIGNMENT]
    )
    return_components: bool = False

    @property
    def batch_size(self) -> int:
        if self.images is not None:
            return len(self.images)
        if self.videos is not None:
            return len(self.videos)
        return len(self.prompts)

    @property
    def is_video(self) -> bool:
        return self.videos is not None


@dataclass
class RewardResponse:
    """
    Response from reward computation.

    Contains both aggregated rewards and optional per-component breakdowns.
    """

    rewards: List[float]
    reward_components: Dict[str, List[float]] = field(default_factory=dict)
    successes: List[bool] = field(default_factory=list)
    errors: List[Optional[str]] = field(default_factory=list)
    compute_time: float = 0.0

    @property
    def batch_size(self) -> int:
        return len(self.rewards)

__all__ = [
    "RewardRequest",
    "RewardResponse",
    "RewardType",
]
