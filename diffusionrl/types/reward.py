"""Shared reward data types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from diffusionrl.types.media import MediaRef
from diffusionrl.utils.batched import Batched, concat_field, max_field


class RewardType(Enum):
    """Types of reward computation."""

    IMAGE_TEXT_ALIGNMENT = "image_text_alignment"
    AESTHETIC = "aesthetic"
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
    prompt_ids: Optional[List[str]] = None
    sample_ids: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
    input_media_refs: Optional[List[List[MediaRef]]] = None
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None
    reward_types: List[RewardType] = field(default_factory=lambda: [RewardType.IMAGE_TEXT_ALIGNMENT])
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
class RewardResponse(Batched):
    """
    Response from reward computation.

    Contains both aggregated rewards and optional per-component reward breakdowns.

    ``compute_time`` is reduced by ``max`` when multiple responses are
    concatenated — it is a wall-clock measurement, so the max across
    parallel-produced responses is the meaningful aggregate.
    """

    rewards: List[float] = concat_field(default_factory=list)
    component_rewards: Dict[str, List[float]] = concat_field(default_factory=dict)
    successes: List[bool] = concat_field(default_factory=list)
    errors: List[Optional[str]] = concat_field(default_factory=list)
    compute_time: float = max_field(default=0.0)

    @property
    def batch_size(self) -> int:
        return len(self.rewards)


__all__ = [
    "RewardRequest",
    "RewardResponse",
    "RewardType",
]
