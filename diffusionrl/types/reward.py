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
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None
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


@dataclass(init=False)
class RewardResponse:
    """
    Response from reward computation.

    Contains both aggregated rewards and optional per-component reward breakdowns.
    """

    rewards: List[float]
    component_rewards: Dict[str, List[float]] = field(default_factory=dict)
    successes: List[bool] = field(default_factory=list)
    errors: List[Optional[str]] = field(default_factory=list)
    compute_time: float = 0.0

    def __init__(
        self,
        rewards: List[float],
        component_rewards: Optional[Dict[str, List[float]]] = None,
        successes: Optional[List[bool]] = None,
        errors: Optional[List[Optional[str]]] = None,
        compute_time: float = 0.0,
    ) -> None:
        self.rewards = list(rewards) if rewards is not None else []
        self.component_rewards = {
            str(name): list(values or [])
            for name, values in dict(component_rewards or {}).items()
        }
        self.successes = list(successes) if successes is not None else []
        self.errors = list(errors) if errors is not None else []
        self.compute_time = float(compute_time)

    @property
    def batch_size(self) -> int:
        return len(self.rewards)

__all__ = [
    "RewardRequest",
    "RewardResponse",
    "RewardType",
]
