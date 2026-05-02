"""ImageReward reward scorer."""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from diffusionrl.config.registration import register_config
from diffusionrl.reward.base import BaseRewardComponentSpec
from diffusionrl.reward.scorers._device import resolve_device
from diffusionrl.types.reward import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class ImageRewardScorer(BaseLocalRewardScorer):
    """ImageReward human preference scorer (BLIP-based, ~300M).

    ImageReward is trained on 137k human preference annotations and outputs
    a scalar score reflecting overall human preference (text-image alignment,
    aesthetics, composition, etc.).

    Reference: https://github.com/THUDM/ImageReward
    """

    canonical_model_name = "image_reward"

    def __init__(self, *, config: "ImageRewardSpec", base_device: str) -> None:
        super().__init__(
            device=resolve_device(config.device, base_device),
            model_version=config.model_version,
        )

    def _load_model(self) -> None:
        try:
            import ImageReward as RM
        except ImportError:
            raise ImportError("image-reward is required for ImageReward reward. Install with: pip install image-reward")

        self.model = RM.load(
            self.model_kwargs.get("model_version", "ImageReward-v1.0"),
            device=self.device,
        )
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        all_rewards: List[float] = []

        for img, prompt in zip(images, prompts):
            # Ensure PIL Image
            if isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                pil_img = Image.fromarray(img).convert("RGB")

            with torch.no_grad():
                score = float(self.model.score(prompt, pil_img))
                # Normalize to ~[0, 1]: raw scores typically in [-2, +2]
                score = (score + 2.0) / 4.0
                all_rewards.append(score)

        return all_rewards


@register_config(
    group="reward/component",
    name="image_reward",
    target="diffusionrl.reward.scorers.image_reward.ImageRewardScorer",
)
class ImageRewardSpec(BaseRewardComponentSpec):
    """Typed config for the ImageReward (BLIP-based, ~300M) reward component.

    ImageReward processes one image at a time, so no batch_size knob.
    """

    weight: float = 1.0
    device: str = "auto"
    model_version: str = "ImageReward-v1.0"
