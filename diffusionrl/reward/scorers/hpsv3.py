"""HPSv3 reward scorer."""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from diffusionrl.types.reward import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class HPSv3RewardScorer(BaseLocalRewardScorer):
    """HPSv3 image-text alignment reward (Qwen2-VL-7B based).

    HPSv3 is a 7B VLM reward model that outputs [mu, sigma] per image.
    We use mu (index 0) as the final score. The model accepts PIL Images
    directly via its internal fetch_image(), so no disk I/O is needed.

    Reference: https://github.com/MizzenAI/HPSv3
    """

    canonical_model_name = "hpsv3"

    def _load_model(self) -> None:
        try:
            from hpsv3 import HPSv3RewardInferencer
        except ImportError:
            raise ImportError("hpsv3 is required for HPSv3 reward. Install from https://github.com/MizzenAI/HPSv3")

        self._hpsv3_inferencer = HPSv3RewardInferencer(
            device=self.device,
        )
        self.model = self._hpsv3_inferencer.model
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        all_rewards: List[float] = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]

            # Ensure PIL Image inputs
            pil_images = []
            for img in batch_images:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                else:
                    pil_images.append(Image.fromarray(img).convert("RGB"))

            with torch.no_grad():
                scores = self._hpsv3_inferencer.reward(batch_prompts, pil_images)
                # scores shape: [B, 2] (mu, sigma); take mu
                if scores.ndim == 2:
                    scores = scores[:, 0]
                # Normalize to ~0-1 range (benchmark max ~11.79)
                scores = scores / 15.0
                all_rewards.extend(scores.cpu().tolist())

        return all_rewards
