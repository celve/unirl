"""CLIP reward scorer."""

from __future__ import annotations

from typing import List

import torch

from diffusionrl.reward.base import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class ClipRewardScorer(BaseLocalRewardScorer):
    """CLIP similarity reward."""

    canonical_model_name = "clip"

    def _load_model(self) -> None:
        try:
            import torch.nn as nn
            import torchvision.transforms as T
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("transformers and torchvision are required for CLIP")

        model_name = self.model_kwargs.get(
            "model_name", "openai/clip-vit-large-patch14"
        )
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        def _get_size(size):
            if isinstance(size, int):
                return (size, size)
            if "height" in size and "width" in size:
                return (size["height"], size["width"])
            if "shortest_edge" in size:
                return size["shortest_edge"]
            raise ValueError(f"Invalid size: {size}")

        config = self.processor.image_processor.to_dict()
        resize = (
            T.Resize(_get_size(config.get("size")))
            if config.get("do_resize")
            else nn.Identity()
        )
        crop = (
            T.CenterCrop(_get_size(config.get("crop_size")))
            if config.get("do_center_crop")
            else nn.Identity()
        )
        normalise = (
            T.Normalize(
                mean=self.processor.image_processor.image_mean,
                std=self.processor.image_processor.image_std,
            )
            if config.get("do_normalize")
            else nn.Identity()
        )
        self._clip_tform = T.Compose([resize, crop, normalise])

        self.model.eval()
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        import numpy as np

        images = request.images
        prompts = request.prompts
        all_rewards: List[float] = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]

            img_arrays = np.array([np.array(img) for img in batch_images])
            pixels = img_arrays.transpose(0, 3, 1, 2)
            pixels = torch.tensor(pixels, dtype=torch.uint8).float() / 255.0
            pixels = self._clip_tform(pixels).to(self.device, dtype=pixels.dtype)

            texts = self.processor(
                text=batch_prompts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(pixel_values=pixels, **texts)
                scores = outputs.logits_per_image.diagonal() / 30
                all_rewards.extend(scores.cpu().tolist())

        return all_rewards
