"""Diversity reward scorer using DINOv2 features."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from PIL import Image

from diffusionrl.reward.base import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class DiversityRewardScorer(BaseLocalRewardScorer):
    """Per-group diversity score using DINOv2 CLS features + mean pairwise cosine distance.

    Unlike typical scorers that evaluate individual samples, this scorer measures
    diversity within groups of samples generated from the same prompt. The score
    is broadcast to all samples within each group.

    Grouping strategy (by priority):
      1. ``request.group_ids`` — native field in the new architecture
      2. Fallback: ``model_kwargs["num_samples_per_prompt"]`` for fixed-size grouping
    """

    canonical_model_name = "diversity"

    def _load_model(self) -> None:
        from torchvision import transforms

        model_path = self.model_kwargs.get("model_path", "facebook/dinov2-base")

        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
            self._diversity_use_hf = True
        except Exception:
            variant = self.model_kwargs.get("dino_variant", "dinov2_vitb14")
            self.model = torch.hub.load("facebookresearch/dinov2", variant)
            self.model = self.model.to(self.device).eval()
            self._diversity_use_hf = False

        self._diversity_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.reward_types = [RewardType.CUSTOM]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images

        # --- Determine groups ---
        groups: List[tuple] = []  # [(start, end), ...]

        if request.group_ids is not None and len(request.group_ids) == len(images):
            # Use native group_ids field (new architecture)
            i = 0
            while i < len(images):
                current_gid = request.group_ids[i]
                j = i + 1
                while j < len(images) and request.group_ids[j] == current_gid:
                    j += 1
                groups.append((i, j))
                i = j
        else:
            # Fallback: fixed-size grouping
            group_size = int(self.model_kwargs.get("num_samples_per_prompt", 1) or 1)
            group_size = max(1, group_size)
            for start in range(0, len(images), group_size):
                groups.append((start, min(start + group_size, len(images))))

        # --- Extract DINOv2 CLS features ---
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch_imgs = images[i : i + self.batch_size]
            pixel_values = torch.stack(
                [
                    self._diversity_transform(
                        img if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB")
                    )
                    for img in batch_imgs
                ]
            ).to(self.device)

            with torch.no_grad():
                if self._diversity_use_hf:
                    outputs = self.model(pixel_values)
                    features = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    features = self.model(pixel_values)  # torch.hub DINOv2 returns CLS directly
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)  # [N, D]
        all_features = F.normalize(all_features, dim=-1)

        # --- Compute diversity score per group, broadcast to each sample ---
        rewards = [0.0] * len(images)
        for start, end in groups:
            group_features = all_features[start:end]  # [K, D]
            K = group_features.shape[0]

            if K <= 1:
                diversity_score = 0.0
            else:
                sim_matrix = group_features @ group_features.T
                mask = torch.triu(
                    torch.ones(K, K, dtype=torch.bool, device=sim_matrix.device),
                    diagonal=1,
                )
                mean_sim = sim_matrix[mask].mean().item()
                diversity_score = 1.0 - mean_sim

            for idx in range(start, end):
                rewards[idx] = diversity_score

        return rewards
