"""HPSv2 reward scorer."""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from diffusionrl.reward.base import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class HPSv2RewardScorer(BaseLocalRewardScorer):
    """HPSv2 image-text alignment reward."""

    canonical_model_name = "hpsv2"

    def _load_model(self) -> None:
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        except ImportError:
            raise ImportError("hpsv2 is required for HPSv2 reward")

        open_clip_path = self.model_kwargs.get(
            "open_clip_path", "./hps_ckpt/open_clip_pytorch_model.bin"
        )
        checkpoint_path = self.model_kwargs.get(
            "checkpoint_path", "./hps_ckpt/HPS_v2.1_compressed.pt"
        )

        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            open_clip_path,
            precision="amp",
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        self._hpsv2_tokenizer = get_tokenizer("ViT-H-14")
        self._hpsv2_preprocess_val = preprocess_val
        self.model = model.to(self.device)
        self.model.eval()
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        all_rewards: List[float] = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]

            for img, prompt in zip(batch_images, batch_prompts):
                try:
                    if isinstance(img, Image.Image):
                        img_pil = img.convert("RGB")
                    else:
                        img_pil = Image.fromarray(img).convert("RGB")

                    image_input = self._hpsv2_preprocess_val(img_pil).unsqueeze(0).to(
                        device=self.device, non_blocking=True
                    )
                    text_input = self._hpsv2_tokenizer([prompt]).to(
                        device=self.device, non_blocking=True
                    )

                    with torch.no_grad():
                        with torch.amp.autocast("cuda"):
                            outputs = self.model(image_input, text_input)
                            image_features = outputs["image_features"]
                            text_features = outputs["text_features"]
                            logits_per_image = image_features @ text_features.T
                            hps_score = torch.diagonal(logits_per_image)

                    all_rewards.append(float(hps_score.item()))
                except Exception:
                    all_rewards.append(0.0)

        return all_rewards
