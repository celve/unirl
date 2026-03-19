"""OCR reward scorer."""

from __future__ import annotations

import os
from typing import List

from PIL import Image
from tqdm import tqdm

from diffusionrl.reward.base import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class OCRRewardScorer(BaseLocalRewardScorer):
    """OCR reward for text rendering tasks."""

    canonical_model_name = "ocr"

    def _load_model(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr is required for OCR reward. Install with: pip install paddleocr"
            )

        try:
            from Levenshtein import distance as levenshtein_distance
        except ImportError:
            raise ImportError(
                "python-Levenshtein is required for OCR reward. Install with: "
                "pip install python-Levenshtein"
            )

        self._ocr_reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self._levenshtein_distance = levenshtein_distance
        self.model = "ocr"
        self.reward_types = [RewardType.CUSTOM]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        import numpy as np

        images = request.images
        prompts = [prompt.split('"')[1] for prompt in request.prompts]

        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have the same length")

        rewards: List[float] = []
        rank = int(os.environ.get("RANK", 0))
        for img, prompt in tqdm(
            zip(images, prompts),
            desc="Computing OCR rewards",
            disable=(rank != 0),
        ):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                result = self._ocr_reader.predict(img)
                recognized_text = ""
                for res in result:
                    recognized_text += "".join(res["rec_texts"])

                recognized_text = recognized_text.replace(" ", "").lower()
                prompt = prompt.replace(" ", "").lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = self._levenshtein_distance(recognized_text, prompt)
                if dist > len(prompt):
                    dist = len(prompt)
            except Exception as e:
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)

            rewards.append(1 - dist / len(prompt))

        return rewards
