"""Multiple-choice exact-match reward scorer for VLM QA tasks."""

from __future__ import annotations

import re
from typing import List

from diffusionrl.config.registration import register_config
from diffusionrl.reward.base import BaseRewardComponentSpec
from diffusionrl.types.reward import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer

_ANSWER_PATTERN = re.compile(
    r"(?:(?:answer|option)\s*(?:is|:)\s*)\(?([A-D])\)?",
    re.IGNORECASE,
)

_STANDALONE_LETTER = re.compile(r"\b([A-D])\b")


def _extract_answer_letter(text: str) -> str:
    text = text.strip()
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    matches = _STANDALONE_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    return ""


class MCExactMatchRewardScorer(BaseLocalRewardScorer):
    """Multiple-choice exact-match reward for VLM QA tasks."""

    canonical_model_name = "mc_exact_match"
    input_kind = "text"

    def __init__(self, *, config: "MCExactMatchSpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "mc_exact_match"
        self.reward_types = [RewardType.CUSTOM]

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MCExactMatchRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = str(meta["answer"]).strip().upper()
            predicted = _extract_answer_letter(text)
            rewards.append(1.0 if predicted == gt else 0.0)
        return rewards


@register_config(
    group="reward/component",
    name="mc_exact_match",
    target="diffusionrl.reward.scorers.mc_exact_match.MCExactMatchRewardScorer",
)
class MCExactMatchSpec(BaseRewardComponentSpec):
    weight: float = 1.0
