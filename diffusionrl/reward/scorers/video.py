"""Video reward scorer."""

from __future__ import annotations

import time
from typing import List, Optional

import torch
from PIL import Image

from diffusionrl.reward.base import BaseRewardScorer
from diffusionrl.types.reward import RewardRequest, RewardResponse

from .registry import resolve_builtin_reward_scorer_class


class VideoRewardScorer(BaseRewardScorer):
    """Specialized reward scorer for video generation."""

    input_kind = "video"

    def __init__(
        self,
        model_name: Optional[str] = "pickscore",
        temporal_weight: float = 0.3,
        alignment_weight: float = 0.7,
        sample_frames: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        timeout: float = 60.0,
        **model_kwargs,
    ) -> None:
        resolved_frame_model = str(model_name or "pickscore").strip().lower()
        super().__init__(
            model_name=resolved_frame_model,
            batch_size=batch_size,
            timeout=timeout,
        )
        self.temporal_weight = temporal_weight
        self.alignment_weight = alignment_weight
        self.sample_frames = sample_frames

        scorer_cls = resolve_builtin_reward_scorer_class(resolved_frame_model)
        self.frame_scorer = scorer_cls(
            model_name=resolved_frame_model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            timeout=timeout,
            **model_kwargs,
        )

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if not request.is_video:
            return self.frame_scorer.compute_rewards(request)

        start = time.time()
        videos = request.videos
        prompts = request.prompts

        try:
            rewards = []
            component_rewards = {
                "alignment": [],
                "temporal": [],
            }

            for video, prompt in zip(videos, prompts):
                frames = self._sample_frames(video)
                frame_request = RewardRequest(
                    images=frames,
                    prompts=[prompt] * len(frames),
                )
                frame_response = self.frame_scorer.compute_rewards(frame_request)
                alignment_reward = sum(frame_response.rewards) / len(frame_response.rewards)
                temporal_reward = self._compute_temporal_consistency(video)
                total_reward = self.alignment_weight * alignment_reward + self.temporal_weight * temporal_reward

                rewards.append(total_reward)
                component_rewards["alignment"].append(alignment_reward)
                component_rewards["temporal"].append(temporal_reward)

            return RewardResponse(
                rewards=rewards,
                component_rewards=component_rewards,
                successes=[True] * len(rewards),
                errors=[None] * len(rewards),
                compute_time=time.time() - start,
            )
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * len(videos),
                successes=[False] * len(videos),
                errors=[str(e)] * len(videos),
                compute_time=time.time() - start,
            )

    def _sample_frames(self, video: torch.Tensor) -> List[Image.Image]:
        from torchvision.transforms.functional import to_pil_image

        if video.dim() == 4:
            video = video.permute(1, 0, 2, 3)
        elif video.dim() == 5:
            video = video.squeeze(0).permute(1, 0, 2, 3)

        num_frames = video.shape[0]
        indices = torch.linspace(0, num_frames - 1, self.sample_frames).long()

        frames = []
        for idx in indices:
            frame = video[idx]
            if frame.max() <= 1.0:
                frame = (frame * 255).byte()
            frames.append(to_pil_image(frame))

        return frames

    def _compute_temporal_consistency(self, video: torch.Tensor) -> float:
        if video.dim() == 4:
            video = video.permute(1, 0, 2, 3)
        elif video.dim() == 5:
            video = video.squeeze(0).permute(1, 0, 2, 3)

        frame_diffs = []
        for i in range(len(video) - 1):
            diff = (video[i] - video[i + 1]).abs().mean()
            frame_diffs.append(diff.item())

        avg_diff = sum(frame_diffs) / len(frame_diffs) if frame_diffs else 0
        return max(0.0, 1 - avg_diff)

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

    def is_available(self) -> bool:
        return self.frame_scorer.is_available()

    def offload(self) -> None:
        self.frame_scorer.offload()

    def onload(self) -> None:
        self.frame_scorer.onload()

    def dispose(self) -> None:
        self.frame_scorer.dispose()
