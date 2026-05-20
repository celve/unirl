from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch

from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import LogProbData, MediaPreview, RolloutSamples
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.types.trajectory_store import TrajectoryStore
from diffusionrl.utils.batched import Batched, concat_field
from diffusionrl.utils.media import tensor_frame_to_pil


@dataclass
class RolloutResponse(Batched):
    """Paired request + sample output from a rollout."""

    request: RolloutRequest = concat_field()
    samples: RolloutSamples = concat_field()

    def split(self) -> List["RolloutResponse"]:
        """Split into one RolloutResponse per unique group ID."""
        group_ids = self.request.prompts.group_ids
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(group_ids):
            groups.setdefault(gid, []).append(i)

        results: List[RolloutResponse] = []
        for gid in dict.fromkeys(group_ids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(self.select(indices))
        return results

    def to_meta(self) -> "RolloutResponseMeta":
        return RolloutResponseMeta(group_ids=self.request.prompts.group_ids, sample_ids=self.request.prompts.sample_ids)

    def attach_media_preview(self, *, max_items: int) -> None:
        """Build a ``MediaPreview`` from this (already-scored) response and
        bind it directly onto ``self.samples.media_preview``.

        Two parallel modality paths:

        - **Image path**: reads canonical 3D ``[C, H, W]`` float tensors
          from ``self.samples.decoded_images``, converts each to PIL via
          ``tensor_frame_to_pil`` (the wandb boundary).
        - **Video path**: reads 4D ``[C, T, H, W]`` per-sample tensors
          (list-of-tensors) or a 5D ``[B, C, T, H, W]`` stacked tensor
          from ``self.samples.decoded_videos``, keeps them as 4D CPU
          float tensors on the preview — NOT pre-built ``wandb.Video``.
          The wandb-side encoding is owned by
          ``DiffusionRLWandBLogger.log_generated_media``.

        Both paths cap at ``max_items`` per modality and pair the per-
        sample media with prompts + rewards. Image-only, video-only, and
        image+video previews are all valid. Writes ``None`` when neither
        modality has decoded media.

        Used by the actor-side rollout pipeline right after reward
        scoring (see ``RolloutPipelineMixin.attach_reward``) and as a
        fallback on the driver side.
        """
        decoded_images = self.samples.decoded_images
        decoded_videos = self.samples.decoded_videos

        has_images = isinstance(decoded_images, list) and bool(decoded_images)
        has_videos = (isinstance(decoded_videos, list) and bool(decoded_videos)) or (
            torch.is_tensor(decoded_videos) and decoded_videos.dim() == 5 and int(decoded_videos.shape[0]) > 0
        )
        if not has_images and not has_videos:
            self.samples.media_preview = None
            return

        limit = max(1, int(max_items))
        prompt_texts = list(self.request.prompts.prompts or [])
        rewards_flat: List[float] = []
        if self.samples.rewards is not None and torch.is_tensor(self.samples.rewards):
            rewards_flat = [float(v) for v in self.samples.rewards.detach().cpu().reshape(-1).tolist()]

        # Per-sample video accessor — normalizes list-of-4D-tensor and
        # stacked-5D-tensor inputs into a single ``video[idx] -> 4D | None``
        # surface.
        def _video_at(idx: int):
            if isinstance(decoded_videos, list):
                if 0 <= idx < len(decoded_videos):
                    vid = decoded_videos[idx]
                    return vid if torch.is_tensor(vid) and vid.dim() == 4 else None
                return None
            if torch.is_tensor(decoded_videos) and decoded_videos.dim() == 5 and idx < int(decoded_videos.shape[0]):
                return decoded_videos[idx]  # [C, T, H, W]
            return None

        # Collect per-sample (image | video) up to ``limit``. We use a
        # single ``selected_indices`` list so prompts / rewards stay
        # aligned across both modalities. Image side drives the iteration
        # when present (legacy behavior); video-only path iterates the
        # video batch dim.
        selected_indices: List[int] = []
        images: List[Any] = []
        videos: List[Any] = []

        if has_images:
            for idx, img in enumerate(decoded_images):
                if len(selected_indices) >= limit:
                    break
                if not torch.is_tensor(img):
                    continue
                # Slice to first 3 channels (drops alpha / model-specific
                # 4th channel) before PIL — wandb wants RGB.
                images.append(tensor_frame_to_pil(img[:3]))
                selected_indices.append(idx)
                if has_videos:
                    vid = _video_at(idx)
                    if vid is not None:
                        videos.append(vid.detach().cpu().to(dtype=torch.float32))
        else:
            n_video = len(decoded_videos) if isinstance(decoded_videos, list) else int(decoded_videos.shape[0])
            for idx in range(n_video):
                if len(selected_indices) >= limit:
                    break
                vid = _video_at(idx)
                if vid is None:
                    continue
                videos.append(vid.detach().cpu().to(dtype=torch.float32))
                selected_indices.append(idx)

        if not selected_indices:
            self.samples.media_preview = None
            return

        prompts_out = [str(prompt_texts[i]) if i < len(prompt_texts) else "" for i in selected_indices]
        reward_values = [float(rewards_flat[i]) if i < len(rewards_flat) else 0.0 for i in selected_indices]
        self.samples.media_preview = MediaPreview(
            images=images,
            videos=videos,
            prompts=prompts_out,
            rewards=reward_values,
        )

    def to_training_batch(
        self,
        *,
        sde_indices: Optional[Set[int]] = None,
    ) -> TrainingBatch:
        """Convert this response into a TrainingBatch.

        Requires advantages to be pre-computed on the samples
        (via RolloutActor.compute_advantages).
        """
        samples = self.samples
        if samples.advantages is None:
            raise ValueError("Cannot create TrainingBatch: advantages not computed on samples.")
        if samples.forward_context is None:
            raise ValueError("Cannot create TrainingBatch: forward_context missing on samples.")

        timesteps = samples.timesteps
        log_probs = samples.log_probs

        if sde_indices is not None and samples.trajectories is not None:
            # Trajectory path (GRPO-style): filter log_probs by sde_indices
            trajectory_store = samples.trajectories
            if log_probs is not None:
                sde_int = {int(i) for i in sde_indices}
                log_probs = LogProbData.from_dict({k: v for k, v in log_probs.data.items() if k in sde_int})
                recorded = {int(k) for k in log_probs.data.keys()}
                if recorded != sde_int:
                    raise AssertionError(f"log_probs keys {sorted(recorded)} != sde_indices {sorted(sde_int)}")
        else:
            # Clean-latents path (NFT-style): no SDE indices to train on
            trajectory_store = TrajectoryStore.from_clean_latents(
                samples.latents,
                total_positions=int(timesteps.shape[0]),
            )
            log_probs = None

        batch = TrainingBatch(
            trajectory_store=trajectory_store,
            timesteps=timesteps,
            advantages=samples.advantages,
            forward_context=samples.forward_context,
            log_probs=log_probs,
            rewards=samples.rewards,
            component_rewards=samples.component_rewards,
            prompts=self.request.prompts,
            step_indices=samples.step_indices,
            target_sde_indices=set(int(i) for i in sde_indices) if sde_indices is not None else None,
        )
        batch.validate()
        return batch


@dataclass
class RolloutResponseMeta(Batched):
    group_ids: List[str] = concat_field()
    sample_ids: List[str] = concat_field()
