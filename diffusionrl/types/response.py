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

        Handles three decode layouts:

        * **Image-only** (``decoded_images`` is a list of 3D tensors):
          converts each to PIL via ``tensor_frame_to_pil``.
        * **Video-only** (``decoded_videos`` is a 5D tensor or list of 4D
          tensors): extracts the middle frame of each clip as a still-image
          preview and keeps the raw 4D tensor for ``wandb.Video``.
        * **Both**: images drive the preview; videos are sliced to the same
          indices.

        Writes ``None`` when neither modality is populated.
        """
        decoded_images = self.samples.decoded_images
        decoded_videos = self.samples.decoded_videos

        has_images = isinstance(decoded_images, list) and bool(decoded_images)
        has_videos = (
            torch.is_tensor(decoded_videos) and decoded_videos.dim() == 5 and decoded_videos.shape[0] > 0
        ) or (isinstance(decoded_videos, list) and bool(decoded_videos))
        if not has_images and not has_videos:
            self.samples.media_preview = None
            return

        limit = max(1, int(max_items))
        prompt_texts = list(self.request.prompts.prompts or [])
        rewards_flat: List[float] = []
        if self.samples.rewards is not None and torch.is_tensor(self.samples.rewards):
            rewards_flat = [float(v) for v in self.samples.rewards.detach().cpu().reshape(-1).tolist()]

        images: List[Any] = []
        videos: List[Any] = []
        prompts_out: List[str] = []
        reward_values: List[float] = []
        selected_indices: List[int] = []

        if has_images:
            for idx, img in enumerate(decoded_images):
                if len(images) >= limit:
                    break
                if not torch.is_tensor(img):
                    continue
                selected_indices.append(idx)
                images.append(tensor_frame_to_pil(img[:3]))
                prompts_out.append(str(prompt_texts[idx]) if idx < len(prompt_texts) else "")
                reward_values.append(float(rewards_flat[idx]) if idx < len(rewards_flat) else 0.0)

        images_were_materialized = bool(images)

        if has_videos:
            if images_were_materialized and selected_indices:
                if torch.is_tensor(decoded_videos):
                    batch = int(decoded_videos.shape[0])
                    for i in selected_indices:
                        if i >= batch:
                            raise ValueError(
                                f"attach_media_preview: image index {i} has no matching decoded_videos "
                                f"row (batch size {batch})."
                            )
                    video_list = [decoded_videos[int(i)] for i in selected_indices]
                else:
                    n_vid = len(decoded_videos)
                    for i in selected_indices:
                        if i >= n_vid:
                            raise ValueError(
                                f"attach_media_preview: image index {i} has no matching decoded_videos "
                                f"entry (len={n_vid})."
                            )
                    video_list = [decoded_videos[int(i)] for i in selected_indices]
            else:
                if torch.is_tensor(decoded_videos):
                    video_list = [decoded_videos[i] for i in range(min(int(decoded_videos.shape[0]), limit))]
                else:
                    video_list = decoded_videos[:limit]
            for idx, vid in enumerate(video_list):
                vid_cpu = vid.detach().cpu() if torch.is_tensor(vid) else vid
                videos.append(vid_cpu)
                if not images_were_materialized:
                    mid = vid_cpu.shape[1] // 2 if torch.is_tensor(vid_cpu) and vid_cpu.dim() == 4 else 0
                    frame = vid_cpu[:3, mid] if torch.is_tensor(vid_cpu) and vid_cpu.dim() == 4 else vid_cpu[:3]
                    images.append(tensor_frame_to_pil(frame))
                    prompts_out.append(str(prompt_texts[idx]) if idx < len(prompt_texts) else "")
                    reward_values.append(float(rewards_flat[idx]) if idx < len(rewards_flat) else 0.0)

        if not images and not videos:
            self.samples.media_preview = None
            return

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
