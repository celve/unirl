from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch

from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import LogProbData, MediaPreview, RolloutSamples
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.types.trajectory_store import TrajectoryStore
from diffusionrl.utils.batched import Batched, concat_field


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

        Reads up to ``max_items`` PIL images from
        ``self.samples.decoded_images``, pairs them with the corresponding
        prompt strings and (already-attached) reward floats, and writes a
        typed ``MediaPreview`` back onto ``self.samples.media_preview``.
        Writes ``None`` when no PIL images are available on the samples.
        Used by the actor-side rollout pipeline right after reward scoring
        (see ``RolloutPipelineMixin.attach_reward``) and as a fallback on
        the driver side.
        """
        decoded_images = self.samples.decoded_images
        if not isinstance(decoded_images, list) or not decoded_images:
            self.samples.media_preview = None
            return

        limit = max(1, int(max_items))
        prompt_texts = list(self.request.prompts.prompts or [])
        rewards_flat: List[float] = []
        if self.samples.rewards is not None and torch.is_tensor(self.samples.rewards):
            rewards_flat = [float(v) for v in self.samples.rewards.detach().cpu().reshape(-1).tolist()]

        images: List[Any] = []
        prompts_out: List[str] = []
        reward_values: List[float] = []
        for idx, img in enumerate(decoded_images):
            if len(images) >= limit:
                break
            if not hasattr(img, "save"):
                continue
            images.append(img)
            prompts_out.append(str(prompt_texts[idx]) if idx < len(prompt_texts) else "")
            reward_values.append(float(rewards_flat[idx]) if idx < len(rewards_flat) else 0.0)

        if not images:
            self.samples.media_preview = None
            return
        self.samples.media_preview = MediaPreview(images=images, prompts=prompts_out, rewards=reward_values)

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

        if samples.trajectories is not None:
            # Trajectory path (GRPO-style)
            trajectory_store = samples.trajectories
            if sde_indices is not None and log_probs is not None:
                sde_int = {int(i) for i in sde_indices}
                log_probs = LogProbData.from_dict({k: v for k, v in log_probs.data.items() if k in sde_int})
                recorded = {int(k) for k in log_probs.data.keys()}
                if recorded != sde_int:
                    raise AssertionError(f"log_probs keys {sorted(recorded)} != sde_indices {sorted(sde_int)}")
        else:
            # Clean-latents path (NFT-style)
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
