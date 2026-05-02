from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.types.forward_context import ForwardContext
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.types.trajectory_store import Trajectory
from diffusionrl.utils.batched import Batched, FieldKind, concat_field, field, max_field, shared_field


def _tensor_bytes(value: Any) -> int:
    """Recursively estimate the total tensor memory in an arbitrary value tree."""
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    if isinstance(value, Batched):
        return sum(_tensor_bytes(getattr(value, f.name)) for f in dataclass_fields(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _tensor_bytes(value.to_dict())
    return 0


@dataclass
class LogProbData(Batched):
    data: Dict[int, torch.Tensor] = concat_field()

    @classmethod
    def from_dict(cls, d: Dict[int, torch.Tensor]) -> "LogProbData":
        return cls(data=dict(d) if d else {})

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        return self.data.get(idx)

    def __contains__(self, idx: int) -> bool:
        return idx in self.data

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> Dict[int, torch.Tensor]:
        return dict(self.data)

    def cast_dtype(self, dtype: torch.dtype) -> "LogProbData":
        return LogProbData(data={k: v.to(dtype=dtype) if v.is_floating_point() else v for k, v in self.data.items()})


@dataclass
class MediaPreview(Batched):
    """Per-rollout wandb media preview: PIL images + prompts + rewards.

    Declaring the three parallel lists as ``concat_field`` lets
    ``Batched.concat`` auto-merge per-shard previews (lists extended) and
    ``Batched.slice(0, n)`` naturally cap the payload size — no custom
    concat/cap code is needed on the owning ``RolloutSamples``.
    """

    images: List[Any] = concat_field(default_factory=list)
    prompts: List[str] = concat_field(default_factory=list)
    rewards: List[float] = concat_field(default_factory=list)

    def __len__(self) -> int:
        return len(self.images)

    def is_empty(self) -> bool:
        return not self.images


@dataclass
class RolloutSamples(Transportable):
    """Lightweight sampler output contract shared across rollout stages.

    ``media_preview`` is a typed, batch-aligned preview payload
    (``MediaPreview`` dataclass) intended only for wandb logging. It is
    declared ``concat_field`` so that ``Batched.concat`` recurses into
    ``MediaPreview.concat`` and auto-merges per-shard previews.

    Heavy per-sample tensors (``latents``, ``decoded_images``,
    ``decoded_videos``) are tagged ``transport=True`` so they route through
    the efficient transport path (TQ) when enabled; small per-sample fields
    (``rewards``, ``advantages``, …) ride the default Ray channel.
    """

    latents: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)
    timesteps: torch.Tensor = shared_field()
    sampling_params: Optional[SamplingParams] = shared_field(default=None)
    prompts: Optional[Prompts] = concat_field(default=None)
    trajectories: Optional[Trajectory] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    log_probs: Optional[LogProbData] = concat_field(default=None)
    forward_context: Optional[ForwardContext] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    step_indices: Optional[torch.Tensor] = shared_field(default=None)
    rewards: Optional[torch.Tensor] = concat_field(default=None)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    decoded_images: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    decoded_videos: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    media_preview: Optional[MediaPreview] = concat_field(default=None)
    # Per-actor sum of per-handle score_and_attach seconds; reduces by max
    # across actors so the driver sees the wall-clock reward share of
    # rollout_phase_s.
    reward_compute_s: float = max_field(default=0.0)

    def cast_dtype(self, dtype: torch.dtype) -> "RolloutSamples":
        """Cast float tensors to *dtype* for transport optimization.

        ``trajectories`` is intentionally NOT cast: trajectory latents must
        preserve their sampler-side storage precision (trajectory_precision,
        default fp16) for train-inference consistency — training-side log-prob
        replay reads the exact same latent values that the sampler wrote, so
        any dtype cast would introduce a bias.
        """
        return RolloutSamples(
            latents=self.latents.to(dtype=dtype) if self.latents.is_floating_point() else self.latents,
            timesteps=self.timesteps,
            sampling_params=self.sampling_params,
            prompts=self.prompts,
            trajectories=self.trajectories,
            log_probs=self.log_probs.cast_dtype(dtype) if self.log_probs is not None else self.log_probs,
            forward_context=self.forward_context.cast_dtype(dtype)
            if self.forward_context is not None
            else self.forward_context,
            step_indices=self.step_indices,
            rewards=self.rewards,
            advantages=self.advantages,
            component_rewards=self.component_rewards,
            decoded_images=self.decoded_images,
            decoded_videos=self.decoded_videos,
            media_preview=self.media_preview,
            reward_compute_s=self.reward_compute_s,
        )

    def cap_media_preview(self, max_items: int) -> None:
        """Truncate ``media_preview`` to at most ``max_items`` entries.

        Driver-side hook used after ``aggregate`` to enforce the final
        per-rollout cap regardless of how many shards contributed. Delegates
        to ``MediaPreview.slice`` (the Batched base) so all three parallel
        lists stay in sync.
        """
        if self.media_preview is None:
            return
        limit = max(1, int(max_items))
        if len(self.media_preview) <= limit:
            return
        self.media_preview = self.media_preview.slice(0, limit)

    def compute_bytes(self) -> int:
        total = 0
        for f in dataclass_fields(self):
            total += _tensor_bytes(getattr(self, f.name))
        return total
