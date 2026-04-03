"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import torch

from diffusionrl.types.batch_ops import (
    concat_columnar_values,
    copy_columnar_mapping,
    pad_columnar_value,
    slice_columnar_value,
)
from diffusionrl.types.sde import SDEConfig
from diffusionrl.types.trajectory_store import TrajectoryStore

if TYPE_CHECKING:
    from torch import device as TorchDevice


@dataclass(frozen=True)
class SamplingSpec:
    """Canonical resolved sampling view built once from SamplingConfig."""

    sampler_dotpath: str
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    seed: int
    replay_sampler_dotpath: Optional[str] = None
    sampling_adapter: Optional[str] = None
    init_same_noise: bool = False
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    sde_config: SDEConfig = field(default_factory=SDEConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampler_kwargs", dict(self.sampler_kwargs or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampler_dotpath": self.sampler_dotpath,
            "num_inference_steps": int(self.num_inference_steps),
            "guidance_scale": float(self.guidance_scale),
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "seed": int(self.seed),
            "replay_sampler_dotpath": self.replay_sampler_dotpath,
            "sampling_adapter": self.sampling_adapter,
            "init_same_noise": bool(self.init_same_noise),
            "sampler_kwargs": dict(self.sampler_kwargs),
            "sde_config": self.sde_config.to_dict(),
        }


@dataclass(frozen=True)
class SamplingRequirements:
    """
    Algorithm-declared sampling contract shared with runtime.

    Algorithm-specific extras (for example ``sde_ratio`` for MixGRPO and
    ``requires_clean_latents`` for NFT) live in the open ``extras`` mapping so
    new algorithms can declare sampler requirements without modifying the core
    contract. The mapping is normalized into an owned snapshot at construction
    time so callers cannot retain references to the original input mapping.
    """

    requires_trajectory: bool = True
    """Whether the algorithm needs full denoising trajectories."""

    requires_log_prob: bool = True
    """Whether the algorithm needs log probabilities at each step."""

    requires_embeddings: bool = True
    """Whether the algorithm needs prompt embeddings in the sampled batch."""

    extras: Dict[str, Any] = field(default_factory=dict)
    """Owned algorithm-specific sampler extras snapshot."""

    def __post_init__(self) -> None:
        raw_extras = self.extras
        extras_copy = dict(raw_extras) if isinstance(raw_extras, Mapping) else {}
        object.__setattr__(self, "extras", extras_copy)

    @property
    def sde_ratio(self) -> float:
        """Ratio of SDE steps (1.0 = all SDE, 0.0 = all ODE)."""
        return float(self.extras.get("sde_ratio", 1.0))

    @property
    def requires_clean_latents(self) -> bool:
        """Whether the algorithm needs clean latents x0."""
        return bool(self.extras.get("requires_clean_latents", False))

    @property
    def forward_diffusion_in_loss(self) -> bool:
        """Whether forward diffusion happens in loss computation."""
        return bool(self.extras.get("forward_diffusion_in_loss", False))

    @property
    def is_mixed_sampling(self) -> bool:
        """Whether this uses mixed SDE/ODE sampling."""
        return 0.0 < self.sde_ratio < 1.0

    @property
    def is_trajectory_based(self) -> bool:
        """Whether this is a trajectory-based algorithm (GRPO, MixGRPO)."""
        return self.requires_trajectory

    @property
    def is_forward_process(self) -> bool:
        """Whether this is a forward process algorithm (NFT)."""
        return self.requires_clean_latents and not self.requires_trajectory

    def to_dict(self) -> Dict[str, bool]:
        """Convert core boolean requirements to a plain dictionary."""
        return {
            "requires_trajectory": bool(self.requires_trajectory),
            "requires_log_prob": bool(self.requires_log_prob),
            "requires_embeddings": bool(self.requires_embeddings),
        }


@dataclass
class LogProbData:
    """
    Standardized log probability storage - always dict internally.

    This provides a consistent interface for log probabilities regardless
    of how they were computed (sparse from MixGRPO or dense from full SDE).

    Attributes:
        data: Dictionary mapping step index to log probability tensor [B]
    """

    data: Dict[int, torch.Tensor]

    @classmethod
    def from_dict(cls, d: Dict[int, torch.Tensor]) -> "LogProbData":
        """
        Create from existing dictionary.

        Args:
            d: Dictionary mapping step index to tensor [B]

        Returns:
            LogProbData instance
        """
        return cls(data=d.copy())

    @classmethod
    def empty(cls) -> "LogProbData":
        """Create empty LogProbData."""
        return cls(data={})

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        """Get log probability for a specific timestep index."""
        return self.data.get(idx)

    def __contains__(self, idx: int) -> bool:
        """Check if timestep index has log probability."""
        return idx in self.data

    def __len__(self) -> int:
        """Number of timesteps with log probabilities."""
        return len(self.data)

    @property
    def sde_indices(self) -> Set[int]:
        """Get set of timestep indices that have log probabilities (SDE steps)."""
        return set(self.data.keys())

    def to_dict(self) -> Dict[int, torch.Tensor]:
        """Convert to plain dictionary."""
        return self.data.copy()

    def slice(self, start: int, end: int) -> "LogProbData":
        """
        Slice log probabilities along batch dimension.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New LogProbData with sliced tensors
        """
        return LogProbData(
            data={k: v[start:end].clone() for k, v in self.data.items()}
        )

    def reindex(self, indices: torch.Tensor) -> "LogProbData":
        """
        Reindex log probabilities along batch dimension using a permutation tensor.

        Args:
            indices: 1-D LongTensor of sample indices (e.g. from torch.randperm)

        Returns:
            New LogProbData with reindexed tensors
        """
        return LogProbData(data={k: v[indices] for k, v in self.data.items()})

    def to_device(self, device: Union[str, "TorchDevice"]) -> "LogProbData":
        """Move all tensors to specified device."""
        return LogProbData(data={k: v.to(device) for k, v in self.data.items()})


@dataclass
class RolloutSamples:
    """Lightweight sampler output contract shared across rollout stages."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    aux: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.aux = dict(self.aux or {})
        self.meta = copy_columnar_mapping(self.meta)
        # Auto-promote legacy aux["trajectories"] tensor to TrajectoryStore.
        # After this, all downstream code only needs to look at "trajectory_store".
        if "trajectory_store" not in self.aux and "trajectories" in self.aux:
            raw = self.aux.pop("trajectories")
            if raw is not None:
                self.aux["trajectory_store"] = TrajectoryStore.from_full(raw)

    @property
    def num_steps(self) -> int:
        trajectory_store = self.aux.get("trajectory_store")
        if trajectory_store is not None:
            return trajectory_store.total_positions - 1
        return int(self.timesteps.shape[0]) - 1

    @property
    def batch_size(self) -> int:
        return int(self.latents.shape[0])

    @property
    def sde_indices(self) -> Set[int]:
        log_probs = self.aux.get("log_probs")
        if log_probs is not None:
            return log_probs.sde_indices
        raw_metadata = self.aux.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        meta_indices = metadata.get("sde_indices")
        if meta_indices is not None:
            return set(int(i) for i in meta_indices)
        return set()

    @property
    def resolved_step_indices(self) -> torch.Tensor:
        step_indices = self.aux.get("step_indices")
        if step_indices is not None:
            return step_indices
        return torch.arange(
            self.timesteps.shape[0], device=self.timesteps.device, dtype=torch.long
        )

    def slice(self, start: int, end: int) -> "RolloutSamples":
        batch_size = self.batch_size
        sliced_aux = {
            key: slice_columnar_value(value, batch_size=batch_size, start=start, end=end)
            for key, value in self.aux.items()
        }
        sliced_meta = {
            key: slice_columnar_value(value, batch_size=batch_size, start=start, end=end)
            for key, value in self.meta.items()
        }
        return RolloutSamples(
            latents=self.latents[start:end].clone(),
            timesteps=self.timesteps,
            aux=sliced_aux,
            meta=sliced_meta,
        )

    def to_device(self, device: Union[str, "TorchDevice"]) -> "RolloutSamples":
        moved_aux: Dict[str, Any] = dict(self.aux)
        trajectory_store = self.aux.get("trajectory_store")
        log_probs = self.aux.get("log_probs")
        fwd_ctx = self.aux.get("forward_context")
        step_indices = self.aux.get("step_indices")
        if trajectory_store is not None:
            moved_aux["trajectory_store"] = trajectory_store.to_device(device)
        if log_probs is not None:
            moved_aux["log_probs"] = log_probs.to_device(device)
        if fwd_ctx is not None:
            moved_aux["forward_context"] = fwd_ctx.to_device(device)
        if step_indices is not None:
            moved_aux["step_indices"] = step_indices.to(device)
        return RolloutSamples(
            latents=self.latents.to(device),
            timesteps=self.timesteps.to(device),
            aux=moved_aux,
            meta=self.meta,
        )

    def validate_contract(
        self,
        *,
        requires_log_probs: bool = False,
        requires_trajectory: bool = False,
        requires_embeddings: bool = False,
    ) -> None:
        if self.latents is None or self.timesteps is None:
            raise ValueError(
                "RolloutSamples contract violation: latents and timesteps must be present."
            )
        if self.timesteps.ndim != 1:
            raise ValueError(
                f"RolloutSamples timesteps must be 1D, got shape={tuple(self.timesteps.shape)}"
            )

        batch_size = int(self.latents.shape[0])
        t_plus_1 = int(self.timesteps.shape[0])
        step_indices = self.aux.get("step_indices")
        if step_indices is None:
            step_indices = torch.arange(
                self.timesteps.shape[0], device=self.timesteps.device, dtype=torch.long
            )
        trajectory_store = self.aux.get("trajectory_store")
        log_probs = self.aux.get("log_probs")
        fwd_ctx = self.aux.get("forward_context")
        raw_metadata = self.aux.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        if step_indices.shape[0] != t_plus_1:
            raise ValueError(
                f"RolloutSamples step_indices length {step_indices.shape[0]} != timesteps length {t_plus_1}"
            )
        if step_indices.numel() > 1 and not bool(
            torch.all(step_indices[1:] > step_indices[:-1])
        ):
            raise ValueError(
                f"RolloutSamples step_indices must be strictly increasing, got {step_indices.tolist()}"
            )

        if requires_trajectory and trajectory_store is None:
            raise ValueError(
                "RolloutSamples contract violation: trajectories required but missing."
            )

        if trajectory_store is not None:
            if trajectory_store.batch_size != batch_size:
                raise ValueError(
                    f"RolloutSamples trajectory_store batch {trajectory_store.batch_size} != latents batch {batch_size}"
                )
            if trajectory_store.is_full and trajectory_store.total_positions != t_plus_1:
                raise ValueError(
                    f"RolloutSamples trajectory_store total_positions {trajectory_store.total_positions} "
                    f"!= timesteps length {t_plus_1}"
                )

        if requires_log_probs and not (log_probs is not None and len(log_probs) > 0):
            raise ValueError(
                "RolloutSamples contract violation: log_probs required but missing."
            )

        if log_probs is not None:
            allowed_steps = set(int(v) for v in step_indices[:-1].tolist())
            for idx, lp in log_probs.data.items():
                if idx not in allowed_steps:
                    raise ValueError(
                        f"RolloutSamples contract violation: log_prob index {idx} not in step_indices[:-1]={sorted(allowed_steps)}"
                    )
                if int(lp.shape[0]) != batch_size:
                    raise ValueError(
                        f"RolloutSamples contract violation: log_prob[{idx}] batch {lp.shape[0]} != {batch_size}"
                    )
            sde_indices = (
                log_probs.sde_indices
                if log_probs is not None
                else set(int(i) for i in metadata.get("sde_indices", []))
            )
            if sde_indices and set(int(i) for i in sde_indices) != set(log_probs.data.keys()):
                raise ValueError(
                    "RolloutSamples contract violation: sde_indices must exactly match log_probs keys"
                )

        if requires_embeddings and fwd_ctx is None:
            raise ValueError(
                "RolloutSamples contract violation: embeddings required but missing."
            )

        if fwd_ctx is not None and fwd_ctx.batch_size > 0 and fwd_ctx.batch_size != batch_size:
            raise ValueError(
                f"RolloutSamples contract violation: forward_context batch {fwd_ctx.batch_size} != {batch_size}"
            )

        if metadata:
            trajectory_format = metadata.get("trajectory_format")
            timestep_type = metadata.get("timestep_type")
            if trajectory_format is None:
                raise ValueError(
                    "RolloutSamples contract violation: aux['metadata'].trajectory_format is required."
                )
            if timestep_type is None:
                raise ValueError(
                    "RolloutSamples contract violation: aux['metadata'].timestep_type is required."
                )
            valid_formats = {"dense_latent", "video_dense_latent", "packed_seq_c4"}
            if trajectory_format not in valid_formats:
                raise ValueError(
                    f"RolloutSamples contract violation: unknown trajectory_format={trajectory_format}"
                )
            if timestep_type not in {"sigma", "timestep"}:
                raise ValueError(
                    f"RolloutSamples contract violation: unknown timestep_type={timestep_type}"
                )


@dataclass
class RolloutRequest:
    """Lightweight request contract shared across rollout stages."""

    prompts: List[str]
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    sampling: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompts = list(self.prompts or [])
        self.sampling = dict(self.sampling or {})
        self.meta = copy_columnar_mapping(self.meta)
        self.inputs = dict(self.inputs or {})

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    def with_seed_offset(self, seed_offset: int) -> "RolloutRequest":
        seed_raw = self.sampling.get("seed")
        if seed_raw is None or int(seed_offset) == 0:
            return self
        import copy

        req = copy.copy(self)
        req.sampling = dict(self.sampling)
        req.sampling["seed"] = int(seed_raw) + int(seed_offset)
        req.meta = copy_columnar_mapping(self.meta)
        req.inputs = dict(self.inputs)
        return req

    @classmethod
    def concat(cls, requests: List["RolloutRequest"]) -> "RolloutRequest":
        if not requests:
            raise ValueError("RolloutRequest.concat requires at least one request.")
        if len(requests) == 1:
            return requests[0]

        first = requests[0]
        for request in requests[1:]:
            if int(request.num_inference_steps) != int(first.num_inference_steps):
                raise ValueError("Cannot concatenate RolloutRequest with different num_inference_steps.")
            if float(request.guidance_scale) != float(first.guidance_scale):
                raise ValueError("Cannot concatenate RolloutRequest with different guidance_scale.")
            if int(request.height) != int(first.height) or int(request.width) != int(first.width):
                raise ValueError("Cannot concatenate RolloutRequest with different geometry.")
            if int(request.num_frames) != int(first.num_frames):
                raise ValueError("Cannot concatenate RolloutRequest with different num_frames.")

        batch_sizes = [request.batch_size for request in requests]
        return cls(
            prompts=[prompt for request in requests for prompt in request.prompts],
            num_inference_steps=int(first.num_inference_steps),
            guidance_scale=float(first.guidance_scale),
            height=int(first.height),
            width=int(first.width),
            num_frames=int(first.num_frames),
            sampling=concat_columnar_values([request.sampling for request in requests], batch_sizes=batch_sizes) or {},
            meta=concat_columnar_values([request.meta for request in requests], batch_sizes=batch_sizes) or {},
            inputs=concat_columnar_values([request.inputs for request in requests], batch_sizes=batch_sizes) or {},
        )

    def pad_to(self, target_size: int) -> "RolloutRequest":
        batch_size = self.batch_size
        if target_size <= batch_size:
            return self
        import copy

        req = copy.copy(self)
        req.prompts = list(
            pad_columnar_value(list(self.prompts), batch_size=batch_size, target_size=target_size)
        )
        req.meta = {
            key: pad_columnar_value(value, batch_size=batch_size, target_size=target_size)
            for key, value in self.meta.items()
        }
        req.inputs = {
            key: pad_columnar_value(value, batch_size=batch_size, target_size=target_size)
            for key, value in self.inputs.items()
        }
        req.sampling = {
            key: pad_columnar_value(value, batch_size=batch_size, target_size=target_size)
            for key, value in self.sampling.items()
            if key != "kwargs"
        }
        kwargs = self.sampling.get("kwargs")
        req.sampling["kwargs"] = {
            key: pad_columnar_value(value, batch_size=batch_size, target_size=target_size)
            for key, value in (kwargs.items() if isinstance(kwargs, dict) else [])
        }
        return req

    def slice_prompts(self, start: int, end: int) -> "RolloutRequest":
        import copy

        req = copy.copy(self)
        batch_size = self.batch_size
        req.prompts = self.prompts[start:end]
        req.meta = {
            key: slice_columnar_value(value, batch_size=batch_size, start=start, end=end)
            for key, value in self.meta.items()
        }
        req.inputs = {
            key: slice_columnar_value(value, batch_size=batch_size, start=start, end=end)
            for key, value in self.inputs.items()
        }
        req.sampling = {
            key: slice_columnar_value(value, batch_size=batch_size, start=start, end=end)
            for key, value in self.sampling.items()
            if key != "kwargs"
        }
        kwargs = self.sampling.get("kwargs")
        req.sampling["kwargs"] = {
            key: slice_columnar_value(value, batch_size=batch_size, start=start, end=end)
            for key, value in (kwargs.items() if isinstance(kwargs, dict) else [])
        }
        return req


__all__ = [
    "RolloutSamples",
    "RolloutRequest",
    "LogProbData",
    "SamplingSpec",
    "SamplingRequirements",
]
