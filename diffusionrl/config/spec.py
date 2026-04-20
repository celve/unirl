"""Framework-level shared spec objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from diffusionrl.types.engine import ROLLOUT_ENGINE_TYPES

if TYPE_CHECKING:
    from diffusionrl.types.sampling import SamplingParams


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
    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0
    replay_sampler_dotpath: Optional[str] = None
    sampling_adapter: Optional[str] = None
    init_same_noise: bool = False
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Resolved specs own a stable sampler_kwargs snapshot; downstream payload
        # builders copy again only when emitting mutable runtime payload dicts.
        object.__setattr__(self, "sampler_kwargs", dict(self.sampler_kwargs or {}))

    def as_shared_payload(self) -> Dict[str, Any]:
        """Serialize shared runtime sampling fields."""
        return {
            "sampler_dotpath": self.sampler_dotpath,
            "num_inference_steps": int(self.num_inference_steps),
            "eta": float(self.eta),
            "sde_type": str(self.sde_type),
            "shift": float(self.shift),
            "guidance_scale": float(self.guidance_scale),
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
        }

    def as_training_sampling_payload(
        self,
        *,
        sampler_engine_type: str,
        precision_settings: Any,
    ) -> Dict[str, Any]:
        """Serialize the training-actor sampling payload."""
        payload = self.as_shared_payload()
        payload.update(
            {
                "sampler_engine_type": sampler_engine_type,
                "replay_sampler_dotpath": self.replay_sampler_dotpath,
                "seed": int(self.seed),
                "sampling_adapter": self.sampling_adapter,
                "init_same_noise": bool(self.init_same_noise),
                "sampler_kwargs": dict(self.sampler_kwargs),
                "autocast_precision": precision_settings.rollout_autocast_precision,
                "trajectory_precision": precision_settings.trajectory_precision,
                "logprob_precision": precision_settings.logprob_precision,
            }
        )
        return payload

    def to_params(self, precision_settings: Any) -> "SamplingParams":
        """Project this config-layer view into the runtime SamplingParams contract.

        Drops dotpath fields (sampler_dotpath, replay_sampler_dotpath,
        sampling_adapter) which flow through engine init payloads, not
        through SamplingParams. ``num_samples_per_prompt`` and ``sde_indices``
        stay at defaults — RolloutPipeline.plan_requests stamps them per step.
        """
        from diffusionrl.types.sampling import SamplingParams, SDEConfig

        return SamplingParams(
            num_inference_steps=int(self.num_inference_steps),
            guidance_scale=float(self.guidance_scale),
            height=int(self.height),
            width=int(self.width),
            num_frames=int(self.num_frames),
            seed=int(self.seed),
            init_same_noise=bool(self.init_same_noise),
            sde_config=SDEConfig(
                eta=float(self.eta),
                sde_type=str(self.sde_type),
                shift=float(self.shift),
            ),
            sampler_kwargs=dict(self.sampler_kwargs),
            autocast_precision=precision_settings.rollout_autocast_precision,
            trajectory_precision=precision_settings.trajectory_precision,
            logprob_precision=precision_settings.logprob_precision,
        )


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model/sampler selection without mutating args."""

    model_dotpath: str
    model_cls: Any
    model_type: str
    sampler_dotpath: str


@dataclass(frozen=True)
class TrainingPlan:
    """Authoritative training batch/update plan derived from explicit config."""

    global_batch_size: int
    local_batch_size: int
    local_mini_batch_size: int
    micro_batch_size: int
    num_updates_per_batch: int
    update_slices: tuple[tuple[int, int], ...]
    mini_batch_slices_per_update: tuple[tuple[tuple[int, int], ...], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_batch_size": self.global_batch_size,
            "local_batch_size": self.local_batch_size,
            "local_mini_batch_size": self.local_mini_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "num_updates_per_batch": self.num_updates_per_batch,
            "update_slices": [[start, end] for start, end in self.update_slices],
            "mini_batch_slices_per_update": [
                [[start, end] for start, end in per_update] for per_update in self.mini_batch_slices_per_update
            ],
        }


@dataclass(frozen=True)
class RolloutInfo:
    """Rollout state shared by validation and entrypoints."""

    mode: str
    rollout_engine: Optional[str]
    training_actor_sampling_mode: bool
    is_sglang_engine: bool
    logprob_source: str
    replay_enabled: bool
    sync_protocol: str
    algorithm_type: str
    max_samples_per_request: Optional[int]
    effective_engine_capabilities: Optional[Dict[str, bool]] = None

    @property
    def sampling_engine(self) -> str:
        if self.training_actor_sampling_mode:
            return "fsdp"
        engine = self.rollout_engine
        if not engine:
            raise ValueError("Dedicated rollout sampling requires rollout.rollout_engine to be set.")
        return engine


@dataclass(frozen=True)
class PlacementSpec:
    rollout_num_nodes: int
    rollout_num_gpus_per_node: int
    training_num_nodes: int
    training_num_gpus_per_node: int
    colocate_rollout: bool
    strategy: str


__all__ = [
    "ModelSpec",
    "PlacementSpec",
    "RolloutInfo",
    "SamplingSpec",
    "TrainingPlan",
    "ROLLOUT_ENGINE_TYPES",
]
