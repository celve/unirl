"""Buffer plugin contracts and built-in filtering plugins."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.buffer.buffer_batch_ops import index_training_batch
from diffusionrl.config.argument_parsing import parse_cli_list
from diffusionrl.types.training_batch import BackwardTrainingBatch, TrainingBatch
from diffusionrl.utils import load_function


@dataclass
class BufferPluginContext:
    """Execution context passed to buffer plugins."""

    rollout_id: int
    metadata: Dict[str, Any]


class BufferPlugin(ABC):
    """Plugin interface for rollout buffer data cleaning and validation."""

    def __init__(self, *, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def process(self, batch: TrainingBatch, *, context: BufferPluginContext) -> TrainingBatch:
        """Transform or filter an incoming training batch."""

    def stats(self) -> Dict[str, Any]:
        return {}


def _sample_finite_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return torch.tensor([bool(torch.isfinite(tensor))], dtype=torch.bool, device=tensor.device)
    if tensor.ndim == 1:
        return torch.isfinite(tensor)
    flat = tensor.reshape(tensor.shape[0], -1)
    return torch.isfinite(flat).all(dim=1)


class FiniteTensorFilterPlugin(BufferPlugin):
    """Drop samples with NaN/Inf in core training tensors."""

    def __init__(self, *, drop_invalid: bool = True) -> None:
        super().__init__()
        self.drop_invalid = bool(drop_invalid)
        self.filtered_samples = 0
        self.rejected_batches = 0

    def process(self, batch: TrainingBatch, *, context: BufferPluginContext) -> TrainingBatch:
        del context
        sample_count = int(batch.batch_size)
        if sample_count <= 0:
            raise ValueError("Empty training batch.")

        mask = torch.ones(sample_count, dtype=torch.bool)

        if isinstance(batch, BackwardTrainingBatch):
            mask &= _sample_finite_mask(batch.trajectories).cpu()
            mask &= _sample_finite_mask(batch.advantages).cpu()
            if batch.rewards is not None:
                mask &= _sample_finite_mask(batch.rewards).cpu()
            mask &= _sample_finite_mask(batch.embeddings.prompt_embeds).cpu()
            for value in batch.log_probs.to_dict().values():
                mask &= _sample_finite_mask(value).cpu()

            if not torch.isfinite(batch.timesteps).all():
                self.rejected_batches += 1
                raise ValueError("Non-finite timesteps detected in BackwardTrainingBatch.")
        else:
            mask &= _sample_finite_mask(batch.clean_latents).cpu()
            mask &= _sample_finite_mask(batch.advantages).cpu()
            if batch.rewards is not None:
                mask &= _sample_finite_mask(batch.rewards).cpu()
            mask &= _sample_finite_mask(batch.embeddings.prompt_embeds).cpu()
            if batch.timesteps is not None and not torch.isfinite(batch.timesteps).all():
                self.rejected_batches += 1
                raise ValueError("Non-finite timesteps detected in ForwardTrainingBatch.")

        valid_indices = mask.nonzero(as_tuple=False).flatten().tolist()
        if len(valid_indices) == sample_count:
            return batch

        dropped = sample_count - len(valid_indices)
        self.filtered_samples += dropped
        if not valid_indices:
            self.rejected_batches += 1
            raise ValueError("All samples invalid after finite-value filtering.")

        if not self.drop_invalid:
            self.rejected_batches += 1
            raise ValueError(f"Found {dropped} invalid samples but drop_invalid=false.")

        return index_training_batch(batch, valid_indices)

    def stats(self) -> Dict[str, Any]:
        return {
            "filtered_samples": self.filtered_samples,
            "rejected_batches": self.rejected_batches,
        }


class RewardRangeFilterPlugin(BufferPlugin):
    """Filter samples with rewards outside the configured range."""

    def __init__(self, *, min_reward: Optional[float], max_reward: Optional[float]) -> None:
        super().__init__()
        self.min_reward = float(min_reward) if min_reward is not None else None
        self.max_reward = float(max_reward) if max_reward is not None else None
        self.filtered_samples = 0
        self.rejected_batches = 0

    def process(self, batch: TrainingBatch, *, context: BufferPluginContext) -> TrainingBatch:
        del context
        rewards = getattr(batch, "rewards", None)
        if rewards is None:
            return batch

        sample_count = int(batch.batch_size)
        mask = torch.ones(sample_count, dtype=torch.bool, device=rewards.device)
        if self.min_reward is not None:
            mask &= rewards >= self.min_reward
        if self.max_reward is not None:
            mask &= rewards <= self.max_reward

        keep_indices = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        if len(keep_indices) == sample_count:
            return batch

        dropped = sample_count - len(keep_indices)
        self.filtered_samples += dropped

        if not keep_indices:
            self.rejected_batches += 1
            raise ValueError(
                "All samples filtered by reward range "
                f"[min={self.min_reward}, max={self.max_reward}]"
            )

        return index_training_batch(batch, keep_indices)

    def stats(self) -> Dict[str, Any]:
        return {
            "filtered_samples": self.filtered_samples,
            "rejected_batches": self.rejected_batches,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
        }


class MinSamplesGuardPlugin(BufferPlugin):
    """Reject batches that are too small after filtering."""

    def __init__(self, *, min_samples: int = 1) -> None:
        super().__init__()
        self.min_samples = int(min_samples)
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}.")
        self.rejected_batches = 0

    def process(self, batch: TrainingBatch, *, context: BufferPluginContext) -> TrainingBatch:
        del context
        if int(batch.batch_size) < self.min_samples:
            self.rejected_batches += 1
            raise ValueError(
                f"Batch size {batch.batch_size} below rollout_buffer_min_samples={self.min_samples}"
            )
        return batch

    def stats(self) -> Dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "rejected_batches": self.rejected_batches,
        }


def normalize_plugin_dotpaths(raw: Any) -> List[str]:
    """Normalize rollout buffer plugin dotpaths to a list of non-empty strings.

    Accepts ``None``, a ``list``/``tuple`` of paths, or a single ``str`` (including
    comma-separated or JSON list forms, via :func:`parse_cli_list`).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return parse_cli_list(raw, item_type=str)
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    raise TypeError(
        "rollout.plugin_dotpaths must be a list of strings or a single "
        f"string; got {type(raw).__name__}"
    )


def build_buffer_plugins(args: Any) -> List[BufferPlugin]:
    """Build built-in and custom rollout-buffer plugins from args."""
    rollout_buffer = args.rollout
    plugins: List[BufferPlugin] = [
        FiniteTensorFilterPlugin(
            drop_invalid=bool(rollout_buffer.drop_invalid)
        )
    ]

    min_reward = rollout_buffer.reward_min
    max_reward = rollout_buffer.reward_max
    if min_reward is not None or max_reward is not None:
        plugins.append(
            RewardRangeFilterPlugin(
                min_reward=min_reward,
                max_reward=max_reward,
            )
        )

    plugins.append(
        MinSamplesGuardPlugin(
            min_samples=int(rollout_buffer.min_samples)
        )
    )

    for path in normalize_plugin_dotpaths(rollout_buffer.plugin_dotpaths):
        target = load_function(path)
        if inspect.isclass(target):
            if hasattr(target, "from_args") and callable(getattr(target, "from_args")):
                plugin_obj = target.from_args(args)
            else:
                try:
                    plugin_obj = target(args=args)
                except TypeError:
                    plugin_obj = target()
        else:
            plugin_obj = target

        if isinstance(plugin_obj, BufferPlugin):
            plugins.append(plugin_obj)
        elif callable(plugin_obj):
            raise TypeError(
                f"Rollout buffer plugin {path} is a plain callable. "
                "Wrap it in a BufferPlugin subclass with process() and stats() methods."
            )
        else:
            raise TypeError(
                f"Invalid rollout buffer plugin {path}: expected BufferPlugin instance/class, "
                f"got {type(plugin_obj).__name__}"
            )

    return plugins


__all__ = [
    "BufferPlugin",
    "BufferPluginContext",
    "FiniteTensorFilterPlugin",
    "MinSamplesGuardPlugin",
    "RewardRangeFilterPlugin",
    "build_buffer_plugins",
    "normalize_plugin_dotpaths",
]
