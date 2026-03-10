"""Rollout buffer actor and plugin chain for rollout->train decoupling."""

from __future__ import annotations

import inspect
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import ray
import torch

from diffusionrl.runtime.pipeline.rollout_pipeline import maybe_partition_training_batch
from diffusionrl.types.sampling import LogProbData, PromptEmbeddings
from diffusionrl.types.training_batch import (
    BackwardTrainingBatch,
    ForwardTrainingBatch,
    TrainingBatch,
)
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


def _put_training_data(
    *,
    train_data: Any,
    dp_size: Optional[int],
    partition_train_data: bool = True,
    partition_mode: str = "data_parallel",
) -> Any:
    mode = str(partition_mode or "data_parallel").strip().lower()
    if mode in ("backend_managed", "replicated", "none"):
        return ray.put(train_data)
    if mode != "data_parallel":
        raise ValueError(
            f"Unsupported rollout buffer partition_mode={partition_mode!r}. "
            "Expected one of: data_parallel, backend_managed, replicated, none."
        )

    partitioned_batches = maybe_partition_training_batch(
        train_data=train_data,
        dp_size=dp_size,
        partition_train_data=bool(partition_train_data),
    )
    if partitioned_batches is not None:
        return [ray.put(part) for part in partitioned_batches]
    return ray.put(train_data)


def _resolve_consumer_spec(
    *,
    default_partition_train_data: bool,
    dp_size: Optional[int],
    consumer_spec: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], bool, str]:
    resolved_dp_size = dp_size
    partition_train_data = bool(default_partition_train_data)
    partition_mode = "data_parallel"

    if isinstance(consumer_spec, dict):
        if consumer_spec.get("dp_size") is not None:
            try:
                resolved_dp_size = int(consumer_spec.get("dp_size"))
            except (TypeError, ValueError):
                resolved_dp_size = dp_size
        if consumer_spec.get("partition_train_data") is not None:
            partition_train_data = bool(consumer_spec.get("partition_train_data"))
        if consumer_spec.get("partition_mode") is not None:
            partition_mode = str(consumer_spec.get("partition_mode")).strip().lower()

    return resolved_dp_size, partition_train_data, partition_mode


@dataclass
class BufferItem:
    """Single buffered training-batch entry in dispatch queue."""

    item_id: str
    rollout_id: int
    batch_ref: Any
    sample_count: int
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupSampleLocator:
    """Pointer to one sample inside a stored rollout TrainingBatch."""

    batch_ref: Any
    sample_idx: int
    rollout_id: int
    created_at: float
    prompt: Optional[str]
    reward: Optional[float]
    modality: str


@dataclass
class BufferPluginContext:
    """Execution context passed to buffer plugins."""

    rollout_id: int
    metadata: Dict[str, Any]


class BufferPlugin(ABC):
    """Plugin interface for rollout buffer data cleaning/validation."""

    def __init__(self, *, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def process(self, batch: TrainingBatch, *, context: BufferPluginContext) -> TrainingBatch:
        """Transform or filter an incoming training batch."""

    def stats(self) -> Dict[str, Any]:
        """Optional plugin-local stats for observability."""
        return {}


def _as_index_tensor(indices: Sequence[int], *, device: torch.device) -> torch.Tensor:
    if len(indices) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(list(indices), dtype=torch.long, device=device)


def _index_if_batched(tensor: Optional[torch.Tensor], idx: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if int(tensor.shape[0]) != int(batch_size):
        # Shared tensor (e.g. FLUX image_ids), keep as-is.
        return tensor
    return tensor.index_select(0, idx.to(tensor.device))


def _index_prompt_embeddings(emb: PromptEmbeddings, idx: torch.Tensor) -> PromptEmbeddings:
    batch_size = int(emb.prompt_embeds.shape[0])
    return PromptEmbeddings(
        prompt_embeds=emb.prompt_embeds.index_select(0, idx.to(emb.prompt_embeds.device)),
        pooled_prompt_embeds=_index_if_batched(emb.pooled_prompt_embeds, idx, batch_size),
        encoder_attention_mask=_index_if_batched(emb.encoder_attention_mask, idx, batch_size),
        negative_prompt_embeds=_index_if_batched(emb.negative_prompt_embeds, idx, batch_size),
        negative_pooled_prompt_embeds=_index_if_batched(
            emb.negative_pooled_prompt_embeds,
            idx,
            batch_size,
        ),
        text_ids=_index_if_batched(emb.text_ids, idx, batch_size),
        image_ids=_index_if_batched(emb.image_ids, idx, batch_size),
    )


def _index_batch(batch: TrainingBatch, keep_indices: Sequence[int]) -> TrainingBatch:
    """Select arbitrary sample indices from a typed training batch."""
    if isinstance(batch, BackwardTrainingBatch):
        idx = _as_index_tensor(keep_indices, device=batch.trajectories.device)
        if idx.numel() == 0:
            raise ValueError("Cannot index BackwardTrainingBatch with empty indices.")

        log_probs = LogProbData.from_dict(
            {
                int(k): v.index_select(0, idx.to(v.device))
                for k, v in batch.log_probs.to_dict().items()
            }
        )

        prompts = None
        if batch.prompts is not None:
            prompts = [batch.prompts[i] for i in keep_indices]

        return BackwardTrainingBatch(
            trajectories=batch.trajectories.index_select(0, idx.to(batch.trajectories.device)),
            log_probs=log_probs,
            timesteps=batch.timesteps,
            advantages=batch.advantages.index_select(0, idx.to(batch.advantages.device)),
            embeddings=_index_prompt_embeddings(batch.embeddings, idx),
            rewards=(
                batch.rewards.index_select(0, idx.to(batch.rewards.device))
                if batch.rewards is not None
                else None
            ),
            prompts=prompts,
            num_steps=batch.num_steps,
            is_partitioned=batch.is_partitioned,
            step_indices=batch.step_indices,
            target_sde_indices=batch.target_sde_indices,
        )

    if isinstance(batch, ForwardTrainingBatch):
        idx = _as_index_tensor(keep_indices, device=batch.clean_latents.device)
        if idx.numel() == 0:
            raise ValueError("Cannot index ForwardTrainingBatch with empty indices.")

        prompts = None
        if batch.prompts is not None:
            prompts = [batch.prompts[i] for i in keep_indices]

        return ForwardTrainingBatch(
            clean_latents=batch.clean_latents.index_select(0, idx.to(batch.clean_latents.device)),
            advantages=batch.advantages.index_select(0, idx.to(batch.advantages.device)),
            embeddings=_index_prompt_embeddings(batch.embeddings, idx),
            rewards=(
                batch.rewards.index_select(0, idx.to(batch.rewards.device))
                if batch.rewards is not None
                else None
            ),
            prompts=prompts,
            timesteps=batch.timesteps,
            is_partitioned=batch.is_partitioned,
        )

    raise TypeError(f"Unsupported training batch type: {type(batch).__name__}")


def _concat_optional_embedding_field(
    tensors: Sequence[Optional[torch.Tensor]],
    *,
    batch_sizes: Sequence[int],
) -> Optional[torch.Tensor]:
    """Concatenate batched embedding fields, preserving shared non-batched tensors."""
    if not tensors or all(t is None for t in tensors):
        return None
    if any(t is None for t in tensors):
        raise ValueError("Inconsistent optional embedding field presence across batches.")

    casted: List[torch.Tensor] = [t for t in tensors if t is not None]
    is_batched = [int(t.shape[0]) == int(bs) for t, bs in zip(casted, batch_sizes)]
    if all(is_batched):
        return torch.cat(casted, dim=0)
    if any(is_batched):
        raise ValueError("Mixed batched/non-batched embedding tensor shapes across batches.")
    return casted[0]


def _concat_prompt_embeddings(embeddings: Sequence[PromptEmbeddings]) -> PromptEmbeddings:
    if not embeddings:
        raise ValueError("Cannot concatenate empty embedding list.")

    batch_sizes = [int(e.prompt_embeds.shape[0]) for e in embeddings]
    return PromptEmbeddings(
        prompt_embeds=torch.cat([e.prompt_embeds for e in embeddings], dim=0),
        pooled_prompt_embeds=_concat_optional_embedding_field(
            [e.pooled_prompt_embeds for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        encoder_attention_mask=_concat_optional_embedding_field(
            [e.encoder_attention_mask for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        negative_prompt_embeds=_concat_optional_embedding_field(
            [e.negative_prompt_embeds for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        negative_pooled_prompt_embeds=_concat_optional_embedding_field(
            [e.negative_pooled_prompt_embeds for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        text_ids=_concat_optional_embedding_field(
            [e.text_ids for e in embeddings],
            batch_sizes=batch_sizes,
        ),
        image_ids=_concat_optional_embedding_field(
            [e.image_ids for e in embeddings],
            batch_sizes=batch_sizes,
        ),
    )


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a.to(b.device), b))


def _concat_batches(batches: Sequence[TrainingBatch]) -> TrainingBatch:
    """Concatenate typed training batches along batch dimension."""
    if not batches:
        raise ValueError("Cannot concatenate empty batch list.")

    first = batches[0]
    if isinstance(first, BackwardTrainingBatch):
        if not all(isinstance(b, BackwardTrainingBatch) for b in batches):
            raise TypeError("Cannot mix BackwardTrainingBatch and ForwardTrainingBatch in concat.")

        typed: List[BackwardTrainingBatch] = [b for b in batches if isinstance(b, BackwardTrainingBatch)]
        key_set = set(int(k) for k in typed[0].log_probs.to_dict().keys())
        for b in typed[1:]:
            if set(int(k) for k in b.log_probs.to_dict().keys()) != key_set:
                raise ValueError("Inconsistent log_prob step indices across batches.")
            if not _tensor_equal(typed[0].timesteps, b.timesteps):
                raise ValueError("Inconsistent timesteps across BackwardTrainingBatch items.")

        log_probs = LogProbData.from_dict(
            {
                int(step): torch.cat(
                    [b.log_probs.to_dict()[int(step)] for b in typed],
                    dim=0,
                )
                for step in sorted(key_set)
            }
        )

        prompts = None
        if all(b.prompts is not None for b in typed):
            prompts = [p for b in typed for p in (b.prompts or [])]

        rewards = None
        if all(b.rewards is not None for b in typed):
            rewards = torch.cat([b.rewards for b in typed if b.rewards is not None], dim=0)

        return BackwardTrainingBatch(
            trajectories=torch.cat([b.trajectories for b in typed], dim=0),
            log_probs=log_probs,
            timesteps=typed[0].timesteps,
            advantages=torch.cat([b.advantages for b in typed], dim=0),
            embeddings=_concat_prompt_embeddings([b.embeddings for b in typed]),
            rewards=rewards,
            prompts=prompts,
            num_steps=typed[0].num_steps,
            is_partitioned=False,
            step_indices=typed[0].step_indices,
            target_sde_indices=typed[0].target_sde_indices,
        )

    if isinstance(first, ForwardTrainingBatch):
        if not all(isinstance(b, ForwardTrainingBatch) for b in batches):
            raise TypeError("Cannot mix ForwardTrainingBatch and BackwardTrainingBatch in concat.")

        typed_f: List[ForwardTrainingBatch] = [b for b in batches if isinstance(b, ForwardTrainingBatch)]
        base_timesteps = typed_f[0].timesteps
        if base_timesteps is not None:
            for b in typed_f[1:]:
                if b.timesteps is None or not _tensor_equal(base_timesteps, b.timesteps):
                    raise ValueError("Inconsistent timesteps across ForwardTrainingBatch items.")

        prompts_f = None
        if all(b.prompts is not None for b in typed_f):
            prompts_f = [p for b in typed_f for p in (b.prompts or [])]

        rewards_f = None
        if all(b.rewards is not None for b in typed_f):
            rewards_f = torch.cat([b.rewards for b in typed_f if b.rewards is not None], dim=0)

        return ForwardTrainingBatch(
            clean_latents=torch.cat([b.clean_latents for b in typed_f], dim=0),
            advantages=torch.cat([b.advantages for b in typed_f], dim=0),
            embeddings=_concat_prompt_embeddings([b.embeddings for b in typed_f]),
            rewards=rewards_f,
            prompts=prompts_f,
            timesteps=base_timesteps,
            is_partitioned=False,
        )

    raise TypeError(f"Unsupported training batch type: {type(first).__name__}")


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
            raise ValueError(
                f"Found {dropped} invalid samples but drop_invalid=false."
            )

        return _index_batch(batch, valid_indices)

    def stats(self) -> Dict[str, Any]:
        return {
            "filtered_samples": self.filtered_samples,
            "rejected_batches": self.rejected_batches,
        }


class RewardRangeFilterPlugin(BufferPlugin):
    """Filter samples with rewards outside configured range."""

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

        return _index_batch(batch, keep_indices)

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
        self.min_samples = max(1, int(min_samples))
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


def _parse_plugin_paths(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, Iterable):
        paths = []
        for item in raw:
            text = str(item).strip()
            if text:
                paths.append(text)
        return paths
    text = str(raw).strip()
    return [text] if text else []


def build_buffer_plugins(args: Any) -> List[BufferPlugin]:
    """Build built-in and custom rollout-buffer plugins from runtime args."""
    plugins: List[BufferPlugin] = [
        FiniteTensorFilterPlugin(
            drop_invalid=bool(getattr(args.rollout, "rollout_buffer_drop_invalid", True))
        )
    ]

    min_reward = getattr(args.rollout, "rollout_buffer_reward_min", None)
    max_reward = getattr(args.rollout, "rollout_buffer_reward_max", None)
    if min_reward is not None or max_reward is not None:
        plugins.append(
            RewardRangeFilterPlugin(
                min_reward=min_reward,
                max_reward=max_reward,
            )
        )

    plugins.append(
        MinSamplesGuardPlugin(
            min_samples=max(1, int(getattr(args.rollout, "rollout_buffer_min_samples", 1)))
        )
    )

    for path in _parse_plugin_paths(getattr(args.rollout, "rollout_buffer_plugin_paths", "")):
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


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutBufferActor:
    """Queue-backed buffer actor for rollout->train decoupling."""

    def __init__(self, args: Any):
        self.args = args
        self.partition_train_data = bool(getattr(args.ray, "partition_train_data", True))
        self.max_queue_size = max(0, int(getattr(args.rollout, "rollout_buffer_max_queue_size", 0)))
        self.plugins = build_buffer_plugins(args)

        self.grouped = bool(getattr(args.rollout, "rollout_buffer_grouped", False))
        default_group_size = max(1, int(getattr(args.algorithm, "num_samples_per_prompt", 1)))
        raw_group_size = getattr(args.rollout, "rollout_buffer_group_size", None)
        if raw_group_size is None:
            self.group_size = default_group_size
        else:
            self.group_size = max(1, int(raw_group_size))

        raw_dispatch_groups = int(getattr(args.rollout, "rollout_buffer_dispatch_groups", 0) or 0)
        prompts_per_batch = getattr(args.algorithm, "prompts_per_batch", None)
        if prompts_per_batch is None:
            raise ValueError("algorithm.prompts_per_batch must be set explicitly.")
        default_dispatch_groups = max(1, int(prompts_per_batch))
        self.dispatch_groups = raw_dispatch_groups if raw_dispatch_groups > 0 else default_dispatch_groups
        self.allow_partial_group = bool(getattr(args.rollout, "rollout_buffer_allow_partial_group", True))
        self.group_ttl_seconds = max(0.0, float(getattr(args.rollout, "rollout_buffer_group_ttl_seconds", 0.0)))
        self.max_pending_samples = max(0, int(getattr(args.rollout, "rollout_buffer_max_pending_samples", 0)))
        self.min_samples = max(1, int(getattr(args.rollout, "rollout_buffer_min_samples", 1)))

        self._dispatch_queue: Deque[BufferItem] = deque()
        self._groups: Dict[str, Deque[GroupSampleLocator]] = {}

        self._counter = 0
        self._dropped_queue_items = 0
        self._dropped_batches = 0
        self._dropped_samples = 0
        self._expired_samples = 0
        self._pending_overflow_drops = 0
        self._pushed_batches = 0
        self._popped_batches = 0
        self._pushed_samples = 0
        self._popped_samples = 0
        self._assembled_batches = 0
        self._assembled_partial_batches = 0

        # Runtime handles are attached by train loop.
        self._rollout_manager = None
        self._training_group = None

    def bind_runtime(self, *, rollout_manager: Any, training_group: Optional[Any] = None) -> Dict[str, bool]:
        """Attach rollout/training handles used by request_rollout()."""
        self._rollout_manager = rollout_manager
        if training_group is not None:
            self._training_group = training_group
        return {
            "has_rollout_manager": self._rollout_manager is not None,
            "has_training_group": self._training_group is not None,
        }

    def bind_training_group(self, training_group: Any) -> Dict[str, bool]:
        """Attach or update training-group handle."""
        self._training_group = training_group
        return {
            "has_rollout_manager": self._rollout_manager is not None,
            "has_training_group": self._training_group is not None,
        }

    def _new_item_id(self) -> str:
        self._counter += 1
        return f"buffer_item_{self._counter}"

    def _pending_samples(self) -> int:
        return sum(len(items) for items in self._groups.values())

    def _ready_groups(self) -> int:
        return sum(1 for items in self._groups.values() if len(items) >= self.group_size)

    def _detect_modality(self, batch: TrainingBatch) -> str:
        if isinstance(batch, BackwardTrainingBatch):
            dims = int(batch.trajectories.ndim)
            if dims >= 6:
                return "video"
            return "image"
        if isinstance(batch, ForwardTrainingBatch):
            dims = int(batch.clean_latents.ndim)
            if dims >= 5:
                return "video"
            return "image"
        return "unknown"

    def _normalize_prompt(self, prompt: Any) -> Optional[str]:
        if prompt is None:
            return None
        text = str(prompt).strip()
        return text if text else None

    def _group_key_for_sample(self, *, prompt: Optional[str], rollout_id: int, sample_idx: int) -> str:
        if prompt is not None:
            return f"prompt::{prompt}"
        return f"fallback::{int(rollout_id)}::{int(sample_idx)}"

    def _maybe_drop_dispatch_head(self) -> None:
        if self.max_queue_size > 0 and len(self._dispatch_queue) >= self.max_queue_size:
            dropped = self._dispatch_queue.popleft()
            self._dropped_queue_items += 1
            self._dropped_batches += 1
            self._dropped_samples += int(dropped.sample_count)

    def _queue_has_capacity(self) -> bool:
        return self.max_queue_size <= 0 or len(self._dispatch_queue) < self.max_queue_size

    def _cleanup_empty_group(self, group_key: str) -> None:
        group = self._groups.get(group_key)
        if group is not None and len(group) == 0:
            del self._groups[group_key]

    def _cleanup_expired_groups(self) -> None:
        if self.group_ttl_seconds <= 0:
            return
        now = time.time()
        cutoff = now - self.group_ttl_seconds
        for key in list(self._groups.keys()):
            group = self._groups.get(key)
            if group is None:
                continue
            while group and group[0].created_at < cutoff:
                group.popleft()
                self._expired_samples += 1
                self._dropped_samples += 1
            self._cleanup_empty_group(key)

    def _drop_oldest_pending_sample(self) -> bool:
        oldest_key: Optional[str] = None
        oldest_ts: Optional[float] = None

        for key, group in self._groups.items():
            if not group:
                continue
            ts = float(group[0].created_at)
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
                oldest_key = key

        if oldest_key is None:
            return False

        group = self._groups[oldest_key]
        group.popleft()
        self._pending_overflow_drops += 1
        self._dropped_samples += 1
        self._cleanup_empty_group(oldest_key)
        return True

    def _enforce_pending_capacity(self) -> None:
        if self.max_pending_samples <= 0:
            return
        while self._pending_samples() > self.max_pending_samples:
            if not self._drop_oldest_pending_sample():
                break

    def _pop_group_samples(self, group_key: str, count: int) -> List[GroupSampleLocator]:
        group = self._groups.get(group_key)
        if group is None:
            return []

        items: List[GroupSampleLocator] = []
        take = max(0, int(count))
        for _ in range(min(take, len(group))):
            items.append(group.popleft())
        self._cleanup_empty_group(group_key)
        return items

    def _select_dispatch_plan(self, *, allow_partial: bool) -> Tuple[List[Tuple[str, int]], bool]:
        plan: List[Tuple[str, int]] = []
        for key in list(self._groups.keys()):
            group = self._groups.get(key)
            if group is None or len(group) < self.group_size:
                continue
            plan.append((key, self.group_size))
            if len(plan) >= self.dispatch_groups:
                return plan, False

        if plan:
            return plan, False

        if not allow_partial:
            return [], False

        for key in list(self._groups.keys()):
            group = self._groups.get(key)
            if group is None or len(group) < self.min_samples:
                continue
            return [(key, min(len(group), self.group_size))], True

        return [], False

    def _materialize_batch_from_locators(self, locators: Sequence[GroupSampleLocator]) -> TrainingBatch:
        if not locators:
            raise ValueError("Cannot materialize batch from empty locator list.")

        batch_cache: Dict[Any, TrainingBatch] = {}
        sample_batches: List[TrainingBatch] = []
        for locator in locators:
            base_batch = batch_cache.get(locator.batch_ref)
            if base_batch is None:
                base_batch = ray.get(locator.batch_ref)
                batch_cache[locator.batch_ref] = base_batch

            sample_idx = int(locator.sample_idx)
            if sample_idx < 0 or sample_idx >= int(base_batch.batch_size):
                raise IndexError(
                    f"Sample index out of range while materializing grouped batch: "
                    f"sample_idx={sample_idx}, batch_size={base_batch.batch_size}"
                )
            sample_batches.append(_index_batch(base_batch, [sample_idx]))

        out = _concat_batches(sample_batches)
        out.validate()
        return out

    def _enqueue_dispatch_item(
        self,
        *,
        rollout_id: int,
        batch: TrainingBatch,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._maybe_drop_dispatch_head()
        item = BufferItem(
            item_id=self._new_item_id(),
            rollout_id=int(rollout_id),
            batch_ref=ray.put(batch),
            sample_count=int(batch.batch_size),
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self._dispatch_queue.append(item)
        self._assembled_batches += 1

    def _promote_ready_groups(self, *, allow_partial: bool) -> int:
        if not self.grouped:
            return 0

        self._cleanup_expired_groups()
        self._enforce_pending_capacity()
        promoted = 0

        while self._queue_has_capacity():
            plan, partial = self._select_dispatch_plan(allow_partial=allow_partial)
            if not plan:
                break

            selected: List[GroupSampleLocator] = []
            selected_group_keys: List[str] = []
            for key, take in plan:
                selected_group_keys.append(key)
                selected.extend(self._pop_group_samples(key, take))

            if not selected:
                break

            batch = self._materialize_batch_from_locators(selected)
            self._enqueue_dispatch_item(
                rollout_id=max(int(s.rollout_id) for s in selected),
                batch=batch,
                metadata={
                    "group_keys": selected_group_keys,
                    "partial_group": bool(partial),
                    "group_size": int(self.group_size),
                },
            )
            if partial:
                self._assembled_partial_batches += 1
            promoted += 1

            # Partial fallback is only used to make progress; keep one partial per call.
            if partial:
                break

        return promoted

    def _push_grouped(
        self,
        *,
        rollout_id: int,
        current: TrainingBatch,
        context: BufferPluginContext,
    ) -> Dict[str, Any]:
        sample_count = int(current.batch_size)
        if sample_count <= 0:
            raise ValueError("Processed batch is empty.")

        created_at = time.time()
        batch_ref = ray.put(current)
        modality = self._detect_modality(current)

        prompts = current.prompts
        if prompts is not None and len(prompts) != sample_count:
            logger.warning(
                "Training batch prompts length %d != batch_size %d; grouped prompt keying disabled for this push.",
                len(prompts),
                sample_count,
            )
            prompts = None

        rewards_tensor = current.rewards
        for sample_idx in range(sample_count):
            prompt = self._normalize_prompt(prompts[sample_idx]) if prompts is not None else None
            reward = (
                float(rewards_tensor[sample_idx].item())
                if rewards_tensor is not None
                else None
            )
            group_key = self._group_key_for_sample(
                prompt=prompt,
                rollout_id=int(rollout_id),
                sample_idx=sample_idx,
            )
            locator = GroupSampleLocator(
                batch_ref=batch_ref,
                sample_idx=sample_idx,
                rollout_id=int(rollout_id),
                created_at=created_at,
                prompt=prompt,
                reward=reward,
                modality=modality,
            )
            if group_key not in self._groups:
                self._groups[group_key] = deque()
            self._groups[group_key].append(locator)

        # Promote full groups immediately; partial fallback happens on pop only.
        promoted = self._promote_ready_groups(allow_partial=False)
        self._pushed_batches += 1
        self._pushed_samples += sample_count

        return {
            "accepted": True,
            "rollout_id": int(rollout_id),
            "sample_count": sample_count,
            "ready_queue_size": len(self._dispatch_queue),
            "pending_groups": len(self._groups),
            "pending_samples": self._pending_samples(),
            "promoted_batches": promoted,
            "grouped_mode": True,
            "metadata": context.metadata,
        }

    def size(self) -> int:
        return len(self._dispatch_queue)

    def clear(self) -> None:
        self._dispatch_queue.clear()
        self._groups.clear()

    def push(
        self,
        *,
        rollout_id: int,
        train_data: TrainingBatch,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Push one training batch through plugin chain into the buffer."""
        context = BufferPluginContext(
            rollout_id=int(rollout_id),
            metadata=dict(metadata or {}),
        )

        try:
            current = train_data
            for plugin in self.plugins:
                current = plugin.process(current, context=context)

            current.validate()

            if self.grouped:
                return self._push_grouped(
                    rollout_id=rollout_id,
                    current=current,
                    context=context,
                )

            sample_count = int(current.batch_size)
            if sample_count <= 0:
                raise ValueError("Processed batch is empty.")

            self._maybe_drop_dispatch_head()
            item = BufferItem(
                item_id=self._new_item_id(),
                rollout_id=int(rollout_id),
                batch_ref=ray.put(current),
                sample_count=sample_count,
                created_at=time.time(),
                metadata=context.metadata,
            )
            self._dispatch_queue.append(item)
            self._pushed_batches += 1
            self._pushed_samples += sample_count

            return {
                "accepted": True,
                "item_id": item.item_id,
                "rollout_id": item.rollout_id,
                "sample_count": sample_count,
                "queue_size": len(self._dispatch_queue),
                "grouped_mode": False,
            }
        except Exception as exc:
            logger.warning("RolloutBufferActor drop rollout_id=%s due to: %s", rollout_id, exc)
            self._dropped_batches += 1
            return {
                "accepted": False,
                "rollout_id": int(rollout_id),
                "error": str(exc),
                "queue_size": len(self._dispatch_queue),
            }

    def pop(
        self,
        *,
        dp_size: Optional[int] = None,
        consumer_spec: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Pop one training batch, optionally partitioned using consumer spec."""
        if self.grouped and not self._dispatch_queue:
            self._promote_ready_groups(allow_partial=self.allow_partial_group)

        if not self._dispatch_queue:
            return None

        item = self._dispatch_queue.popleft()
        self._popped_batches += 1
        self._popped_samples += int(item.sample_count)

        resolved_dp_size, partition_train_data, partition_mode = _resolve_consumer_spec(
            default_partition_train_data=self.partition_train_data,
            dp_size=dp_size,
            consumer_spec=consumer_spec,
        )

        if not partition_train_data or not resolved_dp_size:
            return {
                "item_id": item.item_id,
                "rollout_id": int(item.rollout_id),
                "sample_count": int(item.sample_count),
                "metadata": dict(item.metadata),
                "training_data": item.batch_ref,
                "consumer_spec": {
                    "dp_size": resolved_dp_size,
                    "partition_train_data": partition_train_data,
                    "partition_mode": partition_mode,
                },
            }

        train_data = ray.get(item.batch_ref)
        training_data_ref = _put_training_data(
            train_data=train_data,
            dp_size=resolved_dp_size,
            partition_train_data=partition_train_data,
            partition_mode=partition_mode,
        )
        return {
            "item_id": item.item_id,
            "rollout_id": int(item.rollout_id),
            "sample_count": int(item.sample_count),
            "metadata": dict(item.metadata),
            "training_data": training_data_ref,
            "consumer_spec": {
                "dp_size": resolved_dp_size,
                "partition_train_data": partition_train_data,
                "partition_mode": partition_mode,
            },
        }

    def pop_training_data(
        self,
        *,
        dp_size: Optional[int] = None,
        consumer_spec: Optional[Dict[str, Any]] = None,
        expected_rollout_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Pop next ready training payload with optional rollout-id guard."""
        payload = self.pop(dp_size=dp_size, consumer_spec=consumer_spec)
        if payload is None:
            raise RuntimeError("Rollout buffer is empty; no training data available.")
        if expected_rollout_id is not None:
            got = int(payload.get("rollout_id", -1))
            if got != int(expected_rollout_id):
                raise RuntimeError(
                    "Rollout/training payload mismatch: "
                    f"expected rollout_id={expected_rollout_id}, got {got}. "
                    "Disable strict alignment when using grouped rollout buffer dispatch."
                )
        return payload

    def push_and_pop(
        self,
        *,
        rollout_id: int,
        train_data: TrainingBatch,
        dp_size: Optional[int] = None,
        consumer_spec: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Convenience API: push current rollout and pop next ready batch."""
        push_result = self.push(
            rollout_id=rollout_id,
            train_data=train_data,
            metadata=metadata,
        )
        if not push_result.get("accepted", False):
            raise RuntimeError(
                f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
            )
        return self.pop(dp_size=dp_size, consumer_spec=consumer_spec)

    def get_stats(self) -> Dict[str, Any]:
        plugin_stats = {plugin.name: plugin.stats() for plugin in self.plugins}

        pending_modalities: Dict[str, int] = {}
        reward_sum = 0.0
        reward_count = 0
        for group in self._groups.values():
            for locator in group:
                pending_modalities[locator.modality] = pending_modalities.get(locator.modality, 0) + 1
                if locator.reward is not None:
                    reward_sum += float(locator.reward)
                    reward_count += 1

        avg_pending_reward = reward_sum / reward_count if reward_count > 0 else None

        return {
            "grouped_mode": self.grouped,
            "has_rollout_manager": self._rollout_manager is not None,
            "has_training_group": self._training_group is not None,
            "queue_size": len(self._dispatch_queue),
            "pushed_batches": self._pushed_batches,
            "popped_batches": self._popped_batches,
            "pushed_samples": self._pushed_samples,
            "popped_samples": self._popped_samples,
            "assembled_batches": self._assembled_batches,
            "assembled_partial_batches": self._assembled_partial_batches,
            "dropped_queue_items": self._dropped_queue_items,
            "dropped_batches": self._dropped_batches,
            "dropped_samples": self._dropped_samples,
            "expired_samples": self._expired_samples,
            "pending_overflow_drops": self._pending_overflow_drops,
            "max_queue_size": self.max_queue_size,
            "partition_train_data": self.partition_train_data,
            "plugins": plugin_stats,
            "group_size": self.group_size,
            "dispatch_groups": self.dispatch_groups,
            "allow_partial_group": self.allow_partial_group,
            "group_ttl_seconds": self.group_ttl_seconds,
            "max_pending_samples": self.max_pending_samples,
            "pending_group_count": len(self._groups),
            "ready_group_count": self._ready_groups(),
            "pending_sample_count": self._pending_samples(),
            "pending_modality_counts": pending_modalities,
            "avg_pending_reward": avg_pending_reward,
        }

    def dispose(self) -> None:
        """Release buffered data and runtime handles."""
        self.clear()
        self._rollout_manager = None
        self._training_group = None


def create_rollout_buffer_actor(args: Any):
    """Factory for rollout buffer actor."""
    return RolloutBufferActor.options(num_cpus=1, num_gpus=0).remote(args)


__all__ = [
    "BufferPlugin",
    "BufferPluginContext",
    "FiniteTensorFilterPlugin",
    "RewardRangeFilterPlugin",
    "MinSamplesGuardPlugin",
    "RolloutBufferActor",
    "build_buffer_plugins",
    "create_rollout_buffer_actor",
]
