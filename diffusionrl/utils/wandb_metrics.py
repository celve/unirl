"""Helpers for building structured WandB metrics in train loops."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch

from diffusionrl.types.training_batch import TrainingBatch


def _coerce_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.numel() == 0:
            return None
        if tensor.numel() == 1:
            return float(tensor.item())
        return float(tensor.to(dtype=torch.float32).mean().item())
    return None


def flatten_numeric_metrics(
    payload: Dict[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """Flatten nested dict payload into numeric metrics only."""
    output: Dict[str, float] = {}

    def _walk(node: Dict[str, Any], node_prefix: str) -> None:
        for key, value in node.items():
            metric_key = f"{node_prefix}{key}" if node_prefix else str(key)
            if isinstance(value, dict):
                _walk(value, f"{metric_key}/")
                continue
            scalar = _coerce_scalar(value)
            if scalar is not None:
                output[metric_key] = scalar

    _walk(payload, prefix)
    return output


def _iter_batches(training_data: Any) -> Iterable[TrainingBatch]:
    if isinstance(training_data, list):
        for item in training_data:
            if isinstance(item, TrainingBatch):
                yield item
        return
    if isinstance(training_data, TrainingBatch):
        yield training_data


def _tensor_stats(prefix: str, tensor: Optional[torch.Tensor]) -> Dict[str, float]:
    if tensor is None or (not torch.is_tensor(tensor)) or tensor.numel() == 0:
        return {}
    flat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
        f"{prefix}_min": float(flat.min().item()),
        f"{prefix}_max": float(flat.max().item()),
    }


def _zero_std_group_counts_from_ids(
    rewards: torch.Tensor,
    group_ids: Optional[List[str]],
) -> tuple[int, int]:
    if not isinstance(group_ids, list) or len(group_ids) != int(rewards.shape[0]):
        return 0, 0
    ordered: Dict[str, List[float]] = {}
    rewards_f = rewards.to(dtype=torch.float32).reshape(-1)
    for sample_idx, raw_group_id in enumerate(group_ids):
        group_id = str(raw_group_id).strip()
        if not group_id:
            continue
        ordered.setdefault(group_id, []).append(float(rewards_f[sample_idx].item()))
    if not ordered:
        return 0, 0
    zero_std = 0
    for values in ordered.values():
        if len(values) <= 1:
            continue
        std = torch.tensor(values, dtype=torch.float32).std(unbiased=False)
        if float(std.item()) <= 1e-8:
            zero_std += 1
    return zero_std, len(ordered)


def compute_rollout_batch_metrics(
    *,
    training_data: Any,
) -> Dict[str, float]:
    """Build rollout metrics from typed training batch (or partition list).

    For multi-reward runs, also emits ``reward_<component>_{mean,std,min,max}``
    per key in ``component_rewards``. Component names with ``/`` are flattened
    to ``_`` so the keys stay leaf metrics under the ``rollout/`` prefix.
    """
    metrics: Dict[str, float] = {}

    reward_tensors: List[torch.Tensor] = []
    advantage_tensors: List[torch.Tensor] = []
    component_tensors: Dict[str, List[torch.Tensor]] = {}
    total_samples = 0
    zero_std_groups = 0
    total_groups = 0
    sde_selected = 0
    sde_total = 0

    for batch in _iter_batches(training_data):
        total_samples += int(getattr(batch, "batch_size", 0))

        rewards = getattr(batch, "rewards", None)
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_f = rewards.detach().to(dtype=torch.float32).reshape(-1).cpu()
            reward_tensors.append(rewards_f)
            zero_cnt, group_cnt = _zero_std_group_counts_from_ids(
                rewards_f,
                getattr(batch, "group_ids", None),
            )
            zero_std_groups += zero_cnt
            total_groups += group_cnt

        advantages = getattr(batch, "advantages", None)
        if torch.is_tensor(advantages) and advantages.numel() > 0:
            advantage_tensors.append(advantages.detach().to(dtype=torch.float32).reshape(-1).cpu())

        component_rewards = getattr(batch, "component_rewards", None)
        if isinstance(component_rewards, dict):
            for name, tensor in component_rewards.items():
                if not torch.is_tensor(tensor) or tensor.numel() == 0:
                    continue
                safe_name = str(name).replace("/", "_")
                component_tensors.setdefault(safe_name, []).append(
                    tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
                )

        if batch.has_trajectory_rl_data:
            sde_selected += len(batch.sde_indices)
            sde_total += max(int(batch.trajectory_store.total_positions) - 1, 0)

    metrics["num_samples"] = float(total_samples)

    if reward_tensors:
        rewards_cat = torch.cat(reward_tensors, dim=0)
        metrics.update(_tensor_stats("reward", rewards_cat))

    for safe_name, tensors in component_tensors.items():
        cat = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        metrics.update(_tensor_stats(f"reward_{safe_name}", cat))

    if advantage_tensors:
        advantages_cat = torch.cat(advantage_tensors, dim=0)
        metrics.update(_tensor_stats("advantage", advantages_cat))

    if total_groups > 0:
        metrics["zero_std_group_ratio"] = float(zero_std_groups) / float(total_groups)
        metrics["zero_std_group_count"] = float(zero_std_groups)
        metrics["group_count"] = float(total_groups)

    if sde_total > 0:
        metrics["sde_selected_steps"] = float(sde_selected)
        metrics["sde_total_steps"] = float(sde_total)
        metrics["sde_selected_ratio"] = float(sde_selected) / float(sde_total)

    return metrics


def compute_rollout_resp_metrics(*, resp: Any) -> Dict[str, float]:
    """Build rollout metrics directly from a :class:`RolloutResp`.

    Mirrors :func:`compute_rollout_batch_metrics` but consumes the
    new-design path's ``RolloutResp`` instead of a ``TrainingBatch``
    (or list thereof). Emits the same wandb key shape under the
    ``rollout/`` prefix:

    - ``num_samples``
    - ``reward_{mean,std,min,max}``
    - ``advantage_{mean,std,min,max}``
    - ``reward_<component>_{mean,std,min,max}`` per
      ``resp.component_rewards`` entry (``/`` flattened to ``_``)
    - ``group_count``, ``zero_std_group_ratio``,
      ``zero_std_group_count`` when ``resp.group_ids`` is populated

    Skips the legacy ``has_trajectory_rl_data`` / ``sde_indices`` block
    since neither concept exists on ``RolloutResp``.
    """
    metrics: Dict[str, float] = {}

    metrics["num_samples"] = float(int(getattr(resp, "batch_size", 0)))

    rewards = getattr(resp, "rewards", None)
    if torch.is_tensor(rewards) and rewards.numel() > 0:
        rewards_f = rewards.detach().to(dtype=torch.float32).reshape(-1).cpu()
        metrics.update(_tensor_stats("reward", rewards_f))
        zero_cnt, group_cnt = _zero_std_group_counts_from_ids(
            rewards_f,
            getattr(resp, "group_ids", None),
        )
        if group_cnt > 0:
            metrics["zero_std_group_ratio"] = float(zero_cnt) / float(group_cnt)
            metrics["zero_std_group_count"] = float(zero_cnt)
            metrics["group_count"] = float(group_cnt)

    advantages = getattr(resp, "advantages", None)
    if torch.is_tensor(advantages) and advantages.numel() > 0:
        adv_f = advantages.detach().to(dtype=torch.float32).reshape(-1).cpu()
        metrics.update(_tensor_stats("advantage", adv_f))

    component_rewards = getattr(resp, "component_rewards", None)
    if isinstance(component_rewards, dict):
        for name, tensor in component_rewards.items():
            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                continue
            safe_name = str(name).replace("/", "_")
            cat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"reward_{safe_name}", cat))

    return metrics


_BUFFER_CORE_KEYS = (
    "queue_size",
    "pushed_batches",
    "popped_batches",
    "pushed_samples",
    "popped_samples",
    "dropped_queue_items",
    "dropped_batches",
    "dropped_samples",
)


def build_buffer_metrics(
    stats: Optional[Dict[str, Any]],
    prefix: str = "buffer/",
) -> Dict[str, float]:
    """Extract numeric rollout-buffer health metrics."""
    if not isinstance(stats, dict):
        return {}

    metrics: Dict[str, float] = {}
    for key in _BUFFER_CORE_KEYS:
        scalar = _coerce_scalar(stats.get(key))
        if scalar is not None:
            metrics[f"{prefix}{key}"] = scalar

    plugins = stats.get("plugins")
    if isinstance(plugins, dict):
        metrics.update(flatten_numeric_metrics(plugins, prefix=f"{prefix}plugin/"))

    return metrics


def build_sync_metrics(
    sync_result: Any,
    prefix: str = "sync/",
) -> Dict[str, float]:
    """Flatten weight-sync result into numeric metrics."""
    if sync_result is None:
        return {}

    metrics: Dict[str, float] = {}
    for key in ("elapsed_ms", "version", "rollout_id"):
        scalar = _coerce_scalar(getattr(sync_result, key, None))
        if scalar is not None:
            metrics[f"{prefix}{key}"] = scalar

    extra = getattr(sync_result, "extra", None)
    if isinstance(extra, dict):
        metrics.update(flatten_numeric_metrics(extra, prefix=f"{prefix}extra/"))
    return metrics
