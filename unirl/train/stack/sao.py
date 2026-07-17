"""Composite actor/critic training stack for agentic SAO.

The ordinary :class:`~unirl.train.stack.base.TrainStack` owns one model and
interprets ``num_updates_per_batch`` as disjoint minibatches.  SAO instead needs
two optimizer steps over the *same* critic batch, a fresh value prediction, and
then one actor step.  This stack keeps that ordering on each train worker so
packed token signals never round-trip through the driver.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from unirl.algorithms import AlgorithmStepResult
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.stack.base import TrainStepResult, _align_track_to_model
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.segments.text import TextSegment
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


def _action_parts(trajectory: Sample) -> List[Part]:
    """Return non-empty generated text Parts in temporal order."""
    out: List[Part] = []
    for part in trajectory.gen_parts():
        segment = part.segment
        if not isinstance(segment, TextSegment) or segment.tokens is None:
            continue
        if int(segment.tokens.numel()) == 0:
            continue
        if int(part.batch_size) != 1:
            raise ValueError(
                "SAO expects one trajectory per prompt and one row per generated turn; "
                f"got generated Part batch_size={part.batch_size}."
            )
        out.append(part)
    return out


def _part_token_count(part: Part, *, mask_field: str) -> int:
    segment = part.segment
    if not isinstance(segment, TextSegment) or segment.tokens is None:
        return 0
    # Actor advantages may legitimately be exactly zero; they are signals, not
    # a structural mask.  DIS-rejected and zero-advantage tokens stay in the
    # denominator, so only loss_mask controls actor token eligibility.
    mask = segment.loss_mask if mask_field == "token_advantages" else getattr(segment, mask_field, None)
    if mask is None:
        return int(segment.tokens.numel())
    return int((mask != 0).sum().item())


def _trajectory_replay_cost(trajectory: Sample) -> int:
    """Approximate a trajectory's teacher-forced work in real tokens.

    Each turn is replayed independently in the v1 implementation, so growing
    prompts are intentionally counted repeatedly.  This is also the replay
    amplification quantity surfaced in metrics.
    """
    total = 0
    for part in _action_parts(trajectory):
        segment = part.segment
        assert isinstance(segment, TextSegment) and segment.tokens is not None
        response = int(segment.tokens.numel())
        prompt = part.conditions.get("prompt") if isinstance(part.conditions, dict) else None
        attention_mask = getattr(prompt, "attention_mask", None)
        if isinstance(attention_mask, torch.Tensor):
            prompt_tokens = int(attention_mask.long().sum().item())
        else:
            prompt_tokens = 0
        total += prompt_tokens + response
    return max(1, total)


def plan_trajectory_micros(
    costs: Sequence[int],
    *,
    micro_batch_size: int,
    token_budget: Optional[int] = None,
    target_micros: Optional[int] = None,
) -> List[List[int]]:
    """Build deterministic trajectory-granularity microbatches.

    ``target_micros`` is used after a DP all-reduce(MAX) so every rank executes
    the same number of FSDP forwards.  All ranks receive the same number of
    trajectories, therefore the requested count can always be reached by
    seeding one trajectory per bin.
    """
    n = len(costs)
    if n == 0:
        raise ValueError("plan_trajectory_micros: empty trajectory batch")
    if int(micro_batch_size) < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")

    order = sorted(range(n), key=lambda i: (-int(costs[i]), i))
    if token_budget is None:
        groups = [order[i : i + int(micro_batch_size)] for i in range(0, n, int(micro_batch_size))]
    else:
        if int(token_budget) < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}")
        oversized = [(idx, int(cost)) for idx, cost in enumerate(costs) if int(cost) > int(token_budget)]
        if oversized:
            idx, cost = oversized[0]
            raise ValueError(
                "plan_trajectory_micros: whole trajectory "
                f"at index {idx} has replay cost {cost}, which exceeds token_budget={int(token_budget)}; "
                "SAO trajectories are indivisible and cannot be split across microbatches"
            )
        groups: List[List[int]] = []
        totals: List[int] = []
        for idx in order:
            cost = int(costs[idx])
            placed = False
            for j, current in enumerate(totals):
                if len(groups[j]) < int(micro_batch_size) and current + cost <= int(token_budget):
                    groups[j].append(idx)
                    totals[j] += cost
                    placed = True
                    break
            if not placed:
                groups.append([idx])
                totals.append(cost)

    if target_micros is None or int(target_micros) == len(groups):
        return groups
    k = int(target_micros)
    if not 1 <= k <= n:
        raise ValueError(f"target_micros={k} must be in [1, {n}]")

    # DP parity only ever asks a rank to increase its local micro count. Split
    # existing valid bins in that common path, which cannot violate either the
    # count cap or token budget.
    if k > len(groups):
        expanded = [list(group) for group in groups]
        while len(expanded) < k:
            candidates = [j for j, group in enumerate(expanded) if len(group) > 1]
            if not candidates:
                raise RuntimeError("cannot split trajectory micros to requested parity")
            j = max(
                candidates,
                key=lambda x: (len(expanded[x]), sum(int(costs[i]) for i in expanded[x]), -x),
            )
            expanded.append([expanded[j].pop()])
        return expanded

    # General public-helper path for reducing the number of bins. Fail rather
    # than silently exceed a configured count/token bound when no valid merge
    # exists.
    balanced = [[idx] for idx in order[:k]]
    totals = [int(costs[idx]) for idx in order[:k]]
    for idx in order[k:]:
        candidates = [
            j
            for j in range(k)
            if len(balanced[j]) < int(micro_batch_size)
            and (token_budget is None or totals[j] + int(costs[idx]) <= int(token_budget))
        ]
        if not candidates:
            raise ValueError(
                f"cannot rebalance {len(groups)} valid micros into target_micros={k} "
                "without exceeding micro_batch_size/token_budget"
            )
        j = min(candidates, key=lambda x: (totals[x], x))
        balanced[j].append(idx)
        totals[j] += int(costs[idx])
    return balanced


def _dp_world_size() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return int(dist.get_world_size())


def _global_int(value: int, *, op: dist.ReduceOp = dist.ReduceOp.SUM) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def _all_reduce_named(
    values: Mapping[str, float],
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> Dict[str, float]:
    """All-reduce detached scalar statistics in one collective.

    The SAO loss itself is normalized separately before backward.  This helper
    is deliberately reporting-only: callers pass Python scalars obtained from
    detached algorithm results, so adding diagnostics cannot alter autograd.
    """
    if not values:
        return {}
    keys = sorted(values)
    if not (dist.is_available() and dist.is_initialized()):
        return {key: float(values[key]) for key in keys}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    reduced = tensor.cpu().tolist()
    return dict(zip(keys, (float(value) for value in reduced)))


def _numeric_metrics(result: AlgorithmStepResult) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in result.metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, bool):
            metrics[key] = float(value)
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _add_generic_metric_stats(
    *,
    rows: Sequence[Tuple[Mapping[str, float], int]],
    handled: set[str],
    sums: Dict[str, float],
    minima: Dict[str, float],
    maxima: Dict[str, float],
) -> Dict[str, str]:
    """Accumulate unknown diagnostics by naming convention.

    Counts/sums are additive, fractions and ordinary diagnostics are
    structural-token weighted, and extrema retain their natural reduction.
    The returned mapping records how each public key should be finalized.
    """
    modes: Dict[str, str] = {}
    keys = sorted({key for metrics, _ in rows for key in metrics if key not in handled})
    for key in keys:
        entries = [(metrics[key], float(weight)) for metrics, weight in rows if key in metrics]
        entries = [(value, weight) for value, weight in entries if math.isfinite(value) and weight > 0]
        if not entries:
            continue
        if key.endswith(("_tokens", "_count", "_sum")):
            sums[f"generic/value/{key}"] = sum(value for value, _ in entries)
            modes[key] = "sum"
        elif key.endswith("_min"):
            minima[f"generic/value/{key}"] = min(value for value, _ in entries)
            modes[key] = "min"
        elif key.endswith("_max"):
            maxima[f"generic/value/{key}"] = max(value for value, _ in entries)
            modes[key] = "max"
        else:
            sums[f"generic/numerator/{key}"] = sum(value * weight for value, weight in entries)
            sums[f"generic/denominator/{key}"] = sum(weight for _, weight in entries)
            modes[key] = "mean"
    return modes


def _finalize_generic_metrics(
    *,
    modes: Mapping[str, str],
    sums: Mapping[str, float],
    minima: Mapping[str, float],
    maxima: Mapping[str, float],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, mode in modes.items():
        if mode == "sum":
            metrics[key] = sums[f"generic/value/{key}"]
        elif mode == "min":
            metrics[key] = minima[f"generic/value/{key}"]
        elif mode == "max":
            metrics[key] = maxima[f"generic/value/{key}"]
        else:
            numerator = sums[f"generic/numerator/{key}"]
            denominator = sums[f"generic/denominator/{key}"]
            metrics[key] = numerator / max(1.0, denominator)
    return metrics


_ACTOR_REPORT_KEYS = {
    "policy_loss",
    "dis_structural_tokens",
    "dis_accepted_tokens",
    "dis_accept_fraction",
    "dis_reject_lower_fraction",
    "dis_reject_upper_fraction",
    "dis_nonfinite_fraction",
    "ratio_mean",
    "ratio_std",
    "ratio_min",
    "ratio_max",
    "approx_kl",
    "rollout_replay_logp_absdiff_mean",
    "rollout_replay_logp_absdiff_max",
}


def _global_actor_report(
    results: Sequence[AlgorithmStepResult],
    token_counts: Sequence[int],
    *,
    global_denominator: int,
) -> Tuple[float, Mapping[str, object]]:
    """Return globally token-weighted actor loss and DIS diagnostics."""
    rows = [(_numeric_metrics(result), int(count)) for result, count in zip(results, token_counts)]
    present = {key for metrics, _ in rows for key in metrics}
    sums: Dict[str, float] = {
        "loss_sum": sum(float(result.loss) * int(count) for result, count in zip(results, token_counts))
    }
    # Every rank must enter collectives with identical keys, including a rank
    # whose entire local shard has non-finite ratios. Sentinels are removed
    # after the globally reduced finite count is known.
    minima: Dict[str, float] = {"ratio_min": float("inf")} if "ratio_min" in present else {}
    maxima: Dict[str, float] = {}
    if "ratio_max" in present:
        maxima["ratio_max"] = float("-inf")
    if "rollout_replay_logp_absdiff_max" in present:
        maxima["absdiff_max"] = float("-inf")

    for metrics, count in rows:
        structural = float(count)
        nonfinite = structural * metrics.get("dis_nonfinite_fraction", 0.0)
        finite = max(0.0, structural - nonfinite)
        accepted = metrics.get(
            "dis_accepted_tokens",
            structural * metrics.get("dis_accept_fraction", 0.0),
        )
        sums["dis_accepted"] = sums.get("dis_accepted", 0.0) + accepted
        sums["dis_reject_lower"] = sums.get("dis_reject_lower", 0.0) + structural * metrics.get(
            "dis_reject_lower_fraction", 0.0
        )
        sums["dis_reject_upper"] = sums.get("dis_reject_upper", 0.0) + structural * metrics.get(
            "dis_reject_upper_fraction", 0.0
        )
        sums["dis_nonfinite"] = sums.get("dis_nonfinite", 0.0) + nonfinite
        sums["ratio_count"] = sums.get("ratio_count", 0.0) + finite

        if "ratio_mean" in metrics:
            mean = metrics["ratio_mean"]
            sums["ratio_sum"] = sums.get("ratio_sum", 0.0) + mean * finite
            if "ratio_std" in metrics:
                std = metrics["ratio_std"]
                sums["ratio_square_sum"] = sums.get("ratio_square_sum", 0.0) + (std * std + mean * mean) * finite
        if finite > 0 and "ratio_min" in metrics:
            minima["ratio_min"] = min(minima.get("ratio_min", float("inf")), metrics["ratio_min"])
        if finite > 0 and "ratio_max" in metrics:
            maxima["ratio_max"] = max(maxima.get("ratio_max", float("-inf")), metrics["ratio_max"])
        if "approx_kl" in metrics:
            sums["approx_kl_sum"] = sums.get("approx_kl_sum", 0.0) + metrics["approx_kl"] * finite
        if "rollout_replay_logp_absdiff_mean" in metrics:
            sums["absdiff_sum"] = sums.get("absdiff_sum", 0.0) + metrics["rollout_replay_logp_absdiff_mean"] * finite
        if finite > 0 and "rollout_replay_logp_absdiff_max" in metrics:
            maxima["absdiff_max"] = max(
                maxima.get("absdiff_max", float("-inf")),
                metrics["rollout_replay_logp_absdiff_max"],
            )

    generic_modes = _add_generic_metric_stats(
        rows=rows,
        handled=_ACTOR_REPORT_KEYS,
        sums=sums,
        minima=minima,
        maxima=maxima,
    )
    sums = _all_reduce_named(sums)
    minima = _all_reduce_named(minima, op=dist.ReduceOp.MIN)
    maxima = _all_reduce_named(maxima, op=dist.ReduceOp.MAX)

    denominator = float(global_denominator)
    global_loss = sums["loss_sum"] / denominator
    metrics: Dict[str, object] = _finalize_generic_metrics(
        modes=generic_modes,
        sums=sums,
        minima=minima,
        maxima=maxima,
    )
    if "policy_loss" in present:
        metrics["policy_loss"] = global_loss
    if "dis_structural_tokens" in present:
        metrics["dis_structural_tokens"] = global_denominator
    if "dis_accepted_tokens" in present:
        metrics["dis_accepted_tokens"] = sums["dis_accepted"]
    if "dis_accept_fraction" in present:
        metrics["dis_accept_fraction"] = sums["dis_accepted"] / denominator
    if "dis_reject_lower_fraction" in present:
        metrics["dis_reject_lower_fraction"] = sums["dis_reject_lower"] / denominator
    if "dis_reject_upper_fraction" in present:
        metrics["dis_reject_upper_fraction"] = sums["dis_reject_upper"] / denominator
    if "dis_nonfinite_fraction" in present:
        metrics["dis_nonfinite_fraction"] = sums["dis_nonfinite"] / denominator

    ratio_count = sums["ratio_count"]
    if "ratio_mean" in present:
        ratio_mean = sums.get("ratio_sum", 0.0) / max(1.0, ratio_count)
        metrics["ratio_mean"] = ratio_mean
        if "ratio_std" in present:
            ratio_second = sums.get("ratio_square_sum", 0.0) / max(1.0, ratio_count)
            metrics["ratio_std"] = math.sqrt(max(0.0, ratio_second - ratio_mean * ratio_mean))
    if "ratio_min" in present:
        metrics["ratio_min"] = minima.get("ratio_min", 0.0) if ratio_count > 0 else 0.0
    if "ratio_max" in present:
        metrics["ratio_max"] = maxima.get("ratio_max", 0.0) if ratio_count > 0 else 0.0
    if "approx_kl" in present:
        metrics["approx_kl"] = sums.get("approx_kl_sum", 0.0) / max(1.0, ratio_count)
    if "rollout_replay_logp_absdiff_mean" in present:
        metrics["rollout_replay_logp_absdiff_mean"] = sums.get("absdiff_sum", 0.0) / max(1.0, ratio_count)
    if "rollout_replay_logp_absdiff_max" in present:
        metrics["rollout_replay_logp_absdiff_max"] = maxima.get("absdiff_max", 0.0) if ratio_count > 0 else 0.0
    return global_loss, metrics


_CRITIC_REPORT_KEYS = {
    "critic_loss",
    "value_mse",
    "value_structural_tokens",
    "value_finite_tokens",
    "value_nonfinite_fraction",
    "value_explained_variance",
    "value_prediction_mean",
    "value_target_mean",
    "value_target_std",
}


def _global_critic_report(
    results: Sequence[AlgorithmStepResult],
    token_counts: Sequence[int],
    *,
    global_denominator: int,
) -> Tuple[float, Mapping[str, object]]:
    """Return global critic loss and moment-correct value diagnostics."""
    rows = [(_numeric_metrics(result), int(count)) for result, count in zip(results, token_counts)]
    present = {key for metrics, _ in rows for key in metrics}
    sums: Dict[str, float] = {
        "loss_sum": sum(float(result.loss) * int(count) for result, count in zip(results, token_counts))
    }
    minima: Dict[str, float] = {}
    maxima: Dict[str, float] = {}

    for metrics, count in rows:
        structural = float(count)
        finite = metrics.get(
            "value_finite_tokens",
            structural * (1.0 - metrics.get("value_nonfinite_fraction", 0.0)),
        )
        finite = min(structural, max(0.0, finite))
        sums["value_finite"] = sums.get("value_finite", 0.0) + finite

        prediction_mean = metrics.get("value_prediction_mean")
        target_mean = metrics.get("value_target_mean")
        target_std = metrics.get("value_target_std")
        if prediction_mean is not None:
            sums["prediction_sum"] = sums.get("prediction_sum", 0.0) + prediction_mean * finite
        if target_mean is not None:
            sums["target_sum"] = sums.get("target_sum", 0.0) + target_mean * finite
            if target_std is not None:
                sums["target_square_sum"] = (
                    sums.get("target_square_sum", 0.0) + (target_std * target_std + target_mean * target_mean) * finite
                )
        if prediction_mean is not None and target_mean is not None:
            sums["residual_sum"] = sums.get("residual_sum", 0.0) + (target_mean - prediction_mean) * finite
            # result.loss is the structural-token mean; invalid tokens add zero.
            sums["residual_square_sum"] = (
                sums.get("residual_square_sum", 0.0)
                + float(metrics.get("value_mse", metrics.get("critic_loss", 0.0))) * structural
            )

    generic_modes = _add_generic_metric_stats(
        rows=rows,
        handled=_CRITIC_REPORT_KEYS,
        sums=sums,
        minima=minima,
        maxima=maxima,
    )
    sums = _all_reduce_named(sums)
    minima = _all_reduce_named(minima, op=dist.ReduceOp.MIN)
    maxima = _all_reduce_named(maxima, op=dist.ReduceOp.MAX)

    denominator = float(global_denominator)
    global_loss = sums["loss_sum"] / denominator
    metrics: Dict[str, object] = _finalize_generic_metrics(
        modes=generic_modes,
        sums=sums,
        minima=minima,
        maxima=maxima,
    )
    if "critic_loss" in present:
        metrics["critic_loss"] = global_loss
    if "value_mse" in present:
        metrics["value_mse"] = global_loss
    if "value_structural_tokens" in present:
        metrics["value_structural_tokens"] = global_denominator

    finite = sums.get("value_finite", denominator)
    if "value_finite_tokens" in present:
        metrics["value_finite_tokens"] = finite
    if "value_nonfinite_fraction" in present:
        metrics["value_nonfinite_fraction"] = max(0.0, denominator - finite) / denominator
    if "value_prediction_mean" in present:
        metrics["value_prediction_mean"] = sums.get("prediction_sum", 0.0) / max(1.0, finite)
    target_mean = sums.get("target_sum", 0.0) / max(1.0, finite)
    if "value_target_mean" in present:
        metrics["value_target_mean"] = target_mean
    target_variance = max(
        0.0,
        sums.get("target_square_sum", 0.0) / max(1.0, finite) - target_mean * target_mean,
    )
    if "value_target_std" in present:
        metrics["value_target_std"] = math.sqrt(target_variance)
    if "value_explained_variance" in present:
        residual_mean = sums.get("residual_sum", 0.0) / max(1.0, finite)
        residual_variance = max(
            0.0,
            sums.get("residual_square_sum", 0.0) / max(1.0, finite) - residual_mean * residual_mean,
        )
        metrics["value_explained_variance"] = (
            1.0 - residual_variance / target_variance if target_variance > 0.0 else 0.0
        )
    return global_loss, metrics


def _training_part(trajectories: Sequence[Sample], indices: Sequence[int]) -> Tuple[Part, List[Tuple[int, int]]]:
    """Flatten selected action turns and return their ``(traj, turn)`` keys."""
    parts: List[Part] = []
    keys: List[Tuple[int, int]] = []
    for traj_idx in indices:
        for turn_idx, part in enumerate(_action_parts(trajectories[traj_idx])):
            clean = _part_with_field(part, "primitive", None)
            clean = _part_with_field(clean, "rewards", None)
            clean = _part_with_field(clean, "component_rewards", None)
            clean = _part_with_field(clean, "media_preview", None)
            # The generic algorithm contract still carries a row-level tensor;
            # SAO/value algorithms consume the packed fields instead.
            clean = _part_with_field(clean, "advantages", torch.zeros(1, dtype=torch.float32))
            parts.append(clean)
            keys.append((traj_idx, turn_idx))
    if not parts:
        raise ValueError("SAO training micro contains no generated action tokens")
    return Part.concat(parts), keys


def _set_segment_field(part: Part, name: str, value: torch.Tensor) -> None:
    segment = part.segment
    if not isinstance(segment, TextSegment):
        raise TypeError(f"SAO requires TextSegment, got {type(segment).__name__}")
    setattr(segment, name, value)


class SAOTrainStack(Remote):
    """One distributed SAO update: critic x2, fresh values, actor x1."""

    def __init__(
        self,
        *,
        actor_backend: Any,
        actor_algorithm: Any,
        critic_backend: Any,
        critic_algorithm: Any,
        actor_micro_batch_size: int = 1,
        critic_micro_batch_size: int = 1,
        actor_token_budget: Optional[int] = None,
        critic_token_budget: Optional[int] = None,
        actor_max_grad_norm: float = 1.0,
        critic_max_grad_norm: float = 1.0,
        critic_updates_per_actor: int = 2,
        gamma: float = 1.0,
        gae_alpha: float = 1.5,
        critic_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        if int(critic_updates_per_actor) != 2:
            raise ValueError("SAO paper configuration requires critic_updates_per_actor=2")
        if float(critic_lambda) != 1.0:
            raise ValueError("SAO v1 fixes critic_lambda=1.0 (Monte Carlo targets)")
        if float(gamma) < 0.0 or float(gamma) > 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if float(gae_alpha) <= 0.0:
            raise ValueError(f"gae_alpha must be > 0, got {gae_alpha}")
        if float(actor_max_grad_norm) <= 0.0 or float(critic_max_grad_norm) <= 0.0:
            raise ValueError("actor/critic max_grad_norm must be > 0")

        self.actor_backend = actor_backend
        self.actor_algorithm = actor_algorithm
        self.critic_backend = critic_backend
        self.critic_algorithm = critic_algorithm
        self.actor_micro_batch_size = int(actor_micro_batch_size)
        self.critic_micro_batch_size = int(critic_micro_batch_size)
        self.actor_token_budget = None if actor_token_budget is None else int(actor_token_budget)
        self.critic_token_budget = None if critic_token_budget is None else int(critic_token_budget)
        self.actor_max_grad_norm = float(actor_max_grad_norm)
        self.critic_max_grad_norm = float(critic_max_grad_norm)
        self.critic_updates_per_actor = 2
        self.gamma = float(gamma)
        self.gae_alpha = float(gae_alpha)
        self.critic_lambda = 1.0

    def _plan(
        self,
        trajectories: Sequence[Sample],
        *,
        micro_batch_size: int,
        token_budget: Optional[int],
    ) -> List[List[int]]:
        costs = [_trajectory_replay_cost(t) for t in trajectories]
        local = plan_trajectory_micros(
            costs,
            micro_batch_size=micro_batch_size,
            token_budget=token_budget,
        )
        target = _global_int(len(local), op=dist.ReduceOp.MAX)
        return plan_trajectory_micros(
            costs,
            micro_batch_size=micro_batch_size,
            token_budget=token_budget,
            target_micros=target,
        )

    @staticmethod
    def _attach_critic_targets(trajectories: Sequence[Sample], rewards: torch.Tensor, *, gamma: float) -> int:
        local_tokens = 0
        for trajectory, reward in zip(trajectories, rewards.tolist()):
            parts = _action_parts(trajectory)
            lengths = [int(part.segment.tokens.numel()) for part in parts]  # type: ignore[union-attr]
            total = sum(lengths)
            if total <= 0:
                raise ValueError("SAO trajectory contains no action tokens")
            returns = torch.empty(total, dtype=torch.float32)
            running = float(reward)
            for pos in range(total - 1, -1, -1):
                returns[pos] = running
                running *= float(gamma)
            cursor = 0
            for part, length in zip(parts, lengths):
                segment = part.segment
                assert isinstance(segment, TextSegment)
                target = returns[cursor : cursor + length].clone()
                structural = (
                    segment.loss_mask.detach().to(dtype=torch.float32, device="cpu")
                    if segment.loss_mask is not None
                    else torch.ones(length, dtype=torch.float32)
                )
                _set_segment_field(part, "value_targets", target)
                _set_segment_field(part, "value_mask", structural)
                local_tokens += int((structural != 0).sum().item())
                cursor += length
        return local_tokens

    def _train_side(
        self,
        trajectories: Sequence[Sample],
        plan: Sequence[Sequence[int]],
        *,
        backend: Any,
        algorithm: Any,
        mask_field: str,
        max_grad_norm: float,
        training_progress: float,
    ) -> TrainStepResult:
        local_denominator = sum(
            _part_token_count(part, mask_field=mask_field)
            for trajectory in trajectories
            for part in _action_parts(trajectory)
        )
        global_denominator = _global_int(local_denominator)
        if local_denominator <= 0 or global_denominator <= 0:
            raise ValueError("SAO update has no structurally trainable action tokens")
        dp_size = _dp_world_size()

        backend.model.train()
        backend.zero_grad()
        results: List[AlgorithmStepResult] = []
        micro_counts: List[int] = []
        for micro_idx, indices in enumerate(plan):
            backend.set_grad_sync(micro_idx == len(plan) - 1)
            part, _ = _training_part(trajectories, indices)
            device = next(backend.trainable_module().parameters()).device
            _align_track_to_model(part, device=device)
            micro_count = _part_token_count(part, mask_field=mask_field)
            # FSDP averages gradients across DP ranks.  Scaling the local token
            # mean by local_count * dp/global_count yields the exact global mean.
            loss_scale = micro_count * dp_size / float(global_denominator)
            result = algorithm.compute_loss_and_backward(
                conditions=part.conditions,
                segment=part.segment,
                advantages=part.advantages,
                training_progress=float(training_progress),
                loss_scale=float(loss_scale),
            )
            results.append(result)
            micro_counts.append(micro_count)

        has_backward = any(result.has_backward for result in results)
        grad_norm = float(backend.optimizer_step(max_grad_norm=float(max_grad_norm))) if has_backward else 0.0
        if mask_field == "token_advantages":
            total_loss, metrics = _global_actor_report(
                results,
                micro_counts,
                global_denominator=global_denominator,
            )
        elif mask_field == "value_mask":
            total_loss, metrics = _global_critic_report(
                results,
                micro_counts,
                global_denominator=global_denominator,
            )
        else:
            raise ValueError(f"unsupported SAO training mask field {mask_field!r}")
        return TrainStepResult(
            loss=float(total_loss),
            grad_norm=grad_norm,
            lr=self._current_lr(backend),
            has_backward=has_backward,
            micros=results,
            metrics=metrics,
        )

    def _predict_values(
        self,
        trajectories: Sequence[Sample],
        plan: Sequence[Sequence[int]],
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        predicted: Dict[Tuple[int, int], torch.Tensor] = {}
        self.critic_backend.model.train()
        with torch.no_grad():
            for indices in plan:
                part, keys = _training_part(trajectories, indices)
                device = next(self.critic_backend.trainable_module().parameters()).device
                _align_track_to_model(part, device=device)
                values = self.critic_algorithm.predict_values(conditions=part.conditions, segment=part.segment)
                if part.segment is None or part.segment.lengths is None:
                    raise ValueError("critic prediction requires packed TextSegment lengths")
                chunks = torch.split(values, [int(n) for n in part.segment.lengths.tolist()])
                if len(chunks) != len(keys):
                    raise RuntimeError("critic value chunks do not match generated action turns")
                for key, chunk in zip(keys, chunks):
                    predicted[key] = chunk.detach().float().cpu()
        return predicted

    def _attach_actor_advantages(
        self,
        trajectories: Sequence[Sample],
        rewards: torch.Tensor,
        values: Mapping[Tuple[int, int], torch.Tensor],
    ) -> Tuple[int, List[float]]:
        # Local import keeps the stack importable while algorithms remain an
        # independently testable package leaf.
        from unirl.algorithms.gae import action_token_gae

        local_tokens = 0
        lambdas: List[float] = []
        for traj_idx, (trajectory, reward) in enumerate(zip(trajectories, rewards.tolist())):
            parts = _action_parts(trajectory)
            lengths = [int(part.segment.tokens.numel()) for part in parts]  # type: ignore[union-attr]
            flat_values = torch.cat([values[(traj_idx, turn_idx)] for turn_idx in range(len(parts))])
            total = int(flat_values.numel())
            gae_lambda = 1.0 - 1.0 / (self.gae_alpha * total)
            lambdas.append(gae_lambda)
            result = action_token_gae(
                values=flat_values,
                action_lengths=lengths,
                terminal_reward=float(reward),
                gamma=self.gamma,
                gae_lambda=gae_lambda,
                terminal=True,
            )
            advantages = getattr(result, "advantages", result)
            if not isinstance(advantages, torch.Tensor) or int(advantages.numel()) != total:
                raise RuntimeError("action_token_gae returned misaligned advantages")
            cursor = 0
            for part, length in zip(parts, lengths):
                segment = part.segment
                assert isinstance(segment, TextSegment)
                chunk = advantages[cursor : cursor + length].detach().float().cpu()
                _set_segment_field(part, "token_advantages", chunk)
                local_tokens += _part_token_count(part, mask_field="token_advantages")
                cursor += length
        return local_tokens, lambdas

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_trajectories(
        self,
        trajectories: List[Sample],
        rewards: torch.Tensor,
        *,
        training_progress: float,
    ) -> TrainStepResult:
        """Run two critic updates followed by one actor update."""
        if len(trajectories) != int(rewards.shape[0]):
            raise ValueError(f"trajectory/reward mismatch: {len(trajectories)} != {int(rewards.shape[0])}")
        if not trajectories:
            raise ValueError("SAOTrainStack.train_trajectories received an empty batch")
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("SAOTrainStack requires finite trajectory rewards")

        rewards = rewards.detach().float().cpu()
        critic_tokens = self._attach_critic_targets(trajectories, rewards, gamma=self.gamma)
        critic_plan = self._plan(
            trajectories,
            micro_batch_size=self.critic_micro_batch_size,
            token_budget=self.critic_token_budget,
        )
        critic_results = [
            self._train_side(
                trajectories,
                critic_plan,
                backend=self.critic_backend,
                algorithm=self.critic_algorithm,
                mask_field="value_mask",
                max_grad_norm=self.critic_max_grad_norm,
                training_progress=float(training_progress),
            )
            for _ in range(self.critic_updates_per_actor)
        ]

        values = self._predict_values(trajectories, critic_plan)
        actor_tokens, lambdas = self._attach_actor_advantages(trajectories, rewards, values)
        actor_plan = self._plan(
            trajectories,
            micro_batch_size=self.actor_micro_batch_size,
            token_budget=self.actor_token_budget,
        )
        actor_result = self._train_side(
            trajectories,
            actor_plan,
            backend=self.actor_backend,
            algorithm=self.actor_algorithm,
            mask_field="token_advantages",
            max_grad_norm=self.actor_max_grad_norm,
            training_progress=float(training_progress),
        )
        self.critic_backend.on_rollout_end()
        self.actor_backend.on_rollout_end()

        critic_metrics = aggregate_numeric_metrics([r.metrics for r in critic_results if r.metrics])
        summary = _all_reduce_named(
            {
                "actor_tokens": float(actor_tokens),
                "critic_tokens": float(critic_tokens),
                "gae_lambda_count": float(len(lambdas)),
                "gae_lambda_sum": float(sum(lambdas)),
                "replay_tokens": float(sum(_trajectory_replay_cost(t) for t in trajectories)),
            }
        )
        global_actor_tokens = int(round(summary["actor_tokens"]))
        global_critic_tokens = int(round(summary["critic_tokens"]))
        metrics: Dict[str, object] = dict(actor_result.metrics)
        metrics.update({f"critic/{key}": value for key, value in critic_metrics.items()})
        metrics.update(
            {
                "critic/loss": sum(r.loss for r in critic_results) / len(critic_results),
                "critic/grad_norm": sum(r.grad_norm for r in critic_results) / len(critic_results),
                "critic/updates_per_actor": self.critic_updates_per_actor,
                "critic/tokens": global_critic_tokens,
                "actor/tokens": global_actor_tokens,
                "gae/lambda_mean": summary["gae_lambda_sum"] / max(1.0, summary["gae_lambda_count"]),
                "replay/token_amplification": summary["replay_tokens"] / max(1, global_actor_tokens),
            }
        )
        return replace(actor_result, metrics=metrics)

    @staticmethod
    def _current_lr(backend: Any) -> float:
        groups = getattr(backend.optimizer, "param_groups", None)
        if isinstance(groups, list) and groups:
            return float(groups[0]["lr"])
        return 0.0


__all__ = ["SAOTrainStack", "plan_trajectory_micros"]
