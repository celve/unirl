"""ID-joined flat-row oracle for the E4a grouping equality check.

UniRL groups siblings by reshaping a parent-contiguous ``Part`` into
``(n_groups, branch)`` and reducing along the row. That is only equal to grouping
by lineage ID when the rows are in fact parent-contiguous and uniformly branched.
This module recomputes groups and advantages from the sample IDs alone —
dict-of-lists, no reshape, no ordering assumption — so the two can be compared on
identical recorded payloads. Independence is the point: it must not import the
code under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def group_key(sample_id: str, layer: int) -> str:
    """The ancestor ID at ``layer`` — the first ``layer + 1`` slash-separated segments."""
    segments = sample_id.split("/")
    if layer < 0 or layer >= len(segments):
        raise ValueError(f"group_key: layer {layer} out of range for id {sample_id!r} (depth {len(segments) - 1})")
    return "/".join(segments[: layer + 1])


def default_layer(sample_ids: Sequence[str]) -> int:
    """The parent layer UniRL defaults to: one above the first ID's own depth."""
    if not sample_ids:
        raise ValueError("default_layer: no sample_ids")
    return max(sample_ids[0].count("/") - 1, 0)


def _finite(values: Sequence[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Population mean/std over finite entries; std defaults to 1.0 below two values."""
    finite = _finite(values)
    if not finite:
        return 0.0, 1.0
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, 1.0
    variance = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, math.sqrt(variance)


@dataclass(frozen=True)
class FlatRowGrouping:
    """Groups and advantages derived from IDs alone, plus the structural facts."""

    groups: Dict[str, List[str]]
    advantages: Dict[str, float]
    group_means: Dict[str, float]
    group_stds: Dict[str, float]
    parent_contiguous: bool
    uniform_branch: bool
    duplicate_ids: List[str]


def flat_id_join(
    sample_ids: Sequence[str],
    rewards: Sequence[float],
    *,
    layer: Optional[int] = None,
    normalize: bool = True,
    eps: float = 1e-8,
    use_global_std: bool = False,
) -> FlatRowGrouping:
    """Group by lineage ID and centre each reward within its group."""
    if len(sample_ids) != len(rewards):
        raise ValueError(f"flat_id_join: {len(sample_ids)} ids against {len(rewards)} rewards")
    if not sample_ids:
        raise ValueError("flat_id_join: no rows")

    resolved_layer = default_layer(sample_ids) if layer is None else int(layer)

    seen: Dict[str, int] = {}
    duplicates: List[str] = []
    for sid in sample_ids:
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] == 2:
            duplicates.append(sid)

    groups: Dict[str, List[str]] = {}
    per_group_rewards: Dict[str, List[float]] = {}
    keys_in_row_order: List[str] = []
    for sid, reward in zip(sample_ids, rewards):
        key = group_key(sid, resolved_layer)
        keys_in_row_order.append(key)
        groups.setdefault(key, []).append(sid)
        per_group_rewards.setdefault(key, []).append(float(reward))

    unique_keys = list(dict.fromkeys(keys_in_row_order))
    sizes = {key: len(members) for key, members in groups.items()}
    uniform_branch = len(set(sizes.values())) == 1
    contiguous_expectation = [key for key in unique_keys for _ in range(sizes[key])]
    parent_contiguous = uniform_branch and keys_in_row_order == contiguous_expectation

    _, global_std = _mean_std([float(r) for r in rewards])

    group_means: Dict[str, float] = {}
    group_stds: Dict[str, float] = {}
    advantages: Dict[str, float] = {}
    for key in unique_keys:
        mean, std = _mean_std(per_group_rewards[key])
        group_means[key] = mean
        group_stds[key] = global_std if use_global_std else std
        for sid, reward in zip(groups[key], per_group_rewards[key]):
            if not math.isfinite(reward):
                advantages[sid] = 0.0
                continue
            centered = reward - mean
            advantages[sid] = centered / (group_stds[key] + eps) if normalize else centered

    return FlatRowGrouping(
        groups=groups,
        advantages=advantages,
        group_means=group_means,
        group_stds=group_stds,
        parent_contiguous=parent_contiguous,
        uniform_branch=uniform_branch,
        duplicate_ids=duplicates,
    )


@dataclass(frozen=True)
class OracleComparison:
    """The verdict of one oracle comparison, with every disagreement named."""

    passed: bool
    num_rows: int
    max_abs_advantage_diff: float
    tolerance: float
    mismatched_ids: List[str]
    findings: List[str]

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: {self.num_rows} rows, max|Δadv|={self.max_abs_advantage_diff:.3e} "
            f"(tol {self.tolerance:.1e}), {len(self.mismatched_ids)} mismatched"
        )


def compare_to_oracle(
    sample_ids: Sequence[str],
    rewards: Sequence[float],
    system_advantages: Sequence[float],
    *,
    layer: Optional[int] = None,
    normalize: bool = True,
    eps: float = 1e-8,
    use_global_std: bool = False,
    tolerance: float = 1e-6,
) -> OracleComparison:
    """Check a system's advantages against the ID join on the same recorded payload."""
    if len(sample_ids) != len(system_advantages):
        raise ValueError(f"compare_to_oracle: {len(sample_ids)} ids against {len(system_advantages)} advantages")

    oracle = flat_id_join(
        sample_ids,
        rewards,
        layer=layer,
        normalize=normalize,
        eps=eps,
        use_global_std=use_global_std,
    )

    findings: List[str] = []
    if oracle.duplicate_ids:
        findings.append(f"duplicate sample ids: {sorted(set(oracle.duplicate_ids))[:8]}")
    if not oracle.uniform_branch:
        findings.append(f"non-uniform group sizes: {sorted({len(m) for m in oracle.groups.values()})}")
    if not oracle.parent_contiguous:
        findings.append("rows are not parent-contiguous — the reshape path and the ID join cannot agree by construction")

    mismatched: List[str] = []
    max_diff = 0.0
    for sid, system_adv in zip(sample_ids, system_advantages):
        expected = oracle.advantages[sid]
        diff = abs(float(system_adv) - expected)
        max_diff = max(max_diff, diff)
        if not (diff <= tolerance):
            mismatched.append(sid)
    if mismatched:
        findings.append(f"{len(mismatched)} advantages differ beyond {tolerance:.1e}: {mismatched[:8]}")

    return OracleComparison(
        passed=not findings,
        num_rows=len(sample_ids),
        max_abs_advantage_diff=max_diff,
        tolerance=tolerance,
        mismatched_ids=mismatched,
        findings=findings,
    )


__all__ = [
    "FlatRowGrouping",
    "OracleComparison",
    "compare_to_oracle",
    "default_layer",
    "flat_id_join",
    "group_key",
]
