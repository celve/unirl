"""AgenticTrainer — multi-turn agentic RL over the AgenticRolloutEngine (LIN-519).

A sibling of :class:`~unirl.trainer.ar.ARTrainer` for the AGENTIC path. The rollout is
the rank-0-coordinated
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`, whose ``generate``
returns a FLAT ``List[Sample]`` of variable-depth, independently terminated
trajectories (one per GRPO sibling) — not a single batched Sample. So this trainer
overrides only:

- ``__init__`` — wire the rank-0 coordinator (``set_workers``);
- ``_build_request_sample`` — emit JUST the prompts (no ``fork``: the engine fans the
  ``n`` GRPO siblings internally) with the per-turn ``stop`` on the root control bag;
- ``train_step`` — consume the trajectory list: judge each trajectory's ``<answer>``
  with the reward backend, compute GROUP-relative GRPO advantages over the ``n``
  siblings of each prompt, assign each trajectory's scalar advantage to ALL its
  assistant turns, concatenate every turn into ONE training Part (padded to a DP
  multiple), and run ONE optimizer step.

Everything else — worker construction, weight sync, checkpointing, the ``train`` loop
— is inherited from ``ARTrainer`` / ``BaseTrainer`` unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from omegaconf import OmegaConf

from unirl.algorithms.normalizers import build_group_index_map
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.trajectory_dump import maybe_dump_trajectories

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: Optional[str]) -> str:
    """The last ``<answer>…</answer>`` span. With no tags the fallback is the whole
    text (math/calc verifiers tolerate an unwrapped / ``\\boxed{}`` answer), UNLESS
    ``$REQUIRE_ANSWER_TAG`` is set: then a missing ``<answer>`` scores as no answer
    (empty -> reward 0). Matches AReaL's tongyi judge (no ``<answer>`` -> "No answer
    found.") and stops the policy reward-hacking the LLM judge with unwrapped verbose
    prose (LIN-564: our prose fallback let untagged answers out-score tagged ones, so
    GRPO abandoned the format — tag usage fell 70%->11% and the reward stalled)."""
    if not text:
        return ""
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    if os.environ.get("REQUIRE_ANSWER_TAG", "").lower() in ("1", "true", "yes"):
        return ""
    return text.strip()


def _part_has_metadata(part: Part, key: str) -> bool:
    return any(bool((meta or {}).get(key)) for meta in (part.metadata or []))


_TOOL_DIAGNOSTIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "request_count": ("request_count", "requests"),
    "success_count": ("success_count", "successes"),
    "cache_hit_count": ("cache_hit_count", "cache_hits"),
    "retry_count": ("retry_count", "retries"),
    "recovered_transient_count": (
        "recovered_transient_count",
        "recovered_transient",
        "recovered",
    ),
    "transient_exhausted_count": (
        "transient_exhausted_count",
        "transient_exhausted",
    ),
    "permanent_error_count": ("permanent_error_count", "permanent_error"),
    "auth_error_count": ("auth_error_count", "auth_error"),
}


def _diagnostic_count(diagnostic: Mapping[str, Any], canonical: str) -> int:
    """Read one non-negative diagnostic counter, accepting legacy aliases.

    The first present spelling wins so a producer temporarily emitting both its
    canonical field and an alias cannot double-count one request.
    """
    for key in _TOOL_DIAGNOSTIC_ALIASES[canonical]:
        if key not in diagnostic:
            continue
        try:
            return max(0, int(diagnostic[key]))
        except (TypeError, ValueError):
            return 0
    return 0


def _trajectory_tool_diagnostics(traj: Sample) -> Dict[str, int]:
    """Sum safe per-turn tool counters across one trajectory."""
    totals = {key: 0 for key in _TOOL_DIAGNOSTIC_ALIASES}
    for diagnostic in _trajectory_tool_diagnostic_records(traj):
        for canonical in totals:
            totals[canonical] += _diagnostic_count(diagnostic, canonical)
    return totals


def _trajectory_tool_diagnostic_records(traj: Sample) -> List[Mapping[str, Any]]:
    """Return the sanitized per-call aggregate records carried by a trajectory."""
    records: List[Mapping[str, Any]] = []
    for part in traj.gen_parts():
        for metadata in part.metadata or []:
            value = (metadata or {}).get("tool_diagnostics")
            diagnostics = value if isinstance(value, list) else [value]
            for diagnostic in diagnostics:
                if not isinstance(diagnostic, Mapping):
                    continue
                records.append(diagnostic)
    return records


def _infrastructure_group_exclusion(
    trajs: List[Sample],
    group_ids: List[str],
) -> Tuple[torch.Tensor, List[Optional[str]], List[Dict[str, int]], Dict[str, Tuple[str, ...]]]:
    """Return the trajectory mask/reasons for group-scoped infrastructure loss.

    One exhausted transient or authentication failure makes the reward comparison
    inside that root's GRPO group causally invalid, so every sibling is excluded.
    Permanent content/URL failures remain policy outcomes and do not invalidate a
    group. The returned per-trajectory diagnostics still include all counters for
    operational metrics.
    """
    if len(trajs) != len(group_ids):
        raise ValueError(
            "group_ids must align one-to-one with trajectories; "
            f"got {len(group_ids)} ids for {len(trajs)} trajectories"
        )
    per_trajectory = [_trajectory_tool_diagnostics(traj) for traj in trajs]
    failures: Dict[str, set[str]] = {}
    for group_id, diagnostic in zip(group_ids, per_trajectory):
        reasons = failures.setdefault(str(group_id), set())
        if diagnostic["transient_exhausted_count"] > 0:
            reasons.add("transient_exhausted")
        if diagnostic["auth_error_count"] > 0:
            reasons.add("auth_error")
    failures = {group: reasons for group, reasons in failures.items() if reasons}
    mask = torch.tensor([str(group_id) in failures for group_id in group_ids], dtype=torch.bool)
    reasons: List[Optional[str]] = []
    for group_id in group_ids:
        group_reasons = tuple(sorted(failures.get(str(group_id), ())))
        reasons.append(
            "infrastructure_group:" + "+".join(group_reasons)
            if group_reasons
            else None
        )
    frozen_failures = {group: tuple(sorted(reason)) for group, reason in failures.items()}
    return mask, reasons, per_trajectory, frozen_failures


def _tool_diagnostic_metrics(
    trajs: List[Sample], per_trajectory: List[Dict[str, int]]
) -> Dict[str, Any]:
    """Aggregate tool reliability counters with explicit denominators.

    ``request_count`` is physical upstream attempts and can exceed logical tool
    calls after retries. Cache/single-flight hits make no upstream attempt, so
    their rates use the logical outcome count instead. Bounded tool/provider and
    status breakouts make a Jina 429 distinguishable from a Serper failure.
    """
    totals = {
        key: sum(int(row.get(key, 0)) for row in per_trajectory)
        for key in _TOOL_DIAGNOSTIC_ALIASES
    }
    requests = totals["request_count"]
    trajectories = len(per_trajectory)
    logical_calls = sum(
        totals[key]
        for key in (
            "success_count",
            "transient_exhausted_count",
            "permanent_error_count",
            "auth_error_count",
        )
    )
    request_denominator = max(1, requests)
    logical_denominator = max(1, logical_calls)
    transient_events = totals["recovered_transient_count"] + totals["transient_exhausted_count"]
    metrics: Dict[str, Any] = {
        "agent/tool_request_count": requests,
        "agent/tool_logical_call_count": logical_calls,
        # Mean upstream requests per trajectory. The companion trajectory rate
        # distinguishes broad low-volume use from a concentrated retry storm.
        "agent/tool_request_rate": requests / trajectories if trajectories else 0.0,
        "agent/tool_request_trajectory_rate": (
            sum(row.get("request_count", 0) > 0 for row in per_trajectory)
            / trajectories
            if trajectories
            else 0.0
        ),
    }
    metric_names = {
        "success_count": "success",
        "retry_count": "retry",
        "cache_hit_count": "cache_hit",
        "recovered_transient_count": "recovered_transient",
        "transient_exhausted_count": "transient_exhausted",
        "permanent_error_count": "permanent_error",
        "auth_error_count": "auth_error",
    }
    for canonical, label in metric_names.items():
        metrics[f"agent/tool_{label}_count"] = totals[canonical]
        denominator = request_denominator if canonical == "retry_count" else logical_denominator
        metrics[f"agent/tool_{label}_rate"] = totals[canonical] / denominator
    metrics["agent/tool_transient_recovery_rate"] = (
        totals["recovered_transient_count"] / transient_events if transient_events else 0.0
    )

    grouped: Dict[Tuple[str, str], Counter] = {}
    statuses: Counter = Counter()
    for traj in trajs:
        for diagnostic in _trajectory_tool_diagnostic_records(traj):
            tool = str(diagnostic.get("tool", "unknown"))
            provider = str(diagnostic.get("provider", "unknown"))
            group_keys: List[Tuple[str, str]] = []
            for dimension, name in (("tool", tool), ("provider", provider)):
                safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "unknown"
                group_keys.append((dimension, safe_name))
                bucket = grouped.setdefault((dimension, safe_name), Counter())
                for canonical in _TOOL_DIAGNOSTIC_ALIASES:
                    bucket[canonical] += _diagnostic_count(diagnostic, canonical)
            raw_statuses = diagnostic.get("status_counts", {})
            if isinstance(raw_statuses, Mapping):
                for status, count in raw_statuses.items():
                    status = str(status)
                    if not re.fullmatch(
                        r"(?:(?:http|app|app_status)_[1-5][0-9]{2}|"
                        r"app(?:_status)?_malformed|timeout|connection|auth_config|"
                        r"client_error|http_error|http_unknown|malformed_response|"
                        r"malformed_jina_envelope|empty_jina_content|other)",
                        status,
                    ):
                        continue
                    try:
                        safe_count = max(0, int(count))
                    except (TypeError, ValueError):
                        continue
                    statuses[status] += safe_count
                    for group_key in group_keys:
                        grouped[group_key][f"status::{status}"] += safe_count

    for (dimension, name), bucket in sorted(grouped.items()):
        prefix = f"agent/tool_{dimension}_{name}"
        for canonical in _TOOL_DIAGNOSTIC_ALIASES:
            label = metric_names.get(canonical, canonical.removesuffix("_count"))
            metrics[f"{prefix}_{label}_count"] = int(bucket[canonical])
        for key, count in sorted(bucket.items()):
            if key.startswith("status::"):
                metrics[f"{prefix}_status_{key.removeprefix('status::')}_count"] = int(count)
    for status, count in sorted(statuses.items()):
        metrics[f"agent/tool_status_{status}_count"] = int(count)
    metrics["agent/tool_rate_limited_count"] = int(
        sum(count for status, count in statuses.items() if status.endswith("_429"))
    )
    metrics["agent/tool_server_error_status_count"] = int(
        sum(
            count
            for status, count in statuses.items()
            if re.search(r"_5[0-9]{2}$", status)
        )
    )
    return metrics


def _trajectory_token_counts(
    trajs: List[Sample], *, exclude_answer_rescue_triggers: bool = False
) -> torch.Tensor:
    """Generated-token count for each trajectory, summed across assistant turns.

    AReaL broadcasts the terminal reward onto every token selected by its loss
    mask before batch-level advantage normalization.  Agentic UniRL stores each
    assistant turn as a separate generated ``Part``, so summing their packed
    lengths reconstructs the equivalent per-trajectory multiplicity.  Prompt and
    tool/context parts are not generated parts and therefore do not contribute.
    """
    counts: List[int] = []
    for tr in trajs:
        count = 0
        for gp in tr.gen_parts():
            if exclude_answer_rescue_triggers and _is_answer_rescue_trigger(gp):
                continue
            segment = gp.segment
            if segment is None:
                continue
            loss_mask = getattr(segment, "loss_mask", None)
            if loss_mask is not None:
                count += int(hydrate(loss_mask).count_nonzero().item())
            elif segment.lengths is not None:
                count += int(segment.lengths.sum().item())
            elif getattr(segment, "tokens", None) is not None:
                count += int(segment.tokens.shape[0])
        counts.append(count)
    return torch.tensor(counts, dtype=torch.long)


def _is_answer_repair(part: Part) -> bool:
    """Whether a generated Part is the decoder-prefix repair suffix."""
    return _part_has_metadata(part, "answer_injected")


def _is_answer_rescue(part: Part) -> bool:
    """Whether a full generated Part follows a user-side answer-rescue nudge."""
    return _part_has_metadata(part, "answer_rescued")


def _is_answer_rescue_trigger(part: Part) -> bool:
    """Whether this NEITHER Part caused the controller's answer rescue."""
    return _part_has_metadata(part, "answer_rescue_trigger")


def _intervention_aware_advantage(
    part: Part,
    trajectory_advantage: float,
    *,
    mask_trigger_task_credit: bool,
    trigger_penalty: float,
) -> float:
    """Per-Part credit across the user-rescue intervention boundary.

    Earlier research and the fully sampled rescued answer keep the trajectory's
    task advantage. The NEITHER Part that caused rescue is across the causal cut,
    so it receives no downstream task credit at all; an optional small intervention
    penalty makes rescue explicitly costly without treating its prose as the cause
    of a later correct or incorrect answer.
    """
    if mask_trigger_task_credit and _is_answer_rescue_trigger(part):
        return -float(trigger_penalty)
    return float(trajectory_advantage)


def _prepare_agentic_train_part(part: Part, advantage: float) -> Part:
    """Return the policy-only copy of one generated Part used for training.

    Decoded primitives, terminal rewards, and rollout diagnostics have already
    served their purpose before train assembly. In particular, ``Part.metadata``
    uses ``[]`` as the ordinary generated-Part sentinel, while decoder repairs
    carry a batch-aligned marker list. Mixing those two list shapes makes generic
    ``Part.concat`` ambiguous. Strip diagnostics from the immutable training copy;
    the original trajectory retains them for dumps and repair metrics.
    """
    part = _part_with_field(
        part,
        "advantages",
        torch.full((part.batch_size,), float(advantage), dtype=torch.float32),
    )
    part = _part_with_field(part, "primitive", None)
    part = _part_with_field(part, "rewards", None)
    return _part_with_field(part, "metadata", [])


def _validate_agentic_cfg(kw: dict) -> None:
    """Fail fast on cross-config invariants only jointly visible at the trainer.

    The recipes assert these in prose comments but nothing enforced them, so a
    mismatch silently broke the on-policy ratio or DP scatter. Each check is
    skipped when either side is absent, so it never rejects a config that merely
    omits a key. (The ``env.max_turns == config.max_turns`` check lives in
    ``AgenticRolloutEngine.__init__``, where the built env is in scope.)
    """
    rollout_cfg = kw.get("rollout_cfg")
    ep = OmegaConf.select(rollout_cfg, "config.episode_sampling") if rollout_cfg is not None else None
    if ep is None:
        return
    n_ep = int(OmegaConf.select(ep, "samples_per_prompt") or 1)
    t_ep = OmegaConf.select(ep, "temperature")

    sampling_cfg = kw.get("sampling_cfg")
    if sampling_cfg is not None:
        n_s = int(OmegaConf.select(sampling_cfg, "samples_per_prompt") or 1)
        if n_s != n_ep:
            raise ValueError(
                f"sampling.samples_per_prompt ({n_s}) must equal "
                f"rollout.config.episode_sampling.samples_per_prompt ({n_ep})"
            )
        t_s = OmegaConf.select(sampling_cfg, "temperature")
        if t_ep is not None and t_s is not None and abs(float(t_s) - float(t_ep)) > 1e-9:
            raise ValueError(f"sampling.temperature ({t_s}) must equal episode_sampling.temperature ({t_ep})")

    algorithm_cfg = kw.get("algorithm_cfg")
    t_a = OmegaConf.select(algorithm_cfg, "sampling_temperature") if algorithm_cfg is not None else None
    if t_ep is not None and t_a is not None and abs(float(t_a) - float(t_ep)) > 1e-9:
        raise ValueError(
            f"algorithm.sampling_temperature ({t_a}) must equal episode_sampling.temperature "
            f"({t_ep}) — else replay's tempered log-softmax diverges from the sampler (ratio != 1)"
        )

    cfg = kw.get("cfg")
    batch_size = kw.get("batch_size")
    nd = OmegaConf.select(cfg, "num_devices") if cfg is not None else None
    if nd is not None and batch_size is not None and (int(batch_size) * n_ep) % int(nd) != 0:
        raise ValueError(
            f"batch_size*samples_per_prompt ({int(batch_size) * n_ep}) must be divisible "
            f"by num_devices ({int(nd)}) for the DP scatter"
        )


class AgenticTrainer(ARTrainer):
    """Agentic (multi-turn tool-use) RL trainer over the ``AgenticRolloutEngine``."""

    def __init__(
        self,
        *,
        stop: Optional[List[str]] = None,
        no_stop_trim: bool = False,
        mask_answer_rescue_trigger_task_credit: bool = False,
        answer_rescue_trigger_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        _validate_agentic_cfg(kwargs)
        super().__init__(**kwargs)
        # Per-turn stop: a tool-call turn ends at ``</tool_call>`` and yields to the
        # tool; a final-answer turn runs to EOS. Rides the request root's control bag
        # (``resolve_sampling`` reads ``control["ar"]``).
        self._stop = list(stop) if stop else ["</tool_call>"]
        # SGLang normally trims a matched stop string from decoded text. Opting in
        # preserves the closed tool-call delimiter so decoded conversation context
        # and the replay tokens agree. Keep false absent from the control bag below:
        # the established launcher must retain its exact request shape/semantics.
        self._no_stop_trim = bool(no_stop_trim)
        self._mask_answer_rescue_trigger_task_credit = bool(mask_answer_rescue_trigger_task_credit)
        self._answer_rescue_trigger_penalty = float(answer_rescue_trigger_penalty)
        if self._answer_rescue_trigger_penalty < 0:
            raise ValueError("answer_rescue_trigger_penalty must be non-negative")
        if self._answer_rescue_trigger_penalty and not self._mask_answer_rescue_trigger_task_credit:
            raise ValueError(
                "answer_rescue_trigger_penalty requires mask_answer_rescue_trigger_task_credit=true"
            )
        # Wire the rank-0 coordinator (``AgenticRolloutEngine.set_workers`` — the
        # ``NCCLWeightSync.set_rollout_targets`` shape). ``.workers`` / ``.role_name``
        # are ``Handle`` attributes.
        self.rollout.set_workers(self.rollout.workers, self.rollout.role_name)

    def _build_request_sample(
        self,
        inputs: RolloutInputs,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """The ``P`` prompts as a single root input Part — NO ``fork`` (the agentic
        engine fans the ``n`` GRPO siblings itself) — with the per-turn ``stop`` on the
        root control bag and ``metadata`` (the ground-truth answer) carried for the
        reward judge."""
        del sampling  # the engine's ``episode_sampling`` owns per-turn params + ``n``
        root_ids = [f"r{rollout_id}:{sid}" for sid in inputs.sample_ids]
        ar_control: Dict[str, Any] = {"stop": list(self._stop)}
        if getattr(self, "_no_stop_trim", False):
            ar_control["no_stop_trim"] = True
        text = Part.input(
            root_ids,
            primitive=inputs.primitives["text"],
            metadata=list(inputs.metadata) if inputs.metadata else None,
            control={"ar": ar_control},
        )
        return Sample.request(text)

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One agentic ``rollout → judge → advantage → optimizer step`` pass.

        Returns ``(train_result, mean_reward)`` for the progress line.
        """
        t0 = time.perf_counter()

        # 1) Rollout — the barrier multi-turn generate. On-policy: sync first.
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        trajs: List[Sample] = self.rollout.generate(sample)[0]  # BROADCAST+RANK_ZERO -> [List[Sample]]
        self.rollout.sleep()

        # 2) Per-trajectory scalar reward + GRPO group id (root id). Overridable:
        #    answer-graded here (``<answer>`` -> reward backend), env-sourced in
        #    ``AgenticEnvTrainer`` (ALFWorld etc.).
        rewards, group_ids = self._rewards_and_groups(sample, trajs, rollout_id)

        # 3-6) GROUP-relative advantage -> assign to every turn -> ONE step -> log.
        return self._advantage_train_and_log(
            trajs, rewards, group_ids, rollout_id=rollout_id, training_progress=training_progress, t0=t0
        )

    def _advantage_train_and_log(
        self,
        trajs: List[Sample],
        rewards: torch.Tensor,
        group_ids: List[str],
        *,
        rollout_id: int,
        training_progress: float,
        t0: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TrainStepResult, float]:
        """GROUP-relative GRPO advantage → assign each trajectory's scalar advantage to ALL
        its assistant turns → ONE padded ``train_track`` step → log. Shared by the barrier
        ``train_step`` and the colocate partial-rollout trainer
        (:class:`~unirl.trainer.agentic_partial.AgenticPartialTrainer`); the reward SOURCE
        (answer vs env) is already resolved into ``rewards``/``group_ids`` by the caller.
        ``extra_metrics`` are merged into the logged ``agent/*`` metrics (e.g. the partial
        trainer's committed/carried/dropped counts)."""
        # Keep judge/env rewards immutable for audit, then derive the only tensor
        # allowed to influence policy training. A transient-exhaustion/auth event
        # makes every sibling in that root group incomparable, so all effective
        # rewards in that group become non-finite. Existing NaN crash semantics are
        # preserved independently in ``raw_rewards``.
        raw_rewards = rewards.to(torch.float32).reshape(-1)
        if raw_rewards.numel() != len(trajs):
            raise ValueError(
                "rewards must align one-to-one with trajectories; "
                f"got {raw_rewards.numel()} rewards for {len(trajs)} trajectories"
            )
        (
            infra_invalid,
            exclusion_reasons,
            per_trajectory_diagnostics,
            invalid_groups,
        ) = _infrastructure_group_exclusion(trajs, group_ids)
        effective_rewards = raw_rewards.clone()
        effective_rewards[infra_invalid.to(device=effective_rewards.device)] = float("nan")
        raw_finite = torch.isfinite(raw_rewards)
        finite = torch.isfinite(effective_rewards)
        raw_mean_reward = (
            float(raw_rewards[raw_finite].mean().item())
            if bool(raw_finite.any())
            else 0.0
        )
        mean_reward = (
            float(effective_rewards[finite].mean().item())
            if bool(finite.any())
            else 0.0
        )

        # GROUP-relative GRPO by default; ``token-global`` reconstructs AReaL's
        # batch-level masked normalization by weighting each terminal reward by
        # the number of generated tokens onto which AReaL broadcasts it.
        advantages, token_counts = self._compute_agentic_advantages(
            trajs, effective_rewards, group_ids
        )
        generated_token_counts = _trajectory_token_counts(trajs)

        # Debug: dump every trajectory of this rollout (decoded turns + reward/advantage)
        # when $TRAJ_DUMP_DIR is set — a no-op otherwise, and never raises. Placed BEFORE
        # the loop below frees each gen Part's decoded ``primitive`` for training.
        maybe_dump_trajectories(
            trajs,
            effective_rewards,
            advantages,
            group_ids,
            rollout_id=rollout_id,
            raw_rewards=raw_rewards,
            excluded_from_training=infra_invalid,
            exclusion_reasons=exclusion_reasons,
        )

        # Assign each trajectory's scalar advantage to ALL its assistant turns; gather.
        train_parts: List[Part] = []
        for i, tr in enumerate(trajs):
            if bool(infra_invalid[i]):
                # Do not keep zero-advantage replicas: even replaying them would
                # spend optimizer/normalizer token mass on an invalid comparison.
                continue
            adv_i = float(advantages[i].item())
            for gp in tr.gen_parts():
                # Empty SGLang generations have no connected replay backward.
                # Remove them before DP scatter so every rank executes the same
                # number of real FSDP collectives; synthetic pads are added later.
                if gp.segment is None or gp.segment.lengths is None or int(gp.segment.lengths.sum().item()) == 0:
                    continue
                # Drop decoded/reward/diagnostic fields from the immutable training
                # copy. The reward was already read into trajectory advantages.
                part_advantage = _intervention_aware_advantage(
                    gp,
                    adv_i,
                    mask_trigger_task_credit=self._mask_answer_rescue_trigger_task_credit,
                    trigger_penalty=self._answer_rescue_trigger_penalty,
                )
                train_parts.append(_prepare_agentic_train_part(gp, part_advantage))

        depths = [len(tr.gen_parts()) for tr in trajs]
        repair_counts = [sum(1 for gp in tr.gen_parts() if _is_answer_repair(gp)) for tr in trajs]
        rescue_counts = [sum(1 for gp in tr.gen_parts() if _is_answer_rescue(gp)) for tr in trajs]
        # A repair is a second physical decode/Part for honest log-prob ownership,
        # but semantically continues the same assistant turn. Report both depths so
        # the experiment cannot claim a mechanical +1 as learned tool-use depth.
        logical_depths = [depth - repairs for depth, repairs in zip(depths, repair_counts)]
        autonomous_depths = [depth - rescue for depth, rescue in zip(logical_depths, rescue_counts)]
        # Per-trajectory turn distribution — the workload's depth VARIANCE (a straggler-cut only
        # pays when this is wide; ~uniform means over-sample-and-drop is pure waste). LIN-531.
        logger.info(
            "rollout %d trajectory turns: n=%d physical_mean=%.2f logical_mean=%.2f "
            "autonomous_mean=%.2f injected=%d rescued=%d min=%d max=%d hist=%s",
            rollout_id,
            len(depths),
            (sum(depths) / len(depths)) if depths else 0.0,
            (sum(logical_depths) / len(logical_depths)) if logical_depths else 0.0,
            (sum(autonomous_depths) / len(autonomous_depths)) if autonomous_depths else 0.0,
            sum(repair_counts),
            sum(rescue_counts),
            min(depths, default=0),
            max(depths, default=0),
            dict(sorted(Counter(depths).items())),
        )
        train_rows_before_padding = len(train_parts)
        infra_invalid_count = int(infra_invalid.sum().item())
        if not train_parts and infra_invalid_count == 0:
            # Preserve the historical gen-less/zero-token behavior when this is
            # not an infrastructure-invalid rollout. The U7 exception below
            # intentionally continues through logging so operators can diagnose
            # an all-invalid batch without dispatching an optimizer step.
            logger.warning("AgenticTrainer rollout %d produced no trainable turns.", rollout_id)
            return TrainStepResult(0.0, 0.0, 0.0, False, [], {}), mean_reward
        if train_parts:
            # ONE training Part -> pad to a DP multiple (zero-advantage rows) -> ONE step.
            train_part = Part.concat(train_parts)
            train_part = self._pad_to_dp_multiple(train_part)
            train_rows = int(train_part.batch_size)
            result = self.stack.train_track(
                train_part, training_progress=float(training_progress)
            )
        else:
            # An all-invalid rollout is an expected reliability outcome, not an
            # optimizer error. Still emit rollout/tool diagnostics below, but do
            # not dispatch an empty Part or advance the optimizer step.
            logger.warning(
                "AgenticTrainer rollout %d produced no trainable turns (%d infra-invalid trajectories).",
                rollout_id,
                infra_invalid_count,
            )
            train_rows = 0
            result = TrainStepResult(0.0, 0.0, 0.0, False, [], {})

        # Logging sample: one row per trajectory whose gen frontier carries the reward +
        # advantage (compute_rollout_sample_metrics reads gen_parts). Built from the
        # computed tensors so it is independent of the reward SOURCE (answer vs env).
        log_sample = self._build_log_sample(
            trajs, effective_rewards, advantages, rollout_id
        )
        group_count = len({str(group_id) for group_id in group_ids})
        invalid_trajectory_count = infra_invalid_count
        invalid_group_count = len(invalid_groups)
        finite_cpu = finite.detach().to("cpu")
        valid_token_counts = token_counts[finite_cpu]
        valid_effective_rewards = effective_rewards[finite]
        metrics: Dict[str, Any] = {
            # Override the generic rollout reward panel, whose plain tensor
            # reduction would otherwise turn NaN as soon as one invalid group is
            # present. These are the same valid effective rows used by policy
            # normalization and the progress-line mean.
            "reward_mean": mean_reward,
            "reward_std": (
                float(valid_effective_rewards.std(unbiased=False).item())
                if valid_effective_rewards.numel()
                else 0.0
            ),
            "reward_min": (
                float(valid_effective_rewards.min().item())
                if valid_effective_rewards.numel()
                else 0.0
            ),
            "reward_max": (
                float(valid_effective_rewards.max().item())
                if valid_effective_rewards.numel()
                else 0.0
            ),
            "agent/mean_turns": (sum(depths) / len(depths)) if depths else 0.0,
            "agent/mean_logical_turns": (
                (sum(logical_depths) / len(logical_depths)) if logical_depths else 0.0
            ),
            "agent/mean_autonomous_turns": (
                (sum(autonomous_depths) / len(autonomous_depths)) if autonomous_depths else 0.0
            ),
            "agent/max_turns": max(depths) if depths else 0,
            "agent/answer_injected_count": sum(repair_counts),
            "agent/answer_injected_rate": (
                sum(1 for count in repair_counts if count > 0) / len(repair_counts)
                if repair_counts
                else 0.0
            ),
            "agent/answer_rescued_count": sum(rescue_counts),
            "agent/answer_rescued_rate": (
                sum(1 for count in rescue_counts if count > 0) / len(rescue_counts)
                if rescue_counts
                else 0.0
            ),
            "agent/genless_trajectories": sum(1 for d in depths if d == 0),
            "agent/train_rows": train_rows,
            "agent/train_rows_before_padding": train_rows_before_padding,
            "agent/infra_invalid_groups": invalid_group_count,
            "agent/infra_invalid_group_rate": (
                invalid_group_count / group_count if group_count else 0.0
            ),
            "agent/infra_invalid_trajectories": invalid_trajectory_count,
            "agent/infra_invalid_trajectory_rate": (
                invalid_trajectory_count / len(trajs) if trajs else 0.0
            ),
            "agent/raw_all_trajectory_mean_reward": raw_mean_reward,
            "agent/raw_finite_mean_reward": raw_mean_reward,
            "agent/raw_finite_trajectory_count": int(raw_finite.sum().item()),
            "agent/effective_valid_trajectory_count": int(finite.sum().item()),
            "agent/effective_valid_trajectory_rate": (
                int(finite.sum().item()) / len(trajs) if trajs else 0.0
            ),
            "agent/valid_effective_mean_reward": mean_reward,
            # Short aliases keep dashboards readable while the explicit names
            # above define the denominator unambiguously.
            "agent/raw_mean_reward": raw_mean_reward,
            "agent/effective_mean_reward": mean_reward,
            "agent/mean_gen_tokens": (
                float(generated_token_counts.float().mean().item()) if generated_token_counts.numel() else 0.0
            ),
            "agent/max_gen_tokens": (
                int(generated_token_counts.max().item())
                if generated_token_counts.numel()
                else 0
            ),
            "agent/mean_task_credit_tokens": (
                float(valid_token_counts.float().mean().item()) if valid_token_counts.numel() else 0.0
            ),
        }
        metrics.update(_tool_diagnostic_metrics(trajs, per_trajectory_diagnostics))
        token_valid = finite & (token_counts.to(device=effective_rewards.device) > 0)
        if bool(token_valid.any()):
            token_weights = token_counts.to(
                device=effective_rewards.device, dtype=torch.float64
            )[token_valid]
            token_rewards = effective_rewards.to(torch.float64)[token_valid]
            metrics["agent/token_weighted_reward"] = float(
                (token_rewards * token_weights).sum().div(token_weights.sum()).item()
            )
        if extra_metrics:
            metrics.update(extra_metrics)
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics=metrics,
        )
        return result, mean_reward

    def _rewards_and_groups(
        self, sample: Sample, trajs: List[Sample], rollout_id: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Per-trajectory scalar reward + GRPO group id (root id) — the overridable
        reward step. Base path grades each trajectory's ``<answer>`` against the
        ground truth via the reward backend (MathVerify / LLM judge); the ``n``
        siblings of a prompt share its root id (group-by-root). Subclasses swap the
        reward SOURCE (e.g. :class:`~unirl.trainer.agentic_env.AgenticEnvTrainer`
        reads the environment's per-trajectory return) while keeping the GRPO tail.

        A FRESH flat scoring Sample is required because ``score_and_attach`` rejects
        precomputed frontier rewards; its frontier carries no ``segment`` so the
        truncation-shaping branch is skipped.
        """
        root = sample.parts[0]
        root_meta = root.metadata or [None] * len(root.sample_ids)
        gt_by_root = {sid: (md or {}).get("answer") for sid, md in zip(root.sample_ids, root_meta)}
        questions: List[str] = []
        predictions: List[str] = []
        answers: List[Optional[str]] = []
        group_ids: List[str] = []
        for tr in trajs:
            root_id = tr.parts[0].sample_ids[0]
            q_prim = tr.parts[0].primitive
            questions.append(q_prim.texts[0] if (q_prim is not None and q_prim.texts) else "")
            gens = tr.gen_parts()
            term = ""
            if gens and gens[-1].primitive is not None and gens[-1].primitive.texts:
                term = gens[-1].primitive.texts[0]
            predictions.append(_extract_answer(term))
            answers.append(gt_by_root.get(root_id))
            group_ids.append(root_id)

        m = len(trajs)
        ar_sp = self.sampling_params.get("ar")
        score_in = Part.input(
            [f"score{rollout_id}:{i}" for i in range(m)],
            primitive=Texts(texts=list(questions)),
            metadata=[{"answer": a} for a in answers],
        )
        scoring = (
            Sample.request(score_in)
            .fork(1, sampling_params=ar_sp)  # frontier is a gen Part (reward/adv panels read gen_parts)
            .with_filled_frontier(primitive=Texts(texts=list(predictions)))
        )
        scoring = self.reward.score_and_attach(scoring)
        rewards = hydrate(scoring.parts[-1].rewards).to(torch.float32)
        return rewards, group_ids

    def _build_log_sample(
        self, trajs: List[Sample], rewards: torch.Tensor, advantages: torch.Tensor, rollout_id: int
    ) -> Sample:
        """A flat one-row-per-trajectory Sample whose gen frontier carries the
        per-trajectory reward + advantage, for ``compute_rollout_sample_metrics``
        (reward/advantage distributions). Reward-source agnostic — no reward backend,
        no scoring sample — so it works for both the answer-graded and env-sourced paths."""
        m = len(trajs)
        ar_sp = self.sampling_params.get("ar")
        log_in = Part.input([f"log{rollout_id}:{i}" for i in range(m)], primitive=Texts(texts=[""] * m))
        log_sample = (
            Sample.request(log_in)
            .fork(1, sampling_params=ar_sp)
            .with_filled_frontier(primitive=Texts(texts=[""] * m))
        )
        frontier = _part_with_field(log_sample.parts[-1], "rewards", rewards.to(torch.float32))
        frontier = _part_with_field(frontier, "advantages", advantages.to(torch.float32))
        return log_sample.with_parts([*log_sample.parts[:-1], frontier])

    def _compute_agentic_advantages(
        self, trajs: List[Sample], rewards: torch.Tensor, group_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-trajectory advantages and generated-token multiplicities."""
        # A user-rescue trigger is across the task-credit boundary. Exclude its
        # tokens from token-global normalization as well as masking downstream task
        # advantage on its Part; its separate constant intervention penalty is an
        # auxiliary objective and deliberately does not shift the task mean/std.
        token_counts = _trajectory_token_counts(
            trajs,
            exclude_answer_rescue_triggers=self._mask_answer_rescue_trigger_task_credit,
        )
        return self._group_advantages(rewards, group_ids, token_counts=token_counts), token_counts

    def _group_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: List[str],
        *,
        token_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Group-relative GRPO advantages, ``ARTrainer.compute_advantages`` parity
        (population std), over the ``n`` siblings of each prompt (grouped by root id;
        completion order is fine). ``adv_normalization_scope='global'`` z-scores the
        whole batch; ``normalize_adv_by_std=False`` mean-centers only.

        ``adv_normalization_scope='token-global'`` matches AReaL's batch-level,
        token-masked normalization: each reward is repeated ``token_counts[i]``
        times, and AReaL climb4's exact unbiased std + epsilon are used.
        """
        r = rewards.to(torch.float32)
        # NaN reward = crashed trajectory: excluded from the group's mean/std and given
        # ZERO advantage (neutral), so an env crash neither rewards nor penalizes its
        # actions. All-finite (the answer-graded path) is byte-identical to before.
        finite = torch.isfinite(r)
        if self.adv_normalization_scope == "token-global":
            if token_counts is None:
                raise ValueError("adv_normalization_scope='token-global' requires generated token counts")
            if token_counts.shape != r.shape:
                raise ValueError(
                    "token_counts must align one-to-one with rewards; "
                    f"got {tuple(token_counts.shape)} vs {tuple(r.shape)}"
                )
            weights = token_counts.to(device=r.device, dtype=torch.float64)
            active = finite & torch.isfinite(weights) & (weights > 0)
            if not bool(active.any()):
                return torch.zeros_like(r)

            rr = r.to(torch.float64)
            ww = weights[active]
            factor = ww.sum()
            mean = (rr[active] * ww).sum() / factor
            centered = rr - mean
            if self.normalize_adv_by_std:
                # AReaL climb4 normalizes (reward - .5) * 10 with
                # std_unbiased=true and eps=1e-5. Centering removes the bias;
                # dividing numerator/denominator by 10 makes the equivalent eps
                # on UniRL's raw 0/1 reward scale exactly 1e-6.
                if float(factor.item()) <= 1.0:
                    std = rr.new_ones(())
                else:
                    std = ((centered[active].square() * ww).sum() / (factor - 1.0)).sqrt()
                centered = centered / (std + 1e-6)
            return torch.where(active, centered, torch.zeros_like(centered)).to(torch.float32)
        if self.adv_normalization_scope == "global":
            rf = r[finite]
            mean = rf.mean() if rf.numel() else r.new_zeros(())
            centered = torch.where(finite, r - mean, torch.zeros_like(r))
            if self.normalize_adv_by_std:
                std = rf.std(unbiased=False) if rf.numel() > 1 else r.new_ones(())
                centered = centered / (std + 1e-8)
            return torch.where(finite, centered, torch.zeros_like(centered))
        adv = torch.zeros_like(r)
        for idxs in build_group_index_map(group_ids).values():
            idx = torch.tensor(idxs, dtype=torch.long)
            fin = finite[idx]
            g = r[idx]
            gf = g[fin]
            if gf.numel() == 0:
                continue  # whole group crashed -> zero advantage
            centered = g - gf.mean()
            if self.normalize_adv_by_std:
                std = gf.std(unbiased=False) if gf.numel() > 1 else g.new_ones(())
                centered = centered / (std + 1e-8)
            adv[idx] = torch.where(fin, centered, torch.zeros_like(centered))
        return adv

    def _pad_to_dp_multiple(self, part: Part) -> Part:
        """Pad ``part`` up to a multiple of the train DP size by replicating the
        shortest row with advantage 0 (zero gradient for GRPO/DRPO/CPPO), so the ragged
        Σ-turns batch satisfies ``pytree_chunk``'s divisibility check."""
        dp = int(getattr(self.stack, "dp_size", self.num_devices))
        pad = (-int(part.batch_size)) % dp
        if pad == 0:
            return part
        lengths = part.segment.lengths if part.segment is not None else None
        if lengths is not None and lengths.numel():
            positive = torch.nonzero(lengths > 0, as_tuple=False).flatten()
            if positive.numel() == 0:
                raise ValueError("AgenticTrainer cannot DP-pad an all-zero-token training part")
            src = int(positive[torch.argmin(lengths[positive])].item())
        else:
            src = 0

        # Give real rows an explicit all-active mask before concatenating pads.
        # The pads must still replay/backward on their DP rank for collective
        # parity, but a zero mask keeps them out of AReaL's global token
        # denominator (and makes their connected backward exactly zero).
        segment = part.segment
        if segment is not None and getattr(segment, "tokens", None) is not None and hasattr(segment, "loss_mask"):
            segment = segment.clone()
            if segment.loss_mask is None:
                token_shape = getattr(segment.tokens, "shape", None)
                if not token_shape:
                    raise ValueError("AgenticTrainer cannot build a loss mask without packed token shape metadata")
                # Rollout tensors reach the driver as TensorRef proxies.  The
                # mask is controller-owned metadata, so construct it from the
                # proxy's shape instead of calling a torch op on the proxy.
                segment.loss_mask = torch.ones(int(token_shape[0]), dtype=torch.bool)
            else:
                # Keep pre-existing rollout masks usable when they too arrived
                # as TensorRef proxies; Part.select/concat can then pad them as
                # ordinary packed CPU metadata.
                segment.loss_mask = hydrate(segment.loss_mask).to(dtype=torch.bool)
            part = _part_with_field(part, "segment", segment)
        pad_block = part.select(torch.full((pad,), src, dtype=torch.long))
        pad_block = _part_with_field(pad_block, "advantages", torch.zeros(pad, dtype=torch.float32))
        if pad_block.segment is not None and getattr(pad_block.segment, "loss_mask", None) is not None:
            pad_segment = pad_block.segment.clone()
            pad_segment.loss_mask = torch.zeros_like(pad_segment.loss_mask, dtype=torch.bool)
            pad_block = _part_with_field(pad_block, "segment", pad_segment)
        return Part.concat([part, pad_block])

    def evaluate(self, rollout_id: int) -> float:
        raise NotImplementedError(
            "AgenticTrainer.evaluate is not implemented: the agentic engine returns "
            "List[Sample], not a Sample. Set eval_interval=0 (agentic eval is a follow-up)."
        )
