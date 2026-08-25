"""AReaL-specific asynchronous training and concatenated trajectory assembly."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.rollout.manager import AdmissionPolicy, RolloutManager, keep_within_lag
from unirl.train.stack import TrainStepResult
from unirl.trainer.areal_dump import ARealTrajectoryDumper
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample, unwrap_replicated_int
from unirl.types.advantages import token_weighted_global_normalize
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.types.segments import TextSegment
from unirl.utils.graceful_shutdown import run_with_timeout
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

_PROTOCOL = "areal_deep_research/v1"
_ROLLOUT_SHUTDOWN_TIMEOUT_S = 60.0
_ROLLOUT_DRAIN_TIMEOUT_S = 120.0
_DATA_STATE_FILENAME = "areal_data_state.json"
_REWARD_CLIP = 20.0
_ADVANTAGE_EPS = 1e-5

logger = logging.getLogger(__name__)


class ARealTrajectoryError(ValueError):
    """The rollout cannot be represented as one faithful AReaL training row."""


def _is_failed(trajectory: Sample) -> bool:
    return not trajectory.gen_parts() or not trajectory.parts or trajectory.parts[-1].harness_status == "failed"


def areal_metadata(trajectory: Sample) -> Mapping[str, Any]:
    """Return the terminal AReaL harness metadata for one trajectory."""
    if not trajectory.parts or not trajectory.parts[-1].metadata:
        raise ARealTrajectoryError("trajectory has no terminal harness metadata")
    metadata = trajectory.parts[-1].metadata[0] or {}
    harness = metadata.get("harness")
    if not isinstance(harness, Mapping) or harness.get("protocol") != _PROTOCOL:
        raise ARealTrajectoryError("trajectory is not stamped with the AReaL protocol")
    return harness


def _prompt_ids(part: Part) -> torch.Tensor:
    prompt = part.conditions.get("prompt")
    if not isinstance(prompt, TextTokenCondition) or prompt.input_ids is None or prompt.attention_mask is None:
        raise ARealTrajectoryError("generated turn has no tokenized prompt condition")
    input_ids = hydrate(prompt.input_ids).to(dtype=torch.long, device="cpu")
    attention_mask = hydrate(prompt.attention_mask).to(dtype=torch.bool, device="cpu")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or input_ids.shape[0] != 1:
        raise ARealTrajectoryError("generated prompt condition must contain exactly one aligned row")
    return input_ids[0][attention_mask[0]].clone()


def _output(part: Part) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(part.segment, TextSegment) or part.segment.tokens is None or part.segment.log_probs is None:
        raise ARealTrajectoryError("generated turn has no text tokens and behavior log-probabilities")
    if part.batch_size != 1:
        raise ARealTrajectoryError("AReaL trajectory assembly requires single-row generated turns")
    tokens = hydrate(part.segment.tokens).to(dtype=torch.long, device="cpu").flatten()
    log_probs = hydrate(part.segment.log_probs).to(dtype=torch.float32, device="cpu").flatten()
    if tokens.shape != log_probs.shape:
        raise ARealTrajectoryError("generated tokens and behavior log-probabilities are not aligned")
    return tokens, log_probs


def build_areal_part(trajectory: Sample) -> Part:
    """Build one masked training row from every generated turn in a trajectory."""
    metadata = areal_metadata(trajectory)
    if not isinstance(metadata.get("prediction"), str):
        raise ARealTrajectoryError("completed trajectory has no string prediction")
    generated = trajectory.gen_parts()
    if not generated:
        raise ARealTrajectoryError("trajectory has no generated turns")

    token_chunks: list[torch.Tensor] = []
    log_prob_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    output_versions: list[int] = []
    first_prompt: torch.Tensor | None = None
    previous_prompt: torch.Tensor | None = None
    previous_output: torch.Tensor | None = None

    for turn, part in enumerate(generated):
        prompt = _prompt_ids(part)
        output, log_probs = _output(part)
        if first_prompt is None:
            first_prompt = prompt
        else:
            assert previous_prompt is not None and previous_output is not None
            expected_prefix = torch.cat([previous_prompt, previous_output])
            if prompt.numel() <= expected_prefix.numel() or not torch.equal(
                prompt[: expected_prefix.numel()], expected_prefix
            ):
                raise ARealTrajectoryError(f"turn {turn} prompt is not a strict extension of its parent interaction")
            suffix = prompt[expected_prefix.numel() :]
            token_chunks.append(suffix)
            log_prob_chunks.append(torch.zeros(suffix.numel(), dtype=torch.float32))
            mask_chunks.append(torch.zeros(suffix.numel(), dtype=torch.bool))

        token_chunks.append(output)
        log_prob_chunks.append(log_probs)
        mask_chunks.append(torch.ones(output.numel(), dtype=torch.bool))
        output_versions.append(int(part.output_version) if part.output_version is not None else -1)
        previous_prompt = prompt
        previous_output = output

    assert first_prompt is not None
    if first_prompt.numel() == 0:
        raise ARealTrajectoryError("trajectory has an empty initial policy prompt")
    tokens = torch.cat(token_chunks) if token_chunks else torch.zeros(0, dtype=torch.long)
    log_probs = torch.cat(log_prob_chunks) if log_prob_chunks else torch.zeros(0, dtype=torch.float32)
    loss_mask = torch.cat(mask_chunks) if mask_chunks else torch.zeros(0, dtype=torch.bool)
    if not bool(loss_mask.any()):
        raise ARealTrajectoryError("trajectory has no trainable assistant tokens")
    if tokens.shape != log_probs.shape or tokens.shape != loss_mask.shape:
        raise ARealTrajectoryError("assembled tokens, log-probabilities, and loss mask are not aligned")

    prompt_condition = TextTokenCondition(
        input_ids=first_prompt.unsqueeze(0),
        attention_mask=torch.ones((1, first_prompt.numel()), dtype=torch.long),
    )
    segment = TextSegment.pack(tokens=[tokens], log_probs=[log_probs], loss_mask=[loss_mask])
    row_metadata = {
        "areal": {
            "termination_reason": metadata.get("termination_reason"),
            "policy_call_count": metadata.get("policy_call_count"),
            "output_versions": output_versions,
        }
    }
    return Part(
        sample_ids=[generated[-1].sample_ids[0]],
        segment=segment,
        conditions={"prompt": prompt_condition},
        metadata=[row_metadata],
        sampling_params=generated[0].sampling_params,
    )


class ARealTrainer(BaseTrainer):
    """AReaL trajectory trainer with bounded-staleness asynchronous rollout and training.

    Rollout and training hold disjoint GPU slabs, so generation keeps running while the
    consumer scores and takes an optimizer step. A :class:`AdmissionPolicy` bounds how far
    the producer may run ahead: at most ``max_concurrent_rollouts`` grouped workflows are
    in flight, and the producer may lead the consumer by ``(max_staleness + 1) * batch_size``
    prompt roots. Each optimizer step publishes full weights and stamps the next policy
    version; roots whose earliest turn is more than ``max_staleness`` versions behind are
    rejected rather than trained on.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        reward_transform_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
        stop: Optional[List[str]] = None,
        max_concurrent_rollouts: Optional[int] = None,
        per_worker_inflight: int = 8,
        max_staleness: int = 4,
        train_fraction: float = 0.5,
        trajectory_dump_dir: str = "",
    ) -> None:
        self._validate_config(
            cfg=cfg,
            batch_size=batch_size,
            rollout_cfg=rollout_cfg,
            sampling_cfg=sampling_cfg,
            reward_transform_cfg=reward_transform_cfg,
            algorithm_cfg=algorithm_cfg,
            stack_cfg=stack_cfg,
            sync_cfg=sync_cfg,
            stop=stop,
            max_concurrent_rollouts=max_concurrent_rollouts,
            per_worker_inflight=per_worker_inflight,
            max_staleness=max_staleness,
            train_fraction=train_fraction,
        )
        self._trajectory_dumper = ARealTrajectoryDumper(trajectory_dump_dir)
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)

        try:
            self.batch_size = int(batch_size)
            self.data_source = instantiate(data_source_cfg)
            self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
            self._group_size = total_samples_per_prompt(self.sampling_params)
            self._max_concurrent_rollouts = int(
                batch_size if max_concurrent_rollouts is None else max_concurrent_rollouts
            )
            self._per_worker_inflight = int(per_worker_inflight)
            self._reward_bias = float(reward_transform_cfg.bias)
            self._reward_scale = float(reward_transform_cfg.scale)
            self._rollout_seed = int(cfg.get("seed", 0))
            self._sampling_seed_stride = int(rollout_cfg.config.harness.max_policy_calls) + 1
            self._train_fraction = float(train_fraction)
            self._policy = AdmissionPolicy(
                max_concurrent_roots=self._max_concurrent_rollouts,
                max_staleness=int(max_staleness),
                consumer_batch=self.batch_size,
            )
            self._root_ordinal = 0
            self._validate_slabs(rollout_cfg)

            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
                self.bundle = remote_hydra(bundle_cfg)
                self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
                self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
                self.reward = remote_hydra(reward_cfg)
                self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
                self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
            with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
                self.rollout = remote(**rollout_parsed)
            self._connect_weight_sync()

            indices = [
                index
                for index, rank_info in enumerate(self.rollout.rank_infos)
                if rank_info.tp_rank == 0 and rank_info.pp_rank == 0
            ]
            slots = [self.rollout.slot(index) for index in indices]
            launchers = [lambda sample, slot=slot: slot.launch("generate", sample) for slot in slots]
            self._rollout_manager = RolloutManager(
                self.rollout,
                launchers=launchers,
                capacities=[self._per_worker_inflight] * len(launchers),
                group_size=self._group_size,
                filter_fn=keep_within_lag(self._policy.max_staleness),
                policy=self._policy,
            )
            self._train_version = unwrap_replicated_int(
                self.backend.get_optimizer_step_count(),
                name="backend optimizer step count",
            )
        except BaseException:
            self._shutdown_runtime()
            raise

    @staticmethod
    def _validate_config(
        *,
        cfg: DictConfig,
        batch_size: int,
        rollout_cfg: DictConfig,
        sampling_cfg: DictConfig,
        reward_transform_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        sync_cfg: Optional[DictConfig],
        stop: Optional[List[str]],
        max_concurrent_rollouts: Optional[int],
        per_worker_inflight: int,
        max_staleness: int,
        train_fraction: float,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        concurrent = int(batch_size if max_concurrent_rollouts is None else max_concurrent_rollouts)
        if concurrent < int(batch_size):
            raise ValueError(f"max_concurrent_rollouts must be >= batch_size; got {concurrent} < {batch_size}")
        if int(per_worker_inflight) <= 0:
            raise ValueError(f"per_worker_inflight must be positive; got {per_worker_inflight}")
        if int(max_staleness) < 0:
            raise ValueError(f"max_staleness must be non-negative; got {max_staleness}")
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError(f"train_fraction must leave a non-empty rollout slab; got {train_fraction}")
        if stop:
            raise ValueError("ARealTrainer requires an empty policy stop list")
        reward_bias = float(reward_transform_cfg.get("bias"))
        reward_scale = float(reward_transform_cfg.get("scale"))
        if not math.isfinite(reward_bias):
            raise ValueError(f"reward_transform.bias must be finite; got {reward_bias}")
        if not math.isfinite(reward_scale) or reward_scale <= 0.0:
            raise ValueError(f"reward_transform.scale must be finite and positive; got {reward_scale}")
        num_updates = int(stack_cfg.get("num_updates_per_batch", 1))
        if num_updates != 1:
            raise ValueError(f"ARealTrainer requires stack.num_updates_per_batch=1; got {num_updates}")

        worker_max_concurrency = int(cfg.get("worker_max_concurrency", 1))
        required_concurrency = int(per_worker_inflight) + 2
        if worker_max_concurrency < required_concurrency:
            raise ValueError(
                f"worker_max_concurrency ({worker_max_concurrency}) must be >= per_worker_inflight + 2 "
                f"({required_concurrency}) so control calls (set_stopping/sleep/weight sync) are not "
                "starved by trajectory slots; raise worker_max_concurrency in the recipe or lower "
                "per_worker_inflight"
            )
        if sync_cfg is None:
            raise ValueError("ARealTrainer requires a cross-slab NCCLWeightSync; add a `sync:` block")
        sync_target = str(sync_cfg.get("_target_", ""))
        if not sync_target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"ARealTrainer runs disjoint rollout and training slabs and requires NCCLWeightSync; "
                f"got sync._target_={sync_target!r}"
            )

        episode = rollout_cfg.get("config", {}).get("episode_sampling")
        if episode is None:
            raise ValueError("rollout.config.episode_sampling is required")
        group_size = int(sampling_cfg.get("samples_per_prompt", 1))
        episode_group_size = int(episode.get("samples_per_prompt", 1))
        if group_size != episode_group_size:
            raise ValueError(
                "sampling.samples_per_prompt must equal rollout.config.episode_sampling.samples_per_prompt"
            )
        sampling_temperature = float(sampling_cfg.get("temperature", 1.0))
        episode_temperature = float(episode.get("temperature", 1.0))
        algorithm_temperature = float(algorithm_cfg.get("sampling_temperature", 1.0))
        if abs(sampling_temperature - episode_temperature) > 1e-9:
            raise ValueError("sampling.temperature must equal rollout episode temperature")
        if abs(sampling_temperature - algorithm_temperature) > 1e-9:
            raise ValueError("sampling.temperature must equal algorithm.sampling_temperature")

    def _validate_slabs(self, rollout_cfg: DictConfig) -> None:
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        self._rollout_devices = self.num_devices - self._train_devices
        if self._train_devices <= 0 or self._rollout_devices <= 0:
            raise ValueError(
                f"train_fraction={self._train_fraction} splits {self.num_devices} devices into "
                f"{self._train_devices} train / {self._rollout_devices} rollout; both slabs must be non-empty"
            )
        rows = self.batch_size * self._group_size
        if rows % self._train_devices != 0:
            raise ValueError(
                f"batch_size * samples_per_prompt = {rows} is not divisible by the train slab size "
                f"{self._train_devices}; adjust batch_size / samples_per_prompt / train_fraction"
            )
        tp_size = int(rollout_cfg.get("config", {}).get("inner", {}).get("tp_size", 1))
        if self._rollout_devices % tp_size != 0:
            raise ValueError(f"rollout slab of {self._rollout_devices} devices is not divisible by tp_size={tp_size}")

    def _connect_weight_sync(self) -> None:
        """One-time cross-slab NCCL handshake; the rollout slab stays resident afterwards."""
        addr, port = self.weight_sync.pick_master()[0]
        targets = self.rollout.tp_zero_workers
        self.weight_sync.set_rollout_targets(targets, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(targets) * self.rollout.tp_size,
            tp_size=self.rollout.tp_size,
            pp_size=self.rollout.pp_size,
        )

    def _build_request_sample(self, inputs: Sample, rollout_id: int) -> Sample:
        return prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text"},
            caller="ARealTrainer._build_request_sample",
            root_control={"ar": {"stop": []}},
        )

    def _collect_groups(self, rollout_id: int) -> Tuple[List[List[Sample]], Dict[str, float]]:
        publish_start = time.perf_counter()
        self._rollout_manager.sync_weights(self.weight_sync, output_version=self._train_version)
        publish_seconds = time.perf_counter() - publish_start

        submitted = self._top_up(rollout_id)
        collect_start = time.perf_counter()
        groups = self._rollout_manager.collect(self.batch_size, current_version=self._train_version)
        collect_seconds = time.perf_counter() - collect_start

        stats = self._rollout_manager.stats
        random.Random(self._rollout_seed + rollout_id).shuffle(groups)
        metrics = {
            "runtime/publish_seconds": publish_seconds,
            "runtime/collect_seconds": collect_seconds,
            "runtime/submitted_roots": float(submitted),
            "runtime/pending_roots": float(stats.pending_roots),
            "runtime/running_roots": float(stats.active_roots),
            "runtime/carried_groups": float(stats.ready_groups),
            "runtime/carry_age_seconds": stats.ready_age_seconds,
            "runtime/oldest_running_age_seconds": stats.oldest_active_age_seconds,
            "runtime/accepted_roots": float(stats.accepted),
            "runtime/stale_rejected_roots": float(stats.rejected),
            "runtime/producer_lead_roots": float(stats.lead),
        }
        return groups, metrics

    def _top_up(self, rollout_id: int) -> int:
        """Refill the pending queue to the staleness window so admission never starves."""
        capacity = self._rollout_manager.pending_capacity
        if capacity <= 0:
            return 0
        requests = self._build_request_sample(self.data_source.get_samples(capacity), rollout_id)
        self._rollout_manager.submit(self._build_trajectory_tasks(requests))
        return capacity

    def _build_trajectory_tasks(self, requests: Sample) -> List[Sample]:
        tasks = []
        for prompt in requests.split():
            root_ordinal = self._root_ordinal
            self._root_ordinal += 1
            for sibling_index in range(self._group_size):
                trajectory_index = root_ordinal * self._group_size + sibling_index
                seed_base = (self._rollout_seed + trajectory_index * self._sampling_seed_stride) % ((1 << 63) - 1)
                root = prompt.parts[0]
                control = dict(root.control)
                ar_control = dict(control.get("ar") or {})
                ar_control["sampling_seed_base"] = seed_base
                control["ar"] = ar_control
                metadata = dict(root.metadata[0] or {}) if root.metadata else {}
                metadata.update({"sibling_index": sibling_index, "sampling_seed_base": seed_base})
                seeded_root = _part_with_field(root, "control", control)
                seeded_root = _part_with_field(seeded_root, "metadata", [metadata])
                tasks.append(prompt.with_parts([seeded_root, *prompt.parts[1:]]))
        return tasks

    def train_step(
        self,
        *,
        training_progress: float = 0.0,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        t0 = time.perf_counter()
        groups, runtime_metrics = self._collect_groups(rollout_id)
        trajectories = [trajectory for group in groups for trajectory in group]
        train_rows = self._prepare_training_rows(trajectories, rollout_id)
        eligible = [part is not None for part in train_rows]
        judge_start = time.perf_counter()
        rewards = self._score_trajectories(trajectories, rollout_id, eligible=eligible)
        runtime_metrics["runtime/judge_seconds"] = time.perf_counter() - judge_start
        runtime_metrics.update(self._trajectory_metrics(trajectories))
        dump_summary = self._trajectory_dumper.dump(groups, rewards, rollout_id=rollout_id)
        logger.info(
            "rollout %d dumped %d trajectories with %d identical sibling groups",
            rollout_id,
            dump_summary["trajectories"],
            dump_summary["fully_identical_assistant_groups"],
        )
        advantages, scaled_rewards, token_counts, norm_mean, norm_std = self._areal_advantages(
            train_rows,
            rewards,
        )
        result, mean_reward = self._train_and_log(
            trajectories,
            train_rows,
            rewards,
            scaled_rewards,
            token_counts,
            advantages,
            norm_mean=norm_mean,
            norm_std=norm_std,
            rollout_id=rollout_id,
            training_progress=training_progress,
            t0=t0,
            runtime_metrics=runtime_metrics,
        )
        self._train_version += result.optimizer_updates
        return result, mean_reward

    def _trajectory_metrics(self, trajectories: List[Sample]) -> Dict[str, float]:
        """Per-phase wall time from the harness plus the policy-version spread of the batch."""
        policy_seconds = []
        tool_seconds = []
        versions = []
        for trajectory in trajectories:
            for part in trajectory.gen_parts():
                if part.output_version is not None:
                    versions.append(int(part.output_version))
            metadata = trajectory.parts[-1].metadata if trajectory.parts else None
            harness = (metadata[0] or {}).get("harness") if metadata else None
            if isinstance(harness, Mapping):
                policy_seconds.append(float(harness.get("policy_seconds") or 0.0))
                tool_seconds.append(float(harness.get("tool_seconds") or 0.0))
        return {
            "runtime/policy_seconds_mean": (sum(policy_seconds) / len(policy_seconds)) if policy_seconds else 0.0,
            "runtime/tool_seconds_mean": (sum(tool_seconds) / len(tool_seconds)) if tool_seconds else 0.0,
            "runtime/tool_seconds_max": max(tool_seconds, default=0.0),
            "runtime/version_span": float(max(versions) - min(versions)) if versions else 0.0,
            "runtime/version_lag_max": float(self._train_version - min(versions)) if versions else 0.0,
        }

    def _prepare_training_rows(
        self,
        trajectories: List[Sample],
        rollout_id: int,
    ) -> List[Optional[Part]]:
        prepared: List[Optional[Part]] = []
        for trajectory in trajectories:
            if _is_failed(trajectory):
                prepared.append(None)
                continue
            try:
                prepared.append(build_areal_part(trajectory))
            except ARealTrajectoryError as exc:
                logger.warning("AReaL rollout %d: excluded invalid trajectory (%s)", rollout_id, exc)
                prepared.append(None)
        return prepared

    def _score_trajectories(
        self,
        trajectories: List[Sample],
        rollout_id: int,
        *,
        eligible: List[bool],
    ) -> torch.Tensor:
        rewards = torch.full((len(trajectories),), float("nan"), dtype=torch.float32)
        if len(eligible) != len(trajectories):
            raise RuntimeError("trajectory eligibility cardinality mismatch")
        valid_indices = [
            i for i, trajectory in enumerate(trajectories) if not _is_failed(trajectory) and bool(eligible[i])
        ]
        if not valid_indices:
            logger.warning("AReaL rollout %d: every trajectory failed validation", rollout_id)
            return rewards

        questions: List[str] = []
        predictions: List[str] = []
        answers: List[Any] = []
        for i in valid_indices:
            trajectory = trajectories[i]
            root = trajectory.parts[0]
            metadata = root.metadata[0] if root.metadata else {}
            prompt = root.primitives.get("text")
            questions.append(prompt.texts[0] if isinstance(prompt, Texts) and prompt.texts else "")
            prediction = areal_metadata(trajectory).get("prediction")
            if not isinstance(prediction, str):
                raise RuntimeError("eligible AReaL trajectory has no string prediction")
            predictions.append(prediction)
            answers.append((metadata or {}).get("answer"))

        score_count = len(valid_indices)
        reward_dp_size = max(1, int(self.reward.dp_size))
        score_padding = (-score_count) % reward_dp_size
        if score_padding:
            questions.extend([questions[0]] * score_padding)
            predictions.extend([predictions[0]] * score_padding)
            answers.extend([answers[0]] * score_padding)

        ar_sampling = self.sampling_params.get("ar")
        score_input = Part.input(
            [f"score{rollout_id}:{i}" for i in valid_indices]
            + [f"score{rollout_id}:padding{i}" for i in range(score_padding)],
            primitives={"text": Texts(texts=questions)},
            metadata=[{"answer": answer} for answer in answers],
        )
        scoring = (
            Sample.request(score_input)
            .fork(1, sampling_params=ar_sampling)
            .with_filled_frontier(primitives={"text": Texts(texts=predictions)})
        )
        scored = self.reward.score_and_attach(scoring)
        scored_rewards = scored.parts[-1].rewards
        if scored_rewards is None:
            raise RuntimeError("reward service returned no rewards")
        values = hydrate(scored_rewards).to(torch.float32).flatten()
        if values.numel() != score_count + score_padding:
            raise RuntimeError(
                f"reward service returned {values.numel()} rewards for {score_count + score_padding} scoring rows"
            )
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("reward service returned non-finite rewards")
        rewards[torch.tensor(valid_indices, dtype=torch.long)] = values[:score_count]

        failed = len(trajectories) - len(valid_indices)
        if failed:
            logger.warning(
                "AReaL rollout %d: %d/%d trajectories failed and were excluded",
                rollout_id,
                failed,
                len(trajectories),
            )
        return rewards

    def _areal_advantages(
        self,
        train_rows: List[Optional[Part]],
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        if len(train_rows) != rewards.numel():
            raise RuntimeError("AReaL training-row/reward cardinality mismatch")

        token_counts = torch.zeros(rewards.shape, dtype=torch.long)
        healthy = torch.zeros(rewards.shape, dtype=torch.bool)
        for i, row in enumerate(train_rows):
            if row is None:
                continue
            if row.segment is None or row.segment.loss_mask is None:
                raise RuntimeError("prepared AReaL row has no trajectory loss mask")
            token_counts[i] = int(hydrate(row.segment.loss_mask).to(dtype=torch.bool).count_nonzero().item())
            healthy[i] = bool(torch.isfinite(rewards[i])) and int(token_counts[i]) > 0

        scaled_rewards = ((rewards + self._reward_bias) * self._reward_scale).clamp(
            min=-_REWARD_CLIP,
            max=_REWARD_CLIP,
        )
        advantages, mean, std = token_weighted_global_normalize(
            scaled_rewards,
            token_counts,
            healthy,
            eps=_ADVANTAGE_EPS,
        )
        return advantages, scaled_rewards, token_counts, float(mean), float(std)

    def _train_and_log(
        self,
        trajectories: List[Sample],
        train_rows: List[Optional[Part]],
        rewards: torch.Tensor,
        scaled_rewards: torch.Tensor,
        token_counts: torch.Tensor,
        advantages: torch.Tensor,
        *,
        norm_mean: float,
        norm_std: float,
        rollout_id: int,
        training_progress: float,
        t0: float,
        runtime_metrics: Dict[str, float],
    ) -> Tuple[TrainStepResult, float]:
        if len(train_rows) != len(trajectories):
            raise RuntimeError("trajectory/training-row cardinality mismatch")
        healthy = torch.isfinite(rewards) & (token_counts > 0)
        mean_reward = float(rewards[healthy].mean().item()) if bool(healthy.any()) else 0.0

        prepared: List[Part] = []
        for i, generated in enumerate(train_rows):
            if not bool(healthy[i]):
                continue
            if generated is None:
                raise RuntimeError("finite AReaL trajectory has no prepared training row")
            generated = _part_with_field(
                generated,
                "advantages",
                torch.full((generated.batch_size,), float(advantages[i].item()), dtype=torch.float32),
            )
            generated = _part_with_field(generated, "primitive_metadata", {})
            generated = _part_with_field(generated, "media_preview", None)
            generated = _part_with_field(generated, "primitives", {})
            generated = _part_with_field(generated, "rewards", None)
            generated = _part_with_field(generated, "component_rewards", None)
            prepared.append(generated)

        depths = [len(trajectory.gen_parts()) for trajectory in trajectories]
        logger.info(
            "rollout %d AReaL trajectory calls: n=%d mean=%.2f min=%d max=%d hist=%s",
            rollout_id,
            len(depths),
            (sum(depths) / len(depths)) if depths else 0.0,
            min(depths, default=0),
            max(depths, default=0),
            dict(sorted(Counter(depths).items())),
        )
        update_start = time.perf_counter()
        if prepared:
            train_part = self._pad_to_dp_multiple(Part.concat(prepared))
            result = self.stack.train_track(train_part, training_progress=float(training_progress))
            train_rows_count = int(train_part.batch_size)
        else:
            result = TrainStepResult(0.0, 0.0, 0.0, False, [], {}, optimizer_updates=0)
            train_rows_count = 0
        runtime_metrics["runtime/update_seconds"] = time.perf_counter() - update_start
        step_seconds = time.perf_counter() - t0
        runtime_metrics["runtime/roots_per_minute"] = (
            (len(trajectories) / self._group_size) * 60.0 / max(step_seconds, 1e-9)
        )

        log_sample = self._build_log_sample(
            trajectories,
            rewards,
            scaled_rewards,
            token_counts,
            advantages,
            healthy,
            rollout_id,
        )
        scaled_reward_mean = float(scaled_rewards[healthy].mean().item()) if bool(healthy.any()) else 0.0
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=step_seconds,
            extra_metrics={
                "agent/mean_turns": (sum(depths) / len(depths)) if depths else 0.0,
                "agent/max_turns": max(depths) if depths else 0,
                "agent/failed_trajectories": int((~healthy).sum().item()),
                "agent/train_rows": train_rows_count,
                "objective/scaled_reward_mean": scaled_reward_mean,
                "objective/scaled_reward_token_mean": norm_mean,
                "objective/scaled_reward_token_std": norm_std,
                "objective/active_tokens": int(token_counts[healthy].sum().item()),
                "objective/healthy_trajectories": int(healthy.sum().item()),
                **runtime_metrics,
            },
        )
        return result, mean_reward

    def _build_log_sample(
        self,
        trajectories: List[Sample],
        rewards: torch.Tensor,
        scaled_rewards: torch.Tensor,
        token_counts: torch.Tensor,
        advantages: torch.Tensor,
        healthy: torch.Tensor,
        rollout_id: int,
    ) -> Sample:
        size = rewards.numel()
        if not (
            len(trajectories)
            == size
            == scaled_rewards.numel()
            == token_counts.numel()
            == advantages.numel()
            == healthy.numel()
        ):
            raise RuntimeError("AReaL objective logging tensors are not aligned")
        indices = healthy.nonzero(as_tuple=False).flatten().tolist()
        count = len(indices)
        if count == 0:
            return Sample(parts=[])

        ar_sampling = self.sampling_params.get("ar")
        root = Part.input(
            [f"log{rollout_id}:{i}" for i in indices],
            primitives={"text": Texts(texts=[""] * count)},
        )
        sample = (
            Sample.request(root)
            .fork(1, sampling_params=ar_sampling)
            .with_filled_frontier(primitives={"text": Texts(texts=[""] * count)})
        )
        frontier = _part_with_field(sample.parts[-1], "rewards", rewards[healthy].to(torch.float32))
        frontier = _part_with_field(frontier, "advantages", advantages[healthy].to(torch.float32))
        frontier = _part_with_field(
            frontier,
            "metadata",
            [
                {
                    "areal_objective": {
                        "scaled_reward": float(scaled_rewards[i]),
                        "active_tokens": int(token_counts[i]),
                    }
                }
                for i in indices
            ],
        )
        return sample.with_parts([*sample.parts[:-1], frontier])

    def _pad_to_dp_multiple(self, part: Part) -> Part:
        dp_size = int(getattr(self.stack, "dp_size", self.num_devices))
        pad = (-int(part.batch_size)) % dp_size
        if pad == 0:
            return part
        lengths = part.segment.lengths if part.segment is not None else None
        source = int(torch.argmin(lengths).item()) if lengths is not None and lengths.numel() else 0
        padding = part.select(torch.full((pad,), source, dtype=torch.long))
        padding = _part_with_field(padding, "advantages", torch.zeros(pad, dtype=torch.float32))
        if padding.segment is not None and padding.segment.loss_mask is not None:
            padding.segment.loss_mask = torch.zeros_like(
                hydrate(padding.segment.loss_mask),
                dtype=torch.bool,
            )
        return Part.concat([part, padding])

    def _save_data_state(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
    ) -> None:
        if save_interval <= 0:
            return
        step = rollout_id + 1
        if step % save_interval != 0 and step < num_rollouts:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step}", _DATA_STATE_FILENAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as handle:
                json.dump(self.data_source.state_dict(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _load_data_state(self, load_dir: Optional[str], start_rollout: int) -> None:
        if not load_dir:
            return
        path = os.path.join(os.path.abspath(load_dir), _DATA_STATE_FILENAME)
        if os.path.exists(path):
            with open(path) as handle:
                self.data_source.load_state_dict(json.load(handle))
            logger.info("Restored AReaL dataset cursor from %s", path)
            return
        logger.warning("No %s beside the checkpoint; fast-forwarding %d batches", _DATA_STATE_FILENAME, start_rollout)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        try:
            start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
            self._train_version = unwrap_replicated_int(
                self.backend.get_optimizer_step_count(),
                name="backend optimizer step count",
            )
            self._load_data_state(load_dir, start_rollout)
            self._init_wandb(
                num_rollouts=num_rollouts,
                extra={
                    "areal_execution": "async_bounded_staleness",
                    "max_concurrent_rollouts": self._max_concurrent_rollouts,
                    "max_staleness": self._policy.max_staleness,
                    "pending_window_roots": self._policy.window,
                    "train_devices": self._train_devices,
                    "rollout_devices": self._rollout_devices,
                },
            )

            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self.train_step(
                    training_progress=training_progress,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                self.maybe_save_checkpoint(
                    rollout_id,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
                self._save_data_state(
                    rollout_id,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                )
        finally:
            try:
                self._finish_wandb()
            finally:
                self._shutdown_runtime()

    def shutdown(self) -> None:
        self._shutdown_runtime()

    def _shutdown_runtime(self) -> None:
        if getattr(self, "_runtime_shutdown_done", False):
            return
        self._runtime_shutdown_done = True

        manager = getattr(self, "_rollout_manager", None)
        rollout = getattr(self, "rollout", None)
        shutdown = getattr(rollout, "shutdown", None)
        pool = getattr(self, "pool", None)
        try:
            # Draining waits on in-flight tool calls; abandon it rather than hold the GPUs.
            if manager is not None:
                run_with_timeout(
                    manager.close,
                    timeout=_ROLLOUT_DRAIN_TIMEOUT_S,
                    what="AReaL rollout drain",
                )
        finally:
            try:
                if callable(shutdown):
                    run_with_timeout(
                        shutdown,
                        timeout=_ROLLOUT_SHUTDOWN_TIMEOUT_S,
                        what="AReaL rollout engine shutdown",
                    )
            finally:
                if pool is not None:
                    pool.shutdown()


__all__ = ["ARealTrainer", "ARealTrajectoryError", "areal_metadata", "build_areal_part"]
