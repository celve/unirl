"""Fully asynchronous single-rollout actor/critic training for SAO.

This trainer reuses the resident agentic rollout engine and cross-slab actor
weight synchronization from :mod:`unirl.trainer.agentic_async`, but removes the
same-prompt group barrier.  Every completed trajectory enters a FIFO queue and
the composite SAO stack performs critic x2 -> fresh GAE -> actor x1.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.agentic_async import AsyncAgenticTrainer
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.sample import Part, Sample
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


def _normalize_stop(stop: Optional[Sequence[str]]) -> List[str]:
    """Preserve an explicit empty stop list for free-text environments."""
    return ["</tool_call>"] if stop is None else list(stop)


def _trajectory_versions(trajectory: Sample, fallback: int) -> Tuple[int, int]:
    versions = [int(p.weight_version) for p in trajectory.gen_parts() if p.weight_version is not None]
    if not versions:
        return int(fallback), int(fallback)
    return min(versions), max(versions)


@dataclass(frozen=True)
class _TrajectoryEntry:
    trajectory: Sample
    min_version: int
    max_version: int
    completion_id: int


class _TrajectoryBuffer:
    """FIFO completed-trajectory queue with optional oldest-turn eviction."""

    def __init__(self) -> None:
        self._items: Deque[_TrajectoryEntry] = deque()
        self.evicted = 0

    def put(self, trajectory: Sample, *, completion_version: int, completion_id: int) -> None:
        min_version, max_version = _trajectory_versions(trajectory, completion_version)
        self._items.append(
            _TrajectoryEntry(
                trajectory=trajectory,
                min_version=min_version,
                max_version=max_version,
                completion_id=int(completion_id),
            )
        )

    def size(self) -> int:
        return len(self._items)

    def _evict(self, *, current_version: int, max_oldest_version_lag: Optional[int]) -> None:
        if max_oldest_version_lag is None:
            return
        kept: Deque[_TrajectoryEntry] = deque()
        for item in self._items:
            if int(current_version) - item.min_version > int(max_oldest_version_lag):
                self.evicted += 1
            else:
                kept.append(item)
        self._items = kept

    def drain_fifo(
        self,
        n: int,
        *,
        current_version: int,
        max_oldest_version_lag: Optional[int] = None,
    ) -> Optional[List[Sample]]:
        self._evict(
            current_version=int(current_version),
            max_oldest_version_lag=max_oldest_version_lag,
        )
        if len(self._items) < int(n):
            return None
        return [self._items.popleft().trajectory for _ in range(int(n))]

    def version_bounds(self) -> Optional[Tuple[int, int]]:
        if not self._items:
            return None
        return min(x.min_version for x in self._items), max(x.max_version for x in self._items)


class AsyncSAOTrainer(AsyncAgenticTrainer):
    """Disaggregated agentic SAO with independent actor and critic learners."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        actor_cfg: DictConfig,
        critic_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        stop: Optional[List[str]] = None,
        train_fraction: float = 0.5,
        oversample_batch_size: Optional[int] = None,
        max_oldest_version_lag: Optional[int] = None,
    ) -> None:
        # AsyncAgenticTrainer.__init__ wires only one train-side model.  Build the
        # same two-slab topology here with actor + critic on the train slab.
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = int(batch_size)
        self.balance_shards = False
        self.eval_interval = 0
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None
        # ``None`` means "use the tool-calling default" while an explicit empty
        # list disables stop strings.  Stateful free-text environments such as
        # ALFWorld must be able to generate a complete ``Action: ...`` turn and
        # therefore deliberately pass ``stop: []``.
        self._stop = _normalize_stop(stop)

        self._train_fraction = float(train_fraction)
        self._max_oldest_version_lag = None if max_oldest_version_lag is None else int(max_oldest_version_lag)
        self._weight_version = 0
        self._gt_by_root: Dict[str, Optional[str]] = {}
        self._n = 1
        self._oversample = int(oversample_batch_size) if oversample_batch_size else self.batch_size
        self._submitted_prompt_batches = 0
        self._invalid_trajectories = 0

        if self._oversample < self.batch_size:
            raise ValueError(f"oversample_batch_size={self._oversample} must be >= batch_size={self.batch_size}")
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train devices of "
                f"{self.num_devices}; must leave non-empty train and rollout slabs."
            )
        self._rollout_devices = self.num_devices - self._train_devices
        if self.batch_size % self._train_devices != 0:
            raise ValueError(
                f"SAO trajectory batch_size={self.batch_size} must be divisible by train DP size={self._train_devices}."
            )
        self._validate_sampling(actor_cfg=actor_cfg, rollout_cfg=rollout_cfg, sampling_cfg=sampling_cfg)

        with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
            self.actor_bundle = remote_hydra(actor_cfg.bundle)
            self.actor_pipeline = remote_hydra(actor_cfg.pipeline, bundle=self.actor_bundle)
            self.actor_backend = remote_hydra(actor_cfg.backend, bundle=self.actor_bundle)
            self.actor_algorithm = remote_hydra(actor_cfg.algorithm, pipeline=self.actor_pipeline)

            self.critic_bundle = remote_hydra(critic_cfg.bundle)
            self.critic_pipeline = remote_hydra(critic_cfg.pipeline, bundle=self.critic_bundle)
            self.critic_backend = remote_hydra(critic_cfg.backend, bundle=self.critic_bundle)
            self.critic_algorithm = remote_hydra(critic_cfg.algorithm, pipeline=self.critic_pipeline)

            self.stack = remote_hydra(
                stack_cfg,
                actor_backend=self.actor_backend,
                actor_algorithm=self.actor_algorithm,
                critic_backend=self.critic_backend,
                critic_algorithm=self.critic_algorithm,
            )
            self.reward = remote_hydra(reward_cfg)
            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.actor_backend)

        # Compatibility aliases used by BaseTrainer helpers and inherited async
        # synchronization methods always refer to the actor side.
        self.bundle = self.actor_bundle
        self.pipeline = self.actor_pipeline
        self.backend = self.actor_backend
        self.algorithm = self.actor_algorithm

        with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                raise ValueError("AsyncSAOTrainer requires a separate resident rollout engine")
            self.rollout = remote(**rollout_parsed)

        self.rollout.set_workers(self.rollout.workers, self.rollout.role_name)
        if self.weight_sync is not None:
            self._connect_separate(sync_cfg)

    def _validate_sampling(
        self,
        *,
        actor_cfg: DictConfig,
        rollout_cfg: DictConfig,
        sampling_cfg: DictConfig,
    ) -> None:
        trainer_spp = int(sampling_cfg.get("samples_per_prompt", 1))
        episode = rollout_cfg.get("config", {}).get("episode_sampling", {})
        rollout_spp = int(episode.get("samples_per_prompt", 1))
        if trainer_spp != 1 or rollout_spp != 1:
            raise ValueError(
                "SAO requires exactly one rollout per prompt: trainer and episode "
                f"samples_per_prompt must both be 1, got {trainer_spp} and {rollout_spp}."
            )
        for name, actual, expected in (
            ("trainer top_p", float(sampling_cfg.get("top_p", 1.0)), 1.0),
            ("rollout top_p", float(episode.get("top_p", 1.0)), 1.0),
            ("trainer top_k", int(sampling_cfg.get("top_k", 0)), 0),
            ("rollout top_k", int(episode.get("top_k", 0)), 0),
        ):
            if actual != expected:
                raise ValueError(f"SAO behavior-logprob fidelity requires {name}={expected}, got {actual}")
        trainer_temp = float(sampling_cfg.get("temperature", 1.0))
        rollout_temp = float(episode.get("temperature", 1.0))
        replay_temp = float(actor_cfg.get("algorithm", {}).get("sampling_temperature", trainer_temp))
        if not (trainer_temp == rollout_temp == replay_temp):
            raise ValueError(
                "SAO sampling/replay temperatures must match, got "
                f"trainer={trainer_temp}, rollout={rollout_temp}, replay={replay_temp}."
            )

    def _submit_drive(self, carried: List[Sample], rollout_id: int) -> None:
        super()._submit_drive(carried, rollout_id)
        self._submitted_prompt_batches += 1

    def _pump(self) -> int:
        completed = self.rollout.poll()[0]
        for trajectory in completed:
            self._buffer.put(
                trajectory,
                completion_version=self._weight_version,
                completion_id=self._gen_id,
            )
            self._gen_id += 1
        return len(completed)

    @staticmethod
    def _has_action_tokens(trajectory: Sample) -> bool:
        for part in trajectory.gen_parts():
            tokens = getattr(part.segment, "tokens", None) if part.segment is not None else None
            if tokens is not None and int(hydrate(tokens).numel()) > 0:
                return True
        return False

    def _next_n(self, n: int, rollout_id: int) -> List[Sample]:
        refills = 0
        selected: List[Sample] = []
        while len(selected) < int(n):
            self._pump()
            picked = self._buffer.drain_fifo(
                int(n) - len(selected),
                current_version=self._weight_version,
                max_oldest_version_lag=self._max_oldest_version_lag,
            )
            if picked is not None:
                for trajectory in picked:
                    if self._has_action_tokens(trajectory):
                        selected.append(trajectory)
                    else:
                        self._invalid_trajectories += 1
                continue
            if self.rollout.drained()[0]:
                # A drive can finish in the narrow window after the poll above
                # but before ``drained``. Poll once more before submit resets the
                # worker completion buffers, otherwise those terminal trajectories
                # are silently dropped.
                self._pump()
                picked = self._buffer.drain_fifo(
                    int(n) - len(selected),
                    current_version=self._weight_version,
                    max_oldest_version_lag=self._max_oldest_version_lag,
                )
                if picked is not None:
                    for trajectory in picked:
                        if self._has_action_tokens(trajectory):
                            selected.append(trajectory)
                        else:
                            self._invalid_trajectories += 1
                    continue
                refills += 1
                if refills > self._MAX_REFILLS:
                    raise RuntimeError(
                        f"SAO rollout {rollout_id}: FIFO underflow after {refills} refills "
                        f"(queue={self._buffer.size()}, need={n - len(selected)})."
                    )
                self._submit_drive(self._pending_carried, rollout_id)
                self._pending_carried = []
            else:
                time.sleep(self._POLL_INTERVAL_S)
        return selected

    def _next_batch(self, rollout_id: int) -> List[Sample]:
        return self._next_n(self.batch_size, rollout_id)

    def _score_trajectories(self, trajectories: Sequence[Sample], rollout_id: int) -> torch.Tensor:
        # Environment trajectories already carry their return on the final
        # generated Part.  Use it only when every row has one; otherwise run the
        # answer scorer over the complete batch.
        env_values: List[float] = []
        all_env = True
        for trajectory in trajectories:
            gens = trajectory.gen_parts()
            reward = gens[-1].rewards if gens else None
            if reward is None:
                all_env = False
                break
            env_values.append(float(hydrate(reward).float().reshape(-1)[0].item()))
        if all_env:
            return torch.tensor(env_values, dtype=torch.float32)

        roots = [trajectory.parts[0].sample_ids[0] for trajectory in trajectories]
        request = Sample.request(Part.input(roots, metadata=[{"answer": self._gt_by_root.get(root)} for root in roots]))
        rewards, _ = self._rewards_and_groups(request, list(trajectories), rollout_id)
        return rewards.float()

    def _finite_scored_batch(
        self,
        initial: List[Sample],
        *,
        rollout_id: int,
    ) -> Tuple[List[Sample], torch.Tensor]:
        trajectories: List[Sample] = []
        reward_values: List[torch.Tensor] = []
        candidates = initial
        attempts = 0
        while len(trajectories) < self.batch_size:
            rewards = self._score_trajectories(candidates, rollout_id)
            for trajectory, reward in zip(candidates, rewards):
                if bool(torch.isfinite(reward)):
                    trajectories.append(trajectory)
                    reward_values.append(reward.detach().float().cpu())
                else:
                    self._invalid_trajectories += 1
                if len(trajectories) == self.batch_size:
                    break
            if len(trajectories) == self.batch_size:
                break
            attempts += 1
            if attempts > self._MAX_REFILLS:
                raise RuntimeError("SAO could not assemble a finite-reward learner batch")
            candidates = self._next_n(self.batch_size - len(trajectories), rollout_id)
        return trajectories, torch.stack(reward_values)

    def _train_on_trajectories(
        self,
        trajectories: List[Sample],
        *,
        training_progress: float,
        rollout_id: int,
        t0: float,
    ) -> Tuple[TrainStepResult, float]:
        trajectories, rewards = self._finite_scored_batch(trajectories, rollout_id=rollout_id)
        mean_reward = float(rewards.mean().item())
        result = self.stack.train_trajectories(
            trajectories,
            rewards,
            training_progress=float(training_progress),
        )
        versions = [
            int(part.weight_version)
            for trajectory in trajectories
            for part in trajectory.gen_parts()
            if part.weight_version is not None
        ]
        # The logger's legacy sample panel expects a scalar advantage.  Token
        # advantages live train-side, so use zeros here and report GAE metrics
        # from the stack result instead of inventing a trajectory reduction.
        log_sample = self._build_log_sample(
            trajectories,
            rewards,
            torch.zeros_like(rewards),
            rollout_id,
        )
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={
                "agent/mean_turns": sum(len(t.gen_parts()) for t in trajectories) / len(trajectories),
                "agent/max_turns": max(len(t.gen_parts()) for t in trajectories),
                "async/buffer_trajectories": self._buffer.size(),
                "async/weight_version": self._weight_version,
                "async/version_span": (max(versions) - min(versions)) if versions else 0,
                "async/evicted_trajectories": self._buffer.evicted,
                "async/invalid_trajectories": self._invalid_trajectories,
            },
        )
        self._reset_transport_buffers()
        return result, mean_reward

    def _checkpoint_root(self, save_dir: Optional[str], step: int) -> str:
        base = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        return os.path.join(base, f"checkpoint-{step}")

    def _save_sao_checkpoint(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str,
    ) -> None:
        if save_interval <= 0:
            return
        step = rollout_id + 1
        if step % save_interval != 0 and step < num_rollouts:
            return
        if save_mode not in ("full", "auto"):
            raise ValueError("SAO v1 checkpoints both full models; save_mode must be full|auto")
        root = self._checkpoint_root(save_dir, step)
        self.actor_backend.save(os.path.join(root, "actor"), step=step, mode="full")
        self.critic_backend.save(os.path.join(root, "critic"), step=step, mode="full")
        os.makedirs(root, exist_ok=True)
        state = {
            "wandb_run_id": self.wandb_logger.run_id,
            "optimizer_step": self.wandb_logger.optimizer_step,
            "submitted_prompt_batches": self._submitted_prompt_batches,
            "weight_version": self._weight_version,
        }
        with open(os.path.join(root, "trainer_state.json"), "w") as handle:
            json.dump(state, handle)

    def _load_sao_checkpoint(self, load_dir: Optional[str], *, num_rollouts: int) -> int:
        if not load_dir:
            return 0
        root = os.path.abspath(load_dir)
        actor_step = self.actor_backend.load(os.path.join(root, "actor"))
        critic_step = self.critic_backend.load(os.path.join(root, "critic"))
        if isinstance(actor_step, list):
            actor_step = actor_step[0]
        if isinstance(critic_step, list):
            critic_step = critic_step[0]
        if int(actor_step or 0) != int(critic_step or 0):
            raise RuntimeError(f"SAO checkpoint actor/critic step mismatch: {actor_step} != {critic_step}")
        state_path = os.path.join(root, "trainer_state.json")
        checkpoint_weight_version = 0
        if os.path.exists(state_path):
            with open(state_path) as handle:
                self._resume_state = json.load(handle)
            self._submitted_prompt_batches = int(self._resume_state.get("submitted_prompt_batches", 0))
            checkpoint_weight_version = int(self._resume_state.get("weight_version", 0))

        # Rollout engines are reconstructed on resume and therefore start their
        # version counter at zero. Historical Part versions cannot be compared
        # with that fresh epoch; retaining one would make lag eviction discard
        # newly generated trajectories immediately.
        self._weight_version = 0
        logger.warning(
            "SAO resume starts a fresh rollout-version epoch (checkpoint version=%d); "
            "the completed FIFO and in-flight/carried rollout work are intentionally not restored.",
            checkpoint_weight_version,
        )
        start = int(actor_step or 0)
        if start >= int(num_rollouts):
            logger.warning("SAO checkpoint step %d exhausts num_rollouts=%d", start, num_rollouts)
        return start

    def _sync_resumed_rollout(self, *, resumed: bool) -> None:
        """Push restored actor weights and align the fresh rollout version epoch."""
        if not resumed or self.weight_sync is None:
            return
        self.weight_sync.sync()
        # A newly constructed rollout engine starts at version 0; its first full
        # sync advances it to version 1, so keep the driver-side lag clock aligned.
        self._weight_version = 1

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        interval = max(1, int(weight_sync_interval))
        start_rollout = self._load_sao_checkpoint(load_dir, num_rollouts=num_rollouts)
        consumed = self._submitted_prompt_batches
        for _ in range(consumed):
            self.data_source.get_samples(self._oversample)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "algorithm": "sao",
                "samples_per_prompt": 1,
                "max_oldest_version_lag": self._max_oldest_version_lag,
                "oversample_batch_size": self._oversample,
                "train_fraction": self._train_fraction,
                "weight_sync_interval": interval,
            },
        )
        self._buffer = _TrajectoryBuffer()
        self._pending_carried: List[Sample] = []
        self._gen_id = 0
        self._sync_resumed_rollout(resumed=bool(load_dir))
        self._submit_drive([], start_rollout)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                trajectories = self._next_batch(rollout_id)
                progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._train_on_trajectories(
                    trajectories,
                    training_progress=progress,
                    rollout_id=rollout_id,
                    t0=t0,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                need_save = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                need_sync = self.weight_sync is not None and step % interval == 0
                if need_save or need_sync:
                    carried = self.rollout.abort()[0]
                    self._pump()
                    if need_save:
                        self._save_sao_checkpoint(
                            rollout_id,
                            num_rollouts,
                            save_interval=save_interval,
                            save_dir=save_dir,
                            save_mode=save_mode,
                        )
                    if need_sync:
                        self.weight_sync.sync()
                        self._weight_version += 1
                    self._submit_drive(carried, step)
        finally:
            self.rollout.abort()
            self._finish_wandb()


__all__ = ["AsyncSAOTrainer", "_TrajectoryBuffer", "_trajectory_versions"]
