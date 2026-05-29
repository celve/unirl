import dataclasses
import inspect
import logging
from typing import Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from diffusionrl.distributed.group.placement import placement, remote
from diffusionrl.train.stack import TrainStepResult
from diffusionrl.trainer.base import BaseTrainer
from diffusionrl.types.prompts import RolloutInputs
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import _hydrate_tensor_meta
from diffusionrl.types.sampling import BaseSamplingParams
from diffusionrl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class DiffusionTrainer(BaseTrainer):
    """Reference trainer: train + rollout colocated on the whole pool.

    For separate slabs, open two sibling ``placement`` blocks with
    ``fraction<1.0``. For real-colocate (distinct worker processes on the
    same GPU), nest a ``placement(..., shared_workers=False)`` inside.
    """

    def __init__(
        self,
        *,
        num_devices: int,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(num_devices=num_devices)
        self.batch_size = batch_size

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)

        # Set below from the `sync` block; None trainside (shares the module).
        self.weight_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)

            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)  # for direct sampling
            else:
                self.rollout = remote(**rollout_parsed)  # for vllm / sglang

            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data source batch into a typed :class:`RolloutReq`.

        Expands ``inputs`` by ``sampling_params.samples_per_prompt`` so each
        prompt produces an N-sample GRPO group (sibling samples consecutive,
        sample IDs ``prompt:<gid>:sample:<j>``).

        ``rollout_id`` keys the SDE step scheduler (``get_sde_indices``): the
        resolved indices are stamped onto a per-request copy of the sampling
        params, and the schedule config itself is nulled so only the resolved
        ``sde_indices`` ride to the engine.
        """
        inputs = inputs.expand(self.sampling_params.samples_per_prompt)
        sde_indices = self.sampling_params.resolve_sde_indices(rollout_id)
        sampling_params = dataclasses.replace(self.sampling_params, sde_indices=sde_indices, scheduler=None)
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        ``training_progress`` in ``[0, 1]`` drives clip-range / LR schedules
        inside the algorithm. The reference trainer is stateless — the
        outer training loop owns step counting.

        ``sync_weights`` pushes the latest LoRA into the engine between
        ``wake_up`` and ``generate`` — one wake/sleep instead of two, with this
        ``generate`` already using the fresh adapter.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the single track (0.0 if none), for the log line.
        """
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        resp = self.rollout.generate(req)
        self.rollout.sleep()

        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)

        mean_reward = 0.0
        for track in resp.tracks.values():
            if track.rewards is None:
                continue
            rewards_local = _hydrate_tensor_meta(track.rewards)
            mean_reward = float(rewards_local.to(torch.float32).mean().item())
            break  # single-track for now; revisit if multi-track lands

        for name, track in list(resp.tracks.items()):
            if track.rewards is not None:
                resp.tracks[name] = track.compute_advantages(normalize=True)

        (track,) = resp.tracks.values()
        result = self.stack.train_track(track, training_progress=float(training_progress))
        return result, mean_reward

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        Deferred (out of scope for the first runnable trainer):
        ``num_updates_per_batch`` multi-epoch replay, checkpoint cadence,
        evaluation cadence, structured logging.
        """
        interval = max(1, weight_sync_interval)
        for rollout_id in range(num_rollouts):
            training_progress = rollout_id / max(1, num_rollouts - 1)
            inputs = self.data_source.get_samples(self.batch_size)
            req = self._build_req(inputs, rollout_id)
            # Sync before generate; skip step 0 (nothing trained yet).
            sync_weights = rollout_id > 0 and rollout_id % interval == 0
            result, mean_reward = self.train_step(
                req,
                training_progress=training_progress,
                sync_weights=sync_weights,
            )
            logger.info(
                "rollout %d/%d  reward=%.4f  loss=%.4f  grad_norm=%.4f  lr=%.2e",
                rollout_id + 1,
                num_rollouts,
                mean_reward,
                result.loss,
                result.grad_norm,
                result.lr,
            )
