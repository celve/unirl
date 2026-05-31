import inspect
import logging
import time
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


class VLMTrainer(BaseTrainer):
    """Autoregressive (VLM / LLM) RL trainer: rollout + train colocated.

    Sibling of :class:`~diffusionrl.trainer.diffusion.DiffusionTrainer` for the
    AR path. Structurally identical except ``_build_req`` carries **no SDE step
    scheduling** — that is diffusion-only (``DiffusionSamplingParams`` owns
    ``scheduler`` / ``sde_indices`` / ``resolve_sde_indices``), and
    ``ARSamplingParams`` has none of it. Keeping the AR trainer separate means
    the VLM path never touches diffusion code (no ``hasattr`` guard, no
    ``dataclasses.replace`` of SDE fields).

    Trainside colocate (the qwen_vl recipe): the training pipeline IS the
    sampler, so ``sync_cfg`` is absent and ``weight_sync`` stays ``None``.
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
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
    ) -> None:
        super().__init__(num_devices=num_devices, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # "group" (textbook GRPO, default) or "global" (v1 baseline parity).
        self.adv_normalization_scope = adv_normalization_scope

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
        prompt produces an N-sample GRPO group (sibling samples consecutive).
        AR sampling params ride to the engine untouched — there is no SDE step
        schedule to resolve (that is the diffusion trainer's job).
        """
        inputs = inputs.expand(self.sampling_params.samples_per_prompt)
        req = RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=self.sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )
        return req

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the single track (0.0 if none), for the log line.
        ``rollout_id`` only keys the wandb panels (see :meth:`_log_rollout`).
        """
        t0 = time.perf_counter()
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
            # Hydrate in place so the wandb reward/advantage stats reuse this
            # fetch instead of re-pulling the TensorMeta from the worker.
            track.rewards = _hydrate_tensor_meta(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
            break  # single-track for now; revisit if multi-track lands

        for name, track in list(resp.tracks.items()):
            if track.rewards is not None:
                resp.tracks[name] = track.compute_advantages(normalize=True, scope=self.adv_normalization_scope)

        (track,) = resp.tracks.values()
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self._log_rollout(rollout_id, result, resp, step_time_s=time.perf_counter() - t0)
        return result, mean_reward

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        Deferred: ``num_updates_per_batch`` multi-epoch replay, checkpoint /
        eval cadence.
        """
        interval = max(1, weight_sync_interval)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope},
        )
        try:
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
                    rollout_id=rollout_id,
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
        finally:
            self._finish_wandb()
