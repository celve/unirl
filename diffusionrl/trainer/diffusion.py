import dataclasses
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
        logging_cfg: Optional[DictConfig] = None,
        layout: str = "colocated",
        train_fraction: float = 0.5,
    ) -> None:
        super().__init__(num_devices=num_devices, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._layout = str(layout)
        # Set in _build_train_side: True only for the NFT algorithm, which
        # needs the EMA dual-adapter swap around rollout. Stays False for GRPO
        # so its hot path is untouched.
        self._uses_ema = False

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)

        # Set below from the `sync` block; None trainside (shares the module).
        self.weight_sync = None

        # Construction (_build_train_side / _build_rollout) is shared; only the
        # placement topology and the train→rollout sync wiring differ per layout.
        train_cfgs = dict(
            bundle_cfg=bundle_cfg,
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            reward_cfg=reward_cfg,
            algorithm_cfg=algorithm_cfg,
            stack_cfg=stack_cfg,
        )
        if self._layout == "separate":
            # Two disjoint top-level slabs. A nested placement would carve a
            # sub-slab of the parent (not a disjoint slab), so the train scope
            # must fully exit before the rollout scope opens.
            with placement(self.pool, fraction=train_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                if sync_cfg is not None:
                    # NCCL handler: rollout is cross-slab, wired via the handshake below.
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
            # Rollout slab = the rest. Top-level ``fraction`` is relative to the
            # WHOLE pool (placement.py), so the remainder is ``1 - train_fraction``.
            with placement(self.pool, fraction=1.0 - train_fraction, shared_workers=True):
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=False)
            if self.weight_sync is not None:
                self._connect_separate(sync_cfg)
        else:
            # Single slab: train + rollout are siblings on one Worker.
            with placement(self.pool, fraction=1.0, shared_workers=True):
                self._build_train_side(**train_cfgs)
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=True)
                if sync_cfg is not None:
                    # Colocated handlers (tensor/ipc) take the engine as a local sibling.
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

    def _build_train_side(
        self,
        *,
        bundle_cfg,
        pipeline_cfg,
        backend_cfg,
        reward_cfg,
        algorithm_cfg,
        stack_cfg,
    ) -> None:
        """Build the train-side remotes in the *currently active* placement scope.

        Scope-agnostic: ``remote_hydra`` lands each remote in whatever
        ``placement(...)`` block is open, so both layouts reuse this.
        """
        self.bundle = remote_hydra(bundle_cfg)
        self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
        self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
        self.reward = remote_hydra(reward_cfg)
        # NFT resolves its frozen reference adapter off ``backend.ema`` (the
        # FSDPBackend owns the dual-adapter EMA), so it needs the backend sibling
        # injected alongside ``pipeline``. GRPO takes neither and would reject the
        # extra kwarg, so gate on the algorithm target.
        algo_target = str(algorithm_cfg.get("_target_", ""))
        self._uses_ema = algo_target.endswith("DiffusionNFT")
        algo_extra = {"backend": self.backend} if self._uses_ema else {}
        self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline, **algo_extra)
        self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

    def _build_rollout(self, rollout_cfg, *, allow_pipeline: bool):
        """Build the rollout remote in the currently active placement scope.

        The trainside direct-sampling engine takes ``pipeline`` as a local
        sibling and is only valid colocated (``allow_pipeline=True``); vllm /
        sglang engines take no pipeline and work in either layout.
        """
        rollout_parsed = parse_hydra_cfg(rollout_cfg)
        if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
            if not allow_pipeline:
                raise ValueError(
                    "layout='separate' requires a dedicated-rollout engine "
                    "(vllm/sglang); the trainside direct-sampling engine needs "
                    "the pipeline as a local sibling and cannot live on a "
                    "separate slab."
                )
            return remote(**rollout_parsed, pipeline=self.pipeline)  # direct sampling
        return remote(**rollout_parsed)  # vllm / sglang

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake: hand rank 0 the rollout Worker handles.

        Driver-orchestrated because the rollout slab is cross-slab (not a
        sibling). The LoRA-over-Ray handler (``RemoteLoraWeightSync``) only needs
        the rollout engine's ``(role, workers)`` to push adapters by Ray RPC.
        ``NCCLWeightSync`` additionally rendezvous a broadcast group: ``pick_master``
        on rank 0, hand it the rollout Worker handles, then ``connect`` (rank 0
        fires the rollout joins non-blocking, then joins the group itself).
        """
        if str(sync_cfg.get("_target_", "")).endswith("NCCLWeightSync"):
            addr, port = self.weight_sync.pick_master()[0]
            self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
            self.weight_sync.connect(
                master_addr=addr,
                master_port=port,
                num_rollout_gpus=len(self.rollout.workers),
            )
        else:
            self.weight_sync.set_rollout_targets([(self.rollout.role_name, self.rollout.workers)])

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
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        ``training_progress`` in ``[0, 1]`` drives clip-range / LR schedules
        inside the algorithm. The reference trainer is stateless — the
        outer training loop owns step counting; ``rollout_id`` only keys the
        wandb panels (see :meth:`_log_rollout`).

        ``sync_weights`` pushes the latest LoRA into the engine between
        ``wake_up`` and ``generate`` — one wake/sleep instead of two, with this
        ``generate`` already using the fresh adapter.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the single track (0.0 if none), for the log line.
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        # NFT: sample under the EMA-smoothed ("old") adapter, then restore the
        # trainable ("default") adapter before the loss. No-op for GRPO (gated).
        # Only effective for colocate/trainside where rollout shares the train
        # model; a separate sglang engine samples in its own process (see recipe).
        if self._uses_ema:
            self.backend.apply_eval_ema()
        resp = self.rollout.generate(req)
        if self._uses_ema:
            self.backend.restore_from_eval()
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
                resp.tracks[name] = track.compute_advantages(normalize=True)

        (track,) = resp.tracks.values()
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self._log_rollout(rollout_id, result, resp, step_time_s=time.perf_counter() - t0)
        return result, mean_reward

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        Deferred (out of scope for the first runnable trainer):
        ``num_updates_per_batch`` multi-epoch replay, checkpoint cadence,
        evaluation cadence.
        """
        interval = max(1, weight_sync_interval)
        self._init_wandb(num_rollouts=num_rollouts)
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
