"""UniRL v2 unified-backbone trainside joint trainer (single shared backbone).

One shared bundle/pipeline/backend + TWO :class:`StageAlgorithm` siblings —
``GRPO`` on the ``"ar"`` track + ``FlowGRPO`` on the ``"image"`` track — trained by
:class:`~unirl.train.unified_model_stack.UnifiedModelTrainStack` (one shared LoRA,
ONE optimizer step), fed by a single trainside COMPOSED rollout (e.g. bagel t2ti:
the AR und path plans ``N`` ``<think>`` captions/prompt → diffusion renders ``M``
images/caption). The final-image reward is credit-assigned up the lineage to the
recaption, so a better plan earns higher image reward and the recaption learns.

One ``train_step``::

    rollout.generate(req)          → 2-track RolloutResp {"ar", "image"}
    reward.score_and_attach(image) → score the image track vs the ORIGINAL prompt
    resp.propagate_rewards("mean") → credit-assign image reward up to "ar"
    track.compute_advantages()     → per-track GRPO (ar by prompt, image by caption)
    stack.train_track(ar, image)   → ONE optimizer step over the shared backbone

Combines :class:`~unirl.trainer.unified_model.UnifiedModelTrainer`'s single-backbone
sibling wiring (one ``FSDPBackend`` + two algorithms + ``UnifiedModelTrainStack``)
with :class:`~unirl.trainer.pe.PETrainer`'s trainside composed-rollout front-end.
No vLLM-Omni, no weight sync — the rollout samples the live FSDP modules. Pairs
with ``unirl/train_unified_trainside.py`` + a unified recipe (e.g.
``examples/unified_model/bagel_unified_trainside.yaml``).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import _hydrate_tensor_meta
from unirl.types.sampling import BaseSamplingParams, get_diffusion_params
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names emitted by the composed pipeline (e.g. BagelPipeline._generate_t2ti)
# and consumed by UnifiedModelTrainStack (algorithms dict {"ar", "image"}).
AR_TRACK = "ar"
IMAGE_TRACK = "image"


class UnifiedTrainsideTrainer(BaseTrainer):
    """Single shared backbone + two-algorithm one-step train, trainside composed rollout."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)
        # ComposedSamplingParams(ar=N, diffusion=M) — drives the pipeline's fan-out.
        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)

        # Per-sample latent shape for the driver-authored x_T recipe so the M
        # images of one caption get DISTINCT (reproducible) noise → image-group
        # variance → FlowGRPO gradient. Resolved from the DIFFUSION sub-params
        # (the composed params carry no height/width of their own). ``None`` ⇒
        # engine RNG (DISABLE_DRIVER_XT or a pipeline that opts out).
        self._noise_latent_shape: Optional[List[int]] = (
            None
            if os.environ.get("DISABLE_DRIVER_XT")
            else self._resolve_noise_latent_shape(pipeline_cfg=pipeline_cfg, model_cfg=bundle_cfg)
        )

        # Single shared backbone: bundle → pipeline → backend → two algorithms →
        # one stack, all sibling Remotes (auto-resolve handles). The trainside
        # rollout takes the SAME pipeline so it samples the live FSDP modules.
        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            # Two algorithms over the SAME shared pipeline: ar.stage_attr=ar →
            # pipeline.ar, image.stage_attr=diffusion → pipeline.diffusion.
            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)
            # One stack owns the single backend + both algorithms → one step.
            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )
            # Trainside composed rollout: the pipeline IS the sampler.
            # ``stage_attrs: [ar, diffusion]`` eval-scopes both trainable stages.
            self.rollout = remote(**parse_hydra_cfg(rollout_cfg), pipeline=self.pipeline)

    def _resolve_noise_latent_shape(
        self, *, pipeline_cfg: DictConfig, model_cfg: DictConfig
    ) -> Optional[List[int]]:
        """Per-sample latent shape via the pipeline's ``latent_shape`` classmethod.

        Uses the DIFFUSION sub-params as the sampling spec (the composed params
        have no height/width). Mirrors ``DiffusionTrainer._resolve_noise_latent_shape``;
        ``None`` (or ``NotImplementedError``) ⇒ engines draw their own x_T.
        """
        target = getattr(pipeline_cfg, "_target_", None)
        if not isinstance(target, str):
            return None
        latent_shape_fn = getattr(get_class(target), "latent_shape", None)
        if latent_shape_fn is None:
            return None
        try:
            shape = latent_shape_fn(
                model_config=model_cfg, sampling_spec=get_diffusion_params(self.sampling_params)
            )
        except NotImplementedError:
            return None
        return [int(x) for x in shape]

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """P prompts → typed ``RolloutReq``.

        NO pre-expansion: the composed pipeline fans out ``P → P*N → P*N*M`` from
        ``ComposedSamplingParams`` internally. Stamps the per-rollout SDE window
        onto the diffusion sub-params, ``rollout_id`` into ``stage_config`` (the
        pipeline keys per-image x_T on it), and the driver latent shape (so the
        pipeline's per-image noise recipe regenerates distinct, reproducible x_T).
        """
        diff = get_diffusion_params(self.sampling_params)
        sde_indices = diff.resolve_sde_indices(rollout_id)
        new_diff = dataclasses.replace(diff, sde_indices=sde_indices, scheduler=None)
        sampling_params = dataclasses.replace(self.sampling_params, diffusion=new_diff)
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=sampling_params,
            stage_config={"rollout_id": int(rollout_id)},
            metadata=list(inputs.metadata) if inputs.metadata else [],
            init_noise_latent_shape=self._noise_latent_shape,
        )

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)`` — ``mean_reward`` is the mean
        unnormalized image reward (for the log line). ``sync_weights`` is ignored
        (trainside shares the live FSDP modules — no weight sync).
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        resp = self.rollout.generate(req)
        self.rollout.sleep()

        # 1. Score the IMAGE track vs the ORIGINAL prompt (the recaption is the
        #    means, not the target — we reward final-image↔prompt alignment).
        #    score_and_attach is DP_SCATTER over the P*N*M image track, so expand
        #    the P-prompt req prompt-major to match (mirrors PE / DiffusionTrainer).
        img_track = resp.tracks[IMAGE_TRACK]
        n_track, p = len(img_track.sample_ids), max(1, req.batch_size)
        reward_req = req.repeat_interleave(n_track // p) if n_track > p and n_track % p == 0 else req
        scored = self.reward.score_and_attach(req=reward_req, track=img_track)
        if scored.rewards is not None:
            scored.rewards = _hydrate_tensor_meta(scored.rewards)
        resp.tracks[IMAGE_TRACK] = scored

        # 2. Credit-assign image reward up the lineage → fills "ar" (mean over the
        #    M images of each caption).
        resp = resp.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = resp.tracks[IMAGE_TRACK].rewards
        if di_rewards is not None:
            mean_reward = float(_hydrate_tensor_meta(di_rewards).to(torch.float32).mean().item())

        # 4. Per-track GRPO advantages — "ar" groups by prompt (N captions),
        #    "image" groups by caption (M images).
        for name in (AR_TRACK, IMAGE_TRACK):
            resp.tracks[name] = resp.tracks[name].compute_advantages(normalize=True)

        self._drop_decoded(resp)
        # 5. ONE optimizer step over the shared backbone: both algorithms backward
        #    into the same LoRA, one optimizer step applies both.
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            resp.tracks[AR_TRACK],
            resp.tracks[IMAGE_TRACK],
            training_progress=float(training_progress),
        )
        self._log_rollout(rollout_id, results, resp, step_time_s=time.perf_counter() - t0)
        return results, mean_reward

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``."""
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                results, mean_reward = self.train_step(
                    req, training_progress=training_progress, rollout_id=rollout_id
                )
                ar, im = results[AR_TRACK], results[IMAGE_TRACK]
                logger.info(
                    "rollout %d/%d  reward=%.4f  ar[loss=%.4f gn=%.4f lr=%.2e]  img[loss=%.4f gn=%.4f lr=%.2e]",
                    rollout_id + 1,
                    num_rollouts,
                    mean_reward,
                    ar.loss,
                    ar.grad_norm,
                    ar.lr,
                    im.loss,
                    im.grad_norm,
                    im.lr,
                )
        finally:
            self._finish_wandb()
