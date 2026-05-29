"""diffusionRL v2 PE (Prompt Enhancement) joint trainer.

Two :class:`~diffusionrl.train.stack.TrainStack` siblings — one for the
diffusion model, one for the AR LLM — colocated on the whole pool, sharing the
composed :class:`~diffusionrl.models.pe.pipeline.PEPipeline` as a *trainside*
rollout (the rollout reads the live FSDP modules, so no weight sync).

One ``train_step``::

    rollout.generate(req)           → 2-track RolloutResp {"ar", "diffusion"}
    reward.score_and_attach(image)  → score the "diffusion" (image) track only
    resp.propagate_rewards("mean")  → credit-assign image reward up to "ar"
    track.compute_advantages()      → per-track GRPO (ar by prompt, diff by rewrite)
    {name}.stack.train_track(track) → route each track to its own model

Mirrors :class:`~diffusionrl.trainer.diffusion.DiffusionTrainer` but wires two
of everything and a composed rollout. Deferred (same as the reference trainer):
multi-epoch replay, checkpoint / eval cadence, structured logging.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from diffusionrl.distributed.group.placement import placement, remote
from diffusionrl.models.pe.pipeline import PEPipeline
from diffusionrl.train.stack import TrainStepResult
from diffusionrl.trainer.base import BaseTrainer
from diffusionrl.types.prompts import RolloutInputs
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import _hydrate_tensor_meta
from diffusionrl.types.sampling import BaseSamplingParams
from diffusionrl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names match PEPipeline's output and the per-side attributes on the
# trainer (``self.ar`` / ``self.diffusion``); also the algorithms' stage_attr.
TRACK_NAMES: Tuple[str, ...] = ("ar", "diffusion")


@dataclass
class _Side:
    """The sibling Remotes that make up one trained track."""

    bundle: Any
    pipeline: Any
    backend: Any
    algorithm: Any
    stack: Any


class PETrainer(BaseTrainer):
    """PE joint trainer: two TrainStack siblings + composed trainside rollout."""

    def __init__(
        self,
        *,
        num_devices: int,
        batch_size: int,
        diffusion_cfg: DictConfig,
        ar_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
    ) -> None:
        super().__init__(num_devices=num_devices)
        self.batch_size = batch_size

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        # ComposedSamplingParams(ar=N, diffusion=M) — drives PEPipeline's fan-out.
        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.diffusion = self._wire_side(diffusion_cfg)
            self.ar = self._wire_side(ar_cfg)

            # Composed PE pipeline shares both trained child pipelines in-process,
            # so the rollout samples from the live FSDP modules — no weight sync.
            self.pe_pipeline = remote(
                PEPipeline,
                diffusion_pipeline=self.diffusion.pipeline,
                llm_pipeline=self.ar.pipeline,
            )

            # Mirror DiffusionTrainer's rollout wiring: pass the (composed)
            # pipeline only to engines whose role_cls declares it (trainside).
            # The recipe's rollout block sets ``stage_attrs: [diffusion, ar]``
            # so the trainside engine eval-scopes both trained models.
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                self.rollout = remote(**rollout_parsed, pipeline=self.pe_pipeline)
            else:
                self.rollout = remote(**rollout_parsed)

            self.reward = remote_hydra(reward_cfg)

    def _wire_side(self, cfg: DictConfig) -> _Side:
        """Build one track's bundle → pipeline → backend → algorithm → stack.

        Identical to ``DiffusionTrainer``'s single-side chain; called twice
        (diffusion + ar) inside the shared placement block.
        """
        bundle = remote_hydra(cfg.bundle)
        pipeline = remote_hydra(cfg.pipeline, bundle=bundle)
        backend = remote_hydra(cfg.backend, bundle=bundle)
        algorithm = remote_hydra(cfg.algorithm, pipeline=pipeline)
        stack = remote_hydra(cfg.stack, fsdp_backend=backend, algorithm=algorithm)
        return _Side(bundle=bundle, pipeline=pipeline, backend=backend, algorithm=algorithm, stack=stack)

    def _build_req(self, inputs: RolloutInputs) -> RolloutReq:
        """Turn a data-source batch of ``P`` prompts into a typed ``RolloutReq``.

        No pre-expansion: ``PEPipeline`` fans out ``P → P*N → P*N*M`` internally
        from ``ComposedSamplingParams`` (``ar.samples_per_prompt`` rewrites,
        ``diffusion.samples_per_prompt`` images each). The single-track trainer
        pre-expands here; PE must not, or it would double-count.
        """
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=self.sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)``. ``mean_reward`` is the
        mean unnormalized image reward (for the log line).
        """
        self.rollout.wake_up()
        resp = self.rollout.generate(req)
        self.rollout.sleep()

        # 1. Score the IMAGE track only — the AR track's TextSegment is not
        #    directly scorable; its reward is credit-assigned below.
        scored = self.reward.score_and_attach(req=req, track=resp.tracks["diffusion"])
        # propagate_rewards reshapes child.rewards directly (no hydration), so
        # turn the worker-returned TensorMeta into a real tensor first.
        if scored.rewards is not None:
            scored.rewards = _hydrate_tensor_meta(scored.rewards)
        resp.tracks["diffusion"] = scored

        # 2. Credit-assign image reward up the lineage → fills the "ar" track
        #    (mean over the M images of each rewrite).
        resp = resp.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = resp.tracks["diffusion"].rewards
        if di_rewards is not None:
            mean_reward = float(_hydrate_tensor_meta(di_rewards).to(torch.float32).mean().item())

        # 4. Per-track GRPO advantages — "ar" groups by prompt (N rewrites),
        #    "diffusion" groups by rewrite (M images).
        for name in TRACK_NAMES:
            resp.tracks[name] = resp.tracks[name].compute_advantages(normalize=True)

        # 5. Route each track to its own stack (each DP_ALL-sharded on dispatch).
        results: Dict[str, TrainStepResult] = {
            name: getattr(self, name).stack.train_track(resp.tracks[name], training_progress=float(training_progress))
            for name in TRACK_NAMES
        }
        return results, mean_reward

    def train(self, *, num_rollouts: int) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``."""
        for rollout_id in range(num_rollouts):
            training_progress = rollout_id / max(1, num_rollouts - 1)
            inputs = self.data_source.get_samples(self.batch_size)
            req = self._build_req(inputs)
            results, mean_reward = self.train_step(req, training_progress=training_progress)
            ar, di = results["ar"], results["diffusion"]
            logger.info(
                "rollout %d/%d  reward=%.4f  ar[loss=%.4f gn=%.4f lr=%.2e]  diff[loss=%.4f gn=%.4f lr=%.2e]",
                rollout_id + 1,
                num_rollouts,
                mean_reward,
                ar.loss,
                ar.grad_norm,
                ar.lr,
                di.loss,
                di.grad_norm,
                di.lr,
            )
