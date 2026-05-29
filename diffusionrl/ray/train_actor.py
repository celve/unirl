"""diffusionrl train actor — multi-track wrap-functions path.

Drives the training stack end-to-end.  Each actor holds N parallel
training "tracks" (one per ``cfg.training.tracks.<name>`` entry): each
track owns its own pipeline + bundle, inject-function mutations (LoRA,
NFT, FSDP), optimizer, scheduler, and :class:`StageAlgorithm`.  The
shared :class:`StageTrainStack` holds ``Dict[str, TrainTrack]`` and
exposes :meth:`StageTrainStack.train_track` for one-track-at-a-time
optimizer steps.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Union

import ray
import torch
from omegaconf import DictConfig

from diffusionrl.distributed.tensor.transport import TensorTransportRuntime
from diffusionrl.ray.actor_config import ConfigActor
from diffusionrl.ray.distributed import DistributedMixin
from diffusionrl.ray.mixins import TrainingWeightSyncMixin
from diffusionrl.ray.mixins.rollout_pipeline import RolloutPipelineMixin
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.rollout.engine import chunked_engine_generate_req
from diffusionrl.training import StageTrainStack, TrackMiniBatchResult
from diffusionrl.training.fsdp_utils import (
    fsdp_offload,
    fsdp_onload,
    gather_state_dict,
    load_model_state_dict,
    trainable_params,
)
from diffusionrl.training.track_builder import (
    _iter_optimizer_param_states,
    _resolve_primary_track,
    build_training_tracks,
)
from diffusionrl.training.train_track import TrainTrack
from diffusionrl.transfer.buffer import Buffer, BufferHandle
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.sampling import get_diffusion_params
from diffusionrl.utils import clear_memory as _clear_gpu_memory

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class TrainActor(
    ConfigActor,
    TrainingWeightSyncMixin,
    DistributedMixin,
    RolloutPipelineMixin,
    Buffer,
):
    """Multi-track train actor.

    Each instance holds :class:`TrainTrack` instances per track in
    ``cfg.training.tracks``.  The per-track lifecycle (``_onload_track``
    -> ``train_track`` -> ``_offload_track``) is sequenced inside
    :meth:`train` so only one track's params + optimizer state are
    GPU-resident at peak.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        world_size: int,
        rank: int,
        master_addr: Optional[str],
        master_port: Optional[int],
        seed: Optional[int] = 42,
    ) -> None:
        from diffusionrl.config.instantiate import build, materialize
        from diffusionrl.config.validation import is_direct_sampling
        from diffusionrl.utils import set_seed

        set_seed(seed)

        super().__init__(
            cfg=cfg,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        self._init_weight_sync_state()

        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)
        self._init_distributed()

        if is_direct_sampling(cfg) and len(cfg.training.tracks) > 1:
            raise ValueError(
                "TrainActor: multi-track configs are incompatible with "
                "direct sampling. Use separate sampling for multi-track."
            )

        # ------------------------------------------------------------------
        # Per-track artefacts.
        # ------------------------------------------------------------------
        artefacts = build_training_tracks(cfg, device=self._device, rank=self.rank)
        self.tracks: Dict[str, TrainTrack] = artefacts.tracks
        self.pipelines: Dict[str, object] = artefacts.pipelines
        self.bundles: Dict[str, object] = artefacts.bundles

        self._primary_track = _resolve_primary_track(cfg)

        self._track_use_lora: Dict[str, bool] = dict(artefacts.use_lora)
        self._use_lora: bool = bool(artefacts.use_lora[self._primary_track])

        self._eval_ema_active: Dict[str, bool] = {name: False for name in self.tracks}

        self.model = self.tracks[self._primary_track].stage.trainable_module()

        self.train_stack = StageTrainStack(
            tracks=self.tracks,
            max_grad_norm=float(cfg.training.execution.max_grad_norm),
        )

        # ------------------------------------------------------------------
        # RolloutPipelineMixin host-contract — direct-sampling only.
        # ------------------------------------------------------------------
        self.engine = None
        self._rollout_plan = None
        self._reward_config = None
        self._reward_pipeline: Optional[RewardPipeline] = None
        self._adv_scope = str(getattr(cfg.algorithm, "adv_normalization_scope", "group"))
        self._adv_use_global_std = bool(getattr(cfg.algorithm, "use_global_std", False))
        self._adv_samples_per_prompt = max(
            1,
            int(get_diffusion_params(cfg.sampling).samples_per_prompt),
        )
        # Keep-local data plane (direct sampling only): run_rollout_pipeline stashes
        # this actor's heavy rollout on ``_kept_rollout`` and returns only a light
        # view to the driver; ``train_local`` pops it and trains in place.
        self._keep_local = bool(cfg.training.execution.get("keep_local", False))
        self._kept_rollout = None
        if is_direct_sampling(cfg):
            only_track = next(iter(cfg.training.tracks))
            self.engine = build(
                cfg.rollout.engine,
                pipeline=self.pipelines[only_track],
                stage=self.tracks[only_track].stage,
            )
            self._rollout_plan = materialize(cfg.rollout.plan)
            self._reward_config = cfg.reward
            logger.info(
                "Rank %s: direct-sampling engine installed (%s) for track %r",
                self.rank,
                type(self.engine).__name__,
                only_track,
            )

        logger.info(
            "Rank %s: TrainActor initialized (tracks=%s, primary=%r, direct_sampling=%s)",
            self.rank,
            sorted(self.tracks.keys()),
            self._primary_track,
            self.engine is not None,
        )

    # ------------------------------------------------------------------
    # Distributed env
    # ------------------------------------------------------------------

    def _setup_distributed_env(self) -> None:
        if self.master_addr is None or self.master_port is None:
            raise ValueError("master_addr and master_port must be set")

        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            local_rank = 0
        else:
            device_count = torch.cuda.device_count()
            local_rank = self.rank % device_count if device_count > 0 else 0

        self._write_distributed_env(
            master_addr=self.master_addr,
            master_port=self.master_port,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=local_rank,
        )

    # ------------------------------------------------------------------
    # RolloutPipelineMixin host-contract methods (direct-sampling only)
    # ------------------------------------------------------------------

    def generate(self, req: RolloutReq) -> RolloutResp:
        if self.engine is None:
            raise RuntimeError("TrainActor.generate: direct-sampling engine not installed.")
        if int(req.batch_size) == 0:
            raise ValueError("TrainActor.generate requires non-empty req.")
        return chunked_engine_generate_req(
            self.engine,
            req,
            chunk_size=self._rollout_plan.forward_batch_size,
        )

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        if self._reward_pipeline is None:
            if self._reward_config is None:
                raise RuntimeError("TrainActor._ensure_reward_pipeline: cfg.reward not captured.")
            self._reward_pipeline = RewardPipeline.from_configs(self._reward_config)
        return self._reward_pipeline

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        rollout_step: int,
        resp_or_handle: Union[ray.ObjectRef, RolloutResp],
    ) -> Dict[str, TrackMiniBatchResult]:
        if isinstance(resp_or_handle, ray.ObjectRef):
            resp: RolloutResp = ray.get(resp_or_handle)
        else:
            resp = resp_or_handle
        backend = TensorTransportRuntime.current()
        if backend is not None:
            backend.hydrate(resp)
        return self._train_resp(rollout_step, resp)

    def train_from_buffer(
        self,
        rollout_step: int,
        handle: BufferHandle,
    ) -> Dict[str, TrackMiniBatchResult]:
        resp: RolloutResp = ray.get(handle.actor_handle.pop_buffer.remote(handle))
        backend = TensorTransportRuntime.current()
        if backend is not None:
            backend.hydrate(resp)
        return self._train_resp(rollout_step, resp)

    def train_local(self, rollout_step: int) -> Dict[str, TrackMiniBatchResult]:
        """Train on this actor's locally-cached rollout (keep-local data plane).

        Direct-sampling keep-local: ``run_rollout_pipeline`` stashed the heavy
        multi-track rollout this actor produced on ``self._kept_rollout`` and
        returned only a light per-track view to the driver. Pop that cache,
        concat the per-group shards into this actor's training resp, and run the
        normal train path — the heavy tracks never round-tripped through the
        driver. Advantages were attached upstream, so the cached resp is
        training-ready. No hydrate step needed: keep_local and tensor
        transport are mutually exclusive.
        """
        kept = self._kept_rollout
        self._kept_rollout = None  # pop: never train the same cache twice
        if not kept:
            raise RuntimeError(
                "TrainActor.train_local: no cached rollout for this actor. "
                "keep_local requires run_rollout_pipeline to have run on this "
                "actor during the same rollout step (direct sampling only)."
            )
        resp = RolloutResp.concat(kept)
        if int(resp.batch_size) == 0:
            raise RuntimeError(
                "TrainActor.train_local: cached rollout is empty; every FSDP rank must train on >=1 sample."
            )
        return self._train_resp(rollout_step, resp)

    def _train_resp(self, rollout_step: int, resp: RolloutResp) -> Dict[str, TrackMiniBatchResult]:
        resp = resp.to_device(self._device)
        num_rollouts = int(self._cfg.run.num_rollouts)
        progress = max(0.0, min(1.0, float(rollout_step) / max(1, num_rollouts)))
        self._prepare_segments(resp)

        num_updates = int(self._cfg.training.plan.get("num_updates_per_batch", 1))
        offload_train = bool(self._cfg.training.execution.get("offload_train", False))

        mini_sizes = {name: track.batch_size // num_updates for name, track in resp.tracks.items()}

        results: Dict[str, TrackMiniBatchResult] = {}
        for track_name in self.tracks:
            self._onload_track(track_name)
            try:
                for i in range(num_updates):
                    mini_resp = RolloutResp(
                        tracks={
                            name: track.slice(
                                i * mini_sizes[name],
                                (i + 1) * mini_sizes[name],
                            )
                            for name, track in resp.tracks.items()
                        }
                    )
                    results[track_name] = self.train_stack.train_track(
                        mini_resp,
                        track_name,
                        training_progress=progress,
                    )
            finally:
                if offload_train:
                    self._offload_track(track_name)

        self.train_stack.on_rollout_end()
        return results

    def _prepare_segments(self, resp: RolloutResp) -> None:
        # A training track may host multiple algorithms (HI3 shared-backbone
        # case: one "image" track with {"image": DiffusionGRPO, "ar": ARGRPO}).
        # Each algorithm consumes its own resp-slot named the same as its
        # algorithm key.
        for track in self.tracks.values():
            for alg_key, alg in track.algorithms.items():
                resp_track = resp.tracks.get(alg_key)
                if resp_track is None or resp_track.segment is None:
                    continue
                alg.prepare_segment(conditions=resp_track.conditions, segment=resp_track.segment)

    # ------------------------------------------------------------------
    # Eval-EMA swap
    # ------------------------------------------------------------------

    def apply_eval_ema(self, track_filter: Optional[List[str]] = None) -> None:
        names = list(self.tracks) if track_filter is None else track_filter
        for name in names:
            track = self.tracks.get(name)
            if track is None or track.ema is None:
                continue
            track.ema.apply_shadow()
            self._eval_ema_active[name] = True

    def restore_from_eval(self, track_filter: Optional[List[str]] = None) -> None:
        names = list(self.tracks) if track_filter is None else track_filter
        for name in names:
            if not self._eval_ema_active.get(name, False):
                continue
            track = self.tracks.get(name)
            if track is not None and track.ema is not None:
                track.ema.restore_shadow()
            self._eval_ema_active[name] = False

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        for name, track in self.tracks.items():
            track_dir = os.path.join(path, name)
            checkpoint_path = os.path.join(track_dir, "checkpoint.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Track {name!r} checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self._device)

            model = track.stage.trainable_module()
            load_model_state_dict(model, checkpoint["policy_state_dict"])
            track.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if track.scheduler is not None and "scheduler_state_dict" in checkpoint:
                track.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def save_model(self, path: str) -> None:
        per_track_state: Dict[str, Dict[str, object]] = {}
        for name, track in self.tracks.items():
            model = track.stage.trainable_module()
            model_sd = gather_state_dict(model)
            per_track_state[name] = {
                "policy_state_dict": model_sd,
                "optimizer_state_dict": track.optimizer.state_dict(),
            }
            if track.scheduler is not None:
                per_track_state[name]["scheduler_state_dict"] = track.scheduler.state_dict()

        if self.rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        for name, state in per_track_state.items():
            track_dir = os.path.join(path, name)
            os.makedirs(track_dir, exist_ok=True)
            torch.save(state, os.path.join(track_dir, "checkpoint.pt"))

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def _onload_track(self, name: str) -> None:
        track = self.tracks[name]
        model = track.stage.trainable_module()
        fsdp_onload(model, self._device)
        for state in _iter_optimizer_param_states(track.optimizer):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

    def _offload_track(self, name: str) -> None:
        track = self.tracks[name]
        model = track.stage.trainable_module()
        fsdp_offload(model)
        for state in _iter_optimizer_param_states(track.optimizer):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        _clear_gpu_memory()

    def offload(self) -> None:
        for name in self.tracks:
            self._offload_track(name)

    def onload(self) -> None:
        for name in self.tracks:
            self._onload_track(name)

    # ------------------------------------------------------------------
    # Smoke-only helpers
    # ------------------------------------------------------------------

    def compute_local_param_checksums(
        self,
        *,
        names: List[str],
        prefix: str = "",
        track_name: Optional[str] = None,
    ) -> Dict[str, str]:
        from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
            fingerprint_tensor,
        )
        from diffusionrl.utils.peft_merge import raw_state_dict

        name = track_name or self._primary_track
        model = self.tracks[name].stage.trainable_module()
        target = set(names)
        out: Dict[str, str] = {}
        for raw_name, param in raw_state_dict(model):
            prefixed = prefix + raw_name
            if prefixed in target:
                out[prefixed] = fingerprint_tensor(param)
        return out

    def randomize_weights_for_smoke(self, seed: int = 0, track_name: Optional[str] = None) -> None:
        name = track_name or self._primary_track
        model = self.tracks[name].stage.trainable_module()
        gen = torch.Generator(device=self._device)
        gen.manual_seed(int(seed) + int(self.rank))
        with torch.no_grad():
            for p in trainable_params(model):
                local = p.data
                from torch.distributed.tensor import DTensor

                if isinstance(local, DTensor):
                    shard = local.to_local()
                    shard.copy_(
                        torch.randn(
                            shard.shape,
                            dtype=shard.dtype,
                            device=shard.device,
                            generator=gen,
                        )
                    )
                else:
                    local.copy_(
                        torch.randn(
                            local.shape,
                            dtype=local.dtype,
                            device=local.device,
                            generator=gen,
                        )
                    )
        logger.info(
            "Rank %s: randomize_weights_for_smoke complete (track=%r, seed=%d)",
            self.rank,
            name,
            seed,
        )


__all__ = ["TrainActor", "build_training_tracks"]
