import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Union

import ray
import torch
from omegaconf import DictConfig

from diffusionrl.config.registration import register_config
from diffusionrl.config.validation import validate_precision_type
from diffusionrl.distributed.transfer_queue import tqbridge
from diffusionrl.patches.replay_logprob import ReplayLogProbPatch
from diffusionrl.ray.actor_config import ConfigActor
from diffusionrl.ray.distributed import DistributedMixin
from diffusionrl.ray.mixins import RolloutPipelineMixin, TrainingWeightSyncMixin
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.samplers.engine import chunked_engine_generate
from diffusionrl.samplers.fsdp.engine import FSDPSamplingEngine
from diffusionrl.training.factories import build_lr_scheduler, build_optimizer
from diffusionrl.training.stack import TrainStack
from diffusionrl.transfer.buffer import Buffer, BufferHandle
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import clear_memory as _clear_gpu_memory
from diffusionrl.utils.dtypes import parse_torch_dtype
from diffusionrl.utils.ema import EMAManager

logger = logging.getLogger(__name__)


@register_config(group="training/execution", name="default")
@dataclass
class TrainingExecutionConfig:
    """Per-step training-execution knobs read by the training actor.

    Read sites:
      - ``training/stack.py::TrainStack`` reads ``max_grad_norm``
      - ``ray/train_actor.py::TrainActor.__init__`` reads
        ``training_autocast_precision``
      - ``train.py`` reads ``offload_train`` / ``offload_rollout`` to gate
        per-rollout sleep/wake of the training + rollout groups.
    """

    max_grad_norm: float
    training_autocast_precision: str = "bf16"
    offload_train: bool = False
    offload_rollout: bool = False

    def __post_init__(self) -> None:
        if self.max_grad_norm <= 0:
            raise ValueError(f"TrainingExecutionConfig.max_grad_norm must be > 0; got {self.max_grad_norm!r}")
        validate_precision_type(
            self.training_autocast_precision, field="TrainingExecutionConfig.training_autocast_precision"
        )


def _iter_optimizer_param_states(optimizer: Any) -> Iterable[Dict[str, Any]]:
    """Yield per-parameter optimizer state dicts across plain and Multi optimizers.

    VeOmni's ``MultiOptimizer`` skips ``super().__init__()`` so it has no
    ``state`` attribute, but it exposes the sub-optimizers on
    ``optimizers_dict``. Iterate the union of their ``state.values()``.
    """
    sub_dict = getattr(optimizer, "optimizers_dict", None)
    if sub_dict is not None:
        for sub_opt in sub_dict.values():
            yield from sub_opt.state.values()
    else:
        yield from optimizer.state.values()


@ray.remote(num_gpus=1)
class TrainActor(ConfigActor, TrainingWeightSyncMixin, DistributedMixin, RolloutPipelineMixin, Buffer):
    def __init__(
        self,
        *,
        cfg: DictConfig,
        world_size: int,
        rank: int,
        master_addr: Optional[str],
        master_port: Optional[int],
        seed: int = 42,
    ):
        """Initialize TrainActor from the composed cfg + topology scalars.

        ``cfg`` is installed into ``actor_config._current`` by ``ConfigActor``
        and read on demand for every configurable section. Registered leaves
        with ``_target_`` (``cfg.model``, ``cfg.algorithm``, ``cfg.training.backend``,
        ``cfg.rollout.engine`` in direct-sampling mode) are materialized via
        ``build()``. Schema-only leaves (``cfg.training.optimizer`` /
        ``.lr_scheduler`` / ``.execution`` / ``.plan``) are materialized at
        the read site via ``OmegaConf.to_object`` / ``materialize``.
        ``cfg.reward`` is kept as a ``DictConfig`` and forwarded into
        ``RewardPipeline.from_configs``, which dispatches each component
        through ``build()``.
        """
        from diffusionrl.config.instantiate import build, materialize
        from diffusionrl.utils import set_seed

        set_seed(int(seed))

        super().__init__(
            cfg=cfg,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        self._init_weight_sync_state()

        training_execution: TrainingExecutionConfig = materialize(cfg.training.execution)
        training_autocast_precision = str(training_execution.training_autocast_precision)

        self._use_lora = bool(cfg.model.get("use_lora", False))
        self._reward_config = cfg.reward

        # Distributed must be up before FSDP wraps the model.
        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)
        self._init_distributed()

        # Model bundle built via build(). device/training_only/skip_device_move
        # already live on ModelBundleConfig, so apply the runtime values to the
        # cfg slice in-place rather than threading them as kwargs (model bundle
        # constructors only accept ``config``). ``ModelBundleConfig`` is
        # registered with ``mutable=True`` so post-compose writes succeed without
        # toggling readonly. FSDP2 + cpu_offload=False wants params on GPU
        # before fully_shard wraps them; cpu_offload=True means the bundle
        # skips the device move and lets FSDP2 manage placement.
        cfg.model.device = str(self._device)
        cfg.model.training_only = True
        cfg.model.skip_device_move = bool(cfg.training.backend.get("cpu_offload", False))
        self.model_bundle = build(cfg.model)
        # Backend takes the model bundle + topology as runtime deps.
        self.backend = build(
            cfg.training.backend,
            model_bundle=self.model_bundle,
            topology=materialize(cfg.training.topology),
        )
        self.model = self.backend.model

        self.algorithm = build(cfg.algorithm)
        # Inject the SDE strategy chosen at cfg.sampling.sde_strategy so the
        # algorithm can use it for log-prob replay during training.
        self.algorithm._strategy = build(cfg.sampling.sde_strategy)
        self.model_bundle.set_training_forward_autocast_dtype(
            parse_torch_dtype(
                training_autocast_precision,
                field_name="training_autocast_precision",
            )
        )

        # Wire train-inference consistency debug dump dir through cfg.debug.save_dir.
        # ``GRPOAlgorithm.compute_loss`` reads ``self._debug_output_dir`` for its
        # per-step tensor dump.
        debug_save_dir = cfg.debug.get("save_dir") if cfg.get("debug") else None
        if debug_save_dir:
            self.algorithm._debug_output_dir = str(debug_save_dir)

        self.ema_manager = EMAManager.from_model_and_config(
            model=self.model,
            config=self.algorithm.get_ema_spec(),
            use_lora=self._use_lora,
            uses_sharded_model=True,
            algorithm=self.algorithm,
        )
        from diffusionrl.config.validation import is_direct_sampling

        self.engine: Optional[FSDPSamplingEngine] = None
        self._rollout_plan = materialize(cfg.rollout.plan)
        if is_direct_sampling(cfg):
            self.engine = build(cfg.rollout.engine)
            self.engine.strategy = build(cfg.sampling.sde_strategy)
            self.engine.initialize(self._device)
            self.engine.bind_model(
                model=self.model,
                model_bundle=self.model_bundle,
                strategy=self.engine.strategy,
            )

        optimizer_config = materialize(cfg.training.optimizer)
        scheduler_config = materialize(cfg.training.lr_scheduler)
        self.optimizer = build_optimizer(
            optimizer_config,
            params=self.model.parameters(),
            backend=self.backend,
            actor=self,
        )
        self.lr_scheduler = build_lr_scheduler(
            scheduler_config,
            optimizer=self.optimizer,
            backend=self.backend,
            actor=self,
        )
        self.train_stack = TrainStack(
            backend=self.backend,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            algorithm=self.algorithm,
            ema_manager=self.ema_manager,
            cfg=cfg,
        )
        self.reward_pipeline: Optional[RewardPipeline] = None

        # Replay-logprob patch — required when sglang rollout is configured
        # with ``--sampling.logprob-source replay`` (sglang skips native
        # log_prob output and training must recompute log_prob_old via FSDP
        # forward). Short-circuits internally when ``batch.log_probs`` is
        # already populated.
        self._sampling_config = materialize(cfg.sampling)
        self._replay_logprob_patch = ReplayLogProbPatch()
        self.text_encoder = None
        self.vae = None
        self.scheduler = None

    def _setup_distributed_env(self) -> None:
        """Write env vars for the cross-actor training process group."""
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
        logger.info(
            f"Distributed env setup: rank={self.rank}, world_size={self.world_size}, "
            f"master={self.master_addr}:{self.master_port}"
        )

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        if self._reward_config is None:
            raise RuntimeError("Reward pipeline requested before reward config was extracted.")
        if self.reward_pipeline is None:
            self.reward_pipeline = RewardPipeline.from_configs(self._reward_config)
        return self.reward_pipeline

    def generate(self, request: RolloutRequest) -> RolloutResponse:
        if self.engine is None:
            raise RuntimeError("TrainActor.generate() requires sampling_config to be set at construction.")
        return RolloutResponse(
            request=request,
            samples=chunked_engine_generate(
                self.engine,
                request,
                chunk_size=self._rollout_plan.forward_batch_size,
            ),
        )

    def apply_eval_ema(self) -> None:
        """Swap eval-EMA weights into the model for evaluation."""
        self.ema_manager.apply_eval_ema(self.model)

    def restore_from_eval(self) -> None:
        """Restore training weights after evaluation."""
        self.ema_manager.restore_from_eval(self.model)

    def train(
        self,
        rollout_step: int,
        training_data_handle_or_batch: Union[ray.ObjectRef, TrainingBatch],
    ) -> Dict[str, Any]:
        """Execute one training step."""
        if isinstance(training_data_handle_or_batch, ray.ObjectRef):
            batch: TrainingBatch = ray.get(training_data_handle_or_batch)
        else:
            batch = training_data_handle_or_batch
        return self._train_batch(rollout_step, batch)

    def train_from_buffer(
        self,
        rollout_step: int,
        handle: BufferHandle,
    ) -> Dict[str, Any]:
        """Fetch a TrainingBatch from a remote buffer and train."""
        batch: TrainingBatch = ray.get(handle.actor_handle.pop_buffer.remote(handle))
        return self._train_batch(rollout_step, batch)

    def _maybe_replay_old_log_probs(self, batch: TrainingBatch) -> TrainingBatch:
        """Recompute log_prob_old via FSDP when sglang rollout omitted them.

        Always enables the patch; it short-circuits internally when
        ``batch.log_probs`` is already populated (e.g. the native sglang
        path, or direct_sampling). Only does work when log_probs are missing
        — exactly the sglang ``--sampling.logprob-source replay`` case.

        We can't gate on ``sampling_config.logprob_source`` here because the
        resolved ``SamplingParams`` doesn't carry that field; it lives only
        on the upstream input-config layer. Auto-fill semantics (always-on,
        no-op when populated) are equivalent and avoid the plumbing.
        """
        if self._sampling_config is None:
            return batch
        algorithm_type = getattr(
            self.algorithm,
            "algorithm_type",
            self.algorithm.__class__.__name__.lower().replace("algorithm", ""),
        )
        return self._replay_logprob_patch.maybe_replay_old_log_probs(
            batch=batch,
            enabled=True,
            algorithm_type=algorithm_type,
            sampling_config=self._sampling_config,
            strategy=self.algorithm._strategy,
            model_bundle=self.model_bundle,
            model=self.model,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
        )

    @tqbridge(get=True, put=False)
    def _train_batch(self, rollout_step: int, batch: TrainingBatch) -> Dict[str, Any]:
        """Execute the training stack on a materialized batch."""
        # Batches arrive over Ray (or directly from CPU) and may have any of
        # their tensor fields on the wrong device. Move the whole batch onto
        # this actor's compute device before forward.
        batch = batch.to_device(self._device)
        # Populate batch.log_probs from FSDP forward when configured for replay
        # mode and sglang did not return native log_probs. Must run after the
        # batch is on-device so the FSDP forward in the patch sees the right
        # tensors. No-op for the default native / direct_sampling paths.
        batch = self._maybe_replay_old_log_probs(batch)
        return self.train_stack.train_batch(
            batch=batch,
            rollout_step=rollout_step,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self.backend.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.ema_manager is not None and "ema_state_dict" in checkpoint:
            self.ema_manager.load_state_dict(
                checkpoint["ema_state_dict"],
                algorithm=self.algorithm,
            )

    def save_model(self, path: str) -> None:
        """Save model checkpoint. Must be called collectively across ranks.

        Mirrors ``load_checkpoint`` in reverse: writes ``path/checkpoint.pt``
        with model/optimizer/scheduler/ema state dicts. ``backend.get_state_dict``
        is a distributed gather, so every rank must call this method even
        though only rank 0 writes the file.
        """
        state_dict = self.backend.get_state_dict()
        if self.rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self.ema_manager is not None:
            checkpoint["ema_state_dict"] = self.ema_manager.state_dict()
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))

    def offload(self) -> None:
        if self.reward_pipeline is not None:
            self.reward_pipeline.offload()
        self.backend.offload()
        for state in _iter_optimizer_param_states(self.optimizer):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        _clear_gpu_memory()

    def onload(self) -> None:
        if self.reward_pipeline is not None:
            self.reward_pipeline.onload()
        self.backend.onload()
        for state in _iter_optimizer_param_states(self.optimizer):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)
