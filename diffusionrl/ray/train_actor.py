import os
from dataclasses import replace
from typing import Any, Dict, Iterable, Optional, Union

import ray
import torch

from diffusionrl.algorithms import create_algorithm_from_init_payload
from diffusionrl.config.training_sections import LrSchedulerConfig, OptimizerConfig
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models import create_model_bundle_from_init_payload
from diffusionrl.patches.replay_logprob import ReplayLogProbPatch
from diffusionrl.ray.actor_base import BaseTrainRayActor
from diffusionrl.ray.mixins import RolloutPipelineMixin, TrainingWeightSyncMixin
from diffusionrl.reward.config import RewardSpec
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.samplers.fsdp.engine import FSDPSamplingEngine
from diffusionrl.training import build_lr_scheduler, build_optimizer
from diffusionrl.training.backends import (
    FSDPBackend,
    FSDPBackendConfig,
    TrainBackend,
    TrainBackendConfig,
    VeOmniBackend,
    VeOmniBackendConfig,
)
from diffusionrl.training.stack import TrainStack
from diffusionrl.transfer.buffer import Buffer, BufferHandle
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import clear_memory as _clear_gpu_memory
from diffusionrl.utils.dtypes import parse_torch_dtype
from diffusionrl.utils.ema import EMAManager


def _build_backend_from_config(config: TrainBackendConfig, model_bundle: Any) -> TrainBackend:
    if isinstance(config, FSDPBackendConfig):
        return FSDPBackend(config=config, model_bundle=model_bundle)
    if isinstance(config, VeOmniBackendConfig):
        return VeOmniBackend(config=config, model_bundle=model_bundle)
    raise TypeError(
        "TrainActor received unsupported backend config type: "
        f"{type(config).__name__}. Expected FSDPBackendConfig or VeOmniBackendConfig."
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
class TrainActor(TrainingWeightSyncMixin, BaseTrainRayActor, RolloutPipelineMixin, Buffer):
    def __init__(
        self,
        world_size: int,
        rank: int,
        master_addr: Optional[str],
        master_port: Optional[int],
        mini_batch_size: int,
        micro_batch_size: int,
        max_grad_norm: float,
        backend_config: TrainBackendConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: LrSchedulerConfig,
        reward_spec: RewardSpec,
        algorithm_init_payload: ComponentInitPayload,
        model_init_payload: ComponentInitPayload,
        training_autocast_precision: str = "bf16",
        sampling_config: Any = None,
        seed: int = 42,
    ):
        # Per-actor determinism setup: must run BEFORE any CUDA op so that
        # cuDNN / cuBLAS / deterministic-algorithm flags are in effect for
        # the subsequent model construction, FSDP wrap, and training ops.
        from diffusionrl.utils import set_seed

        set_seed(int(seed))

        BaseTrainRayActor.__init__(self, world_size, rank, master_addr, master_port)
        Buffer.__init__(self)
        self._init_weight_sync_state()
        self._use_lora = bool(model_init_payload.component_config.use_lora)

        self.mini_batch_size = mini_batch_size
        self.micro_batch_size = micro_batch_size
        self._reward_spec = reward_spec

        # Distributed must be up before FSDP wraps the model.
        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)
        self._init_distributed()

        # Override the model bundle config so the model is created on this
        # actor's GPU. With FSDP2 + cpu_offload=False we want params on the
        # GPU before fully_shard wraps them; if cpu_offload is on, the model
        # bundle should skip the device move and let FSDP2 manage placement.
        skip_device_move = bool(getattr(backend_config, "cpu_offload", False))
        runtime_model_config = replace(
            model_init_payload.component_config,
            device=self._device,
            training_only=True,
            skip_device_move=skip_device_move,
        )
        runtime_model_init_payload = replace(
            model_init_payload,
            component_config=runtime_model_config,
        )
        model_bundle = create_model_bundle_from_init_payload(runtime_model_init_payload)
        self.model_bundle = model_bundle
        self.backend: TrainBackend = _build_backend_from_config(backend_config, model_bundle)
        self.model = self.backend.model

        self.algorithm = create_algorithm_from_init_payload(algorithm_init_payload)
        forward_plugin = model_bundle.forward_plugin()
        self.algorithm._forward_plugin = forward_plugin
        if hasattr(forward_plugin, "autocast_dtype"):
            forward_plugin.autocast_dtype = parse_torch_dtype(
                training_autocast_precision,
                field_name="training_autocast_precision",
            )

        self.ema_manager = EMAManager.from_model_and_spec(
            model=self.model,
            spec=self.algorithm.get_ema_spec(),
            use_lora=self._use_lora,
            uses_sharded_model=True,
            algorithm=self.algorithm,
        )
        self.engine: Optional[FSDPSamplingEngine] = None
        if sampling_config is not None:
            self.engine = FSDPSamplingEngine(sampling_config)
            self.engine.initialize(self._device)
            self.engine.bind_model(model=self.model, model_bundle=self.model_bundle)

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
            max_grad_norm=max_grad_norm,
        )
        self.reward_pipeline: Optional[RewardPipeline] = None

        # Replay-logprob patch — required when sglang rollout is configured with
        # ``--sampling.logprob-source replay`` (sglang skips native log_prob
        # output and training must recompute log_prob_old via FSDP forward).
        # Without it batch.log_probs stays None for every rollout, GRPO loss
        # returns 0.0 with no gradient, and the policy never updates (the
        # symptom observed in run ``kwqghmlv``: reward flat at baseline 0.755
        # for 70+ rollouts).
        self._sampling_config = sampling_config
        self._replay_logprob_patch = ReplayLogProbPatch()
        self.text_encoder = None
        self.vae = None
        self.scheduler = None

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        if self._reward_spec is None:
            raise RuntimeError("Reward pipeline requested before reward spec initialization.")
        if self.reward_pipeline is None:
            self.reward_pipeline = RewardPipeline.from_spec(self._reward_spec)
        return self.reward_pipeline

    def generate(self, request: RolloutRequest) -> RolloutResponse:
        if self.engine is None:
            raise RuntimeError("TrainActor.generate() requires sampling_config to be set at construction.")
        output = self.engine.generate(request)
        return RolloutResponse(request=request, samples=output)

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
            model_bundle=self.model_bundle,
            model=self.model,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
        )

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
            mini_batch_size=self.mini_batch_size,
            micro_batch_size=self.micro_batch_size,
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
