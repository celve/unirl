"""
diffusionrl Training Actor - Manages model training via pluggable train backends.
"""
import inspect
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import ray
import torch
import torch.distributed as dist

from diffusionrl.types.sampling import RolloutRequest, RolloutOutput
from diffusionrl.types.training_batch import (
    BackwardTrainingBatch,
    TrainingBatch,
)
import torch.nn as nn

from diffusionrl.patches.replay_logprob import ReplayLogProbPatch
from diffusionrl.ray.utils.actor_sampling import TrainingActorSamplingService
from diffusionrl.utils import clear_memory as _clear_gpu_memory
from diffusionrl.utils.weight_sync_checkpoint import (
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
)
from diffusionrl.runtime.training import (
    TrainExecutor,
    TrainExecutorConfig,
    create_train_backend,
    resolve_grad_accum,
)
from diffusionrl.utils import load_function

from .base import BaseTrainRayActor, log_gpu_state, log_resource_ids

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class TrainingActor(BaseTrainRayActor):
    """
    Training Actor - Manages model training with backend-driven runtime.

    This actor handles:
    - Model loading and backend wrapping
    - Optimizer and scheduler management
    - Training step execution
    - Checkpoint saving and loading
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
        """
        Initialize training actor.

        Args:
            world_size: Total number of training actors
            rank: This actor's rank
            master_addr: Master node address
            master_port: Master node port
        """
        super().__init__(world_size, rank, master_addr, master_port)

        self.model = None
        self.model_bundle = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = None
        self._is_initialized = False
        self._is_offloaded = False
        self._device = None
        self._use_lora = False
        self._fsdp_cpu_offload = False
        self._train_backend = None
        self._train_backend_name = "fsdp"
        self._train_backend_capabilities: Dict[str, Any] = {}

        # NFT-specific: EMA updater for dual adapter mechanism
        self._ema_updater = None
        self._use_ema = False
        self._ema_decay = 0.001
        self._loss_type = "grpo"
        self._loss_path = None
        self._guidance_scale = 3.5  # Default CFG scale for training
        # NFT timestep handling
        self._nft_timestep_mode = "random"  # random or all
        self._nft_shuffle_timesteps = True
        self._nft_apply_shift = False

        # Training config (read from config in init)
        self._max_grad_norm = 1.0
        self._gradient_accumulation_steps = 1
        self._num_inner_epochs = 1
        self._replay_log_probs = False

        # Sampling support (training-actor sampling mode)
        self._sampling_config: Dict[str, Any] = {}
        self._sampler = None
        self._sampling_service = TrainingActorSamplingService()
        self._actor_sampling_executor = self._sampling_service.executor
        self._replay_logprob_patch = ReplayLogProbPatch()
        self._sampling_ready = False
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self._weights_update_groups: Dict[str, Any] = {}

    def _log_resource_ids(self, tag: str) -> None:
        log_resource_ids(tag, self.rank)

    def _log_gpu_state(self, tag: str) -> None:
        log_gpu_state(tag, self.rank, device=self._device, offloaded=self._is_offloaded)

    @staticmethod
    def _candidate_sglang_python_paths() -> List[Path]:
        candidates: List[Path] = []
        env_path = os.getenv("SGLANG_PYTHON_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root.parent / "sglang" / "python")
        cwd = Path.cwd()
        candidates.append(cwd / "sglang" / "python")
        candidates.append(cwd.parent / "sglang" / "python")

        dedup: List[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(candidate)
        return dedup

    @classmethod
    def _ensure_sglang_importable(cls) -> None:
        try:
            import sglang.srt.utils  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        for candidate in cls._candidate_sglang_python_paths():
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                import sglang.srt.utils  # noqa: F401
                return
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "Cannot import sglang.srt.utils for IPC/NCCL tensor sync. "
            "Set SGLANG_PYTHON_PATH to sglang/python."
        )

    def init(self, config: dict) -> None:
        """
        Initialize training environment.

        Args:
            config: Configuration dictionary containing:
                - model_config: dict with model_path, pretrained_model_saved_path
                - optimizer_config: dict with lr, betas, weight_decay
                - scheduler_config: dict with scheduler type and params
                - loss_config: dict with loss_type, clip_range, kl_coef, etc.
                - training_config: dict with max_grad_norm, gradient_accumulation_steps
                - train_backend/train_backend_path/train_backend_kwargs

        Note:
            Algorithm instantiation for advantage computation happens in
            RolloutManager, not here. Training uses loss_fn directly.
        """
        logger.info(f"Rank {self.rank}: Initializing training actor...")

        backend_config = config.get("train_backend_config", {}) or {}
        if not isinstance(backend_config, dict):
            backend_config = {}

        backend_name = str(
            config.get("train_backend", backend_config.get("name", "fsdp")) or "fsdp"
        ).strip().lower()
        backend_path = config.get("train_backend_path", backend_config.get("backend_path"))
        backend_kwargs = config.get("train_backend_kwargs", backend_config.get("kwargs", {})) or {}
        if not isinstance(backend_kwargs, dict):
            logger.warning("train_backend_kwargs is not a dict; resetting to empty dict.")
            backend_kwargs = {}

        self._train_backend = create_train_backend(
            backend_name,
            backend_path=backend_path,
            backend_kwargs=backend_kwargs,
        )
        self._train_backend_name = self._train_backend.name
        self._train_backend_capabilities = self._train_backend.capabilities.as_dict()

        # Initialize distributed
        self._init_distributed(backend=self._train_backend.capabilities.distributed_backend)

        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)

        # Backend-specific pre-load hook (e.g. CPU-offload behavior).
        self._train_backend.before_model_load(self)

        # Load model (don't move to GPU if CPU offload is enabled)
        self._load_model(config.get("model_config", {}))

        # Backend model wrapping (FSDP/Megatron/others).
        self._train_backend.wrap_model(self)

        # Create optimizer
        self._create_optimizer(config.get("optimizer_config", {}))

        # Create scheduler
        self._create_scheduler(config.get("scheduler_config", {}))

        # Load loss function
        self._load_loss(config.get("loss_config", {}))

        # Load training config
        training_config = config.get("training_config", {})
        self._max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self._gradient_accumulation_steps = resolve_grad_accum(training_config)
        self._num_inner_epochs = max(1, int(training_config.get("num_inner_epochs", 1)))
        self._replay_log_probs = bool(training_config.get("replay_log_probs", False))

        # Sampling config (used when training actors perform sampling)
        sampling_config = config.get("sampling_config", {}) or {}
        if not isinstance(sampling_config, dict):
            logger.warning("sampling_config is not a dict; resetting to empty dict.")
            sampling_config = {}
        self._sampling_config = sampling_config

        # Note: Algorithm is not instantiated here because:
        # - Advantage computation happens in RolloutManager (sampling phase)
        # - Training only uses loss_fn for gradient computation
        # This is consistent with the functional dispatch pattern used in slime.

        self._is_initialized = True
        logger.info(
            f"Rank {self.rank}: Training actor initialized "
            f"(backend={self._train_backend_name}, "
            f"max_grad_norm={self._max_grad_norm}, "
            f"gradient_accumulation_steps={self._gradient_accumulation_steps}, "
            f"num_inner_epochs={self._num_inner_epochs})"
        )
        self._log_resource_ids("training_init")
        self._log_gpu_state("training_init")

    def _load_model(self, model_config: dict) -> None:
        """Load the model for training."""
        if "model_path" not in model_config:
            raise ValueError("model_config must contain model_path")

        model_cls = load_function(model_config["model_path"])

        # Build kwargs for model constructor
        # Pass through LoRA configuration for models that support it
        model_kwargs = {
            "pretrained_path": model_config.get("pretrained_model_saved_path", ""),
            "device": self._device,
            # Training only needs transformer, skip VAE/text_encoders to save memory
            "training_only": True,
            # Skip device move if using FSDP CPU offload (FSDP manages device placement)
            "skip_device_move": getattr(self, '_fsdp_cpu_offload', False),
        }

        # Pass LoRA parameters only when explicitly enabled
        use_lora = bool(model_config.get("use_lora", False))
        self._use_lora = use_lora
        model_kwargs["use_lora"] = use_lora
        if use_lora:
            if "lora_rank" in model_config:
                model_kwargs["lora_rank"] = model_config["lora_rank"]
            if "lora_alpha" in model_config:
                model_kwargs["lora_alpha"] = model_config["lora_alpha"]
            if "lora_target_modules" in model_config:
                model_kwargs["lora_target_modules"] = model_config["lora_target_modules"]

        self.model_bundle = model_cls(**model_kwargs)

        # Get the transformer for training
        # Note: Device placement is handled by model bundle based on skip_device_move flag
        self.model = self.model_bundle.transformer
        self.model.train()

        # Enable gradient checkpointing (activation checkpointing) only when explicitly requested
        # Note: For LoRA models, this should be done in the model bundle before PEFT wrapping
        use_gradient_checkpointing = bool(model_config.get("use_gradient_checkpointing", False))
        logger.info(
            "Rank %s: Gradient checkpointing %s (scope=transformer)",
            self.rank,
            "enabled" if use_gradient_checkpointing else "disabled",
        )
        if use_gradient_checkpointing:
            gc_enabled = False
            # Create a custom checkpoint function with use_reentrant=False
            # This is required for CFG (batched forward) to work correctly with gradient checkpointing
            def non_reentrant_checkpoint(func, *args, use_reentrant=False, **kwargs):
                return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)

            # Try direct method first
            if hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing(gradient_checkpointing_func=non_reentrant_checkpoint)
                gc_enabled = True
            # For PeftModel, try to enable on base model
            elif hasattr(self.model, "get_base_model"):
                base_model = self.model.get_base_model()
                if hasattr(base_model, "enable_gradient_checkpointing"):
                    base_model.enable_gradient_checkpointing(gradient_checkpointing_func=non_reentrant_checkpoint)
                    gc_enabled = True
            if gc_enabled:
                logger.info(f"Rank {self.rank}: Gradient checkpointing enabled (non-reentrant)")

        logger.info(f"Rank {self.rank}: Model loaded (lora_rank={model_config.get('lora_rank', 'N/A')}, training_only=True)")

    def _create_optimizer(self, optimizer_config: dict) -> None:
        """Create optimizer."""
        if self._train_backend is not None:
            backend_optimizer = self._train_backend.build_optimizer(self, optimizer_config)
            if backend_optimizer is not None:
                self.optimizer = backend_optimizer
                logger.info(
                    "Rank %s: Optimizer created by backend=%s",
                    self.rank,
                    self._train_backend_name,
                )
                return

        lr = optimizer_config.get("learning_rate", 1e-6)
        betas = (
            optimizer_config.get("adam_beta1", 0.9),
            optimizer_config.get("adam_beta2", 0.999),
        )
        eps = optimizer_config.get("adam_epsilon", 1e-8)
        weight_decay = optimizer_config.get("weight_decay", 0.0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        logger.info(f"Rank {self.rank}: Optimizer created (lr={lr})")

    def _create_scheduler(self, scheduler_config: dict) -> None:
        """Create learning rate scheduler."""
        if self._train_backend is not None:
            backend_scheduler = self._train_backend.build_scheduler(self, scheduler_config)
            if backend_scheduler is not None:
                self.lr_scheduler = backend_scheduler
                logger.info(
                    "Rank %s: Scheduler created by backend=%s",
                    self.rank,
                    self._train_backend_name,
                )
                return

        scheduler_type = scheduler_config.get("type", "constant")
        warmup_steps = scheduler_config.get("warmup_steps", 0)
        total_steps = scheduler_config.get("total_steps", 1000)

        if scheduler_type == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: 1.0,
            )
        elif scheduler_type == "linear":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda,
            )
        elif scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=0,
            )
        else:
            self.lr_scheduler = None

    _LOSS_RUNTIME_KEYS = {
        "use_ema",
        "ema_decay",
        "nft_timestep_mode",
        "nft_shuffle_timesteps",
        "nft_apply_shift",
        "decay_type",
        "ema_flat_steps",
        "ema_uprate",
        "ema_uphold",
        "old_adapter_name",
        "new_adapter_name",
    }

    @classmethod
    def _split_loss_kwargs(cls, loss_config: dict) -> tuple[dict, dict]:
        """Split loss kwargs into constructor kwargs and actor runtime kwargs."""
        extra = loss_config.get("loss_kwargs")
        if not isinstance(extra, dict):
            return {}, {}

        ctor_kwargs: Dict[str, Any] = {}
        runtime_kwargs: Dict[str, Any] = {}
        for key, value in extra.items():
            if key in cls._LOSS_RUNTIME_KEYS:
                runtime_kwargs[key] = value
            else:
                ctor_kwargs[key] = value
        return ctor_kwargs, runtime_kwargs

    @staticmethod
    def _filter_constructor_kwargs(loss_cls: type, kwargs: dict) -> dict:
        """Drop kwargs that are not accepted by the target loss constructor."""
        try:
            sig = inspect.signature(loss_cls.__init__)
        except (TypeError, ValueError):
            return dict(kwargs)

        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)

        allowed = {
            name
            for name, param in params.items()
            if name != "self"
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in kwargs.items() if key in allowed}

    @staticmethod
    def _collect_custom_loss_kwargs(loss_config: dict) -> dict:
        """Collect custom loss kwargs from config while removing actor-runtime keys."""
        reserved_keys = {
            "loss_type",
            "loss_path",
            "loss_kwargs",
            "guidance_scale",
        }
        kwargs = {
            key: value
            for key, value in loss_config.items()
            if key not in reserved_keys and value is not None
        }
        extra = loss_config.get("loss_kwargs")
        if isinstance(extra, dict):
            kwargs.update(
                {
                    key: value
                    for key, value in extra.items()
                    if key not in TrainingActor._LOSS_RUNTIME_KEYS
                }
            )
        return kwargs

    def _load_loss(self, loss_config: dict) -> None:
        """Load loss function based on loss_type."""
        from diffusionrl.losses import LOSS_REGISTRY, get_loss

        self._loss_type = str(loss_config.get("loss_type", "grpo"))
        self._loss_path = loss_config.get("loss_path")
        self._guidance_scale = float(loss_config.get("guidance_scale", 3.5))
        self._ema_updater = None

        ctor_loss_kwargs, runtime_loss_kwargs = self._split_loss_kwargs(loss_config)
        self._nft_timestep_mode = str(runtime_loss_kwargs.get("nft_timestep_mode", "random"))
        self._nft_shuffle_timesteps = bool(runtime_loss_kwargs.get("nft_shuffle_timesteps", True))
        self._nft_apply_shift = bool(runtime_loss_kwargs.get("nft_apply_shift", False))
        self._ema_decay = float(runtime_loss_kwargs.get("ema_decay", 0.001))
        self._use_ema = bool(runtime_loss_kwargs.get("use_ema", self._loss_type == "nft"))
        ema_decay_type = str(runtime_loss_kwargs.get("decay_type", "constant"))
        ema_flat_steps = int(runtime_loss_kwargs.get("ema_flat_steps", 0))
        ema_uprate = float(runtime_loss_kwargs.get("ema_uprate", 0.001))
        ema_uphold = float(runtime_loss_kwargs.get("ema_uphold", 0.5))
        old_adapter_name = str(runtime_loss_kwargs.get("old_adapter_name", "old"))
        new_adapter_name = str(runtime_loss_kwargs.get("new_adapter_name", "default"))

        if self._loss_type == "nft" and not self._loss_path:
            nft_kwargs = {
                "beta": 0.1,
                "adv_clip_max": 5.0,
                "adv_mode": "raw",
                "use_adaptive_weight": True,
                "shift": loss_config.get("shift", 3.0),
                "kl_coef": loss_config.get("kl_coef", 0.0),
            }
            nft_kwargs.update(ctor_loss_kwargs)
            loss_cls = LOSS_REGISTRY.get("nft")
            if loss_cls is not None:
                nft_kwargs = self._filter_constructor_kwargs(loss_cls, nft_kwargs)
            self.loss_fn = get_loss(
                loss_type="nft",
                **nft_kwargs,
            )
            if self._use_ema:
                from diffusionrl.utils import DualAdapterEMA
                self._ema_updater = DualAdapterEMA(
                    decay=self._ema_decay,
                    decay_type=ema_decay_type,
                    flat_steps=ema_flat_steps,
                    uprate=ema_uprate,
                    uphold=ema_uphold,
                    old_adapter_name=old_adapter_name,
                    new_adapter_name=new_adapter_name,
                )
        elif self._loss_type == "grpo" and not self._loss_path:
            grpo_kwargs = {
                "clip_range": loss_config.get("clip_range", 1e-4),
                "clip_range_mode": loss_config.get("clip_range_mode", "constant"),
                "use_kl_penalty": loss_config.get("use_kl_penalty", True),
                "kl_coef": loss_config.get("kl_coef", 0.01),
                "eta": loss_config.get("eta", 1.0),
                "sde_type": loss_config.get("sde_type", "sde"),
                "ignore_last": loss_config.get("ignore_last", False),
                "frozen_init_timesteps": loss_config.get("frozen_init_timesteps", 0),
            }
            grpo_kwargs.update(ctor_loss_kwargs)
            loss_cls = LOSS_REGISTRY.get("grpo")
            if loss_cls is not None:
                grpo_kwargs = self._filter_constructor_kwargs(loss_cls, grpo_kwargs)
            self.loss_fn = get_loss(
                loss_type="grpo",
                **grpo_kwargs,
            )
        else:
            custom_kwargs = self._collect_custom_loss_kwargs(loss_config)
            self.loss_fn = get_loss(
                loss_type=self._loss_type,
                loss_path=self._loss_path,
                **custom_kwargs,
            )

        if hasattr(self.loss_fn, "model_type"):
            self.loss_fn.model_type = getattr(self.model_bundle, "model_type", "default")
        if hasattr(self.loss_fn, "_forward_plugin"):
            forward_plugin_fn = getattr(self.model_bundle.__class__, "forward_plugin", None)
            if not callable(forward_plugin_fn):
                raise ValueError(
                    f"Model bundle {self.model_bundle.__class__.__name__} must define "
                    "classmethod forward_plugin() for loss forward dispatch."
                )
            forward_plugin = forward_plugin_fn()
            if forward_plugin is None:
                raise ValueError(
                    f"Model bundle {self.model_bundle.__class__.__name__}.forward_plugin() "
                    "returned None; expected a forward plugin instance."
                )
            self.loss_fn._forward_plugin = forward_plugin

        logger.info(
            "Rank %s: Loss function loaded (loss_type=%s, loss_path=%s)",
            self.rank,
            self._loss_type,
            self._loss_path,
        )

    def _maybe_replay_old_log_probs(self, batch: BackwardTrainingBatch) -> BackwardTrainingBatch:
        return self._replay_logprob_patch.maybe_replay_old_log_probs(
            batch=batch,
            enabled=self._replay_log_probs,
            loss_type=self._loss_type,
            sampling_config=self._sampling_config,
            model_bundle=self.model_bundle,
            model=self.model,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            guidance_scale=self._guidance_scale,
        )

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        seed: Optional[int] = None,
        decode_for_reward: bool = False,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> RolloutOutput:
        return self._sampling_service.generate(
            self,
            prompts=prompts,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            text_ids=text_ids,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            decode_for_reward=decode_for_reward,
            sde_indices=sde_indices,
            **kwargs,
        )

    def generate_batch(self, requests: List[RolloutRequest]) -> List[RolloutOutput]:
        return self._sampling_service.generate_batch(self, requests)

    def get_train_backend_info(self) -> Dict[str, Any]:
        if self._train_backend is None:
            return {
                "name": self._train_backend_name,
                "capabilities": dict(self._train_backend_capabilities),
            }
        return dict(self._train_backend.backend_info(self))

    def get_buffer_consumer_spec(self) -> Dict[str, Any]:
        if self._train_backend is None:
            dp_size = int(getattr(self, "world_size", 1))
            return {
                "dp_size": dp_size,
                "partition_train_data": True,
                "partition_mode": "data_parallel",
            }
        return dict(self._train_backend.buffer_consumer_spec(self))

    def _build_train_executor(self) -> TrainExecutor:
        dp_size = int(getattr(self, "world_size", 1))
        if self._train_backend is not None:
            dp_size = int(self._train_backend.data_parallel_size(self))
        config = TrainExecutorConfig(
            rank=self.rank,
            dp_size=dp_size,
            device=self._device,
            use_fsdp=bool(self._train_backend and self._train_backend.uses_sharded_model()),
            loss_type=self._loss_type,
            guidance_scale=self._guidance_scale,
            max_grad_norm=self._max_grad_norm,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            num_inner_epochs=self._num_inner_epochs,
            use_ema=self._use_ema,
            ema_updater=self._ema_updater,
            nft_timestep_mode=self._nft_timestep_mode,
            nft_shuffle_timesteps=self._nft_shuffle_timesteps,
            nft_apply_shift=self._nft_apply_shift,
            clip_grad_norm_fn=(
                (lambda *, model, max_grad_norm: self._train_backend.clip_grad_norm(
                    self,
                    model=model,
                    max_grad_norm=max_grad_norm,
                ))
                if self._train_backend is not None
                else None
            ),
        )
        return TrainExecutor(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            config=config,
        )

    def train(
        self,
        rollout_id: int,
        batch_or_ref: Union[ray.ObjectRef, TrainingBatch],
    ) -> Dict[str, Any]:
        """
        Execute one training step with gradient accumulation support.

        Args:
            rollout_id: Current rollout iteration number
            batch_or_ref: Either an ObjectRef containing typed TrainingBatch,
                or the TrainingBatch directly (Ray auto-resolves ObjectRefs
                when passed to remote methods)

        Returns:
            Dictionary of training metrics
        """
        if not self._is_initialized:
            raise RuntimeError("Actor not initialized. Call init() first.")

        self._log_gpu_state("training_train_start")
        # Handle both cases: ObjectRef or actual batch (Ray auto-resolves ObjectRefs)
        if isinstance(batch_or_ref, ray.ObjectRef):
            batch: TrainingBatch = ray.get(batch_or_ref)
        else:
            batch = batch_or_ref

        executor = self._build_train_executor()
        batch = executor.prepare_batch(batch)
        if batch is None:
            self._log_gpu_state("training_train_skipped")
            return executor.skipped_metrics(rollout_id)

        self._log_gpu_state("training_after_batch_to_device")
        if isinstance(batch, BackwardTrainingBatch):
            batch = self._maybe_replay_old_log_probs(batch)

        if self._train_backend is not None:
            backend_metrics = self._train_backend.run_train_step(
                self,
                rollout_id=rollout_id,
                batch=batch,
                executor=executor,
            )
            if backend_metrics is not None:
                self._log_gpu_state("training_train_end")
                return backend_metrics

        metrics = executor.execute_prepared_batch(rollout_id=rollout_id, batch=batch)
        self._log_gpu_state("training_train_end")
        return metrics

    # --- State IO methods ---

    def _get_backend(self):
        if self._train_backend is None:
            raise RuntimeError(
                "Training backend is not initialized. "
                "Call TrainingActor.init() before state IO operations."
            )
        return self._train_backend

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights for syncing to rollout actors."""
        was_offloaded = self._is_offloaded
        if was_offloaded:
            self.onload()
        try:
            backend = self._get_backend()
            return backend.get_state_dict(
                self,
                lora_only=bool(self._use_lora),
                rank0_only=True,
            )
        finally:
            if was_offloaded:
                self.offload()

    def export_weights_to_path(
        self,
        checkpoint_path: str,
        export_format: str = "state_dict",
    ) -> Optional[str]:
        """Export synchronized weights to a shared path."""
        backend = self._get_backend()
        exported_path = backend.export_weights_to_path(
            self, checkpoint_path, export_format=export_format,
        )
        if exported_path is not None:
            return exported_path

        state_dict = self.get_weights()
        if self.rank != 0:
            return None
        if export_format == "state_dict":
            return publish_checkpoint_atomic(state_dict, checkpoint_path)
        if export_format == "sglang_transformer_safetensors":
            return publish_sglang_transformer_checkpoint_atomic(
                state_dict, checkpoint_path, module_name="transformer",
            )
        raise ValueError(
            f"Unsupported export_format={export_format}. "
            "Expected one of: state_dict, sglang_transformer_safetensors"
        )

    def update_weights(self) -> None:
        """Broadcast weights from rank 0 to all other ranks."""
        backend = self._get_backend()
        backend.broadcast_parameters(self)

    def get_node_ip_and_free_port(self, start_port: int = 26000) -> Dict[str, Union[str, int]]:
        """Return current node IP and a free port for temporary process groups."""
        del start_port
        master_address = None
        try:
            import ray._private.services

            master_address = str(ray._private.services.get_node_ip_address())
        except Exception:
            master_address = self._get_current_node_ip()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            master_port = int(sock.getsockname()[1])

        return {"master_address": str(master_address), "master_port": int(master_port)}

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        """Initialize rank-0 custom process group for rollout weight broadcast."""
        if self.rank != 0:
            return
        if group_name in self._weights_update_groups:
            return
        self._ensure_sglang_importable()
        try:
            from sglang.srt.utils.common import init_custom_process_group
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sglang init_custom_process_group for NCCL weight sync."
            ) from exc
        self._weights_update_groups[group_name] = init_custom_process_group(
            backend=str(backend),
            init_method=f"tcp://{master_address}:{int(master_port)}",
            world_size=int(world_size),
            rank=0,
            group_name=str(group_name),
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        """Destroy rank-0 custom process group for rollout weight broadcast."""
        if self.rank != 0:
            return
        pg = self._weights_update_groups.pop(group_name, None)
        if pg is not None:
            dist.destroy_process_group(pg)

    @staticmethod
    def _to_rollout_dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).replace("torch.", "")

    def _iter_weight_sync_buckets(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        bucket_size_mb: int,
    ):
        bucket_size_bytes = max(1, int(bucket_size_mb) * 1024 * 1024)
        current_bucket: List[tuple[str, torch.Tensor]] = []
        current_bytes = 0

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            staged = tensor.detach().to(device=self._device, non_blocking=False).contiguous()
            tensor_bytes = staged.numel() * staged.element_size()
            if current_bucket and current_bytes + tensor_bytes > bucket_size_bytes:
                yield current_bucket
                current_bucket = []
                current_bytes = 0
            current_bucket.append((str(name), staged))
            current_bytes += tensor_bytes

        if current_bucket:
            yield current_bucket

    def _iter_weight_sync_buckets_with_last(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        bucket_size_mb: int,
    ):
        bucket_iter = iter(
            self._iter_weight_sync_buckets(state_dict, bucket_size_mb=bucket_size_mb)
        )
        try:
            current_bucket = next(bucket_iter)
        except StopIteration:
            return

        while True:
            try:
                next_bucket = next(bucket_iter)
                is_last = False
            except StopIteration:
                next_bucket = None
                is_last = True

            yield current_bucket, is_last

            if next_bucket is None:
                break
            current_bucket = next_bucket

    def _flatten_tensor_bucket_payload(self, named_tensors: List[tuple[str, torch.Tensor]]) -> tuple[List[str], List[dict]]:
        self._ensure_sglang_importable()
        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except Exception:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()

        if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            grouped = {"_all": named_tensors}
        else:
            grouped: Dict[torch.dtype, List[tuple[str, torch.Tensor]]] = {}
            for name, tensor in named_tensors:
                grouped.setdefault(tensor.dtype, []).append((name, tensor))

        serialized_payloads: List[str] = []
        long_lived_payloads: List[dict] = []
        for grouped_named_tensors in grouped.values():
            bucket = FlattenedTensorBucket(named_tensors=grouped_named_tensors)
            payload = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
            long_lived_payloads.append(payload)
            serialized_payloads.append(
                MultiprocessingSerializer.serialize(payload, output_str=True)
            )
        return serialized_payloads, long_lived_payloads

    def sync_weights_to_rollout_ipc(
        self,
        *,
        rollout_manager: Any,
        target_modules: Optional[List[str]] = None,
        bucket_size_mb: int = 256,
        flush_cache: bool = True,
        tp_payload_count: int = 1,
    ) -> Dict[str, int]:
        """Synchronize weights to rollout actors via serialized tensor IPC payloads."""
        was_offloaded = self._is_offloaded
        if was_offloaded:
            self.onload()
        try:
            backend = self._get_backend()
            state_dict = backend.get_state_dict(
                self,
                lora_only=bool(self._use_lora),
                rank0_only=True,
            )
            if self.rank != 0:
                return {"rank": int(self.rank), "buckets": 0, "payloads": 0}

            total_buckets = 0
            total_payloads = 0
            for named_tensors, is_last_bucket in self._iter_weight_sync_buckets_with_last(
                state_dict,
                bucket_size_mb=bucket_size_mb,
            ):
                total_buckets += 1
                serialized_payloads, long_lived_payloads = self._flatten_tensor_bucket_payload(named_tensors)
                try:
                    for payload_idx, serialized_payload in enumerate(serialized_payloads):
                        flush_this_payload = bool(
                            flush_cache
                            and is_last_bucket
                            and payload_idx == len(serialized_payloads) - 1
                        )
                        payload_batch = [serialized_payload] * max(1, int(tp_payload_count))
                        ref = rollout_manager.update_weights_from_tensor.remote(
                            serialized_named_tensors=payload_batch,
                            target_modules=target_modules,
                            load_format="flattened_bucket",
                            flush_cache=flush_this_payload,
                        )
                        ray.get(ref)
                        total_payloads += 1
                finally:
                    del long_lived_payloads
                    del serialized_payloads
                    del named_tensors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return {
                "rank": int(self.rank),
                "buckets": int(total_buckets),
                "payloads": int(total_payloads),
            }
        finally:
            if was_offloaded:
                self.offload()

    def sync_weights_to_rollout_nccl(
        self,
        *,
        rollout_manager: Any,
        group_name: str,
        target_modules: Optional[List[str]] = None,
        bucket_size_mb: int = 256,
        flush_cache: bool = True,
    ) -> Dict[str, int]:
        """Synchronize weights to rollout actors via custom NCCL broadcast group."""
        was_offloaded = self._is_offloaded
        if was_offloaded:
            self.onload()
        try:
            backend = self._get_backend()
            state_dict = backend.get_state_dict(
                self,
                lora_only=bool(self._use_lora),
                rank0_only=True,
            )
            if self.rank != 0:
                return {"rank": int(self.rank), "buckets": 0, "broadcast_tensors": 0}

            pg = self._weights_update_groups.get(group_name)
            if pg is None:
                raise RuntimeError(
                    f"NCCL weight update group is not initialized: group_name={group_name!r}"
                )

            total_buckets = 0
            total_tensors = 0
            for named_tensors, is_last_bucket in self._iter_weight_sync_buckets_with_last(
                state_dict,
                bucket_size_mb=bucket_size_mb,
            ):
                total_buckets += 1
                names = [name for name, _ in named_tensors]
                dtypes = [self._to_rollout_dtype_name(tensor.dtype) for _, tensor in named_tensors]
                shapes = [list(tensor.shape) for _, tensor in named_tensors]

                ref = rollout_manager.update_weights_from_distributed.remote(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=str(group_name),
                    target_modules=target_modules,
                    flush_cache=bool(flush_cache and is_last_bucket),
                )
                handles = [
                    dist.broadcast(tensor, src=0, group=pg, async_op=True)
                    for _, tensor in named_tensors
                ]
                for handle in handles:
                    handle.wait()
                ray.get(ref)
                total_tensors += len(named_tensors)

                del named_tensors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return {
                "rank": int(self.rank),
                "buckets": int(total_buckets),
                "broadcast_tensors": int(total_tensors),
            }
        finally:
            if was_offloaded:
                self.offload()

    def save_model(self, path: str) -> None:
        """Save model checkpoint (backend-safe collective)."""
        was_offloaded = self._is_offloaded
        if was_offloaded:
            self.onload()
        try:
            backend = self._get_backend()
            state_dict = backend.get_state_dict(self, lora_only=False, rank0_only=True)
            if self.rank != 0:
                return
            os.makedirs(path, exist_ok=True)
            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.lr_scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
            logger.info("Checkpoint saved to %s", path)
        finally:
            if was_offloaded:
                self.offload()

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        backend = self._get_backend()
        backend.load_state_dict(self, checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Checkpoint loaded from %s", path)

    # --- Memory methods ---

    @staticmethod
    def _safe_to_device(component: nn.Module, device, name: str) -> None:
        if component is None or not hasattr(component, "to"):
            return
        try:
            component.to(device)
        except Exception as e:
            logger.warning("Could not move %s to %s: %s", name, device, e)

    def _move_aux_components(self, device, include_transformer: bool) -> None:
        if include_transformer and self.model is not None:
            self._safe_to_device(self.model, device, "model")
        if self.text_encoder is not None:
            self._safe_to_device(self.text_encoder, device, "text_encoder")
        if self.vae is not None:
            self._safe_to_device(self.vae, device, "vae")

        if self.model_bundle is not None:
            for attr_name, component in self.model_bundle.iter_offloadable_modules(
                include_transformer=include_transformer
            ):
                if not include_transformer and component is self.model:
                    continue
                self._safe_to_device(component, device, f"model_bundle.{attr_name}")

        if self._sampler is not None:
            for attr_name, component in self._actor_sampling_executor.iter_reflection_modules(
                self._sampler, include_transformer=include_transformer,
            ):
                if not include_transformer and component is self.model:
                    continue
                self._safe_to_device(component, device, f"sampler.{attr_name}")

    def offload(self) -> None:
        """Offload model and optimizer to CPU."""
        backend = getattr(self, "_train_backend", None)
        if backend is not None:
            try:
                if backend.offload(self):
                    self._is_offloaded = True
                    _clear_gpu_memory()
                    logger.info("Rank %s: Offload handled by backend=%s", self.rank, backend.name)
                    self._log_gpu_state("training_offload")
                    return
            except Exception as exc:
                logger.warning("Rank %s: backend offload failed (%s), falling back to default flow", self.rank, exc)

        if getattr(self, "_fsdp_cpu_offload", False):
            self._move_aux_components("cpu", include_transformer=False)
            self._is_offloaded = True
            _clear_gpu_memory()
            logger.info("Rank %s: FSDP CPU offload mode - just clearing cache", self.rank)
            return

        self._is_offloaded = True
        self._move_aux_components("cpu", include_transformer=False)

        if self.model is not None:
            self.model.to("cpu")

        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        _clear_gpu_memory()
        logger.info("Rank %s: Model and optimizer offloaded to CPU", self.rank)
        self._log_gpu_state("training_offload")

    def onload(self) -> None:
        """Load model and optimizer back to GPU."""
        backend = getattr(self, "_train_backend", None)
        if backend is not None:
            try:
                if backend.onload(self):
                    self._is_offloaded = False
                    logger.info("Rank %s: Onload handled by backend=%s", self.rank, backend.name)
                    self._log_gpu_state("training_onload")
                    return
            except Exception as exc:
                logger.warning("Rank %s: backend onload failed (%s), falling back to default flow", self.rank, exc)

        if getattr(self, "_fsdp_cpu_offload", False):
            if self._device is not None:
                self._move_aux_components(self._device, include_transformer=False)
            self._is_offloaded = False
            logger.info("Rank %s: FSDP CPU offload mode - skipping manual onload", self.rank)
            return

        if self.model is not None:
            self.model.to(self._device)

        if self._device is not None:
            self._move_aux_components(self._device, include_transformer=False)

        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._device)

        self._is_offloaded = False
        logger.info("Rank %s: Model and optimizer loaded to GPU", self.rank)
        self._log_gpu_state("training_onload")

    def clear_memory(self) -> None:
        """Clear GPU cache without full offload."""
        torch.cuda.empty_cache()
        logger.debug("Rank %s: GPU cache cleared", self.rank)

    def onload_weights(self) -> None:
        self.onload()

    def onload_post_update(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def onload_runtime_cache(self) -> None:
        # Diffusion training actor currently does not hold a KV/cache stage.
        return None

    def health_check(self) -> bool:
        """Check if actor is healthy.

        Returns True if:
        - Actor is initialized (normal state), OR
        - Actor is in offloaded state (healthy but GPU resources freed)
        """
        if self._is_offloaded:
            # Offloaded state is considered healthy - just not ready for training
            return True
        return self._is_initialized

    def is_offloaded(self) -> bool:
        """Check if actor is currently offloaded to CPU."""
        return self._is_offloaded
