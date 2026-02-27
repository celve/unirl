"""
diffusionrl Training Actor - Manages model training via pluggable train backends.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

import ray
import torch

from diffusionrl.types.sampling import RolloutRequest, RolloutOutput
from diffusionrl.types.training_batch import (
    BackwardTrainingBatch,
    TrainingBatch,
)
from diffusionrl.patches.replay_logprob import ReplayLogProbPatch
from diffusionrl.ray.utils.actor_sampling import TrainingActorSamplingService
from diffusionrl.ray.utils.memory import TrainingActorMemoryService
from diffusionrl.ray.utils.state_io import TrainingActorStateIOService
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
        self._use_fsdp = True  # legacy alias; derived from selected backend
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
        self._memory_service = TrainingActorMemoryService()
        self._state_io_service = TrainingActorStateIOService()
        self._replay_logprob_patch = ReplayLogProbPatch()
        self._sampling_ready = False
        self.text_encoder = None
        self.vae = None
        self.scheduler = None

    def _log_resource_ids(self, tag: str) -> None:
        log_resource_ids(tag, self.rank)

    def _log_gpu_state(self, tag: str) -> None:
        log_gpu_state(tag, self.rank, device=self._device, offloaded=self._is_offloaded)

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
                - legacy use_fsdp/fsdp_config (backward compatibility)

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

        # Transitional compatibility: map legacy keys into backend kwargs.
        if backend_name == "fsdp":
            legacy_fsdp_config = config.get("fsdp_config", {}) or {}
            if isinstance(legacy_fsdp_config, dict):
                for key, value in legacy_fsdp_config.items():
                    backend_kwargs.setdefault(key, value)
            backend_kwargs.setdefault("use_fsdp", bool(config.get("use_fsdp", True)))

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
        self._use_fsdp = bool(self._train_backend.uses_sharded_model())

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

    @staticmethod
    def _collect_custom_loss_kwargs(loss_config: dict) -> dict:
        """Collect custom loss kwargs from config while removing actor-runtime keys."""
        reserved_keys = {
            "loss_type",
            "loss_path",
            "loss_kwargs",
            "use_ema",
            "ema_decay",
            "guidance_scale",
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
        kwargs = {
            key: value
            for key, value in loss_config.items()
            if key not in reserved_keys and value is not None
        }
        extra = loss_config.get("loss_kwargs")
        if isinstance(extra, dict):
            kwargs.update(extra)
        return kwargs

    def _load_loss(self, loss_config: dict) -> None:
        """Load loss function based on loss_type."""
        from diffusionrl.losses import get_loss

        self._loss_type = loss_config.get("loss_type", "grpo")
        self._loss_path = loss_config.get("loss_path")
        self._use_ema = loss_config.get("use_ema", False)
        self._ema_decay = loss_config.get("ema_decay", 0.001)
        self._guidance_scale = loss_config.get("guidance_scale", 3.5)

        if self._loss_type == "nft" and not self._loss_path:
            # NFT loss with forward process optimization
            self.loss_fn = get_loss(
                loss_type="nft",
                beta=loss_config.get("beta", 0.1),
                adv_clip_max=loss_config.get("adv_clip_max", 5.0),
                adv_mode=loss_config.get("adv_mode", "raw"),
                use_adaptive_weight=loss_config.get("use_adaptive_weight", True),
                shift=loss_config.get("shift", 3.0),
                kl_coef=loss_config.get("kl_coef", 0.0),
            )
            # NFT timestep handling
            self._nft_timestep_mode = loss_config.get("nft_timestep_mode", "random")
            self._nft_shuffle_timesteps = loss_config.get("nft_shuffle_timesteps", True)
            self._nft_apply_shift = loss_config.get("nft_apply_shift", False)
            # Enable EMA by default for NFT
            self._use_ema = loss_config.get("use_ema", True)

            # Initialize EMA updater for dual adapter mechanism
            if self._use_ema:
                from diffusionrl.utils import DualAdapterEMA
                decay_type = loss_config.get("decay_type", "constant")
                self._ema_updater = DualAdapterEMA(
                    decay=self._ema_decay,
                    decay_type=decay_type,
                    # DiffusionNFT decay_type=1 defaults
                    flat_steps=loss_config.get("ema_flat_steps", 0),
                    uprate=loss_config.get("ema_uprate", 0.001),
                    uphold=loss_config.get("ema_uphold", 0.5),
                    old_adapter_name=loss_config.get("old_adapter_name", "old"),
                    new_adapter_name=loss_config.get("new_adapter_name", "default"),
                )
            # Bind model_type for adapter selection in NFT loss.
            if hasattr(self.loss_fn, "model_type"):
                self.loss_fn.model_type = getattr(self.model_bundle, "model_type", "default")
            if hasattr(self.loss_fn, "_forward_plugin"):
                self.loss_fn._forward_plugin = None
        elif self._loss_type == "grpo" and not self._loss_path:
            # Built-in GRPO loss.
            self.loss_fn = get_loss(
                loss_type="grpo",
                clip_range=loss_config.get("clip_range", 1e-4),
                clip_range_mode=loss_config.get("clip_range_mode", "constant"),
                use_kl_penalty=loss_config.get("use_kl_penalty", True),
                kl_coef=loss_config.get("kl_coef", 0.01),
                eta=loss_config.get("eta", 1.0),
                sde_type=loss_config.get("sde_type", "sde"),
                ignore_last=loss_config.get("ignore_last", False),
                frozen_init_timesteps=loss_config.get("frozen_init_timesteps", 0),
            )
        else:
            # Custom/registered loss. Use explicit loss_path when provided.
            custom_kwargs = self._collect_custom_loss_kwargs(loss_config)
            self.loss_fn = get_loss(
                loss_type=self._loss_type,
                loss_path=self._loss_path,
                **custom_kwargs,
            )
            # Keep NFT defaults if user selects nft loss_type via custom path.
            if self._loss_type == "nft":
                self._nft_timestep_mode = loss_config.get("nft_timestep_mode", "random")
                self._nft_shuffle_timesteps = loss_config.get("nft_shuffle_timesteps", True)
                self._nft_apply_shift = loss_config.get("nft_apply_shift", False)
                self._use_ema = loss_config.get("use_ema", True)

        # Bind model metadata for losses that implement model-aware forward plugins.
        if hasattr(self.loss_fn, "model_type"):
            self.loss_fn.model_type = getattr(self.model_bundle, "model_type", "default")
        if hasattr(self.loss_fn, "_forward_plugin"):
            self.loss_fn._forward_plugin = None

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
        return {
            "name": self._train_backend_name,
            "capabilities": dict(self._train_backend_capabilities),
        }

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

        metrics = executor.execute_prepared_batch(rollout_id=rollout_id, batch=batch)
        self._log_gpu_state("training_train_end")
        return metrics

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights for syncing to rollout actors."""
        return self._state_io_service.get_weights(self)

    def export_weights_to_path(
        self,
        checkpoint_path: str,
        export_format: str = "state_dict",
    ) -> Optional[str]:
        """Export synchronized weights to a shared path."""
        return self._state_io_service.export_weights_to_path(
            self,
            checkpoint_path,
            export_format=export_format,
        )

    def update_weights(self) -> None:
        """Broadcast weights from rank 0 to all other ranks."""
        self._state_io_service.update_weights(self)

    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        self._state_io_service.save_model(self, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        self._state_io_service.load_checkpoint(self, path)

    def offload(self) -> None:
        """Offload model and optimizer to CPU."""
        self._memory_service.offload(self)

    def onload(self) -> None:
        """Load model and optimizer back to GPU."""
        self._memory_service.onload(self)

    def clear_memory(self) -> None:
        """Clear GPU cache without offloading."""
        self._memory_service.clear_memory(self)

    def onload_weights(self) -> None:
        self.onload()

    def onload_post_update(self) -> None:
        self._memory_service.onload_post_update(self)

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
