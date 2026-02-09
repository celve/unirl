"""
diffusionrl Training Actor - Manages model training with FSDP.
"""
import logging
import os
import inspect
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .base import BaseTrainRayActor, log_resource_ids, log_gpu_state, tensor_to_pil
from diffusionrl.types import (
    GRPOTrainingBatch,
    NFTTrainingBatch,
    TrainingBatch,
    TimestepData,
    PromptEmbeddings,
    SamplerOutput,
    InferenceRequest,
)
from diffusionrl.utils import load_function, clear_memory
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.weight_sync_checkpoint import publish_checkpoint_atomic

logger = logging.getLogger(__name__)


@contextmanager
def sampling_eval_context(modules: List[nn.Module]):
    """Temporarily switch modules to eval/no-grad and restore all training flags."""
    original_modes = [(module, bool(module.training)) for module in modules if isinstance(module, nn.Module)]
    param_states: List[Tuple[torch.nn.Parameter, bool]] = []
    seen_params: Set[int] = set()
    for module, _ in original_modes:
        for param in module.parameters(recurse=True):
            pid = id(param)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            param_states.append((param, bool(param.requires_grad)))
            if param.requires_grad:
                param.requires_grad_(False)
    for module, _ in original_modes:
        module.eval()

    try:
        # Keep sampling in no_grad (not inference_mode) to avoid FSDP grad_fn/AccumulateGrad
        # assertion failures when the same actor returns to training afterwards.
        with torch.no_grad():
            yield
    finally:
        for module, was_training in original_modes:
            module.train(was_training)
        for param, requires_grad in param_states:
            param.requires_grad_(requires_grad)


@ray.remote(num_gpus=1)
class TrainingActor(BaseTrainRayActor):
    """
    Training Actor - Manages model training with FSDP support.

    This actor handles:
    - Model loading and FSDP wrapping
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
        self._use_fsdp = True
        self._use_lora = False

        # NFT-specific: EMA updater for dual adapter mechanism
        self._ema_updater = None
        self._use_ema = False
        self._ema_decay = 0.001
        self._loss_type = "grpo"
        self._guidance_scale = 3.5  # Default CFG scale for training
        # NFT timestep handling
        self._nft_timestep_mode = "random"  # random or all
        self._nft_shuffle_timesteps = True
        self._nft_apply_shift = False

        # Training config (read from config in init)
        self._max_grad_norm = 1.0
        self._gradient_accumulation_steps = 1
        self._num_inner_epochs = 1
        self._fastvideo_replay_log_probs = False

        # Sampling support (training-actor sampling mode)
        self._sampling_config: Dict[str, Any] = {}
        self._sampler = None
        self._replay_sampler = None
        self._replay_sampler_path: Optional[str] = None
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
                - model_config: dict with model_path, pretrained_model_path
                - optimizer_config: dict with lr, betas, weight_decay
                - scheduler_config: dict with scheduler type and params
                - loss_config: dict with loss_type, clip_range, kl_coef, etc.
                - training_config: dict with max_grad_norm, gradient_accumulation_steps
                - use_fsdp: Whether to use FSDP
                - fsdp_config: FSDP configuration

        Note:
            Algorithm instantiation for advantage computation happens in
            RolloutManager, not here. Training uses loss_fn directly.
        """
        logger.info(f"Rank {self.rank}: Initializing training actor...")

        # Initialize distributed
        self._init_distributed(backend="nccl")

        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)

        # Check if FSDP CPU offload is enabled (affects model loading)
        fsdp_config = config.get("fsdp_config", {})
        self._fsdp_cpu_offload = fsdp_config.get("cpu_offload", False)

        # Load model (don't move to GPU if CPU offload is enabled)
        self._load_model(config.get("model_config", {}))

        # FSDP wrap
        self._use_fsdp = config.get("use_fsdp", True)
        if self._use_fsdp:
            self._wrap_fsdp(fsdp_config)

        # Create optimizer
        self._create_optimizer(config.get("optimizer_config", {}))

        # Create scheduler
        self._create_scheduler(config.get("scheduler_config", {}))

        # Load loss function
        self._load_loss(config.get("loss_config", {}))

        # Load training config
        training_config = config.get("training_config", {})
        self._max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self._gradient_accumulation_steps = self._resolve_grad_accum(training_config)
        self._num_inner_epochs = max(1, int(training_config.get("num_inner_epochs", 1)))
        self._fastvideo_replay_log_probs = bool(
            training_config.get("fastvideo_replay_log_probs", False)
        )

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
            f"(max_grad_norm={self._max_grad_norm}, "
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
            "pretrained_path": model_config.get("pretrained_model_path", ""),
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

    def _wrap_fsdp(self, fsdp_config: dict) -> None:
        """Wrap model with FSDP."""
        # Get sharding strategy
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = strategy_map.get(
            fsdp_config.get("sharding_strategy", "FULL_SHARD"),
            ShardingStrategy.FULL_SHARD,
        )

        # Get backward prefetch
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = prefetch_map.get(
            fsdp_config.get("backward_prefetch", "BACKWARD_PRE"),
            BackwardPrefetch.BACKWARD_PRE,
        )

        # CPU offload
        cpu_offload = None
        if fsdp_config.get("cpu_offload", False):
            cpu_offload = CPUOffload(offload_params=True)

        # Mixed precision
        mixed_precision = None
        if fsdp_config.get("mixed_precision", True):
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

        # Auto wrap policy
        # Note: transformer_auto_wrap_policy requires functools.partial in newer PyTorch
        auto_wrap_policy = None
        if hasattr(self.model_bundle, "get_no_split_modules"):
            no_split_modules = self.model_bundle.get_no_split_modules()
            if no_split_modules:
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=no_split_modules,
                )

        # Wrap with FSDP
        self.model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self._device,
            use_orig_params=True,
        )

        logger.info(f"Rank {self.rank}: Model wrapped with FSDP")

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

    def _load_loss(self, loss_config: dict) -> None:
        """Load loss function based on loss_type."""
        from diffusionrl.losses import get_loss

        self._loss_type = loss_config.get("loss_type", "grpo")
        self._use_ema = loss_config.get("use_ema", False)
        self._ema_decay = loss_config.get("ema_decay", 0.001)
        self._guidance_scale = loss_config.get("guidance_scale", 3.5)

        if self._loss_type == "nft":
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
        else:
            # GRPO loss (default)
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

        logger.info(f"Rank {self.rank}: Loss function loaded ({self._loss_type})")

    def _safe_to_device(self, component: nn.Module, device: Union[str, torch.device], name: str) -> None:
        if component is None or not hasattr(component, "to"):
            return
        try:
            component.to(device)
        except Exception as e:
            logger.warning("Could not move %s to %s: %s", name, device, e)

    def _iter_reflection_modules(
        self,
        obj: Any,
        include_transformer: bool,
    ) -> List[Tuple[str, nn.Module]]:
        if obj is None:
            return []
        known_names = {
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
            "image_encoder",
        }
        results: List[Tuple[str, nn.Module]] = []
        for name, value in obj.__dict__.items():
            if not isinstance(value, nn.Module):
                continue
            base_name = name.lstrip("_").lower()
            if not include_transformer and "transformer" in base_name:
                continue
            if base_name in known_names or any(token in base_name for token in ("encoder", "vae", "transformer")):
                results.append((name, value))
        return results

    def _move_aux_components(self, device: Union[str, torch.device], include_transformer: bool) -> None:
        # Direct refs
        if include_transformer and self.model is not None:
            self._safe_to_device(self.model, device, "model")
        if self.text_encoder is not None:
            self._safe_to_device(self.text_encoder, device, "text_encoder")
        if self.vae is not None:
            self._safe_to_device(self.vae, device, "vae")

        # Model bundle refs
        if self.model_bundle is not None:
            for attr_name, component in self.model_bundle.iter_offloadable_modules(
                include_transformer=include_transformer
            ):
                if not include_transformer and component is self.model:
                    continue
                self._safe_to_device(component, device, f"model_bundle.{attr_name}")

        # Sampler refs (may hold encoders)
        if self._sampler is not None:
            for attr_name, component in self._iter_reflection_modules(
                self._sampler, include_transformer=include_transformer
            ):
                if not include_transformer and component is self.model:
                    continue
                self._safe_to_device(component, device, f"sampler.{attr_name}")

    def _ensure_sampling_components(self) -> None:
        if self._sampling_ready:
            return
        if self.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")

        # Lazy-load aux components needed for sampling
        try:
            self.model_bundle.load_aux_components()
        except Exception as e:
            logger.warning(f"Failed to load auxiliary components: {e}")
            raise

        self.text_encoder = getattr(self.model_bundle, "text_encoder", None)
        self.vae = getattr(self.model_bundle, "vae", None)
        self.scheduler = getattr(self.model_bundle, "scheduler", None)

        sampler_path = self._sampling_config.get("sampler_path")
        if not sampler_path:
            raise ValueError("sampling_config must provide sampler_path for training-actor sampling")

        sampler_cls = load_function(sampler_path)
        sampler_kwargs = dict(self._sampling_config.get("sampler_kwargs", {}))
        extra_kwargs = {}
        if hasattr(self.model_bundle, "get_sampler_extra_kwargs"):
            extra_kwargs = self.model_bundle.get_sampler_extra_kwargs() or {}
        for key, value in extra_kwargs.items():
            sampler_kwargs.setdefault(key, value)

        self._sampler = sampler_cls(
            model=self.model,
            text_encoder=self.text_encoder,
            vae=self.vae,
            eta=self._sampling_config.get("eta", 1.0),
            sde_type=self._sampling_config.get("sde_type", "sde"),
            shift=self._sampling_config.get("shift", 3.0),
            **sampler_kwargs,
        )

        self._sampling_ready = True

    def _resolve_replay_sampler_path(self) -> str:
        replay_path = self._sampling_config.get("replay_sampler_path")
        if replay_path:
            return replay_path

        sampler_path = self._sampling_config.get("sampler_path")
        if sampler_path and "fastvideo" not in sampler_path.lower():
            return sampler_path

        # Experimental/ad-hoc bridge: FastVideo rollout does not emit old log_probs.
        # Recompute old log_probs with an FSDP sampler implementation on training actors.
        model_type = getattr(self.model_bundle, "model_type", None)
        fallback = {
            "flux": "diffusionrl.samplers.fsdp.flux_sampler.FluxSampler",
            "hunyuan": "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler",
            "sd3": "diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler",
        }
        resolved = fallback.get(model_type)
        if not resolved:
            raise RuntimeError(
                "fastvideo_replay_log_probs requires replay_sampler_path or a known model_type fallback. "
                f"model_type={model_type!r}"
            )
        return resolved

    def _build_replay_sampler(self) -> None:
        if self._replay_sampler is not None:
            return
        if self.model is None:
            raise RuntimeError("Model not initialized for replay sampler")

        sampler_path = self._resolve_replay_sampler_path()
        sampler_cls = load_function(sampler_path)
        init_sig = inspect.signature(sampler_cls.__init__)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
        )

        sampler_kwargs = dict(self._sampling_config.get("sampler_kwargs", {}) or {})
        base_kwargs: Dict[str, Any] = {
            "model": self.model,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
            "scheduler": self.scheduler,
            "eta": self._sampling_config.get("eta", 1.0),
            "sde_type": self._sampling_config.get("sde_type", "sde"),
            "shift": self._sampling_config.get("shift", 3.0),
            **sampler_kwargs,
        }
        filtered_kwargs: Dict[str, Any] = {}
        for key, value in base_kwargs.items():
            if key in init_sig.parameters or accepts_kwargs:
                filtered_kwargs[key] = value

        self._replay_sampler = sampler_cls(**filtered_kwargs)
        self._replay_sampler_path = sampler_path
        logger.warning(
            "Enabled experimental replay sampler for old log_probs: %s",
            sampler_path,
        )

    def _maybe_replay_old_log_probs(self, batch: GRPOTrainingBatch) -> GRPOTrainingBatch:
        if not self._fastvideo_replay_log_probs:
            return batch
        if self._loss_type != "grpo":
            return batch
        if len(batch.log_probs) > 0:
            return batch

        self._build_replay_sampler()
        replay_sampler = self._replay_sampler
        replay_fn = getattr(replay_sampler, "compute_log_prob_for_training", None)
        if replay_fn is None:
            raise RuntimeError(
                "Replay sampler does not implement compute_log_prob_for_training; "
                f"sampler_path={self._replay_sampler_path}"
            )

        # Experimental/ad-hoc path: recover old log_probs from rollout trajectory.
        allowed_steps = set(int(v) for v in batch.resolved_step_indices[:-1].tolist())
        target_steps = sorted(int(i) for i in batch.sde_indices if int(i) in allowed_steps)
        if not target_steps:
            raise RuntimeError(
                "fastvideo_replay_log_probs enabled but no target SDE steps were provided in batch."
            )

        replay_sig = inspect.signature(replay_fn)
        replayed: Dict[int, torch.Tensor] = {}
        for step_idx in target_steps:
            pos = batch.get_position_for_step(step_idx)
            arg_map: Dict[str, Any] = {
                "latents": batch.trajectories[:, pos],
                "prev_latents": batch.trajectories[:, pos + 1],
                "prompt_embeds": batch.embeddings.prompt_embeds,
                "pooled_prompt_embeds": batch.embeddings.pooled_prompt_embeds,
                "encoder_attention_mask": batch.embeddings.encoder_attention_mask,
                "text_ids": batch.embeddings.text_ids,
                "image_ids": batch.embeddings.image_ids,
                "timestep_index": int(step_idx),
                "sigma_schedule": batch.timesteps,
                "guidance_scale": self._guidance_scale,
            }
            call_kwargs = {
                name: value
                for name, value in arg_map.items()
                if name in replay_sig.parameters
            }
            missing_required = [
                name
                for name, param in replay_sig.parameters.items()
                if (
                    param.default is inspect.Parameter.empty
                    and param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                    and name not in call_kwargs
                )
            ]
            if missing_required:
                raise RuntimeError(
                    f"Replay sampler missing required args {missing_required} for step={step_idx}. "
                    f"sampler_path={self._replay_sampler_path}"
                )
            with torch.no_grad():
                old_log_prob = replay_fn(**call_kwargs)
            if not torch.is_tensor(old_log_prob):
                raise RuntimeError(
                    "Replay sampler must return torch.Tensor for old_log_prob, "
                    f"got {type(old_log_prob).__name__}"
                )
            replayed[int(step_idx)] = old_log_prob.detach()

        batch.log_probs = type(batch.log_probs).from_dict(replayed)
        return batch

    def encode_prompt(self, prompts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        if self.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")
        if not hasattr(self.model_bundle, "encode_prompt_for_inference"):
            raise RuntimeError("Model bundle does not support inference prompt encoding")
        return self.model_bundle.encode_prompt_for_inference(prompts, **kwargs)

    def _tensor_to_pil(self, images: torch.Tensor) -> List[Any]:
        return tensor_to_pil(images)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("VAE not available for decoding")

        with torch.no_grad():
            if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'scaling_factor'):
                scaling_factor = self.vae.config.scaling_factor
            else:
                scaling_factor = 0.18215

            latents_float = latents.to(dtype=torch.float32)
            decoded = self.vae.to(torch.float32).decode(latents_float / scaling_factor).sample
            return (decoded + 1) / 2

    def _iter_sampling_mode_modules(self) -> List[nn.Module]:
        """Collect modules whose train/eval state must be restored around sampling."""
        modules: List[nn.Module] = []
        seen: Set[int] = set()
        for component in (self.model, self.text_encoder, self.vae):
            if isinstance(component, nn.Module):
                ident = id(component)
                if ident not in seen:
                    modules.append(component)
                    seen.add(ident)

        if self.model_bundle is not None and hasattr(self.model_bundle, "iter_offloadable_modules"):
            for _name, component in self.model_bundle.iter_offloadable_modules(include_transformer=True):
                if isinstance(component, nn.Module):
                    ident = id(component)
                    if ident not in seen:
                        modules.append(component)
                        seen.add(ident)
        return modules

    @contextmanager
    def _sampling_eval_context(self):
        """
        Sampling context that restores module training flags after generation.
        """
        modules = self._iter_sampling_mode_modules()
        with sampling_eval_context(modules):
            yield

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
    ) -> SamplerOutput:
        if not self._is_initialized:
            raise RuntimeError("Actor not initialized. Call init() first.")

        if self._is_offloaded:
            self.onload()

        self._ensure_sampling_components()

        # Defaults from sampling config
        num_inference_steps = num_inference_steps or self._sampling_config.get("num_inference_steps", 50)
        guidance_scale = guidance_scale or self._sampling_config.get("guidance_scale", 7.5)
        height = height or self._sampling_config.get("height", 256)
        width = width or self._sampling_config.get("width", 256)
        num_frames = num_frames or self._sampling_config.get("num_frames", 16)

        # Sampling adapter (NFT old adapter support)
        sampling_adapter = kwargs.pop("sampling_adapter", None)
        if sampling_adapter is None:
            sampling_adapter = self._sampling_config.get("sampling_adapter")

        # Optional shared noise
        init_same_noise = kwargs.pop("init_same_noise", self._sampling_config.get("init_same_noise", False))
        num_samples_per_prompt = kwargs.pop(
            "num_samples_per_prompt",
            self._sampling_config.get("num_samples_per_prompt", 1),
        )

        # Seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(seed)

        with self._sampling_eval_context():
            # Encode prompts if needed
            if prompts is not None and prompt_embeds is None:
                encoded = self.encode_prompt(prompts)
                prompt_embeds = encoded.get("prompt_embeds")
                pooled_prompt_embeds = encoded.get("pooled_prompt_embeds", pooled_prompt_embeds)
                negative_prompt_embeds = encoded.get("negative_prompt_embeds")
                negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")
                if text_ids is None:
                    text_ids = encoded.get("text_ids")
            else:
                negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
                negative_pooled_prompt_embeds = kwargs.pop("negative_pooled_prompt_embeds", None)

            if sampling_adapter and self.model is not None:
                with switch_adapter(self.model, sampling_adapter):
                    output = self._sampler.sample(
                        prompts=prompts,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        encoder_attention_mask=encoder_attention_mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        generator=generator,
                        sde_indices=sde_indices,
                        text_ids=text_ids,
                        init_same_noise=init_same_noise,
                        num_samples_per_prompt=num_samples_per_prompt,
                        **kwargs,
                    )
            else:
                output = self._sampler.sample(
                    prompts=prompts,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    encoder_attention_mask=encoder_attention_mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    generator=generator,
                    sde_indices=sde_indices,
                    text_ids=text_ids,
                    init_same_noise=init_same_noise,
                    num_samples_per_prompt=num_samples_per_prompt,
                    **kwargs,
                )

        # Decode latents for reward if requested
        if decode_for_reward:
            try:
                decoded = self._decode_latents(output.latents)
                decoded_images = self._tensor_to_pil(decoded)
                output = SamplerOutput(
                    latents=output.latents,
                    timesteps=output.timesteps,
                    trajectories=output.trajectories,
                    log_probs=output.log_probs,
                    embeddings=output.embeddings,
                    decoded_images=decoded_images,
                    metadata=output.metadata,
                    contract_version=output.contract_version,
                    step_indices=output.step_indices,
                )
            except Exception as e:
                logger.warning(f"Failed to decode latents: {e}")

        # Move tensors to CPU for Ray serialization
        output = output.to_device("cpu")
        return output

    def generate_batch(self, requests: List[InferenceRequest]) -> List[SamplerOutput]:
        outputs = []
        for request in requests:
            output = self.generate(
                prompts=request.prompts,
                prompt_embeds=request.prompt_embeds,
                pooled_prompt_embeds=request.pooled_prompt_embeds,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )
            outputs.append(output)
        return outputs

    def sample_batch(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> SamplerOutput:
        """Control-plane sampling RPC boundary."""
        return self.generate(prompts=prompts, **kwargs)

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

        # Validate batch type BEFORE calling methods (to give helpful error)
        if not isinstance(batch, (GRPOTrainingBatch, NFTTrainingBatch)):
            raise TypeError(
                f"Unsupported batch type: {type(batch).__name__}. "
                f"Expected GRPOTrainingBatch or NFTTrainingBatch. "
                f"Legacy dict format is not supported - use typed batches."
            )

        # Shard batch by rank to restore data-parallel semantics
        world_size = max(1, getattr(self, "world_size", 1))
        if world_size > 1 and not getattr(batch, "is_partitioned", False):
            batch_size = batch.batch_size
            per_rank = batch_size // world_size
            remainder = batch_size % world_size

            if per_rank == 0:
                logger.error(
                    f"Rank {self.rank}: batch_size={batch_size} too small for world_size={world_size}; skipping train step"
                )
                self._log_gpu_state("training_train_skipped")
                return {
                    "loss": 0.0,
                    "grad_norm": 0.0,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "rollout_id": rollout_id,
                    "skipped": True,
                }

            if remainder != 0 and self.rank == 0:
                logger.warning(
                    "Batch size %d not divisible by world_size %d; dropping %d samples for even sharding",
                    batch_size,
                    world_size,
                    remainder,
                )

            start = self.rank * per_rank
            end = start + per_rank
            batch = batch.slice(start, end)

        # Move batch to device using typed interface
        batch = batch.to_device(self._device)
        self._log_gpu_state("training_after_batch_to_device")
        if isinstance(batch, GRPOTrainingBatch):
            batch = self._maybe_replay_old_log_probs(batch)

        # Validate batch
        batch.validate()

        if isinstance(batch, NFTTrainingBatch) and self._gradient_accumulation_steps > 1 and self._nft_timestep_mode != "all":
            logger.warning(
                f"gradient_accumulation_steps={self._gradient_accumulation_steps} "
                "is ignored for NFT loss (single forward pass)"
            )

        inner_metrics: List[Dict[str, Any]] = []
        total_timesteps = 0
        last_grad_accum = 1

        for inner_epoch_id in range(self._num_inner_epochs):
            # Handle different batch types using isinstance dispatch
            has_backward = False

            if isinstance(batch, NFTTrainingBatch):
                # NFT: random timestep (single pass) or all timesteps (DiffusionNFT style)
                self.optimizer.zero_grad()
                total_loss, all_metrics, num_timesteps, actual_grad_accum, has_backward = \
                    self._train_nft_typed(batch)
                if not has_backward:
                    total_loss.backward()
                    has_backward = True
            else:
                # Must be GRPOTrainingBatch (type already validated above)
                # GRPO: iterate over SDE timesteps with gradient accumulation
                total_loss, all_metrics, num_timesteps, actual_grad_accum, has_backward = \
                    self._train_grpo_with_accumulation(batch)

            # Conditional optimizer step (only if we had valid backwards)
            if has_backward:
                # Gradient clipping (using configurable max_grad_norm)
                if self._use_fsdp:
                    grad_norm = self.model.clip_grad_norm_(self._max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self._max_grad_norm,
                    )

                # Optimizer step
                self.optimizer.step()

                # Scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            else:
                grad_norm = 0.0
                logger.warning("No valid timesteps to train, skipping optimizer step")

            # EMA update (for NFT dual adapter mechanism)
            if self._use_ema and self._ema_updater is not None:
                ema_success = self._ema_updater.update(self.model)
                all_metrics["ema_updated"] = ema_success

            last_grad_accum = actual_grad_accum
            total_timesteps += num_timesteps

            step_metrics = {
                "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": self.optimizer.param_groups[0]["lr"],
                "num_timesteps_trained": num_timesteps,
                "gradient_accumulation_steps": actual_grad_accum,  # Actual used value
                **all_metrics,
            }
            inner_metrics.append(step_metrics)

        metrics = self._aggregate_numeric_metrics(inner_metrics)
        metrics.update(
            {
                "rollout_id": rollout_id,
                "loss_type": self._loss_type,
                "num_inner_epochs": self._num_inner_epochs,
                "num_timesteps_trained": total_timesteps,
                "gradient_accumulation_steps": last_grad_accum,
                "effective_gradient_accumulation_steps": last_grad_accum * self._num_inner_epochs,
            }
        )
        sampling_weight_version = getattr(batch, "sampling_weight_version", None)
        if sampling_weight_version is not None:
            metrics["sampling_weight_version"] = int(sampling_weight_version)

        self._log_gpu_state("training_train_end")
        return metrics

    def _aggregate_numeric_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate numeric metrics from repeated inner-epoch updates."""
        aggregated: Dict[str, float] = {}
        if not metrics_list:
            return aggregated

        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        for key in all_keys:
            values: List[float] = []
            for metrics in metrics_list:
                if key not in metrics:
                    continue
                value = metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.mean().item()
                if isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def _resolve_grad_accum(self, training_config: dict) -> int:
        """Compute gradient accumulation steps with optional auto mode."""
        raw = training_config.get("gradient_accumulation_steps", 1)
        if isinstance(raw, int):
            return max(1, raw)
        if isinstance(raw, str) and raw.lower() == "auto":
            prompts_per_batch = training_config.get("prompts_per_batch", 1)
            k = training_config.get("num_samples_per_prompt", 1)
            world_size = training_config.get("world_size", 1)
            batch_size = training_config.get("batch_size", 1)
            grad_steps_per_epoch = training_config.get("gradient_steps_per_epoch", 1)
            try:
                total_gen = prompts_per_batch * k
                target_per_update = total_gen / max(1, grad_steps_per_epoch)
                denom = batch_size * max(1, world_size)
                accum = int((target_per_update + denom - 1) // denom)
                accum = max(1, accum)
                logger.info(
                    "Auto gradient_accumulation_steps=%d (prompts_per_batch=%d, k=%d, world_size=%d, batch_size=%d, grad_steps_per_epoch=%d)",
                    accum, prompts_per_batch, k, world_size, batch_size, grad_steps_per_epoch
                )
                return accum
            except Exception as e:
                logger.warning(f"Auto gradient accumulation failed ({e}), fallback to 1")
                return 1
        try:
            return max(1, int(raw))
        except Exception:
            logger.warning(f"Invalid gradient_accumulation_steps={raw}, fallback to 1")
            return 1

    def _train_grpo_with_accumulation(self, batch: GRPOTrainingBatch) -> tuple:
        """
        Train using GRPO loss with gradient accumulation over micro-batches.

        This implements the micro-batch gradient accumulation pattern:
        1. Split batch into micro-batches based on gradient_accumulation_steps
        2. Forward/backward on each micro-batch with scaled loss
        3. Accumulate gradients across micro-batches
        4. Single optimizer.step() at the end

        Reference: DanceGRPO/fastvideo/train_grpo_flux.py:603-613

        Args:
            batch: GRPOTrainingBatch with full batch data

        Returns:
            Tuple of (avg_loss, all_metrics, num_timesteps, actual_grad_accum, has_backward)
            - num_timesteps: Actual SDE timesteps per sample (NOT multiplied by micro-batches)
            - actual_grad_accum: Actual gradient accumulation steps used
            - has_backward: Whether any backward pass was performed
        """
        batch_size = batch.batch_size
        grad_accum = self._gradient_accumulation_steps

        # Validate batch_size is divisible by gradient_accumulation_steps
        if batch_size % grad_accum != 0:
            logger.warning(
                f"batch_size {batch_size} not divisible by gradient_accumulation_steps {grad_accum}, "
                f"falling back to grad_accum=1"
            )
            grad_accum = 1

        micro_batch_size = batch_size // grad_accum
        num_micro_batches = grad_accum

        self.optimizer.zero_grad()
        total_loss_accum = 0.0
        has_backward = False

        # Calculate actual timesteps per sample (not multiplied by micro-batches).
        available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
        valid_step_indices = sorted(int(i) for i in batch.sde_indices if int(i) in available_steps)
        num_timesteps_per_sample = len(valid_step_indices)
        if num_timesteps_per_sample == 0:
            logger.warning("No valid SDE timesteps in batch, skipping GRPO backward")
            return 0.0, {}, 0, grad_accum, False

        # Collect metrics from each micro-batch for proper aggregation
        micro_batch_metrics_list: List[Dict[str, Any]] = []

        for mbs_id in range(num_micro_batches):
            # Slice batch to get micro-batch
            start = mbs_id * micro_batch_size
            end = start + micro_batch_size
            micro_batch = batch.slice(start, end)

            # Process micro-batch over all timesteps with immediate backward.
            # This avoids retaining a full-timestep autograd graph in memory.
            micro_loss_sum = 0.0
            micro_metrics: Dict[str, Any] = {}
            metric_sums: Dict[str, float] = {}
            metric_counts: Dict[str, int] = {}

            for t_idx in valid_step_indices:
                loss_t, metrics_t = self._compute_grpo_timestep_loss(micro_batch, t_idx)
                scaled_loss = loss_t / (num_micro_batches * num_timesteps_per_sample)
                scaled_loss.backward()
                has_backward = True
                micro_loss_sum += loss_t.detach().item()

                # Keep per-timestep metrics for debug plus aggregated metrics for logging.
                for key, value in metrics_t.items():
                    val = value.item() if isinstance(value, torch.Tensor) else value
                    metric_key = f"t{t_idx}_{key}"
                    if metric_key not in micro_metrics:
                        micro_metrics[metric_key] = val
                    if isinstance(val, (int, float)):
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(val)
                        metric_counts[key] = metric_counts.get(key, 0) + 1

            for key, total in metric_sums.items():
                count = metric_counts.get(key, 0)
                if count > 0:
                    micro_metrics[key] = total / count

            micro_loss = micro_loss_sum / num_timesteps_per_sample
            micro_batch_metrics_list.append(micro_metrics)
            total_loss_accum += micro_loss

        # Aggregate metrics: average numeric values across micro-batches
        all_metrics: Dict[str, Any] = {}
        if micro_batch_metrics_list:
            keys = micro_batch_metrics_list[0].keys()
            for key in keys:
                values = [m.get(key) for m in micro_batch_metrics_list if m.get(key) is not None]
                if values and isinstance(values[0], (int, float)):
                    all_metrics[key] = sum(values) / len(values)
                else:
                    # Non-numeric: take last micro-batch's value
                    all_metrics[key] = micro_batch_metrics_list[-1].get(key)

        # Average loss for logging
        avg_loss = total_loss_accum / num_micro_batches if num_micro_batches > 0 else 0.0

        return avg_loss, all_metrics, num_timesteps_per_sample, grad_accum, has_backward

    def _train_grpo_typed(self, batch: GRPOTrainingBatch) -> tuple:
        """
        Train using GRPO loss with typed batch (iterate over timesteps).

        This is the core training logic for a single batch. Does NOT call
        backward() - that's handled by the caller for gradient accumulation.

        Args:
            batch: GRPOTrainingBatch to train on

        Returns:
            Tuple of (total_loss, all_metrics, num_timesteps)
        """
        total_loss = torch.tensor(0.0, device=self._device, requires_grad=True)
        all_metrics: Dict[str, Any] = {}
        num_timesteps = 0
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        # Iterate over SDE timesteps
        available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
        for t_idx in sorted(int(i) for i in batch.sde_indices if int(i) in available_steps):

            loss_t, metrics_t = self._compute_grpo_timestep_loss(batch, t_idx)

            total_loss = total_loss + loss_t
            num_timesteps += 1

            # Store per-timestep metrics
            for key, value in metrics_t.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                if f"t{t_idx}_{key}" not in all_metrics:
                    all_metrics[f"t{t_idx}_{key}"] = val
                if isinstance(val, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(val)
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        # Average loss over timesteps
        if num_timesteps > 0:
            total_loss = total_loss / num_timesteps
            for key, value in metric_sums.items():
                count = metric_counts.get(key, 0)
                if count > 0:
                    all_metrics[key] = value / count

        return total_loss, all_metrics, num_timesteps

    def _compute_grpo_timestep_loss(
        self,
        batch: GRPOTrainingBatch,
        timestep_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute GRPO loss and metrics for one timestep."""
        timestep_data = batch.get_timestep_data_by_step(timestep_idx)

        if hasattr(self.loss_fn, "compute_timestep"):
            return self.loss_fn.compute_timestep(
                model=self.model,
                timestep_data=timestep_data,
                advantages=batch.advantages,
                embeddings=batch.embeddings,
                guidance_scale=self._guidance_scale,
            )

        return self.loss_fn.compute(
            model=self.model,
            samples=batch.to_loss_dict(),
            timestep_idx=batch.get_position_for_step(timestep_idx),
            advantages=batch.advantages,
            prompt_embeds=batch.embeddings.prompt_embeds,
            pooled_prompt_embeds=batch.embeddings.pooled_prompt_embeds,
        )

    def _train_nft_typed(self, batch: NFTTrainingBatch) -> tuple:
        """Train using NFT loss with typed batch."""
        # All-timestep mode: iterate through provided timesteps with accumulation
        if self._nft_timestep_mode == "all" and batch.timesteps is not None:
            timesteps = batch.timesteps
            if isinstance(timesteps, torch.Tensor):
                timesteps = timesteps.detach()
            else:
                timesteps = torch.tensor(timesteps, device=batch.advantages.device)

            # Flatten and drop terminal t=0 if present (sampler schedules often include it)
            timesteps = timesteps.flatten()
            if timesteps.numel() > 1 and torch.isclose(
                timesteps[-1],
                torch.zeros((), device=timesteps.device, dtype=timesteps.dtype),
                atol=1e-8,
            ).item():
                timesteps = timesteps[:-1]

            if timesteps.numel() == 0:
                logger.warning("NFT all-timestep mode: empty timesteps, falling back to random mode")
            else:
                if self._nft_shuffle_timesteps:
                    perm = torch.randperm(timesteps.numel(), device=timesteps.device)
                    timesteps = timesteps[perm]

                effective_grad_accum = max(1, self._gradient_accumulation_steps) * timesteps.numel()
                total_loss = torch.zeros((), device=batch.advantages.device)
                metrics_sum: Dict[str, float] = {}

                for t in timesteps:
                    if hasattr(self.loss_fn, "compute_batch"):
                        loss, metrics = self.loss_fn.compute_batch(
                            model=self.model,
                            batch=batch,
                            timestep_values=t,
                            apply_shift=self._nft_apply_shift,
                        )
                    else:
                        loss, metrics = self.loss_fn.compute(
                            model=self.model,
                            samples=batch.to_loss_dict(),
                            timestep_idx=0,
                            advantages=batch.advantages,
                            prompt_embeds=batch.embeddings.prompt_embeds,
                            pooled_prompt_embeds=batch.embeddings.pooled_prompt_embeds,
                            text_ids=batch.embeddings.text_ids,
                            image_ids=batch.embeddings.image_ids,
                            timestep_values=t,
                            apply_shift=self._nft_apply_shift,
                        )

                    (loss / effective_grad_accum).backward()
                    total_loss = total_loss + loss.detach()

                    for key, value in metrics.items():
                        metric_val = value.item() if isinstance(value, torch.Tensor) else float(value)
                        metrics_sum[key] = metrics_sum.get(key, 0.0) + metric_val

                # Average metrics across timesteps
                all_metrics: Dict[str, Any] = {}
                for key, value in metrics_sum.items():
                    all_metrics[key] = value / timesteps.numel()

                return (
                    total_loss / timesteps.numel(),
                    all_metrics,
                    timesteps.numel(),
                    effective_grad_accum,
                    True,
                )

        # Random timestep mode (single pass)
        if hasattr(self.loss_fn, "compute_batch"):
            loss, metrics = self.loss_fn.compute_batch(
                model=self.model,
                batch=batch,
            )
        else:
            # Fall back to legacy dict interface
            loss, metrics = self.loss_fn.compute(
                model=self.model,
                samples=batch.to_loss_dict(),
                timestep_idx=0,
                advantages=batch.advantages,
                prompt_embeds=batch.embeddings.prompt_embeds,
                pooled_prompt_embeds=batch.embeddings.pooled_prompt_embeds,
                text_ids=batch.embeddings.text_ids,
                image_ids=batch.embeddings.image_ids,
            )

        # Convert tensor metrics to Python floats
        all_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                all_metrics[key] = value.item()
            else:
                all_metrics[key] = value

        return loss, all_metrics, 1, 1, False

    # Note: Legacy dict methods (_train_grpo, _train_nft, _move_batch_to_device,
    # _get_timestep_samples) have been removed. Use typed batches (GRPOTrainingBatch,
    # NFTTrainingBatch) instead.

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current model weights for syncing to inference actors.

        IMPORTANT: For FSDP, this method MUST be called on ALL ranks simultaneously
        because state_dict() triggers an ALLGATHER collective. Only rank 0 returns
        the full state dict (rank0_only=True), other ranks return empty dict after
        participating in the collective.
        """
        # If model is offloaded to CPU, temporarily load it back to GPU
        # FSDP requires the model to be on compute device for state_dict()
        was_offloaded = self._is_offloaded
        if was_offloaded:
            self.onload()

        try:
            # LoRA-only sync path (reduce transfer size). Falls back to full sync on failure.
            if self._use_lora:
                try:
                    if self._use_fsdp:
                        # FSDP requires collective full state dict; filter to LoRA keys afterward.
                        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                            full_state_dict = self.model.state_dict()

                        if self.rank != 0:
                            return {}

                        lora_state = {
                            k: (v.cpu() if v.is_cuda else v)
                            for k, v in full_state_dict.items()
                            if "lora" in k.lower()
                        }
                        if lora_state:
                            return lora_state
                        logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")

                        full_state_dict = {k: v.cpu() if v.is_cuda else v for k, v in full_state_dict.items()}
                        return full_state_dict
                    else:
                        # Non-FSDP: try PEFT helper if available
                        try:
                            from peft.utils import get_peft_model_state_dict

                            base_model = self.model
                            if hasattr(base_model, "module"):
                                base_model = base_model.module

                            adapter_names = []
                            if hasattr(base_model, "peft_config"):
                                adapter_names = list(base_model.peft_config.keys())
                            if not adapter_names:
                                adapter_names = [getattr(base_model, "active_adapter", "default")]

                            lora_state = {}
                            for adapter_name in adapter_names:
                                lora_state.update(
                                    get_peft_model_state_dict(base_model, adapter_name=adapter_name)
                                )

                            if lora_state:
                                return {k: v.cpu() if v.is_cuda else v for k, v in lora_state.items()}
                        except Exception as e:
                            logger.warning(f"PEFT LoRA-only sync failed; falling back to key filter: {e}")

                        # Fallback: filter LoRA keys from local state_dict
                        local_state = self.model.state_dict()
                        lora_state = {
                            k: (v.cpu() if v.is_cuda else v)
                            for k, v in local_state.items()
                            if "lora" in k.lower()
                        }
                        if lora_state:
                            return lora_state
                        logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")

                except Exception as e:
                    logger.warning(f"LoRA-only sync failed; falling back to full sync: {e}")

            if self._use_fsdp:
                # For FSDP, we need to gather the full state dict
                # This is a collective operation - all ranks must participate!
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = self.model.state_dict()

                # Only rank 0 has the full state dict, others have empty dict
                if self.rank != 0:
                    return {}

                # Explicitly move to CPU for Ray serialization
                # (offload_to_cpu=True may not always work as expected)
                state_dict = {k: v.cpu() if v.is_cuda else v for k, v in state_dict.items()}
            else:
                # Move weights to CPU for serialization across Ray actors
                # This is necessary because RolloutManager may be CPU-only
                state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

            return state_dict
        finally:
            # Restore offload state if it was offloaded before
            if was_offloaded:
                self.offload()

    def export_weights_to_path(self, checkpoint_path: str) -> Optional[str]:
        """
        Export synchronized weights to a shared path.

        All ranks should call this method. Only rank 0 writes the file.
        """
        state_dict = self.get_weights()
        if self.rank != 0:
            return None
        return publish_checkpoint_atomic(state_dict, checkpoint_path)

    def update_weights(self) -> None:
        """Broadcast weights from rank 0 to all other ranks."""
        if self._use_fsdp:
            # FSDP handles synchronization internally
            return

        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def save_model(self, path: str) -> None:
        """
        Save model checkpoint.

        IMPORTANT: For FSDP, this method MUST be called on ALL ranks simultaneously
        because state_dict() triggers an ALLGATHER collective. Only rank 0 actually
        saves the checkpoint, but all ranks must participate in the collective.
        """
        # If model is offloaded to CPU, temporarily load it back to GPU
        # FSDP requires the model to be on compute device for state_dict()
        was_offloaded = self._is_offloaded
        if was_offloaded:
            self.onload()

        try:
            if self._use_fsdp:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                # All ranks must participate in the collective
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = self.model.state_dict()

                # Only rank 0 saves the checkpoint
                if self.rank != 0:
                    return
            else:
                if self.rank != 0:
                    return
                state_dict = self.model.state_dict()

            os.makedirs(path, exist_ok=True)

            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }

            if self.lr_scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

            torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
            logger.info(f"Checkpoint saved to {path}")
        finally:
            # Restore offload state if it was offloaded before
            if was_offloaded:
                self.offload()

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        if self._use_fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
                self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {path}")

    def offload(self) -> None:
        """Offload model and optimizer to CPU."""
        # When using FSDP CPU offload, FSDP manages device placement automatically
        # Don't manually move model to CPU as it's already managed by FSDP
        if getattr(self, '_fsdp_cpu_offload', False):
            # Still offload auxiliary components (text encoders/vae) if present
            self._move_aux_components("cpu", include_transformer=False)
            self._is_offloaded = True
            clear_memory()
            logger.info(f"Rank {self.rank}: FSDP CPU offload mode - just clearing cache")
            return

        self._is_offloaded = True

        # Offload auxiliary components if they were loaded for sampling
        self._move_aux_components("cpu", include_transformer=False)

        if self.model is not None:
            self.model.to("cpu")

        # Offload optimizer states
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        clear_memory()
        logger.info(f"Rank {self.rank}: Model and optimizer offloaded to CPU")
        self._log_gpu_state("training_offload")

    def onload(self) -> None:
        """Load model and optimizer back to GPU."""
        # When using FSDP CPU offload, FSDP manages device placement automatically
        # Don't manually move model to GPU as it conflicts with FSDP's offloading
        if getattr(self, '_fsdp_cpu_offload', False):
            if self._device is not None:
                self._move_aux_components(self._device, include_transformer=False)
            self._is_offloaded = False
            logger.info(f"Rank {self.rank}: FSDP CPU offload mode - skipping manual onload")
            return

        if self.model is not None:
            self.model.to(self._device)

        # Load auxiliary components back to GPU
        if self._device is not None:
            self._move_aux_components(self._device, include_transformer=False)

        # Load optimizer states back to GPU
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._device)

        self._is_offloaded = False
        logger.info(f"Rank {self.rank}: Model and optimizer loaded to GPU")
        self._log_gpu_state("training_onload")

    def clear_memory(self) -> None:
        """Clear GPU cache without offloading."""
        torch.cuda.empty_cache()
        logger.debug(f"Rank {self.rank}: GPU cache cleared")

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
