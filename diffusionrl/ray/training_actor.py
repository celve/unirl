"""
diffusionrl Training Actor - Manages model training via pluggable train backends.
"""
import importlib
import logging
import os
import socket
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn

from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.distributed.weight_sync_checkpoint import (
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
)
from diffusionrl.models import create_model_bundle_from_init_payload
from diffusionrl.patches.replay_logprob import ReplayLogProbPatch
from diffusionrl.ray.actor_config import TrainingActorConfig
from diffusionrl.ray.training_actor_sampling import ActorSamplingExecutor
from diffusionrl.reward.actor_local import ActorLocalRewardPrecompute
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.samplers.fsdp.sampler_runner import finalize_sampling_output
from diffusionrl.training import (
    TrainExecutor,
    TrainExecutorConfig,
    TrainingWorkflow,
    create_train_backend_from_init_payload,
)
from diffusionrl.types.sampling import LogProbData, RolloutRequest, RolloutSamples
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import clear_memory as _clear_gpu_memory
from diffusionrl.utils.dtypes import inject_model_dtype_kwarg, parse_torch_dtype

from .actor_base import BaseTrainRayActor, log_gpu_state, log_resource_ids

logger = logging.getLogger(__name__)


def _build_synthetic_debug_training_batch(
    *,
    model: nn.Module,
    algorithm: Any,
    model_type: str,
    batch_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    latent_channels: int = 16,
    vae_scale_factor: int = 8,
    max_sequence_length: int = 256,
) -> TrainingBatch:
    """Build a conservative synthetic train_only batch for pre-release debugging.

    This path is intentionally narrow: only image-style SD3 training batches are
    synthesized. Other model families have materially different latent contracts
    (for example video layouts or FLUX-specific packed inputs), so train_only
    should use ``--debug.debug-load-path`` there rather than guessing.
    """
    from diffusionrl.types.training_batch import TrainingBatch
    from diffusionrl.types.trajectory_store import TrajectoryStore

    normalized_model_type = str(model_type or "").strip().lower()
    if normalized_model_type != "sd3":
        raise NotImplementedError(
            "Synthetic train_only batches are only supported for model_type='sd3'. "
            "Use --debug.debug-load-path with a saved rollout payload for other models."
        )

    config = getattr(model, "config", None)
    if config is not None:
        joint_dim = int(getattr(config, "joint_attention_dim", 4096))
        pooled_dim = int(getattr(config, "pooled_projection_dim", 2048))
        latent_channels = int(getattr(config, "in_channels", latent_channels))
    else:
        joint_dim, pooled_dim = 4096, 2048

    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32

    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    from diffusionrl.types.forward_context import get_forward_context_cls

    prompt_embeds = torch.randn(batch_size, max_sequence_length, joint_dim, dtype=dtype)
    pooled_prompt_embeds = torch.randn(batch_size, pooled_dim, dtype=dtype)
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

    try:
        ctx_cls = get_forward_context_cls(normalized_model_type)
    except KeyError:
        ctx_cls = get_forward_context_cls("default")
    from dataclasses import fields as _dc_fields

    valid_fields = {f.name for f in _dc_fields(ctx_cls)}
    ctx_kwargs = {
        "guidance_scale": 7.0,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
    }
    fwd_ctx = ctx_cls(**{k: v for k, v in ctx_kwargs.items() if k in valid_fields})

    advantages = torch.randn(batch_size, dtype=torch.float32)
    rewards = torch.randn(batch_size, dtype=torch.float32)
    prompts = [f"debug_prompt_{i}" for i in range(batch_size)]

    if bool(getattr(algorithm, "is_forward_process", lambda: False)()):
        timesteps = torch.linspace(
            1.0, 0.0, steps=num_inference_steps + 1, dtype=torch.float32
        )
        batch = TrainingBatch(
            trajectory_store=TrajectoryStore.from_clean_latents(
                torch.randn(
                    batch_size, latent_channels, latent_h, latent_w, dtype=dtype
                ),
                total_positions=num_inference_steps + 1,
            ),
            timesteps=timesteps,
            advantages=advantages,
            forward_context=fwd_ctx,
            rewards=rewards,
            prompts=prompts,
        )
        batch.validate()
        return batch

    step_indices = torch.arange(num_inference_steps + 1, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, steps=num_inference_steps + 1, dtype=torch.float32)
    batch = TrainingBatch(
        trajectory_store=TrajectoryStore.from_full(
            torch.randn(
                batch_size,
                num_inference_steps + 1,
                latent_channels,
                latent_h,
                latent_w,
                dtype=dtype,
            ),
        ),
        log_probs=LogProbData.from_dict(
            {
                step_idx: torch.randn(batch_size, dtype=torch.float32)
                for step_idx in range(num_inference_steps)
            }
        ),
        timesteps=timesteps,
        advantages=advantages,
        forward_context=fwd_ctx,
        rewards=rewards,
        prompts=prompts,
        step_indices=step_indices,
        target_sde_indices=set(range(num_inference_steps)),
    )
    batch.validate()
    return batch

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
        self.algorithm = None
        self._is_initialized = False
        self._is_offloaded = False
        self._device = None
        self._use_lora = False
        self._fsdp_cpu_offload = False
        self._train_backend = None
        self._train_backend_name = "fsdp"
        self._train_backend_capabilities: Dict[str, Any] = {}
        self._resolved_train_topology: Dict[str, Any] = {
            "actor_count": int(world_size),
            "world_size": int(world_size),
            "dp_size": int(world_size),
            "partition_mode": "data_parallel",
        }
        self._resolved_training_plan: Dict[str, Any] = {
            "global_batch_size": int(world_size),
            "local_batch_size": 1,
            "local_mini_batch_size": 1,
            "micro_batch_size": 1,
            "num_updates_per_batch": 1,
            "update_slices": [[0, 1]],
            "mini_batch_slices_per_update": [[[0, 1]]],
        }

        # EMA runtime ----------------------------------------------------------
        self._ema_manager = None
        self._algorithm_type = "grpo"
        self._algorithm_dotpath = None
        # Sample-level shuffle before training (analogous to Flow-Factory inner-epoch shuffle)
        self._shuffle_samples = True
        self._shuffle_seed: Optional[int] = None

        # Training config (read from config in init)
        self._max_grad_norm = 1.0
        self._micro_batch_size = 1
        self._local_mini_batch_size = 1
        self._num_updates_per_batch = 1
        self._replay_enabled = False

        # Sampling support (training-actor sampling mode)
        self._sampling_config: Dict[str, Any] = {}
        self._sampler = None
        self._actor_sampling_executor = ActorSamplingExecutor()
        self._replay_logprob_patch = ReplayLogProbPatch()
        self._sampling_ready = False
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self._weights_update_groups: Dict[str, Any] = {}
        self._update_weight_handler = None
        self._rollout_actors: List[Any] = []
        self._lora_initialized_on_rollout = False
        self._reward_config: Dict[str, Any] = {}
        self._reward_schema: Optional[RewardSchema] = None
        self._local_reward_runtime: Optional[ActorLocalRewardPrecompute] = None
        self._training_workflow = TrainingWorkflow()

    def _log_resource_ids(self, tag: str) -> None:
        log_resource_ids(tag, self.rank)

    def _log_gpu_state(self, tag: str) -> None:
        log_gpu_state(tag, self.rank, device=self._device, offloaded=self._is_offloaded)

    @staticmethod
    def _require_dict_config(config: TrainingActorConfig, key: str) -> Dict[str, Any]:
        value = getattr(config, key)
        if not isinstance(value, dict):
            raise ValueError(
                f"TrainingActor.init requires config.{key} dict, got: {type(value).__name__}"
            )
        return value

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
            importlib.import_module("sglang.srt.utils")
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
                importlib.import_module("sglang.srt.utils")
                return
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "Cannot import sglang.srt.utils for IPC/NCCL tensor sync. "
            "Set SGLANG_PYTHON_PATH to sglang/python."
        )

    def init(self, config: TrainingActorConfig) -> None:
        """
        Initialize training environment.

        Args:
            config: Typed TrainingActorConfig containing model/training/backend
                sub-config dicts plus algorithm_init_payload.

        Note:
            The driver-side rollout runtime and TrainingActor each instantiate
            the same Algorithm class locally for their own role. Training owns
            the train-side algorithm instance.
        """
        logger.info(f"Rank {self.rank}: Initializing training actor...")

        if not isinstance(config, TrainingActorConfig):
            raise ValueError(
                "TrainingActor.init requires TrainingActorConfig, "
                f"got: {type(config).__name__}"
            )

        train_backend_init_payload = config.train_backend_init_payload
        if not isinstance(train_backend_init_payload, ComponentInitPayload):
            raise ValueError(
                "TrainingActor.init requires config.train_backend_init_payload as "
                "ComponentInitPayload, "
                f"got: {type(train_backend_init_payload).__name__}"
            )
        self._train_backend = create_train_backend_from_init_payload(
            train_backend_init_payload
        )
        self._train_backend_name = self._train_backend.name
        self._train_backend_capabilities = self._train_backend.capabilities.as_dict()

        topology_config = self._require_dict_config(config, "topology_config")
        if int(topology_config["world_size"]) != int(self.world_size):
            raise ValueError(
                "TrainingActor topology_config.world_size must match the actor-group world_size. "
                f"Got topology_config.world_size={topology_config['world_size']}, "
                f"actor.world_size={self.world_size}."
            )
        self._resolved_train_topology = {
            key: int(value)
            if key
            in {
                "actor_count",
                "world_size",
                "dp_size",
                "dp_replicate_size",
                "dp_shard_size",
                "tp_size",
                "pp_size",
                "sp_size",
                "ep_size",
            }
            else value
            for key, value in dict(topology_config).items()
        }
        self._resolved_training_plan = {
            key: int(value)
            if key
            in {
                "global_batch_size",
                "local_batch_size",
                "local_mini_batch_size",
                "micro_batch_size",
                "num_updates_per_batch",
            }
            else value
            for key, value in dict(self._require_dict_config(config, "training_plan_config")).items()
        }

        # Initialize distributed
        self._init_distributed(backend=self._train_backend.capabilities.distributed_backend)

        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)

        # Backend-specific pre-load hook (e.g. CPU-offload behavior).
        self._train_backend.before_model_load(self)

        model_init_payload = config.model_init_payload
        self._load_model(model_init_payload)

        # Backend model wrapping (FSDP/Megatron/others).
        self._train_backend.wrap_model(self)

        optimizer_config = self._require_dict_config(config, "optimizer_config")
        self._create_optimizer(optimizer_config)

        scheduler_config = self._require_dict_config(config, "scheduler_config")
        self._create_scheduler(scheduler_config)

        # Load training config
        training_config = self._require_dict_config(config, "training_config")
        self._max_grad_norm = float(training_config["max_grad_norm"])
        self._micro_batch_size = int(
            self._resolved_training_plan["micro_batch_size"]
        )
        self._local_mini_batch_size = int(
            self._resolved_training_plan["local_mini_batch_size"]
        )
        self._num_updates_per_batch = int(
            self._resolved_training_plan["num_updates_per_batch"]
        )
        self._replay_enabled = bool(training_config["replay_enabled"])
        self._algorithm_type = str(training_config["algorithm_type"])
        self._guidance_scale = float(training_config["guidance_scale"])
        self._shuffle_samples = bool(training_config.get("shuffle_samples", True))
        raw_shuffle_seed = training_config.get("shuffle_seed", None)
        self._shuffle_seed = (
            int(raw_shuffle_seed) if raw_shuffle_seed is not None else None
        )

        algorithm_init_payload = config.algorithm_init_payload
        self._load_algorithm(
            algorithm_init_payload,
            training_autocast_precision=str(
                training_config.get("training_autocast_precision", "bf16")
            ),
            debug_output_dir=training_config.get("debug_output_dir"),
        )

        # Sampling config (used when training actors perform sampling)
        sampling_config = self._require_dict_config(config, "sampling_config")
        self._sampling_config = sampling_config
        reward_config = self._require_dict_config(config, "reward_config")
        self._reward_config = reward_config
        self._reward_schema = RewardSchema(**reward_config)

        self._is_initialized = True
        logger.info(
            f"Rank {self.rank}: Training actor initialized "
            f"(backend={self._train_backend_name}, "
            f"max_grad_norm={self._max_grad_norm}, "
            f"micro_batch_size={self._micro_batch_size}, "
            f"local_mini_batch_size={self._local_mini_batch_size}, "
            f"num_updates_per_batch={self._num_updates_per_batch}, "
            f"training_plan={self._resolved_training_plan})"
        )
        self._log_resource_ids("training_init")
        self._log_gpu_state("training_init")

    def _uses_rollout_local_reward(self) -> bool:
        # Local reward follows the active sampling host rather than the
        # training/update path itself. In direct actor-sampling mode the
        # TrainingActor temporarily acts as that host, so it needs the same
        # actor-local reward attach step that dedicated rollout actors use.
        return bool(self._reward_schema is not None and self._reward_schema.uses_sampling_actor_execution)

    def _ensure_local_reward_runtime(self) -> ActorLocalRewardPrecompute:
        if not self._uses_rollout_local_reward():
            raise RuntimeError("Local reward runtime requested but reward_location!='sampling_actor'.")
        if self._local_reward_runtime is None:
            self._local_reward_runtime = ActorLocalRewardPrecompute(self._reward_schema)
        return self._local_reward_runtime

    def _attach_local_reward_to_output(
        self,
        *,
        output: RolloutSamples,
        prompts: List[str],
        prompt_ids: Optional[List[str]],
        sample_ids: Optional[List[str]],
        group_ids: Optional[List[str]],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
        collect_media_preview: bool,
        samples_per_prompt: int,
    ) -> RolloutSamples:
        if not self._uses_rollout_local_reward():
            return output
        return self._ensure_local_reward_runtime().attach_to_output(
            output=output,
            prompts=list(prompts),
            prompt_ids=prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
            collect_media_preview=collect_media_preview,
            samples_per_prompt=int(samples_per_prompt),
        )

    def _load_model(self, model_init_payload: ComponentInitPayload) -> None:
        """Load the model for training."""
        from diffusionrl.models.config import ModelBundleConfig

        model_config = model_init_payload.component_config
        if not isinstance(model_config, ModelBundleConfig):
            raise ValueError(
                "model_init_payload.component_config must be a ModelBundleConfig, "
                f"got: {type(model_config).__name__}"
            )

        runtime_model_config = replace(
            model_config,
            device=self._device,
            training_only=True,
            skip_device_move=getattr(self, "_fsdp_cpu_offload", False),
        )

        use_lora = bool(runtime_model_config.use_lora)
        self._use_lora = use_lora
        runtime_model_init_payload = replace(
            model_init_payload,
            component_config=runtime_model_config,
        )

        self.model_bundle = create_model_bundle_from_init_payload(
            runtime_model_init_payload
        )

        # Get the transformer for training
        # Note: Device placement is handled by model bundle based on skip_device_move flag
        self.model = self.model_bundle.transformer
        self.model.train()

        # Enable gradient checkpointing (activation checkpointing) only when explicitly requested
        # Note: For LoRA models, this should be done in the model bundle before PEFT wrapping
        use_gradient_checkpointing = bool(
            runtime_model_config.use_gradient_checkpointing
        )
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

        logger.info(
            "Rank %s: Model loaded (lora_rank=%s, training_only=True)",
            self.rank,
            runtime_model_config.lora_rank if use_lora else "N/A",
        )

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

        lr = float(optimizer_config["learning_rate"])
        betas = (
            float(optimizer_config["adam_beta1"]),
            float(optimizer_config["adam_beta2"]),
        )
        eps = float(optimizer_config["adam_epsilon"])
        weight_decay = float(optimizer_config["weight_decay"])

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        logger.info(f"Rank {self.rank}: Optimizer created (lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})")

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

        scheduler_type = str(scheduler_config["type"])
        warmup_steps = int(scheduler_config["warmup_steps"])
        total_steps = int(scheduler_config["total_steps"])

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

    def _load_algorithm(
        self,
        algorithm_init_payload: ComponentInitPayload,
        *,
        training_autocast_precision: str,
        debug_output_dir: Optional[str],
    ) -> None:
        """Load the train-side Algorithm instance."""
        if not isinstance(algorithm_init_payload, ComponentInitPayload):
            raise ValueError(
                "TrainingActor.init requires config.algorithm_init_payload as ComponentInitPayload, "
                f"got: {type(algorithm_init_payload).__name__}"
            )

        self._algorithm_dotpath = str(algorithm_init_payload.component_dotpath or "")

        if not self._algorithm_dotpath:
            raise ValueError(
                "algorithm_init_payload.component_dotpath must be set before TrainingActor.init."
            )

        self.algorithm = create_algorithm_from_init_payload(algorithm_init_payload)

        logger.info(
            "Rank %s: Train-side algorithm loaded from %s (type=%s)",
            self.rank, self._algorithm_dotpath, type(self.algorithm).__name__,
        )

        if hasattr(self.algorithm, "model_type"):
            self.algorithm.model_type = getattr(self.model_bundle, "model_type", "default")
        if hasattr(self.algorithm, "_forward_plugin"):
            forward_plugin_fn = getattr(self.model_bundle.__class__, "forward_plugin", None)
            if not callable(forward_plugin_fn):
                raise ValueError(
                    f"Model bundle {self.model_bundle.__class__.__name__} must define "
                    "classmethod forward_plugin() for algorithm forward dispatch."
                )
            forward_plugin = forward_plugin_fn()
            if forward_plugin is None:
                raise ValueError(
                    f"Model bundle {self.model_bundle.__class__.__name__}.forward_plugin() "
                    "returned None; expected a forward plugin instance."
                )
            self.algorithm._forward_plugin = forward_plugin
            if hasattr(forward_plugin, "autocast_dtype"):
                forward_plugin.autocast_dtype = parse_torch_dtype(
                    training_autocast_precision,
                    field_name="training_config.training_autocast_precision",
                )
                logger.info(
                    "Rank %s: forward_plugin.autocast_dtype set from "
                    "training_config.training_autocast_precision=%s -> %s",
                    self.rank,
                    training_autocast_precision,
                    forward_plugin.autocast_dtype,
                )

        if debug_output_dir and hasattr(self.algorithm, "_debug_output_dir"):
            self.algorithm._debug_output_dir = debug_output_dir
            logger.info("Rank %s: Debug output dir set on algorithm: %s", self.rank, debug_output_dir)

        logger.info(
            "Rank %s: Algorithm loaded (type=%s, path=%s)",
            self.rank,
            self._algorithm_type,
            self._algorithm_dotpath,
        )

        from diffusionrl.utils.ema import EMAManager

        ema_spec = self.algorithm.get_ema_spec()
        self._ema_manager = EMAManager.from_model_and_spec(
            model=self.model,
            spec=ema_spec,
            use_lora=self._use_lora,
            uses_sharded_model=bool(self._train_backend and self._train_backend.uses_sharded_model()),
            algorithm=self.algorithm,
        )

    def apply_eval_ema(self) -> bool:
        """Swap eval-EMA weights into the model for evaluation."""
        if self._ema_manager is None:
            return False
        applied = self._ema_manager.apply_eval_ema(self.model)
        if applied:
            logger.debug("Rank %s: EMA weights applied for eval", self.rank)
        return applied

    def restore_from_eval(self) -> bool:
        """Restore training weights after evaluation."""
        if self._ema_manager is None:
            return False
        restored = self._ema_manager.restore_from_eval(self.model)
        if restored:
            logger.debug("Rank %s: Training weights restored after eval", self.rank)
        return restored

    def _maybe_replay_old_log_probs(self, batch: TrainingBatch) -> TrainingBatch:
        return self._replay_logprob_patch.maybe_replay_old_log_probs(
            batch=batch,
            enabled=self._replay_enabled,
            algorithm_type=self._algorithm_type,
            sampling_config=self._sampling_config,
            model_bundle=self.model_bundle,
            model=self.model,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
        )

    def generate(self, request: RolloutRequest) -> RolloutSamples:
        output = self._actor_sampling_executor.generate_raw(self, request)
        prompts = request.prompts
        return finalize_sampling_output(
            output=output,
            request=request,
            host_label="training-actor sampling",
            decode_latents_fn=lambda latents: self._actor_sampling_executor.decode_latents(self, latents),
            local_reward_attach_fn=(
                lambda current_output: self._attach_local_reward_to_output(
                    output=current_output,
                    prompts=prompts,
                    prompt_ids=request.meta.get("prompt_ids"),
                    sample_ids=request.meta.get("sample_ids"),
                    group_ids=request.meta.get("group_ids"),
                    prompt_metadata=request.meta.get("prompt_metadata"),
                    collect_media_preview=bool(
                        request.sampling.get("collect_media_preview", False)
                    ),
                    samples_per_prompt=max(
                        1, int(request.sampling.get("samples_per_prompt", 1))
                    ),
                )
            ),
            move_output_to_cpu=True,
        )

    def get_train_backend_info(self) -> Dict[str, Any]:
        if self._train_backend is None:
            return {
                "name": self._train_backend_name,
                "capabilities": dict(self._train_backend_capabilities),
                "topology": dict(self._resolved_train_topology),
                "training_plan": dict(self._resolved_training_plan),
            }
        info = dict(self._train_backend.backend_info(self))
        info["topology"] = dict(self._resolved_train_topology)
        info["training_plan"] = dict(self._resolved_training_plan)
        return info

    def get_expected_global_batch_size(self) -> int:
        return int(self._resolved_training_plan["global_batch_size"])

    def _build_train_executor(self) -> TrainExecutor:
        dp_size = int(self._resolved_train_topology["dp_size"])
        config = TrainExecutorConfig(
            rank=self.rank,
            dp_size=dp_size,
            device=self._device,
            use_fsdp=bool(self._train_backend and self._train_backend.uses_sharded_model()),
            algorithm_type=self._algorithm_type,
            max_grad_norm=self._max_grad_norm,
            micro_batch_size=self._micro_batch_size,
            local_mini_batch_size=self._local_mini_batch_size,
            num_updates_per_batch=self._num_updates_per_batch,
            training_plan=dict(self._resolved_training_plan),
            ema_manager=self._ema_manager,
            shuffle_samples=self._shuffle_samples,
            shuffle_seed=self._shuffle_seed,
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
            algorithm=self.algorithm,
            config=config,
        )

    def train(
        self,
        rollout_id: int,
        training_data_handle_or_batch: Union[ray.ObjectRef, TrainingBatch],
    ) -> Dict[str, Any]:
        """
        Execute one training step with gradient accumulation support.

        Args:
            rollout_id: Current rollout iteration number
            training_data_handle_or_batch: Either a buffer-provided training-data
                handle (typically an ObjectRef in the Ray-backed path) or the
                typed TrainingBatch directly.

        Returns:
            Dictionary of training metrics
        """
        if not self._is_initialized:
            raise RuntimeError("Actor not initialized. Call init() first.")

        self._log_gpu_state("training_train_start")
        if isinstance(training_data_handle_or_batch, ray.ObjectRef):
            batch: TrainingBatch = ray.get(training_data_handle_or_batch)
        else:
            batch = training_data_handle_or_batch

        # Intended boundary: the actor keeps Ray/process-local concerns
        # (ObjectRef resolution, device/resource logging, backend handles),
        # while TrainingWorkflow owns the readable materialized-batch business chain.
        metrics = self._training_workflow.execute(
            rollout_id=rollout_id,
            batch=batch,
            build_executor=self._build_train_executor,
            on_prepared_batch=lambda: self._log_gpu_state("training_after_batch_to_device"),
            replay_batch=self._maybe_replay_old_log_probs,
            backend_train_step=(
                (
                    lambda *, rollout_id, batch, executor: self._train_backend.run_train_step(
                        self,
                        rollout_id=rollout_id,
                        batch=batch,
                        executor=executor,
                    )
                )
                if self._train_backend is not None
                else None
            ),
        )
        if bool(metrics.get("skipped", False)):
            self._log_gpu_state("training_train_skipped")
            return metrics
        self._log_gpu_state("training_train_end")
        return metrics

    def create_debug_training_batch(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        num_inference_steps: int,
        latent_channels: int = 16,
        vae_scale_factor: int = 8,
        max_sequence_length: int = 256,
    ) -> TrainingBatch:
        """Create a synthetic debug batch with the algorithm's expected type."""
        model_type = (
            self.model_bundle.model_type
            if self.model_bundle is not None and hasattr(self.model_bundle, "model_type")
            else ""
        )
        return _build_synthetic_debug_training_batch(
            model=self.model,
            algorithm=self.algorithm,
            model_type=model_type,
            batch_size=batch_size,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            latent_channels=latent_channels,
            vae_scale_factor=vae_scale_factor,
            max_sequence_length=max_sequence_length,
        )

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

    def setup_weight_sync(self, config: dict) -> None:
        """Configure a handler-based rollout weight-sync path."""
        from argparse import Namespace

        from diffusionrl.utils.fsdp_update_weights_utils import (
            UpdateWeightFromCheckpoint,
            UpdateWeightFromDistributed,
            UpdateWeightFromTensor,
        )

        mode = str(config.get("mode", ""))
        rollout_actors = config.get("rollout_actors", [])
        self._rollout_actors = list(rollout_actors) if rollout_actors else []
        self._lora_initialized_on_rollout = False

        handler_args = Namespace(
            target_modules=config.get("target_modules"),
            flush_cache=config.get("flush_cache", True),
            update_weight_buffer_size=config.get("bucket_size_mb", 256) * 1024 * 1024,
            rollout_num_gpus_per_engine=config.get("rollout_num_gpus_per_engine", 1),
            rollout_num_gpus=config.get("rollout_num_gpus", len(self._rollout_actors)),
        )

        if mode == "tensor_payload":
            self._update_weight_handler = UpdateWeightFromTensor(handler_args, self.model)
        elif mode == "nccl_broadcast":
            self._update_weight_handler = UpdateWeightFromDistributed(handler_args, self.model)
        elif mode == "checkpoint_path":
            handler_args.weight_sync_dir = config.get("weight_sync_dir", "/tmp/diffusionrl_wsync")
            handler_args.export_format = config.get("export_format", "state_dict")
            handler_args.rollout_runtime = config.get("rollout_runtime")
            handler_args.rollout_target = config.get("rollout_runtime")
            self._update_weight_handler = UpdateWeightFromCheckpoint(handler_args, self.model)
        else:
            raise ValueError(f"Unknown weight sync mode: {mode!r}")

        self._update_weight_handler.connect_rollout_engines(self._rollout_actors, None)
        logger.info(
            "Rank %s: configured weight sync handler mode=%s rollout_actors=%d",
            self.rank,
            mode,
            len(self._rollout_actors),
        )

    def _extract_lora_tensors_with_alpha(self) -> dict:
        """Extract LoRA A/B tensors plus per-layer alpha scalars."""
        from diffusionrl.utils.peft_merge import _strip_peft_prefix, _to_full_tensor

        result = {}
        adapter_name = "default"
        for raw_name, param in self.model.state_dict().items():
            name = _strip_peft_prefix(raw_name)
            if ".lora_A." in name:
                prefix, adapter_suffix = name.split(".lora_A.", 1)
                adapter, *_rest = adapter_suffix.split(".", 1)
                if adapter == adapter_name:
                    result[prefix + ".lora_A"] = _to_full_tensor(param).cpu()
            elif ".lora_B." in name:
                prefix, adapter_suffix = name.split(".lora_B.", 1)
                adapter, *_rest = adapter_suffix.split(".", 1)
                if adapter == adapter_name:
                    result[prefix + ".lora_B"] = _to_full_tensor(param).cpu()

        peft_cfg = getattr(self.model, "peft_config", {}).get(adapter_name)
        if peft_cfg is not None:
            alpha_val = torch.tensor(float(peft_cfg.lora_alpha))
            for key in list(result.keys()):
                if key.endswith(".lora_A"):
                    prefix = key[: -len(".lora_A")]
                    result[prefix + ".alpha"] = alpha_val
        return result

    def sync_weights_to_rollout(self) -> None:
        """Synchronize weights through the configured rollout weight-sync handler."""
        if self._update_weight_handler is None:
            raise RuntimeError(
                "Weight sync handler not configured. "
                "Call setup_weight_sync() before sync_weights_to_rollout()."
            )

        if self._use_lora and not self._lora_initialized_on_rollout:
            lora_tensors = self._extract_lora_tensors_with_alpha()
            if self.rank == 0:
                for actor in self._rollout_actors:
                    ray.get(actor.set_lora_from_tensors.remote("default", lora_tensors))
            self._lora_initialized_on_rollout = True

        self._update_weight_handler.update_weights()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def teardown_weight_sync(self) -> None:
        """Release handler state after rollout weight-sync finishes."""
        self._update_weight_handler = None
        self._rollout_actors = []
        self._lora_initialized_on_rollout = False
        logger.info("Rank %s: weight sync handler torn down", self.rank)

    @staticmethod
    def _to_rollout_dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).replace("torch.", "")

    def _iter_weight_sync_buckets(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        bucket_size_mb: int,
        staging_device: Optional[torch.device] = None,
    ):
        bucket_size_bytes = max(1, int(bucket_size_mb) * 1024 * 1024)
        current_bucket: List[tuple[str, torch.Tensor]] = []
        current_bytes = 0
        if staging_device is None:
            staging_device = self._device

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            staged = tensor.detach().to(device=staging_device, non_blocking=False).contiguous()
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
        staging_device: Optional[torch.device] = None,
    ):
        bucket_iter = iter(
            self._iter_weight_sync_buckets(
                state_dict,
                bucket_size_mb=bucket_size_mb,
                staging_device=staging_device,
            )
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

        has_cuda = any(tensor.is_cuda for _, tensor in named_tensors)
        if has_cuda:
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
            if has_cuda:
                serialized_payloads.append(
                    MultiprocessingSerializer.serialize(payload, output_str=True)
                )
            else:
                serialized_payloads.append(
                    self._serialize_cpu_payload(payload)
                )
        return serialized_payloads, long_lived_payloads

    @staticmethod
    def _serialize_cpu_payload(obj: dict) -> str:
        """Serialize payload using regular pickle to avoid fd-sharing.

        ForkingPickler (used by MultiprocessingSerializer) reduces CPU tensor
        storages via Unix fd-sharing (DupFd/resource_sharer), which requires
        the receiver to connect back to the sender with a matching authkey.
        This breaks when sender (Ray training actor) and receiver (SGLang GPU
        worker) are in different process trees.  Regular pickle serializes CPU
        storage bytes inline, avoiding the fd-sharing path entirely.
        """
        import io
        import pickle

        import pybase64

        buf = io.BytesIO()
        pickle.Pickler(buf).dump(obj)
        return pybase64.b64encode(buf.getvalue()).decode("utf-8")

    def sync_weights_to_rollout_ipc(
        self,
        *,
        rollout_weight_sink: Any,
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
                staging_device=torch.device("cpu"),
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
                        rollout_weight_sink.update_weights_from_tensor(
                            serialized_named_tensors=payload_batch,
                            target_modules=target_modules,
                            load_format="flattened_bucket",
                            flush_cache=flush_this_payload,
                        )
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
        rollout_weight_sink: Any,
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

                rollout_weight_sink.update_weights_from_distributed(
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
            if self._ema_manager is not None:
                checkpoint["ema_state_dict"] = self._ema_manager.state_dict()
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
        if self._ema_manager is not None and "ema_state_dict" in checkpoint:
            self._ema_manager.load_state_dict(
                checkpoint["ema_state_dict"],
                algorithm=self.algorithm,
            )
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
        if self._local_reward_runtime is not None:
            self._local_reward_runtime.offload()
        backend = getattr(self, "_train_backend", None)
        if backend is not None:
            if backend.offload(self):
                self._is_offloaded = True
                _clear_gpu_memory()
                logger.info("Rank %s: Offload handled by backend=%s", self.rank, backend.name)
                self._log_gpu_state("training_offload")
                return

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
        if self._local_reward_runtime is not None:
            self._local_reward_runtime.onload()
        backend = getattr(self, "_train_backend", None)
        if backend is not None:
            if backend.onload(self):
                self._is_offloaded = False
                logger.info("Rank %s: Onload handled by backend=%s", self.rank, backend.name)
                self._log_gpu_state("training_onload")
                return

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
