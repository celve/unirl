"""
diffusionrl Arguments - Configuration parameters for training.

Default resolution order:
1) Explicit CLI flags
2) YAML values from --config
3) Dataclass field defaults
4) validate_args() normalize/derive rewrites

New contributors: start from parse_args() -> validate_args().
"""
import argparse
import copy
import inspect
import json
import logging
import os
import sys
import warnings
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_args, get_origin

from diffusionrl.config.paths import (
    is_probably_local_weight_sync_dir,
    normalize_repo_relative_paths,
    repo_root,
    resolve_repo_relative_path,
)
from diffusionrl.config.validation import (
    validate_colocate_fractions,
    validate_dotpath,
    validate_dynamic_dotpaths,
    validate_grouped_configs,
    validate_model_specific_logic,
    validate_resolved_engine_algorithm_contract,
    validate_reward_and_rollout_buffer_config,
    validate_rollout_layout,
    validate_runtime_mode_constraints,
)
from diffusionrl.models import list_model_types, resolve_model_bundle_path
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"
ENV_DATA_ROOT = "DIFFUSIONRL_DATA_ROOT"
ENV_MODEL_ROOT = "DIFFUSIONRL_MODEL_ROOT"

DEFAULT_MODEL_PATH = "diffusionrl.models.hunyuan.HunyuanModelBundle"
DEFAULT_SAMPLER_PATH = "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler"
@dataclass
class ModelConfig:
    """Model/runtime identity and checkpoint paths."""

    model_type: str = field(default="hunyuan",
        metadata={"help": "Model architecture type (hunyuan, flux, sd3, mochi, wan2.1, bagel)"})
    model_path: str = field(default=DEFAULT_MODEL_PATH,
        metadata={"help": "Python dotpath to ModelBundle class. Auto-resolved from model_type"})
    pretrained_model_saved_path: str = field(default="",
        metadata={"help": "Path to pretrained model weights (local path or HuggingFace ID)"})
    vae_saved_path: Optional[str] = field(default=None,
        metadata={"help": "Path to separate VAE checkpoint, if not bundled with the model"})
    text_encoder_path: Optional[str] = field(default=None,
        metadata={"help": "Path to separate text encoder checkpoint, if not bundled"})

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError(
                "model_path must be set. It is usually auto-resolved from model_type. "
                "Set --model.model-type (hunyuan, flux, sd3, mochi) or provide "
                "--model.model-path explicitly."
            )


@dataclass
class SamplingConfig:
    """Sampling engine, sampler, and denoising controls."""

    sampler_path: str = field(default=DEFAULT_SAMPLER_PATH,
        metadata={"help": "Python dotpath to Sampler class (auto-resolved from model_type)"})
    sampler_engine_type: Optional[str] = field(default=None,
        metadata={"help": "Rollout engine type: fsdp or sglang (auto-resolved from model)"})
    sampling_mode: str = field(default="rollout_actor",
        metadata={"help": "Where sampling runs: rollout_actor or training_actor"})
    max_samples_per_request: Optional[int] = field(default=None,
        metadata={"help": "Generated-sample cap per training-actor direct-sampling request; rollout_total_samples stays prompts_per_rollout*samples_per_prompt"})
    logprob_source: str = field(default="replay",
        metadata={"help": "SGLang log-prob mode: replay (training-side) or native (engine-side)"})
    replay_log_probs: bool = field(default=False,
        metadata={"help": "Replay old log-probs on training actor (for SGLang replay mode)"})
    replay_sampler_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to replay sampler, if different from sampler_path"})
    num_inference_steps: int = field(default=50,
        metadata={"help": "Number of denoising steps during sampling"})
    eta: float = field(default=1.0,
        metadata={"help": "SDE noise coefficient (eta=0 is ODE, eta=1 is full SDE)"})
    sde_type: str = field(default="sde",
        metadata={"help": "SDE formulation: sde, dance, flux_dance, dpm2, etc."})
    time_shift: float = field(default=3.0,
        metadata={"help": "Time-shift parameter for the sampling timestep schedule (model-specific)"})
    guidance_scale: float = field(default=7.5,
        metadata={"help": "Classifier-free guidance scale (0.0 = no guidance)"})
    sde_ratio: float = field(default=1.0,
        metadata={"help": "Fraction of steps that use SDE (0.0 = all ODE, 1.0 = all SDE)"})
    timestep_fraction: Any = field(default=1.0,
        metadata={"help": "Fraction of total timesteps to train on. "
                          "Single float x means [0, x) range; "
                          "tuple (x, y) means [x, y) range (e.g. (0.2, 0.8) = timesteps 20%%-80%%)"})
    sampling_adapter: Optional[str] = field(default=None,
        metadata={"help": "Sampling adapter type for special modes (e.g. 'old' for NFT)"})
    init_same_noise: bool = field(default=False,
        metadata={"help": "Use identical initial noise for all samples of the same prompt"})
    fsdp_num_gpus: int = field(default=1,
        metadata={"help": "Number of GPUs per FSDP rollout actor"})
    fsdp_inference_sharding_strategy: str = field(default="NO_SHARD",
        metadata={"help": "FSDP sharding for inference: NO_SHARD, FULL_SHARD, SHARD_GRAD_OP"})
    sp_size: int = field(default=1,
        metadata={"help": "Sequence parallelism size for inference"})
    tp_size: int = field(default=1,
        metadata={"help": "Tensor parallelism size for SGLang inference engine"})
    engine_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Additional kwargs passed to the rollout engine"})

    def validate(self) -> None:
        sampling_mode = str(self.sampling_mode or "rollout_actor").strip().lower()
        if sampling_mode not in ("rollout_actor", "training_actor"):
            raise ValueError(
                "sampling_mode must be one of rollout_actor/training_actor, "
                f"got: {self.sampling_mode}"
            )
        mode = str(self.logprob_source).strip().lower()
        if mode not in ("replay", "native"):
            raise ValueError(
                f"logprob_source must be one of replay/native, got: {self.logprob_source}"
            )
        if self.max_samples_per_request is not None and int(self.max_samples_per_request) < 1:
            raise ValueError("max_samples_per_request must be >= 1 when set.")
        if not self.sampler_path:
            raise ValueError(
                "sampler_path must be set. It is usually auto-resolved from model_type. "
                "Set --model.model-type or provide --sampling.sampler-path explicitly."
            )


@dataclass
class RewardConfig:
    """Reward path, reward model, and reward pool controls."""

    reward_path: Optional[str] = field(default="diffusionrl.reward.local.LocalRewardWorker",
        metadata={"help": "Python dotpath to RewardWorker class"})
    reward_model_saved_path: Optional[str] = field(default=None,
        metadata={"help": "Path to reward model weights (local path or HuggingFace ID)"})
    reward_model_name: str = field(default="hpsv2",
        metadata={"help": "Reward model name: hpsv2, pickscore, clip, ocr, etc."})
    reward_batch_size: int = field(default=8,
        metadata={"help": "Batch size for reward model inference"})
    reward_timeout: float = field(default=60.0,
        metadata={"help": "Timeout in seconds for reward computation per batch"})
    use_http_reward: bool = field(default=False,
        metadata={"help": "Use external HTTP reward service instead of local model"})
    reward_service_url: Optional[str] = field(default=None,
        metadata={"help": "URL of HTTP reward service (when use_http_reward=true)"})
    reward_models: Optional[List[str]] = field(default=None,
        metadata={"help": "List of reward model names for multi-reward setup"})
    reward_weights: Optional[List[float]] = field(default=None,
        metadata={"help": "Weights for each reward model in multi-reward aggregation"})
    component_aggregation: str = field(default="weighted_sum",
        metadata={"help": "Multi-reward aggregation method: weighted_sum"})
    component_mix_stage: str = field(default="reward",
        metadata={"help": "Multi-reward mixing: reward (mix rewards) or advantage (mix advantages)"})
    reward_dedicated_num_gpus: int = field(default=0,
        metadata={"help": "Total GPUs for dedicated reward actors (0 = CPU reward)"})
    reward_dedicated_num_nodes: int = field(default=0,
        metadata={"help": "Number of nodes for dedicated reward actors (mutually exclusive with num_gpus)"})
    reward_dedicated_num_gpus_per_node: int = field(default=0,
        metadata={"help": "GPUs per node for dedicated reward actors"})
    reward_dedicated_gpus_per_actor: int = field(default=1,
        metadata={"help": "GPUs per individual reward actor"})
    reward_service_urls: Optional[List[str]] = field(default=None,
        metadata={"help": "List of HTTP reward service URLs for load balancing"})
    reward_location: str = field(default="manager",
        metadata={"help": "Where reward-model inference runs: manager or sampling_actor"})
    local_reward_device: str = field(default="cpu",
        metadata={"help": "Device for local (non-HTTP, non-dedicated) reward workers: cpu, auto, or cuda"})
    allow_local_reward_cuda_contention: bool = field(default=False,
        metadata={"help": "Allow local_reward_device=cuda without dedicated reward GPUs (may contend with rollout/training GPUs)"})

    def validate(self) -> None:
        if self.component_mix_stage not in ("reward", "advantage"):
            raise ValueError(
                f"component_mix_stage must be one of reward/advantage, got: {self.component_mix_stage}"
            )
        reward_location = str(self.reward_location or "manager").strip().lower()
        if reward_location not in ("manager", "sampling_actor"):
            raise ValueError(
                "reward_location must be one of manager/sampling_actor, "
                f"got: {self.reward_location}"
            )
        local_reward_device = str(self.local_reward_device or "cpu").strip().lower()
        if local_reward_device not in ("cpu", "auto", "cuda"):
            raise ValueError(
                "local_reward_device must be one of cpu/auto/cuda, "
                f"got: {self.local_reward_device}"
            )
        has_http_reward = bool(
            self.use_http_reward
            or self.reward_service_url
            or self.reward_service_urls
        )
        if not has_http_reward and not self.reward_path:
            raise ValueError(
                "reward_path must be set for local/ray reward workers. "
                "Available: diffusionrl.reward.local.LocalRewardWorker, "
                "or provide a custom RewardWorker dotpath."
            )


@dataclass
class RayConfig:
    """Ray resource layout, colocate/offload, and weight sync controls."""

    ray_address: Optional[str] = field(default=None,
        metadata={"help": "Ray cluster address (None = auto-detect or start local)"})
    rollout_num_nodes: int = field(default=1,
        metadata={"help": "Number of nodes for rollout actors"})
    rollout_num_gpus_per_node: int = field(default=4,
        metadata={"help": "GPUs per node for rollout actors"})
    training_num_nodes: int = field(default=1,
        metadata={"help": "Number of nodes for training actors"})
    training_num_gpus_per_node: int = field(default=4,
        metadata={"help": "GPUs per node for training actors"})
    colocate_rollout_training: bool = field(default=False,
        metadata={"help": "Run rollout and training on same GPUs (requires offload)"})
    placement_strategy: str = field(default="PACK",
        metadata={"help": "Ray placement group strategy: PACK or SPREAD"})
    colocate_training_gpu_fraction: float = field(default=0.4,
        metadata={"help": "GPU memory fraction for training when colocated"})
    colocate_rollout_gpu_fraction: float = field(default=0.4,
        metadata={"help": "GPU memory fraction for rollout when colocated"})
    allow_noset_multi_gpu_inference: bool = field(default=False,
        metadata={"help": "Allow multi-GPU rollout actors (experimental NOSET layout)"})
    partition_train_data: bool = field(default=True,
        metadata={"help": "Partition training data across rollout actors"})
    offload: bool = field(default=False,
        metadata={"help": "Enable model offload for both training and rollout"})
    offload_train: Optional[bool] = field(default=None,
        metadata={"help": "Enable model offload for training actors (None = auto)"})
    offload_rollout: Optional[bool] = field(default=None,
        metadata={"help": "Enable model offload for rollout actors (None = auto)"})
    weight_sync_mode: str = field(default="auto",
        metadata={"help": "Weight sync mode: auto, tensor_payload, nccl_broadcast, checkpoint_path "
                          "(legacy: ipc, nccl, object_ref)"})
    weight_sync_dir: str = field(default="outputs/weight_sync",
        metadata={"help": "Directory for checkpoint-based weight sync (use shared FS for multi-node)"})
    weight_sync_bucket_mb: int = field(default=256,
        metadata={"help": "Weight sync tensor bucket size (MB) for ipc/nccl strategies"})
    weight_sync_flush_cache: bool = field(default=True,
        metadata={"help": "Whether rollout side flushes runtime cache after each weight sync bucket"})

    def validate(self) -> None:
        _valid_modes = (
            "auto", "tensor_payload", "nccl_broadcast", "checkpoint_path",
            "ipc", "nccl", "object_ref",
        )
        if self.weight_sync_mode not in _valid_modes:
            raise ValueError(
                f"weight_sync_mode must be one of {'/'.join(_valid_modes)}, "
                f"got: {self.weight_sync_mode!r}. "
                "Use 'auto' (recommended) to let diffusionrl choose the best mode."
            )


@dataclass
class WindowSchedulerConfig:
    """MixGRPO timestep/window scheduler."""

    timestep_strategy: str = field(default="all",
        metadata={"help": "Timestep selection strategy: all (use all) or window (sliding window)"})
    window_strategy: str = field(default="progressive",
        metadata={"help": "Window progression: progressive, random, decay, exp_decay"})
    window_group_size: int = field(default=4,
        metadata={"help": "Number of timesteps per window group"})
    window_iters_per_group: int = field(default=25,
        metadata={"help": "Training iterations before advancing the window"})
    window_max_iters_per_group: Optional[int] = field(default=10,
        metadata={"help": "Maximum iters per group for decay strategy (MixGRPO default: 10)"})
    window_min_iters_per_group: Optional[int] = field(default=1,
        metadata={"help": "Minimum iters per group for decay strategy (MixGRPO default: 1)"})
    window_overlap: bool = field(default=False,
        metadata={"help": "Allow overlap between adjacent window groups"})
    window_overlap_step: int = field(default=1,
        metadata={"help": "Number of overlapping timesteps between adjacent windows"})
    window_roll_back: bool = field(default=False,
        metadata={"help": "Roll back to earlier windows after reaching the end"})
    window_training: bool = field(default=False,
        metadata={"help": "Train only on SDE (stochastic) timestep steps"})


@dataclass
class AlgorithmConfig:
    """Algorithm controls normalized into one algorithm-owned config surface."""

    # Algorithm selection
    algorithm_type: str = field(default="grpo",
        metadata={"help": "Built-in algorithm family: grpo, nft, or mix_grpo"})
    algorithm_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to Algorithm class (auto-resolved from algorithm_type when omitted)"})
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Canonical extension surface for algorithm-specific kwargs. Both rollout and training instantiate algorithms from the same normalized algorithm_kwargs payload."})

    # Advantage and policy objective
    clip_range: float = field(default=1e-4,
        metadata={"help": "PPO clipping range for policy ratio. Smaller = more conservative"})
    clip_schedule: str = field(default="constant",
        metadata={"help": "Clip range schedule: constant, linear_decay, cosine_decay"})
    kl_coef: float = field(default=0.01,
        metadata={"help": "KL divergence penalty coefficient"})
    use_kl_penalty: bool = field(default=True,
        metadata={"help": "Add KL penalty term to the loss"})
    adv_normalization: str = field(default="group",
        metadata={"help": "Advantage normalization: global or group"})
    samples_per_prompt: int = field(default=4,
        metadata={"help": "Number of generated samples per prompt for GRPO"})
    prompts_per_rollout: Optional[int] = field(default=None,
        metadata={"help": "Number of unique prompts per rollout step. Required."})
    adv_norm_eps: float = field(default=1e-8,
        metadata={"help": "Epsilon for numerical stability in advantage normalization"})
    adv_clip_abs: Optional[float] = field(default=None,
        metadata={"help": "Max absolute advantage value (None = no clipping)"})
    trimmed_ratio: float = field(default=0.0,
        metadata={"help": "Trim ratio for grouped advantage stats (MixGRPO-style outlier trimming)"})
    use_global_std: bool = field(default=False,
        metadata={"help": "Use global (cross-prompt) std for advantage normalization"})
    skip_last_timestep: bool = field(default=False,
        metadata={"help": "Skip last timestep (t->0) in algorithm objective (can be numerically unstable)"})
    skip_initial_timesteps: int = field(default=0,
        metadata={"help": "Skip first N timesteps in algorithm objective computation (frozen warmup)"})

    # Evaluation EMA
    eval_ema_decay: float = field(default=0.9,
        metadata={"help": "EMA decay rate for evaluation model (warmup schedule: min((1+step)/(10+step), decay))"})
    eval_ema_update_interval: int = field(default=1,
        metadata={"help": "Update evaluation EMA every N optimizer steps"})

    # Sub-configuration
    window: WindowSchedulerConfig = field(default_factory=WindowSchedulerConfig)

    def validate(self) -> None:
        if not self.algorithm_type and not self.algorithm_path:
            raise ValueError(
                "algorithm_type or algorithm_path must be set. "
                "Available built-ins: grpo, nft, mix_grpo."
            )
        if self.samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be >= 1.")
        if self.prompts_per_rollout is None:
            raise ValueError("prompts_per_rollout must be set explicitly.")
        if self.prompts_per_rollout < 1:
            raise ValueError("prompts_per_rollout must be >= 1.")
        if not (0.0 <= self.trimmed_ratio < 0.5):
            raise ValueError("trimmed_ratio must be in [0.0, 0.5).")
        window_cfg = self.window
        if (
            window_cfg.window_max_iters_per_group is not None
            and window_cfg.window_min_iters_per_group is not None
            and window_cfg.window_min_iters_per_group > window_cfg.window_max_iters_per_group
        ):
            raise ValueError(
                "window_min_iters_per_group must be <= window_max_iters_per_group."
            )


@dataclass
class TrainingConfig:
    """Optimizer, LoRA/FSDP, and core training runtime controls."""

    # Optimizer and update schedule
    gradient_accumulation_batch_size: Optional[int] = field(default=None,
        metadata={"help": "Per-GPU micro-batch size used inside one optimizer update. Set to null/None to disable extra gradient accumulation and use the effective update batch size directly"})
    multi_update_batch_size: Optional[int] = field(default=None,
        metadata={"help": "Per-GPU update-chunk size for multi_update. If >= the local rollout batch size, validation normalizes update_mode back to single_update"})
    update_mode: str = field(default="single_update",
        metadata={"help": "Training update schedule: single_update does one optimizer step per rollout pass; multi_update does one optimizer step per update chunk"})
    learning_rate: float = field(default=1e-6,
        metadata={"help": "Peak learning rate for the optimizer"})
    adam_beta1: float = field(default=0.9,
        metadata={"help": "Adam optimizer beta1 (first moment decay)"})
    adam_beta2: float = field(default=0.999,
        metadata={"help": "Adam optimizer beta2 (second moment decay)"})
    adam_epsilon: float = field(default=1e-8,
        metadata={"help": "Adam optimizer epsilon for numerical stability"})
    weight_decay: float = field(default=1e-4,
        metadata={"help": "Weight decay (L2 regularization) coefficient"})
    max_grad_norm: float = field(default=1.0,
        metadata={"help": "Max gradient norm for gradient clipping"})
    warmup_steps: int = field(default=0,
        metadata={"help": "Number of learning rate warmup steps"})
    lr_scheduler_type: str = field(default="constant",
        metadata={"help": "LR scheduler type: constant, linear, cosine"})

    # Train backend
    train_backend: str = field(default="fsdp",
        metadata={"help": "Training backend name (fsdp/veomni built-in; megatron scaffold requires actor_class_path in train_backend_kwargs); or custom via train_backend_path"})
    train_backend_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to custom TrainBackend class (overrides built-in backend selection)"})
    train_backend_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Extra kwargs for selected train backend; accepts JSON string (CLI) or mapping (YAML)"})

    # LoRA
    use_lora: bool = field(default=False,
        metadata={"help": "Enable LoRA (Low-Rank Adaptation) for parameter-efficient training"})
    lora_rank: int = field(default=16,
        metadata={"help": "LoRA rank (lower = fewer parameters, higher = more expressive)"})
    lora_alpha: int = field(default=16,
        metadata={"help": "LoRA alpha scaling factor (effective scale = alpha/rank)"})
    lora_target_modules: Optional[str] = field(default=None,
        metadata={"help": "Comma-separated LoRA target modules (e.g. 'to_q,to_k,to_v')"})

    # FSDP
    fsdp_sharding_strategy: str = field(default="FULL_SHARD",
        metadata={"help": "Legacy compatibility knob. diffusionrl train_backend=fsdp is fully_shard-only; keep FULL_SHARD"})
    fsdp_cpu_offload: bool = field(default=False,
        metadata={"help": "Offload FSDP parameters and gradients to CPU"})

    # Memory optimization
    use_gradient_checkpointing: bool = field(default=False,
        metadata={"help": "Enable gradient checkpointing to save memory at the cost of compute"})

    def validate(self) -> None:
        if (
            self.gradient_accumulation_batch_size is not None
            and int(self.gradient_accumulation_batch_size) < 1
        ):
            raise ValueError("gradient_accumulation_batch_size must be >= 1 when set.")
        if self.multi_update_batch_size is not None and int(self.multi_update_batch_size) < 1:
            raise ValueError("multi_update_batch_size must be >= 1 when set.")
        mode = str(self.update_mode).strip().lower()
        if mode not in {"single_update", "multi_update"}:
            raise ValueError(
                "update_mode must be one of single_update/multi_update, "
                f"got: {self.update_mode!r}"
            )
        backend = str(self.train_backend or "fsdp").strip().lower()
        supported = {"fsdp", "megatron", "veomni"}
        if backend not in supported and not self.train_backend_path:
            raise ValueError(
                f"Unsupported train_backend={self.train_backend!r}. "
                f"Expected one of {sorted(supported)} or provide --training.train-backend-path."
            )
        sharding = str(self.fsdp_sharding_strategy or "FULL_SHARD").strip().upper()
        if sharding != "FULL_SHARD":
            raise ValueError(
                "fsdp_sharding_strategy is a legacy compatibility argument and "
                "must be FULL_SHARD in the current FSDP2 backend. "
                f"Got: {self.fsdp_sharding_strategy!r}"
            )


@dataclass
class RolloutLoggingConfig:
    """Rollout loop/buffer, checkpoint/eval, and logging controls."""

    # Rollout buffer actor (data-centric handoff with validation/filter plugins)
    rollout_buffer_max_queue_size: int = field(default=0,
        metadata={"help": "Max rollout buffer queue size (0 = unbounded)"})
    rollout_buffer_drop_invalid: bool = field(default=True,
        metadata={"help": "Drop samples that fail validation checks"})
    rollout_buffer_reward_min: Optional[float] = field(default=None,
        metadata={"help": "Minimum reward threshold for sample filtering (None = no filter)"})
    rollout_buffer_reward_max: Optional[float] = field(default=None,
        metadata={"help": "Maximum reward threshold for sample filtering (None = no filter)"})
    rollout_buffer_min_samples: int = field(default=1,
        metadata={"help": "Minimum samples required before dispatching a batch"})
    rollout_buffer_grouped: bool = field(default=False,
        metadata={"help": "Group samples by explicit group_ids in the rollout buffer"})
    rollout_buffer_group_size: Optional[int] = field(default=None,
        metadata={"help": "Optional explicit samples per logical group; when unset, infer from incoming group_ids batches"})
    rollout_buffer_dispatch_groups: int = field(default=0,
        metadata={"help": "Number of prompt-groups merged into one training batch (0 = prompts_per_rollout)"})
    rollout_buffer_allow_partial_group: bool = field(default=True,
        metadata={"help": "Allow dispatching groups with fewer samples than group_size"})
    rollout_buffer_group_ttl_seconds: float = field(default=0.0,
        metadata={"help": "Time-to-live for incomplete groups in seconds (0 = no timeout)"})
    rollout_buffer_max_pending_samples: int = field(default=0,
        metadata={"help": "Max pending samples in buffer before blocking rollout (0 = unbounded)"})
    rollout_buffer_plugin_paths: str = field(default="",
        metadata={"help": "Comma-separated dotpaths to rollout buffer filter plugins"})

    # Rollout loop
    num_rollout: int = field(default=1000,
        metadata={"help": "Total number of rollout iterations (outer-loop steps; analogous to global step)"})
    start_rollout_id: int = field(default=0,
        metadata={"help": "Starting rollout step/ID for resuming training (acts like an outer-loop global step)"})
    async_pipeline: bool = field(default=False,
        metadata={"help": "Enable async rollout/training overlap (separate mode only)"})
    async_max_inflight: int = field(default=1,
        metadata={"help": "Max in-flight rollout futures in async pipeline"})
    update_weights_interval: int = field(default=1,
        metadata={"help": "Sync weights from training to rollout every N steps"})

    # Checkpointing
    output_dir: str = field(default="outputs",
        metadata={"help": "Output directory for checkpoints, logs, and generated samples"})
    save_steps: int = field(default=100,
        metadata={"help": "Save a checkpoint every N training steps (0 disables periodic saves)"})
    resume_from_checkpoint: Optional[str] = field(default=None,
        metadata={"help": "Path to checkpoint directory to resume training from"})

    # Evaluation
    eval_steps: int = field(default=100,
        metadata={"help": "Run evaluation every N training steps (0 disables periodic eval)"})
    eval_batch_size: int = field(default=4,
        metadata={"help": "Batch size for evaluation"})

    # Logging
    logging_steps: int = field(default=10,
        metadata={"help": "Log metrics every N training steps (0 disables periodic step logging)"})
    logging_dir: Optional[str] = field(default=None,
        metadata={"help": "Directory for WandB logs/artifacts (defaults to output_dir/logs)"})
    report_to_wandb: bool = field(default=False,
        metadata={"help": "Enable WandB reporting (true/false)"})
    project_name: str = field(default="diffusionrl",
        metadata={"help": "Project name for WandB"})
    run_name: Optional[str] = field(default=None,
        metadata={"help": "Run name for WandB (auto-generated if None)"})
    wandb_log_media: bool = field(default=False,
        metadata={"help": "Log generated media previews to WandB (true/false)"})
    wandb_media_max_items: int = field(default=8,
        metadata={"help": "Maximum generated media items to log per rollout when wandb_log_media=true"})
    wandb_tags: Optional[str] = field(default=None,
        metadata={"help": "Comma-separated tags for WandB run (e.g. 'exp1,baseline'). Defaults to 'diffusionrl-reproduce' if not set."})
    wandb_entity: Optional[str] = field(default=None,
        metadata={"help": "WandB entity (team or username). If not set, uses the default entity of the logged-in user."})

    def validate(self) -> None:
        if self.num_rollout < 1:
            raise ValueError("num_rollout must be >= 1.")
        if self.save_steps < 0:
            raise ValueError("save_steps must be >= 0 (0 disables periodic saves).")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must be >= 0 (0 disables periodic eval).")
        if self.logging_steps < 0:
            raise ValueError("logging_steps must be >= 0 (0 disables periodic step logging).")
        if self.update_weights_interval < 1:
            raise ValueError("update_weights_interval must be >= 1.")
        if not isinstance(self.report_to_wandb, bool):
            raise ValueError(
                "report_to_wandb must be a boolean (true/false). "
                f"Got: {self.report_to_wandb!r}"
            )
        if int(self.wandb_media_max_items) < 1:
            raise ValueError("wandb_media_max_items must be >= 1.")


@dataclass
class DebugConfig:
    """Debug runtime mode and intermediate artifact controls."""

    debug_mode: str = field(default="none",
        metadata={"help": "Debug mode: none or train_only"})
    debug_save_dir: str = field(default="outputs/debug",
        metadata={"help": "Directory for debug artifacts and saved rollout payloads"})
    debug_save_intermediates: bool = field(default=False,
        metadata={"help": "Save rollout debug payloads during the normal training loop"})
    debug_load_path: Optional[str] = field(default=None,
        metadata={"help": "Path to debug payload/training batch for train_only mode"})
    debug_num_rollouts: int = field(default=1,
        metadata={"help": "Number of train_only iterations to run"})
    debug_output_dir: Optional[str] = field(default=None,
        metadata={"help": "Directory to dump per-step SDE tensors for train-inference consistency debugging. "
                          "Sampling tensors saved to <dir>/sampling/, training tensors to <dir>/training/."})

    def validate(self) -> None:
        mode = str(self.debug_mode or "none").strip().lower()
        if mode not in ("none", "train_only"):
            raise ValueError(
                "debug_mode must be one of: none, train_only. "
                f"Got: {self.debug_mode!r}"
            )
        if int(self.debug_num_rollouts) < 1:
            raise ValueError("debug_num_rollouts must be >= 1.")
        if mode == "train_only" and not self.debug_load_path:
            logger.info(
                "debug_mode=train_only without --debug.debug-load-path: "
                "will generate synthetic training data."
            )


_GROUP_CONFIG_TYPES = {
    "model": ModelConfig,
    "sampling": SamplingConfig,
    "reward": RewardConfig,
    "ray": RayConfig,
    "algorithm": AlgorithmConfig,
    "training": TrainingConfig,
    "rollout": RolloutLoggingConfig,
    "debug": DebugConfig,
}
_GROUP_CONFIG_NAMES = set(_GROUP_CONFIG_TYPES.keys())


def _build_flat_field_path_index() -> tuple[Dict[str, str], Dict[str, tuple[str, Optional[str]]]]:
    """Build flat-field owner/path mappings for grouped dataclasses.

    Returns:
        owners: field_name -> group_name
        nested_path: field_name -> (group_name, sub_field_name or None)
            e.g. "clip_range" -> ("algorithm", None)
    """
    owners: Dict[str, str] = {}
    nested_path: Dict[str, tuple[str, Optional[str]]] = {}
    for config_name, config_type in _GROUP_CONFIG_TYPES.items():
        for config_field in fields(config_type):
            ft = config_field.type
            # Check if this field is itself a dataclass (sub-config)
            if isinstance(ft, type) and is_dataclass(ft):
                # Register all leaf fields of the sub-dataclass
                for sub_field in fields(ft):
                    if sub_field.name in owners:
                        raise ValueError(
                            f"Duplicated grouped config field '{sub_field.name}' in "
                            f"{owners[sub_field.name]} and {config_name}.{config_field.name}."
                        )
                    owners[sub_field.name] = config_name
                    nested_path[sub_field.name] = (config_name, config_field.name)
            else:
                if config_field.name in owners:
                    raise ValueError(
                        f"Duplicated grouped config field '{config_field.name}' in "
                        f"{owners[config_field.name]} and {config_name}."
                    )
                owners[config_field.name] = config_name
                nested_path[config_field.name] = (config_name, None)
    return owners, nested_path


_, _FLAT_FIELD_PATH_INDEX = _build_flat_field_path_index()
_TOP_LEVEL_FIELD_NAMES: set[str] = set()
_GROUP_SUBCONFIG_NAMES: Dict[str, set[str]] = {}


def is_training_actor_sampling_mode(args: Any) -> bool:
    """Return whether training actors should directly handle sampling."""
    return str(getattr(args.sampling, "sampling_mode", "rollout_actor") or "rollout_actor").strip().lower() == "training_actor"

@dataclass
class TrainingArguments:
    """All configuration parameters for GRPO training."""

    # ========== Paths (Dynamic Loading) ==========
    data_source_path: str = field(default="diffusionrl.data.DefaultDataSource",
        metadata={"help": "Python dotpath to DataSource class for loading training/eval prompt streams"})

    # ========== Grouped Configuration ==========
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutLoggingConfig = field(default_factory=RolloutLoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ========== Data Configuration ==========
    data_path: Optional[str] = field(default="data/samples/prompts_toy.json",
        metadata={"help": "Path to training prompt data file (JSON, JSONL, or TXT). JSON items should provide text via 'prompt' or 'caption'."})
    eval_data_path: Optional[str] = field(default=None,
        metadata={"help": "Optional path to evaluation prompt data file. If unset, eval uses data_path with deterministic ordering."})
    # ========== Video/Image Configuration ==========
    height: int = field(default=256,
        metadata={"help": "Generated image/video height in pixels"})
    width: int = field(default=256,
        metadata={"help": "Generated image/video width in pixels"})
    num_frames: int = field(default=16,
        metadata={"help": "Number of video frames to generate (video models only)"})
    fps: int = field(default=8,
        metadata={"help": "Video frame rate (video models only)"})

    # ========== Seed ==========
    seed: int = field(default=42,
        metadata={"help": "Random seed for reproducibility"})

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten grouped + top-level args to a single dictionary."""
        flat: Dict[str, Any] = {}
        for info in fields(type(self)):
            if info.name in _GROUP_CONFIG_NAMES:
                continue
            flat[info.name] = getattr(self, info.name)
        for owner, config_type in _GROUP_CONFIG_TYPES.items():
            config_obj = getattr(self, owner)
            for info in fields(config_type):
                ft = info.type
                if isinstance(ft, type) and is_dataclass(ft):
                    # Expand sub-dataclass fields
                    sub_obj = getattr(config_obj, info.name)
                    for sub_info in fields(ft):
                        flat[sub_info.name] = getattr(sub_obj, sub_info.name)
                else:
                    flat[info.name] = getattr(config_obj, info.name)
        return flat


_TOP_LEVEL_FIELD_NAMES = {
    info.name for info in fields(TrainingArguments) if info.name not in _GROUP_CONFIG_NAMES
}
for _group_name, _group_type in _GROUP_CONFIG_TYPES.items():
    _GROUP_SUBCONFIG_NAMES[_group_name] = {
        info.name
        for info in fields(_group_type)
        if isinstance(info.type, type) and is_dataclass(info.type)
    }


def _resolve_field_default(field_info: Any) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return None


def _resolve_cli_field_type(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    inner_types = [t for t in get_args(field_type) if t is not type(None)]
    if len(inner_types) == 1:
        return inner_types[0]
    return field_type


def _parse_cli_bool(value: Any) -> bool:
    """Parse boolean CLI values with strict validation."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true/false (or 1/0, yes/no)."
    )


def _parse_cli_timestep_fraction(value: Any) -> Any:
    """Parse timestep_fraction CLI value.

    Accepts:
    - A single float: 0.6  -> returns 0.6
    - A comma-separated pair: "0.2,0.8" -> returns (0.2, 0.8)
    - A JSON-style list: "[0.2, 0.8]" -> returns (0.2, 0.8)
    - Already a list/tuple (from YAML): [0.2, 0.8] -> returns (0.2, 0.8)
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise argparse.ArgumentTypeError(
            f"timestep_fraction tuple must have exactly 2 elements, got {len(value)}"
        )
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 1.0
    # Try JSON list: "[0.2, 0.8]"
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid timestep_fraction value: {value!r}. Error: {exc}"
            ) from exc
        if isinstance(parsed, list) and len(parsed) == 2:
            return (float(parsed[0]), float(parsed[1]))
        raise argparse.ArgumentTypeError(
            f"timestep_fraction list must have exactly 2 elements, got: {parsed!r}"
        )
    # Try comma-separated: "0.2,0.8"
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid timestep_fraction value: {value!r}. Error: {exc}"
                ) from exc
        raise argparse.ArgumentTypeError(
            f"timestep_fraction comma-separated value must have exactly 2 elements, got {len(parts)}"
        )
    # Single float
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid timestep_fraction value: {value!r}. Error: {exc}"
        ) from exc


def _parse_cli_json_object(value: Any) -> Dict[str, Any]:
    """Parse a CLI value into a JSON object."""
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a JSON object string, got: {value!r}. Error: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"Expected a JSON object string, got {type(parsed).__name__}."
        )
    return parsed


def _parse_cli_list(value: Any, *, item_type: Any = str) -> List[Any]:
    """Parse comma-separated or JSON list CLI values into typed Python lists."""
    if isinstance(value, list):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise argparse.ArgumentTypeError(
                    f"Expected a list value, got: {value!r}. Error: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise argparse.ArgumentTypeError(
                    f"Expected a list value, got {type(parsed).__name__}."
                )
            raw_items = list(parsed)
        else:
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

    normalized_item_type = _resolve_cli_field_type(item_type)
    parsed_items: List[Any] = []
    for raw in raw_items:
        if normalized_item_type == bool:
            parsed_items.append(_parse_cli_bool(raw))
        elif normalized_item_type == int:
            try:
                parsed_items.append(int(raw))
            except Exception as exc:
                raise argparse.ArgumentTypeError(
                    f"Expected integer list item, got: {raw!r}"
                ) from exc
        elif normalized_item_type == float:
            try:
                parsed_items.append(float(raw))
            except Exception as exc:
                raise argparse.ArgumentTypeError(
                    f"Expected float list item, got: {raw!r}"
                ) from exc
        else:
            parsed_items.append(str(raw))
    return parsed_items


def _resolve_field_help_text(field_info) -> str:
    """Extract help text from field metadata, with fallback to auto-generated."""
    help_text = (field_info.metadata or {}).get("help")
    if help_text:
        return help_text
    default = _resolve_field_default(field_info)
    return f"{field_info.name} (default: {default})"


def _collect_field_specs_from_dataclass(
    config_type, group_key: str, seen_names: set, specs: list,
) -> None:
    """Recursively collect (name, type, default, help, group_key) from a dataclass."""
    for field_info in fields(config_type):
        ft = field_info.type
        # Check if this field is itself a dataclass (sub-config)
        resolved_type = None
        if isinstance(ft, type) and is_dataclass(ft):
            resolved_type = ft

        if resolved_type is not None:
            # Recurse into sub-dataclass fields with dotted group key
            sub_key = f"{group_key}.{field_info.name}"
            _collect_field_specs_from_dataclass(resolved_type, sub_key, seen_names, specs)
        else:
            if field_info.name in seen_names:
                raise ValueError(
                    f"Duplicated parser argument field '{field_info.name}' between "
                    "TrainingArguments and grouped configs."
                )
            specs.append(
                (
                    field_info.name,
                    _resolve_cli_field_type(field_info.type),
                    _resolve_field_default(field_info),
                    _resolve_field_help_text(field_info),
                    group_key,
                )
            )
            seen_names.add(field_info.name)


def _collect_cli_field_specs() -> List[tuple[str, Any, Any, str, str]]:
    """Return (name, type, default, help, group_key) for every CLI-exposed field.

    group_key is used for argparse grouping:
      - "" for TrainingArguments top-level
      - "model", "sampling", etc. for group configs
      - "algorithm.window" for nested sub-configs
    """
    specs: List[tuple[str, Any, Any, str, str]] = []
    seen_names: set[str] = set()

    for field_info in fields(TrainingArguments):
        if field_info.name in _GROUP_CONFIG_NAMES:
            continue
        specs.append(
            (
                field_info.name,
                _resolve_cli_field_type(field_info.type),
                _resolve_field_default(field_info),
                _resolve_field_help_text(field_info),
                "",  # top-level
            )
        )
        seen_names.add(field_info.name)

    for group_name, config_type in _GROUP_CONFIG_TYPES.items():
        _collect_field_specs_from_dataclass(config_type, group_name, seen_names, specs)

    return specs


# Display names for argument groups in --help output
_GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "": "General",
    "model": "Model Configuration",
    "sampling": "Sampling & Inference",
    "reward": "Reward Configuration",
    "ray": "Ray & Resource Layout",
    "algorithm": "Algorithm & Advantage",
    "algorithm.window": "Window/Timestep Scheduler",
    "training": "Training & Optimization",
    "rollout": "Rollout, Checkpointing & Logging",
    "debug": "Debug Mode & Artifact Saving",
}


def _load_yaml_mapping(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return a mapping."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for --config support. Install it with: pip install pyyaml"
        )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")
    return data


def _merge_yaml_overrides(
    raw_args: Dict[str, Any],
    yaml_data: Dict[str, Any],
    defaults: Dict[str, Any],
    explicit_cli_keys: set[str],
    action_by_dest: Dict[str, argparse.Action],
    allow_unknown_config_keys: bool,
) -> None:
    """Apply YAML values to raw_args for keys the user did NOT explicitly set on CLI."""
    flattened_yaml = _flatten_yaml_mapping(yaml_data)
    all_known_keys = set(defaults.keys())
    reported_cli_overrides: set[str] = set()
    for key, value in flattened_yaml.items():
        cli_key = key.replace("-", "_")
        if cli_key not in all_known_keys:
            message = f"Unknown key '{key}' in YAML config (no matching CLI argument)."
            if not allow_unknown_config_keys:
                raise ValueError(
                    message
                    + " Remove/fix the key, or pass --allow-unknown-config-keys "
                    "to ignore unknown YAML keys."
                )
            warnings.warn(
                message + " Ignoring because --allow-unknown-config-keys is set.",
                stacklevel=3,
            )
            continue
        # Only apply YAML value if user did not explicitly set via CLI
        if cli_key in explicit_cli_keys:
            if cli_key not in reported_cli_overrides:
                warnings.warn(
                    f"YAML key '{key}' ignored because CLI explicitly set '{cli_key}' (CLI takes precedence).",
                    stacklevel=3,
                )
                reported_cli_overrides.add(cli_key)
            continue
        if raw_args.get(cli_key) == defaults.get(cli_key):
            raw_args[cli_key] = _coerce_yaml_value(
                key=key,
                cli_key=cli_key,
                value=value,
                action_by_dest=action_by_dest,
            )


def _is_yaml_container_path(parts: List[str]) -> bool:
    if len(parts) == 1 and parts[0] in _GROUP_CONFIG_NAMES:
        return True
    if len(parts) == 2 and parts[0] in _GROUP_SUBCONFIG_NAMES:
        return parts[1] in _GROUP_SUBCONFIG_NAMES[parts[0]]
    return False


def _resolve_yaml_leaf_dest(parts: List[str]) -> Optional[str]:
    if not parts:
        return None
    if len(parts) == 1:
        key = parts[0]
        if key in _TOP_LEVEL_FIELD_NAMES:
            return key
        return None

    group = parts[0]
    leaf = parts[-1]
    path = _FLAT_FIELD_PATH_INDEX.get(leaf)
    if path is None:
        return None
    owner, sub_name = path
    if owner != group:
        return None
    if len(parts) == 2:
        # Accept both group.leaf and group.sub_leaf for contributor convenience.
        return leaf
    if len(parts) == 3 and sub_name is not None and parts[1] == sub_name:
        return leaf
    return None


def _flatten_yaml_mapping(
    yaml_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten nested YAML mapping into parser destination keys."""
    flattened: Dict[str, Any] = {}
    origins: Dict[str, str] = {}

    def _value_repr(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(value)

    def _assign(dest_key: str, value: Any, *, source_path: str) -> None:
        if dest_key in flattened:
            previous_source = origins[dest_key]
            if previous_source != source_path:
                raise ValueError(
                    "Conflicting YAML keys map to the same argument destination "
                    f"'{dest_key}': '{previous_source}'={_value_repr(flattened[dest_key])} "
                    f"and '{source_path}'={_value_repr(value)}. "
                    "Keep only one style (prefer grouped keys)."
                )
        flattened[dest_key] = value
        origins[dest_key] = source_path

    def _walk(node: Dict[str, Any], prefix: List[str]) -> None:
        for raw_key, value in node.items():
            key = str(raw_key).replace("-", "_")
            key_parts = [part for part in key.split(".") if part]
            if not key_parts:
                continue
            parts = prefix + key_parts
            # Only grouped style is supported for grouped fields in YAML.
            # Example: use `training: {train_backend: fsdp}` (or `training.train_backend`)
            # instead of legacy flat key `train_backend`.
            if len(parts) == 1:
                flat_key = parts[0]
                if flat_key not in _TOP_LEVEL_FIELD_NAMES and flat_key in _FLAT_FIELD_PATH_INDEX:
                    owner, sub_name = _FLAT_FIELD_PATH_INDEX[flat_key]
                    expected = (
                        f"{owner}.{sub_name}.{flat_key}"
                        if sub_name is not None
                        else f"{owner}.{flat_key}"
                    )
                    raise ValueError(
                        f"Unsupported flat YAML key '{flat_key}'. "
                        f"Use grouped YAML key '{expected}' instead."
                    )

            if isinstance(value, dict):
                leaf_dest = _resolve_yaml_leaf_dest(parts)
                if leaf_dest is not None and not _is_yaml_container_path(parts):
                    _assign(leaf_dest, value, source_path=".".join(parts))
                    continue
                if _is_yaml_container_path(parts):
                    _walk(value, parts)
                    continue
                _assign(".".join(parts), value, source_path=".".join(parts))
                continue

            leaf_dest = _resolve_yaml_leaf_dest(parts)
            if leaf_dest is not None:
                _assign(leaf_dest, value, source_path=".".join(parts))
            else:
                _assign(".".join(parts), value, source_path=".".join(parts))

    _walk(yaml_data, [])
    return flattened


def _coerce_yaml_value(
    *,
    key: str,
    cli_key: str,
    value: Any,
    action_by_dest: Dict[str, argparse.Action],
) -> Any:
    """Coerce YAML value using argparse converter for the destination key."""
    if cli_key in {"algorithm_kwargs", "train_backend_kwargs"}:
        return _parse_cli_json_object(value)

    action = action_by_dest.get(cli_key)
    converter = getattr(action, "type", None) if action is not None else None
    if converter is None or value is None:
        return value
    if converter is str and not isinstance(value, str):
        return value
    try:
        return converter(value)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"Invalid value for YAML key '{key}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid value for YAML key '{key}': {exc}") from exc


def _build_cli_option_strings(field_name: str, group_key: str) -> List[str]:
    """Build CLI option strings.

    Grouped fields use dotted CLI style only (e.g. --training.train-backend).
    """
    field_opt = field_name.replace("_", "-")
    flat_option = f"--{field_opt}"

    if not group_key:
        return [flat_option]

    dotted_group = ".".join(part.replace("_", "-") for part in group_key.split("."))
    dotted_option = f"--{dotted_group}.{field_opt}"
    return [dotted_option]


def _build_add_argument_kwargs(field_type: Any, default: Any, help_text: str, *, field_name: str = "") -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "default": default,
        "help": help_text,
    }
    # Field-specific custom parsers
    if field_name == "timestep_fraction":
        kwargs["type"] = _parse_cli_timestep_fraction
    elif field_type == bool:
        kwargs["type"] = _parse_cli_bool
    elif field_type == int:
        kwargs["type"] = int
    elif field_type == float:
        kwargs["type"] = float
    elif field_type is dict or get_origin(field_type) is dict:
        kwargs["type"] = _parse_cli_json_object
    elif get_origin(field_type) is list:
        item_type = get_args(field_type)[0] if get_args(field_type) else str
        kwargs["type"] = (
            lambda value, item_type=item_type: _parse_cli_list(value, item_type=item_type)
        )
    else:
        kwargs["type"] = str
    return kwargs


def _collect_explicit_cli_destinations(argv: List[str], parser: argparse.ArgumentParser) -> set[str]:
    """Collect parser destination names explicitly provided via CLI options."""
    explicit: set[str] = set()
    option_to_action = getattr(parser, "_option_string_actions", {})
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_to_action.get(option)
        if action is not None:
            explicit.add(action.dest)
    return explicit


_ANSI_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "orange": "\033[38;5;208m",
    "red": "\033[31m",
}


def _supports_color_output() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _color(text: str, name: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    prefix = _ANSI_COLORS.get(name, "")
    if not prefix:
        return text
    return f"{prefix}{text}{_ANSI_COLORS['reset']}"


def _format_config_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def _infer_normalize_source() -> str:
    """Best-effort infer of normalize rewrite source function name."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        skip = {
            "_infer_normalize_source",
            "_set_normalized_attr",
            "_set_normalized_dict_item",
            "_trace_normalize_change",
        }
        while frame is not None:
            name = frame.f_code.co_name
            if name not in skip:
                return name
            frame = frame.f_back
    finally:
        # Avoid reference cycles in frame objects.
        del frame
    return "<unknown>"


def _normalize_trace_enabled(args: Any) -> bool:
    return bool(getattr(args, "_print_normalize_changes", False))


def _print_normalize_header_once(args: Any) -> None:
    if not _normalize_trace_enabled(args):
        return
    if bool(getattr(args, "_normalize_change_header_printed", False)):
        return
    use_color = _supports_color_output()
    print(_color("[Normalize Changes] rewritten fields during validation", "bold", enabled=use_color))
    setattr(args, "_normalize_change_header_printed", True)


def _trace_normalize_change(
    args: Any,
    key: str,
    before: Any,
    after: Any,
    *,
    source: Optional[str] = None,
) -> None:
    if before == after or not _normalize_trace_enabled(args):
        return
    _print_normalize_header_once(args)
    use_color = _supports_color_output()
    before_text = _format_config_value(before)
    after_text = _format_config_value(after)
    source_text = str(source or _infer_normalize_source())
    print(
        f"  {_color(key, 'orange', enabled=use_color)}: "
        f"{_color(before_text, 'red', enabled=use_color)} "
        f"{_color('->', 'orange', enabled=use_color)} "
        f"{_color(after_text, 'green', enabled=use_color)} "
        f"{_color(f'(source={source_text})', 'cyan', enabled=use_color)}"
    )


def _set_normalized_attr(
    args: Any,
    owner: Any,
    attr: str,
    value: Any,
    *,
    key: str,
    source: Optional[str] = None,
) -> None:
    before = getattr(owner, attr)
    if before == value:
        return
    setattr(owner, attr, value)
    _trace_normalize_change(args, key, before, value, source=source or _infer_normalize_source())


def _set_normalized_dict_item(
    args: Any,
    mapping: Dict[str, Any],
    item: str,
    value: Any,
    *,
    key: str,
    source: Optional[str] = None,
) -> None:
    before = mapping.get(item, "<missing>")
    if before == value:
        return
    mapping[item] = value
    _trace_normalize_change(args, key, before, value, source=source or _infer_normalize_source())


def _print_config_views(
    *,
    after_flat: Dict[str, Any],
    print_resolved_config: bool,
) -> None:
    if not print_resolved_config:
        return

    use_color = _supports_color_output()

    if print_resolved_config:
        print(_color("[Resolved Config] final runtime values", "bold", enabled=use_color))
        for key in sorted(after_flat.keys()):
            value_text = _format_config_value(after_flat.get(key))
            print(f"  {_color(key, 'cyan', enabled=use_color)}: {value_text}")


def parse_args(argv: Optional[List[str]] = None) -> TrainingArguments:
    """Parse command line arguments and return TrainingArguments.

    Supports ``--config path/to/config.yaml`` for YAML-based configuration.
    CLI arguments override YAML values when both are provided.
    Grouped fields in YAML must use grouped keys (for example ``training.train_backend``
    or nested ``training: {train_backend: ...}``).
    """
    parser = argparse.ArgumentParser(
        description="diffusionrl training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --config: YAML configuration file (parsed before other args)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override YAML values.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print resolved runtime config after normalization/validation.",
    )
    parser.add_argument(
        "--allow-unknown-config-keys",
        action="store_true",
        help="Allow unknown keys in --config YAML (default is fail-fast).",
    )

    # Build argument groups for organized --help output
    _arg_groups: Dict[str, argparse._ArgumentGroup] = {}
    for field_name, field_type, default, help_text, group_key in _collect_cli_field_specs():
        # Get or create the argument group
        if group_key not in _arg_groups:
            display_name = _GROUP_DISPLAY_NAMES.get(group_key, group_key)
            _arg_groups[group_key] = parser.add_argument_group(display_name)
        group = _arg_groups[group_key]

        # Build CLI option names (dotted style for grouped args).
        option_strings = _build_cli_option_strings(field_name, group_key)
        add_kwargs = _build_add_argument_kwargs(field_type, default, help_text, field_name=field_name)
        add_kwargs["dest"] = field_name
        group.add_argument(*option_strings, **add_kwargs)

    cli_argv = list(argv) if argv is not None else sys.argv[1:]
    parsed_args = parser.parse_args(cli_argv)
    explicit_cli_keys = _collect_explicit_cli_destinations(cli_argv, parser)
    action_by_dest = {
        action.dest: action for action in parser._actions if getattr(action, "dest", None)
    }

    raw_args = vars(parsed_args)
    print_resolved_config = bool(raw_args.get("print_resolved_config", False))
    allow_unknown_config_keys = bool(raw_args.get("allow_unknown_config_keys", False))

    # YAML config merging: CLI values take precedence over YAML
    if raw_args.get("config"):
        defaults: Dict[str, Any] = {}
        for action in parser._actions:
            dest = getattr(action, "dest", None)
            if not dest or dest == "help" or dest in defaults:
                continue
            defaults[dest] = action.default
        yaml_data = _load_yaml_mapping(raw_args["config"])
        _merge_yaml_overrides(
            raw_args,
            yaml_data,
            defaults,
            explicit_cli_keys,
            action_by_dest=action_by_dest,
            allow_unknown_config_keys=allow_unknown_config_keys,
        )

    # Remove --config from raw_args (not a TrainingArguments field)
    raw_args.pop("config", None)
    raw_args.pop("print_resolved_config", None)
    raw_args.pop("allow_unknown_config_keys", None)

    grouped_kwargs: Dict[str, Dict[str, Any]] = {name: {} for name in _GROUP_CONFIG_TYPES}
    # sub_kwargs: group_name -> sub_field_name -> {leaf_field: value}
    sub_kwargs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    top_level_kwargs: Dict[str, Any] = {}
    for key, value in raw_args.items():
        path = _FLAT_FIELD_PATH_INDEX.get(key)
        if path is not None:
            group_name, sub_name = path
            if sub_name is not None:
                sub_kwargs.setdefault(group_name, {}).setdefault(sub_name, {})[key] = value
            else:
                grouped_kwargs[group_name][key] = value
        else:
            top_level_kwargs[key] = value

    for group_name, group_type in _GROUP_CONFIG_TYPES.items():
        kwargs = dict(grouped_kwargs[group_name])
        # Build sub-dataclass instances
        for info in fields(group_type):
            ft = info.type
            if isinstance(ft, type) and is_dataclass(ft):
                sub_data = sub_kwargs.get(group_name, {}).get(info.name, {})
                kwargs[info.name] = ft(**sub_data)
        top_level_kwargs[group_name] = group_type(**kwargs)

    args = TrainingArguments(**top_level_kwargs)

    # Validate and normalize arguments
    args = validate_args(args, print_normalize_changes=True)
    post_validate_flat = args.to_flat_dict()
    _print_config_views(
        after_flat=post_validate_flat,
        print_resolved_config=print_resolved_config,
    )

    return args

def _resolve_model_runtime_contract(
    args: TrainingArguments,
    *,
    explicit_sampler_path: bool,
    explicit_sampler_engine_type: bool,
) -> Any:
    """Resolve model path/type and model-declared runtime defaults."""
    if args.model.model_path == DEFAULT_MODEL_PATH:
        resolved_model_path = resolve_model_bundle_path(args.model.model_type)
        if not resolved_model_path:
            raise ValueError(
                f"Unknown model_type={args.model.model_type!r}. "
                f"Discovered model types: {list_model_types()}. "
                "Provide --model.model-path explicitly for custom models."
            )
        _set_normalized_attr(
            args,
            args.model,
            "model_path",
            resolved_model_path,
            key="model.model_path",
        )
        logger.info(
            "Auto-resolved model_type=%s to model_path=%s",
            args.model.model_type,
            args.model.model_path,
        )

    validate_dotpath(args.model.model_path, label="model")
    model_cls = load_function(args.model.model_path)

    declared_model_type_fn = getattr(model_cls, "declared_model_type", None)
    if callable(declared_model_type_fn):
        declared_model_type = declared_model_type_fn()
        if isinstance(declared_model_type, str) and declared_model_type.strip():
            normalized_declared = declared_model_type.strip().lower()
            if str(args.model.model_type).strip().lower() != normalized_declared:
                logger.info(
                    "Aligning model_type=%s to declared model_type=%s from model_path=%s",
                    args.model.model_type,
                    normalized_declared,
                    args.model.model_path,
                )
            _set_normalized_attr(
                args,
                args.model,
                "model_type",
                normalized_declared,
                key="model.model_type",
            )

    model_defaults: Dict[str, Optional[str]] = {"sampler_path": None, "sampler_engine_type": None}
    sampler_path_fn = getattr(model_cls, "default_sampler_path", None)
    if callable(sampler_path_fn):
        model_defaults["sampler_path"] = sampler_path_fn()
    engine_type_fn = getattr(model_cls, "default_sampler_engine", None)
    if callable(engine_type_fn):
        model_defaults["sampler_engine_type"] = engine_type_fn()

    if args.sampling.sampler_path == DEFAULT_SAMPLER_PATH and not explicit_sampler_path:
        model_sampler_path = model_defaults.get("sampler_path")
        if model_sampler_path:
            _set_normalized_attr(
                args,
                args.sampling,
                "sampler_path",
                model_sampler_path,
                key="sampling.sampler_path",
            )
            logger.info(
                "Auto-mapped model_path=%s to sampler_path=%s",
                args.model.model_path,
                args.sampling.sampler_path,
            )
        else:
            logger.warning(
                "Model %s does not declare default_sampler_path(); keeping sampler_path=%s.",
                args.model.model_path,
                args.sampling.sampler_path,
            )

    if args.sampling.sampler_engine_type is None:
        model_engine_type = model_defaults.get("sampler_engine_type")
        if model_engine_type:
            _set_normalized_attr(
                args,
                args.sampling,
                "sampler_engine_type",
                model_engine_type,
                key="sampling.sampler_engine_type",
            )
            logger.info(
                "Auto-selected sampler_engine_type=%s from model_path=%s",
                args.sampling.sampler_engine_type,
                args.model.model_path,
            )
        else:
            raise ValueError(
                f"Model {args.model.model_path} does not declare default_sampler_engine(). "
                "Provide --sampling.sampler-engine-type explicitly."
            )

    return model_cls


def _normalize_sampling_basics(args: TrainingArguments) -> tuple[bool, bool, str]:
    """Normalize sampling mode, engine kwargs, and sglang mode."""
    if not isinstance(args.sampling.engine_kwargs, dict):
        logger.warning("engine_kwargs is not a dict. Resetting to empty dict.")
        _set_normalized_attr(
            args,
            args.sampling,
            "engine_kwargs",
            {},
            key="sampling.engine_kwargs",
        )

    sampling_mode = str(args.sampling.sampling_mode or "rollout_actor").strip().lower()
    _set_normalized_attr(
        args,
        args.sampling,
        "sampling_mode",
        sampling_mode,
        key="sampling.sampling_mode",
    )
    direct_sampling = sampling_mode == "training_actor"

    is_sglang_engine = str(args.sampling.sampler_engine_type).lower() == "sglang"
    logprob_source = str(args.sampling.logprob_source or "replay").strip().lower()
    _set_normalized_attr(
        args,
        args.sampling,
        "logprob_source",
        logprob_source,
        key="sampling.logprob_source",
    )
    if is_sglang_engine:
        _set_normalized_dict_item(
            args,
            args.sampling.engine_kwargs,
            "logprob_source",
            logprob_source,
            key="sampling.engine_kwargs.logprob_source",
        )
    return direct_sampling, is_sglang_engine, logprob_source


def _apply_training_actor_sampling_overrides(
    args: TrainingArguments,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Apply training-actor direct-sampling-only constraints and overrides."""
    if not training_actor_sampling_mode:
        return

    logger.warning(
        "sampling_mode='training_actor' is an experimental path. "
        "Default production path remains dedicated rollout actors."
    )
    if args.sampling.sampler_engine_type != "fsdp":
        raise ValueError(
            "sampling_mode='training_actor' requires sampler_engine_type='fsdp'. "
            f"Got sampler_engine_type={args.sampling.sampler_engine_type}."
        )
    backend_name = str(getattr(args.training, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_path = str(getattr(args.training, "train_backend_path", "") or "").strip().lower()
    veomni_like_backend = backend_name == "veomni" or ("veomni" in backend_path)
    if backend_name != "fsdp" and not veomni_like_backend:
        raise ValueError(
            "sampling_mode='training_actor' currently supports train_backend='fsdp' "
            "or VeOmni-native backends. "
            f"Got train_backend={getattr(args.training, 'train_backend', None)!r}, "
            f"train_backend_path={getattr(args.training, 'train_backend_path', None)!r}."
        )

    if args.ray.rollout_num_nodes != 0 or args.ray.rollout_num_gpus_per_node != 0:
        raise ValueError(
            "sampling_mode='training_actor' requires rollout_num_nodes=0 and "
            "rollout_num_gpus_per_node=0 (no separate rollout actors). "
            f"Got rollout_num_nodes={args.ray.rollout_num_nodes}, "
            f"rollout_num_gpus_per_node={args.ray.rollout_num_gpus_per_node}."
        )

    if not args.ray.colocate_rollout_training:
        raise ValueError(
            "sampling_mode='training_actor' requires colocate_rollout_training=true. "
            "Set --ray.colocate-rollout-training."
        )

    if args.ray.offload or args.ray.offload_train or args.ray.offload_rollout:
        raise ValueError(
            "sampling_mode='training_actor' is incompatible with offload. "
            "Set --ray.offload=false --ray.offload-train=false --ray.offload-rollout=false."
        )


def _normalize_replay_mode(
    args: TrainingArguments,
    *,
    training_actor_sampling_mode: bool,
    is_sglang_engine: bool,
    logprob_source: str,
) -> tuple[bool, bool, str]:
    """Validate replay flags for sglang/non-sglang engines."""
    replay_enabled = bool(getattr(args.sampling, "replay_log_probs", False))
    replay_guard = (
        (not training_actor_sampling_mode)
        and getattr(args.algorithm, "algorithm_type", "grpo") == "grpo"
    )

    if is_sglang_engine and replay_guard:
        if logprob_source == "replay" and not replay_enabled:
            raise ValueError(
                "sampler_engine_type='sglang' with logprob_source='replay' requires "
                "--sampling.replay-log-probs=true. Set it explicitly."
            )
        if logprob_source == "native" and replay_enabled:
            raise ValueError(
                "logprob_source='native' is incompatible with replay_log_probs=true. "
                "Set --sampling.replay-log-probs=false when using native log_prob mode."
            )

    if replay_enabled and not replay_guard:
        raise ValueError(
            "replay_log_probs=true is only valid for "
            "sampling_mode='rollout_actor' + algorithm_type='grpo'. "
            "Either disable replay_log_probs or adjust your config."
        )

    return replay_guard, replay_enabled, logprob_source


def _apply_colocate_and_offload_rules(
    args: TrainingArguments,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Normalize offload and colocate flags."""
    if training_actor_sampling_mode:
        # Training-actor sampling has no separate rollout actors; keep offload disabled.
        _set_normalized_attr(args, args.ray, "offload", False, key="ray.offload")
        _set_normalized_attr(args, args.ray, "offload_train", False, key="ray.offload_train")
        _set_normalized_attr(args, args.ray, "offload_rollout", False, key="ray.offload_rollout")
        return

    if args.ray.offload:
        _set_normalized_attr(args, args.ray, "offload_train", True, key="ray.offload_train")
        _set_normalized_attr(args, args.ray, "offload_rollout", True, key="ray.offload_rollout")

    if args.ray.colocate_rollout_training:
        if args.ray.offload_train is None:
            _set_normalized_attr(args, args.ray, "offload_train", True, key="ray.offload_train")
        if args.ray.offload_rollout is None:
            _set_normalized_attr(args, args.ray, "offload_rollout", True, key="ray.offload_rollout")
        validate_colocate_fractions(args)

    if args.ray.offload_train is None:
        _set_normalized_attr(args, args.ray, "offload_train", False, key="ray.offload_train")
    if args.ray.offload_rollout is None:
        _set_normalized_attr(args, args.ray, "offload_rollout", False, key="ray.offload_rollout")


def _normalize_training_misc(args: TrainingArguments) -> None:
    """Validate misc training knobs that affect downstream components."""
    if args.algorithm.adv_normalization not in {"global", "group"}:
        raise ValueError(
            "algorithm.adv_normalization must be 'global' or 'group'. "
            f"Got: {args.algorithm.adv_normalization!r}."
        )

    if isinstance(args.training.lora_target_modules, str):
        stripped = args.training.lora_target_modules.strip()
        if stripped:
            _set_normalized_attr(
                args,
                args.training,
                "lora_target_modules",
                [s.strip() for s in stripped.split(",") if s.strip()],
                key="training.lora_target_modules",
            )
        else:
            _set_normalized_attr(
                args,
                args.training,
                "lora_target_modules",
                None,
                key="training.lora_target_modules",
            )

    _normalize_training_batch_geometry(args)


def _maybe_positive_int(value: Any) -> Optional[int]:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    if resolved < 1:
        return None
    return resolved


def _resolve_training_dp_size(args: TrainingArguments) -> int:
    default_dp_size = max(
        1,
        int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node),
    )
    backend = str(getattr(args.training, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_kwargs = getattr(args.training, "train_backend_kwargs", {})
    if not isinstance(backend_kwargs, dict):
        backend_kwargs = {}

    if backend == "veomni":
        return _maybe_positive_int(backend_kwargs.get("dp_size")) or default_dp_size

    if backend == "megatron":
        hinted_dp_size = _maybe_positive_int(backend_kwargs.get("dp_size"))
        if hinted_dp_size is not None:
            return hinted_dp_size
        tp_size = _maybe_positive_int(backend_kwargs.get("tp_size")) or 1
        pp_size = _maybe_positive_int(backend_kwargs.get("pp_size")) or 1
        sp_size = _maybe_positive_int(backend_kwargs.get("sp_size")) or 1
        denom = max(1, tp_size * pp_size * sp_size)
        return max(1, default_dp_size // denom)

    return default_dp_size


def _resolve_nominal_local_training_batch_size(args: TrainingArguments) -> int:
    total_samples = max(
        1,
        int(args.algorithm.prompts_per_rollout) * int(args.algorithm.samples_per_prompt),
    )
    dp_size = _resolve_training_dp_size(args)
    if total_samples % dp_size != 0:
        raise ValueError(
            "Nominal rollout batch size must be divisible by the effective training dp_size. "
            f"Got total_samples={total_samples}, dp_size={dp_size}. "
            "Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend topology."
        )
    return max(1, total_samples // dp_size)


def _normalize_training_batch_geometry(args: TrainingArguments) -> None:
    """Normalize training batch-geometry knobs onto explicit runtime semantics."""
    local_batch_size = _resolve_nominal_local_training_batch_size(args)
    update_mode = str(args.training.update_mode or "single_update").strip().lower()
    gradient_accumulation_batch_size = (
        max(1, int(args.training.gradient_accumulation_batch_size))
        if args.training.gradient_accumulation_batch_size is not None
        else None
    )
    multi_update_batch_size = (
        max(1, int(args.training.multi_update_batch_size))
        if args.training.multi_update_batch_size is not None
        else None
    )

    if update_mode == "multi_update":
        if multi_update_batch_size is None:
            raise ValueError(
                "update_mode='multi_update' requires training.multi_update_batch_size."
            )

        if multi_update_batch_size >= local_batch_size:
            _set_normalized_attr(
                args,
                args.training,
                "update_mode",
                "single_update",
                key="training.update_mode",
            )
            _set_normalized_attr(
                args,
                args.training,
                "multi_update_batch_size",
                None,
                key="training.multi_update_batch_size",
            )
            update_mode = "single_update"
            multi_update_batch_size = None
        elif local_batch_size % multi_update_batch_size != 0:
            raise ValueError(
                "multi_update requires the nominal local training batch size to be divisible by "
                "multi_update_batch_size. "
                f"Got local_batch_size={local_batch_size}, "
                f"multi_update_batch_size={multi_update_batch_size}."
            )

    if update_mode == "single_update":
        normalized_grad_accum_batch_size = (
            local_batch_size
            if gradient_accumulation_batch_size is None
            else min(gradient_accumulation_batch_size, local_batch_size)
        )
        if args.training.gradient_accumulation_batch_size != normalized_grad_accum_batch_size:
            _set_normalized_attr(
                args,
                args.training,
                "gradient_accumulation_batch_size",
                normalized_grad_accum_batch_size,
                key="training.gradient_accumulation_batch_size",
            )
        gradient_accumulation_batch_size = normalized_grad_accum_batch_size
        if args.training.multi_update_batch_size is not None:
            _set_normalized_attr(
                args,
                args.training,
                "multi_update_batch_size",
                None,
                key="training.multi_update_batch_size",
            )

        if local_batch_size % gradient_accumulation_batch_size != 0:
            raise ValueError(
                "single_update requires the nominal local training batch size to be divisible by "
                "gradient_accumulation_batch_size. "
                f"Got local_batch_size={local_batch_size}, "
                f"gradient_accumulation_batch_size={gradient_accumulation_batch_size}."
            )
    else:
        normalized_grad_accum_batch_size = (
            multi_update_batch_size
            if gradient_accumulation_batch_size is None
            else min(
                gradient_accumulation_batch_size,
                multi_update_batch_size,
            )
        )
        if args.training.gradient_accumulation_batch_size != normalized_grad_accum_batch_size:
            _set_normalized_attr(
                args,
                args.training,
                "gradient_accumulation_batch_size",
                normalized_grad_accum_batch_size,
                key="training.gradient_accumulation_batch_size",
            )
        gradient_accumulation_batch_size = normalized_grad_accum_batch_size

        if multi_update_batch_size % gradient_accumulation_batch_size != 0:
            raise ValueError(
                "multi_update requires multi_update_batch_size to be divisible by "
                "gradient_accumulation_batch_size. "
                f"Got multi_update_batch_size={multi_update_batch_size}, "
                f"gradient_accumulation_batch_size={gradient_accumulation_batch_size}."
            )


def _validate_direct_sampling_batch_geometry(
    args: TrainingArguments,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate prompt-batch splitting for training-actor direct sampling."""
    max_samples_per_request = getattr(args.sampling, "max_samples_per_request", None)
    if max_samples_per_request is None:
        return

    if not training_actor_sampling_mode:
        raise ValueError(
            "sampling.max_samples_per_request is only valid when "
            "sampling.sampling_mode='training_actor'."
        )

    max_samples_per_request = int(max_samples_per_request)
    prompts_per_rollout = getattr(args.algorithm, "prompts_per_rollout", None)
    if prompts_per_rollout is None:
        raise ValueError(
            "sampling.max_samples_per_request requires algorithm.prompts_per_rollout to be set explicitly."
        )
    prompts_per_rollout = int(prompts_per_rollout)
    if max_samples_per_request < 1:
        raise ValueError("sampling.max_samples_per_request must be >= 1.")
    if prompts_per_rollout < 1:
        raise ValueError("algorithm.prompts_per_rollout must be >= 1.")


def _normalize_algorithm_kwargs_payload(args: TrainingArguments) -> None:
    """Normalize algorithm_kwargs into a dictionary payload."""
    raw = getattr(args.algorithm, "algorithm_kwargs", {})
    if raw is None:
        _set_normalized_attr(
            args,
            args.algorithm,
            "algorithm_kwargs",
            {},
            key="algorithm.algorithm_kwargs",
        )
        return
    if isinstance(raw, dict):
        _set_normalized_attr(
            args,
            args.algorithm,
            "algorithm_kwargs",
            dict(raw),
            key="algorithm.algorithm_kwargs",
        )
        return
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            _set_normalized_attr(
                args,
                args.algorithm,
                "algorithm_kwargs",
                {},
                key="algorithm.algorithm_kwargs",
            )
            return
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid algorithm_kwargs: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "algorithm_kwargs must decode to a JSON object, "
                f"got: {type(parsed).__name__}"
            )
        _set_normalized_attr(
            args,
            args.algorithm,
            "algorithm_kwargs",
            dict(parsed),
            key="algorithm.algorithm_kwargs",
        )
        return
    raise ValueError(
        "algorithm_kwargs must be a JSON object (YAML mapping) "
        f"or JSON object string, got: {type(raw).__name__}"
    )


def _normalize_algorithm_path(args: TrainingArguments) -> None:
    """Resolve algorithm.algorithm_path from algorithm_type exactly once."""
    raw_algorithm_path = getattr(args.algorithm, "algorithm_path", None)
    if isinstance(raw_algorithm_path, str) and raw_algorithm_path.strip():
        normalized = raw_algorithm_path.strip()
        _set_normalized_attr(
            args,
            args.algorithm,
            "algorithm_path",
            normalized,
            key="algorithm.algorithm_path",
        )
        return

    from diffusionrl.algorithms import DEFAULT_ALGORITHM_PATHS

    algorithm_type = str(getattr(args.algorithm, "algorithm_type", "") or "").strip()
    resolved = DEFAULT_ALGORITHM_PATHS.get(algorithm_type)
    if not resolved:
        raise ValueError(
            f"Cannot resolve algorithm.algorithm_path for algorithm_type={algorithm_type!r}. "
            "Provide --algorithm.algorithm-path explicitly or register this algorithm_type."
        )
    _set_normalized_attr(
        args,
        args.algorithm,
        "algorithm_path",
        resolved,
        key="algorithm.algorithm_path",
    )


def _normalize_train_backend_config(args: TrainingArguments) -> None:
    """Normalize train-backend selection and backend kwargs JSON payload."""
    backend = str(getattr(args.training, "train_backend", "fsdp") or "fsdp").strip().lower()
    _set_normalized_attr(
        args,
        args.training,
        "train_backend",
        backend,
        key="training.train_backend",
    )

    backend_path = getattr(args.training, "train_backend_path", None)
    supported = {"fsdp", "megatron", "veomni"}
    if backend not in supported and not backend_path:
        raise ValueError(
            f"Unsupported train_backend={backend!r}. "
            f"Expected one of {sorted(supported)} or provide --training.train-backend-path."
        )
    if backend in {"megatron"} and not backend_path:
        logger.warning(
            "train_backend=%s is currently a scaffold backend: launch/topology interfaces are wired, "
            "but runtime training flow is not fully implemented yet. "
            "Use train_backend_kwargs.actor_class_path to provide a Megatron-dedicated actor.",
            backend,
        )

    raw = getattr(args.training, "train_backend_kwargs", {})
    if raw is None:
        parsed: Dict[str, Any] = {}
    elif isinstance(raw, dict):
        parsed = dict(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            parsed = {}
            if backend == "megatron" and not backend_path:
                logger.warning(
                    "train_backend=%s without actor_class_path will fail at runtime. "
                    "Set train_backend_kwargs with actor_class_path.",
                    backend,
                )
        else:
            try:
                parsed_obj = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Invalid train_backend_kwargs: {exc}") from exc
            if not isinstance(parsed_obj, dict):
                raise ValueError("train_backend_kwargs must decode to a JSON object.")
            parsed = dict(parsed_obj)
    else:
        raise ValueError(
            "train_backend_kwargs must be a JSON object (YAML mapping) "
            f"or JSON object string, got: {type(raw).__name__}"
        )

    if backend == "veomni":
        mode = str(parsed.get("data_parallel_mode", "fsdp2") or "fsdp2").strip().lower()
        if mode == "ddp":
            raise ValueError(
                "train_backend=veomni in diffusionRL does not support data_parallel_mode='ddp'. "
                "Use data_parallel_mode='fsdp2'."
            )
        if mode != "fsdp2":
            raise ValueError(
                "train_backend=veomni in diffusionRL now targets FSDP2 only. "
                "Set data_parallel_mode='fsdp2' or omit this field."
            )
    if backend == "megatron" and not backend_path and not str(parsed.get("actor_class_path", "")).strip():
        logger.warning(
            "train_backend=%s requires actor_class_path in train_backend_kwargs "
            "to launch a Megatron-specific training actor.",
            backend,
        )

    _set_normalized_attr(
        args,
        args.training,
        "train_backend_kwargs",
        parsed,
        key="training.train_backend_kwargs",
    )

def _normalize_weight_sync(args: TrainingArguments, *, training_actor_sampling_mode: bool) -> None:
    weight_sync_mode = getattr(args.ray, "weight_sync_mode", "auto")
    train_backend = str(getattr(args.training, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_path = str(getattr(args.training, "train_backend_path", "") or "").strip().lower()
    veomni_like_backend = train_backend == "veomni" or ("veomni" in backend_path)
    sampler_engine_type = str(getattr(args.sampling, "sampler_engine_type", "") or "").lower()
    is_multi_node = (
        int(getattr(args.ray, "rollout_num_nodes", 1)) > 1
        or int(getattr(args.ray, "training_num_nodes", 1)) > 1
        or int(getattr(args.reward, "reward_dedicated_num_nodes", 0)) > 1
    )

    if weight_sync_mode == "auto":
        if not training_actor_sampling_mode and sampler_engine_type == "sglang":
            if is_multi_node:
                _set_normalized_attr(
                    args,
                    args.ray,
                    "weight_sync_mode",
                    "nccl_broadcast",
                    key="ray.weight_sync_mode",
                )
            else:
                # Single-node sglang runs are more robust with checkpoint-based
                # sync under Ray actor-local CUDA_VISIBLE_DEVICES mapping.
                _set_normalized_attr(
                    args,
                    args.ray,
                    "weight_sync_mode",
                    "checkpoint_path",
                    key="ray.weight_sync_mode",
                )
        elif (
            not training_actor_sampling_mode
            and (sampler_engine_type == "fsdp" or veomni_like_backend)
        ):
            _set_normalized_attr(
                args,
                args.ray,
                "weight_sync_mode",
                "checkpoint_path",
                key="ray.weight_sync_mode",
            )
        else:
            _set_normalized_attr(
                args,
                args.ray,
                "weight_sync_mode",
                "checkpoint_path",
                key="ray.weight_sync_mode",
            )

    # Legacy alias normalization.
    _alias_map = {"ipc": "tensor_payload", "nccl": "nccl_broadcast"}
    if args.ray.weight_sync_mode in _alias_map:
        _set_normalized_attr(
            args,
            args.ray,
            "weight_sync_mode",
            _alias_map[args.ray.weight_sync_mode],
            key="ray.weight_sync_mode",
        )
    if args.ray.weight_sync_mode == "object_ref":
        logger.warning("weight_sync_mode='object_ref' is deprecated, using 'checkpoint_path'")
        _set_normalized_attr(
            args,
            args.ray,
            "weight_sync_mode",
            "checkpoint_path",
            key="ray.weight_sync_mode",
        )

    if (
        args.ray.weight_sync_mode in {"tensor_payload", "nccl_broadcast"}
        and sampler_engine_type != "sglang"
    ):
        raise ValueError(
            "weight_sync_mode in {tensor_payload,nccl_broadcast} currently requires "
            f"sampler_engine_type='sglang'. Got sampler_engine_type={sampler_engine_type!r}."
        )

    if (
        args.ray.weight_sync_mode == "checkpoint_path"
        and sampler_engine_type == "sglang"
    ):
        root = repo_root(env_repo_root=ENV_REPO_ROOT)
        default_sync_dir = resolve_repo_relative_path("outputs/weight_sync", root)
        if os.path.realpath(args.ray.weight_sync_dir) == os.path.realpath(default_sync_dir):
            _set_normalized_attr(
                args,
                args.ray,
                "weight_sync_dir",
                "/dev/shm/diffusionrl_weight_sync",
                key="ray.weight_sync_dir",
            )
            logger.info(
                "Auto-switched weight_sync_dir to %s for sglang checkpoint sync performance.",
                args.ray.weight_sync_dir,
            )

    if (
        args.ray.weight_sync_mode == "checkpoint_path"
        and is_multi_node
        and is_probably_local_weight_sync_dir(
            args.ray.weight_sync_dir,
            root=repo_root(env_repo_root=ENV_REPO_ROOT),
        )
    ):
        raise ValueError(
            "weight_sync_mode=checkpoint_path in multi-node mode requires a shared filesystem path. "
            f"Got local-only weight_sync_dir={args.ray.weight_sync_dir}. "
            "Use a shared mount (e.g. /mnt/shared/... or NFS path)."
        )


def _normalize_debug_config(args: TrainingArguments) -> str:
    """Normalize debug mode and enforce mode-specific constraints."""
    debug_mode = str(getattr(args.debug, "debug_mode", "none") or "none").strip().lower()
    _set_normalized_attr(
        args,
        args.debug,
        "debug_mode",
        debug_mode,
        key="debug.debug_mode",
    )

    if debug_mode == "train_only":
        if bool(getattr(args.rollout, "async_pipeline", False)):
            raise ValueError(
                "debug_mode=train_only does not support async_pipeline yet. "
                "Set --rollout.async-pipeline=false."
            )

    if bool(getattr(args.debug, "debug_save_intermediates", False)) and bool(getattr(args.rollout, "async_pipeline", False)):
        raise ValueError(
            "debug_save_intermediates=true is not supported with async_pipeline yet. "
            "Set --rollout.async-pipeline=false."
        )

    return debug_mode

def validate_args(
    args: TrainingArguments,
    *,
    print_normalize_changes: bool = False,
) -> TrainingArguments:
    """
    Validate and normalize arguments for colocate/offload logic.

    Args:
        args: TrainingArguments instance to validate

    Returns:
        Validated and normalized TrainingArguments
    """
    setattr(args, "_print_normalize_changes", bool(print_normalize_changes))
    setattr(args, "_normalize_change_header_printed", False)
    setattr(
        args,
        "_normalize_trace_callback",
        lambda key, before, after, source=None: _trace_normalize_change(
            args,
            key,
            before,
            after,
            source=source,
        ),
    )

    explicit_sampler_path = args.sampling.sampler_path != DEFAULT_SAMPLER_PATH
    explicit_sampler_engine_type = getattr(args.sampling, "sampler_engine_type", None) is not None

    validate_grouped_configs(args)
    debug_mode = _normalize_debug_config(args)

    normalize_repo_relative_paths(
        args,
        env_repo_root=ENV_REPO_ROOT,
        env_data_root=ENV_DATA_ROOT,
        env_model_root=ENV_MODEL_ROOT,
    )

    model_cls = _resolve_model_runtime_contract(
        args,
        explicit_sampler_path=explicit_sampler_path,
        explicit_sampler_engine_type=explicit_sampler_engine_type,
    )
    training_actor_sampling_mode, is_sglang_engine, logprob_source = _normalize_sampling_basics(args)
    _normalize_algorithm_kwargs_payload(args)
    _normalize_algorithm_path(args)
    _normalize_train_backend_config(args)
    if debug_mode == "train_only":
        # Keep train_only focused on train-side imports only. Rollout/reward/data
        # extensions are not exercised on this path and should not block replay.
        validate_dotpath(args.model.model_path, label="model")
        validate_dotpath(args.sampling.sampler_path, label="sampler")
        validate_dotpath(args.algorithm.algorithm_path, label="algorithm")
        if getattr(args.training, "train_backend_path", None):
            validate_dotpath(args.training.train_backend_path, label="train_backend")
        if getattr(args.sampling, "replay_sampler_path", None):
            validate_dotpath(args.sampling.replay_sampler_path, label="replay_sampler")
    else:
        validate_dynamic_dotpaths(args)
    _apply_training_actor_sampling_overrides(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
    )
    replay_guard, _, logprob_source = _normalize_replay_mode(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
        is_sglang_engine=is_sglang_engine,
        logprob_source=logprob_source,
    )
    _apply_colocate_and_offload_rules(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
    )

    if debug_mode != "train_only":
        validate_reward_and_rollout_buffer_config(args)
        validate_rollout_layout(
            args,
            training_actor_sampling_mode=training_actor_sampling_mode,
        )
    _normalize_training_misc(args)
    _validate_direct_sampling_batch_geometry(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
    )
    validate_model_specific_logic(args, model_cls=model_cls)
    if debug_mode != "train_only":
        validate_resolved_engine_algorithm_contract(
            args,
            training_actor_sampling_mode=training_actor_sampling_mode,
            is_sglang_engine=is_sglang_engine,
            replay_guard=replay_guard,
            logprob_source=logprob_source,
        )

    if args.sampling.sampling_adapter:
        _set_normalized_dict_item(
            args,
            args.sampling.engine_kwargs,
            "sampling_adapter",
            args.sampling.sampling_adapter,
            key="sampling.engine_kwargs.sampling_adapter",
        )
    _normalize_weight_sync(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
    )
    if debug_mode != "train_only":
        validate_runtime_mode_constraints(
            args,
            training_actor_sampling_mode=training_actor_sampling_mode,
            model_cls=model_cls,
        )

    for transient in (
        "_print_normalize_changes",
        "_normalize_change_header_printed",
        "_normalize_trace_callback",
    ):
        if hasattr(args, transient):
            delattr(args, transient)

    return args


def get_default_args() -> TrainingArguments:
    """Get default arguments without parsing command line."""
    return TrainingArguments()
