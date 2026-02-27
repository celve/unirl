"""
diffusionrl Arguments - Configuration parameters for training.

Reference: slime/utils/arguments.py
"""
import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_args, get_origin

from diffusionrl.models import list_model_types, resolve_model_bundle_path
from diffusionrl.utils.misc import load_function
from diffusionrl.config.validation import (
    DEFAULT_LOSS_TYPE_REQUIREMENTS,
    is_probably_local_weight_sync_dir,
    normalize_repo_relative_paths,
    repo_root,
    resolve_repo_relative_path,
    validate_algorithm_loss_consistency,
    validate_algorithm_kwargs_json,
    validate_colocate_fractions,
    validate_dotpath,
    validate_dynamic_dotpaths,
    validate_grouped_configs,
    validate_loss_kwargs_json,
    validate_model_specific_logic,
    validate_resolved_engine_loss_contract,
    validate_reward_and_rollout_buffer_config,
    validate_rollout_layout,
    validate_runtime_mode_constraints,
)

logger = logging.getLogger(__name__)

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"
ENV_DATA_ROOT = "DIFFUSIONRL_DATA_ROOT"
ENV_MODEL_ROOT = "DIFFUSIONRL_MODEL_ROOT"

# Local model path -> HuggingFace ID fallback mapping.
# When a script specifies a repo-relative local path (e.g., "models/local/sd3.5-medium")
# and the path does not exist on disk, we fall back to the corresponding HF repo ID.
# This allows GitHub users to run scripts without local models — diffusers will
# download from HuggingFace automatically.
_LOCAL_TO_HF_FALLBACK: dict[str, str] = {
    "models/local/sd3.5-medium":    "stabilityai/stable-diffusion-3.5-medium",
    "models/local/flux":            "black-forest-labs/FLUX.1-dev",
    "models/local/hunyuan-video":   "hunyuanvideo-community/HunyuanVideo",
    "models/local/wan2.1":          "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "models/local/qwen-image":      "Qwen/Qwen-Image",
    "models/local/qwen-image-edit": "Qwen/Qwen-Image-Edit",
    "models/local/sd-v1-4":         "CompVis/stable-diffusion-v1-4",
    "models/local/skyreels-i2v":    "xzyhku/SkyReels-V1-I2V",
    "models/local/bagel":           "ByteDance-Seed/BAGEL-7B-MoT",
    "models/local/qwen2-vl":        "Qwen/Qwen2-VL-2B-Instruct",
}

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
                "Set --model-type (hunyuan, flux, sd3, mochi) or provide --model-path explicitly."
            )


@dataclass
class SamplingConfig:
    """Sampling engine, sampler, and denoising controls."""

    sampler_path: str = field(default=DEFAULT_SAMPLER_PATH,
        metadata={"help": "Python dotpath to Sampler class (auto-resolved from model_type)"})
    sampler_engine_type: Optional[str] = field(default=None,
        metadata={"help": "Rollout engine type: fsdp or sglang (auto-resolved from model)"})
    training_actor_direct_sampling: bool = field(default=False,
        metadata={"help": "Training actors handle sampling directly (FSDP-only, no rollout actors)"})
    sglang_logprob_mode: str = field(default="replay",
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
    shift: float = field(default=3.0,
        metadata={"help": "Timestep schedule shift parameter (model-specific)"})
    guidance_scale: float = field(default=7.5,
        metadata={"help": "Classifier-free guidance scale (0.0 = no guidance)"})
    mixed_sampling: bool = field(default=False,
        metadata={"help": "Mix SDE and ODE steps during sampling"})
    sde_ratio: float = field(default=1.0,
        metadata={"help": "Fraction of steps that use SDE when mixed_sampling=true"})
    timestep_fraction: float = field(default=1.0,
        metadata={"help": "Fraction of total timesteps to train on (e.g. 0.6 = last 60%%)"})
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
        mode = str(self.sglang_logprob_mode).strip().lower()
        if mode not in ("replay", "native"):
            raise ValueError(
                f"sglang_logprob_mode must be one of replay/native, got: {self.sglang_logprob_mode}"
            )
        if not self.sampler_path:
            raise ValueError(
                "sampler_path must be set. It is usually auto-resolved from model_type. "
                "Set --model-type or provide --sampler-path explicitly."
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
    reward_aggregation: str = field(default="weighted_sum",
        metadata={"help": "Multi-reward aggregation method: weighted_sum"})
    reward_mix_mode: str = field(default="reward_aggr",
        metadata={"help": "Multi-reward mixing: reward_aggr (mix rewards) or advantage_aggr (mix advantages)"})
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
    local_reward_device: str = field(default="cpu",
        metadata={"help": "Device for local (non-HTTP, non-dedicated) reward workers: cpu, auto, or cuda"})
    allow_local_reward_cuda_contention: bool = field(default=False,
        metadata={"help": "Allow local_reward_device=cuda without dedicated reward GPUs (may contend with rollout/training GPUs)"})

    def validate(self) -> None:
        if self.reward_mix_mode not in ("reward_aggr", "advantage_aggr"):
            raise ValueError(
                f"reward_mix_mode must be one of reward_aggr/advantage_aggr, got: {self.reward_mix_mode}"
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
        metadata={"help": "Weight sync mode: auto, object_ref, or checkpoint_path"})
    weight_sync_dir: str = field(default="outputs/weight_sync",
        metadata={"help": "Directory for checkpoint-based weight sync (use shared FS for multi-node)"})

    def validate(self) -> None:
        if self.weight_sync_mode not in ("auto", "object_ref", "checkpoint_path"):
            raise ValueError(
                f"weight_sync_mode must be one of auto/object_ref/checkpoint_path, "
                f"got: {self.weight_sync_mode!r}. "
                "Use 'auto' (recommended) to let diffusionrl choose the best mode, "
                "'checkpoint_path' for SGLang/multi-node, or 'object_ref' for single-node FSDP."
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
    """Algorithm, loss, and timestep/window scheduler controls."""

    # Dynamic loading path
    algorithm_path: str = field(default="diffusionrl.algorithms.grpo.GRPOAlgorithm",
        metadata={"help": "Python dotpath to Algorithm class (GRPOAlgorithm, NFTAlgorithm, MixGRPOAlgorithm)"})

    # Advantage and policy objective
    clip_range: float = field(default=1e-4,
        metadata={"help": "PPO clipping range for policy ratio. Smaller = more conservative"})
    clip_range_mode: str = field(default="constant",
        metadata={"help": "Clip range schedule: constant, linear_decay, cosine_decay"})
    kl_coef: float = field(default=0.01,
        metadata={"help": "KL divergence penalty coefficient"})
    use_kl_penalty: bool = field(default=True,
        metadata={"help": "Add KL penalty term to the loss"})
    advantage_type: str = field(default="group",
        metadata={"help": "Advantage normalization: global, group (per-prompt), or per_prompt (tracked)"})
    num_samples_per_prompt: int = field(default=4,
        metadata={"help": "Number of generated samples per prompt for GRPO"})
    prompts_per_batch: int = field(default=1,
        metadata={"help": "Number of unique prompts per rollout step"})
    advantage_epsilon: float = field(default=1e-8,
        metadata={"help": "Epsilon for numerical stability in advantage normalization"})
    advantage_clip_max: Optional[float] = field(default=None,
        metadata={"help": "Max absolute advantage value (None = no clipping)"})
    use_per_prompt_stat_tracker: bool = field(default=False,
        metadata={"help": "Track per-prompt running statistics for advantage normalization"})
    per_prompt_mode: str = field(default="running",
        metadata={"help": "Per-prompt stats mode: running (EMA tracker) or batch (per-batch stats)"})
    per_prompt_buffer_size: int = field(default=16,
        metadata={"help": "Buffer size for per-prompt running statistics"})
    per_prompt_min_count: int = field(default=2,
        metadata={"help": "Minimum samples before per-prompt stats are used"})
    use_global_std: bool = field(default=False,
        metadata={"help": "Use global (cross-prompt) std for advantage normalization"})
    use_running_stats: bool = field(default=False,
        metadata={"help": "Use running mean/std for advantage normalization"})
    running_stats_warmup: int = field(default=0,
        metadata={"help": "Number of warmup steps before using running stats"})

    # Loss selection and generic loss knobs
    loss_type: str = field(default="grpo",
        metadata={"help": "Loss function type: grpo or nft"})
    loss_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to custom loss class (overrides loss_type)"})
    algorithm_kwargs_json: str = field(default="",
        metadata={"help": "JSON string of extra kwargs passed to algorithm.from_args()"})
    loss_kwargs_json: str = field(default="",
        metadata={"help": "JSON string of extra kwargs passed to the loss function"})
    ignore_last: bool = field(default=False,
        metadata={"help": "Skip last timestep (t->0) in loss (can be numerically unstable)"})
    frozen_init_timesteps: int = field(default=0,
        metadata={"help": "Skip first N timesteps in loss computation (frozen warmup)"})

    # Sub-configuration
    window: WindowSchedulerConfig = field(default_factory=WindowSchedulerConfig)

    def validate(self) -> None:
        if not self.algorithm_path:
            raise ValueError(
                "algorithm_path must be set. Available: "
                "diffusionrl.algorithms.grpo.GRPOAlgorithm, "
                "diffusionrl.algorithms.nft.NFTAlgorithm, "
                "diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm"
            )
        if self.num_samples_per_prompt < 1:
            raise ValueError("num_samples_per_prompt must be >= 1.")
        if self.prompts_per_batch < 1:
            raise ValueError("prompts_per_batch must be >= 1.")
        if str(self.advantage_type).lower() == "group" and self.num_samples_per_prompt < 2:
            raise ValueError(
                "advantage_type='group' requires num_samples_per_prompt >= 2. "
                "With 1 sample per prompt, group normalization is ill-defined and can produce NaN advantages."
            )
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
    batch_size: int = field(default=4,
        metadata={"help": "Per-GPU training batch size"})
    gradient_accumulation_steps: str = field(default="1",
        metadata={"help": "Gradient accumulation steps (integer or 'auto')"})
    gradient_steps_per_epoch: int = field(default=1,
        metadata={"help": "Gradient steps per epoch when gradient_accumulation_steps='auto'"})
    num_inner_epochs: int = field(default=1,
        metadata={"help": "Number of repeated update passes per rollout batch"})
    learning_rate: float = field(default=1e-6,
        metadata={"help": "Peak learning rate for the optimizer"})
    adam_beta1: float = field(default=0.9,
        metadata={"help": "Adam optimizer beta1 (first moment decay)"})
    adam_beta2: float = field(default=0.999,
        metadata={"help": "Adam optimizer beta2 (second moment decay)"})
    adam_epsilon: float = field(default=1e-8,
        metadata={"help": "Adam optimizer epsilon for numerical stability"})
    weight_decay: float = field(default=0.0,
        metadata={"help": "Weight decay (L2 regularization) coefficient"})
    max_grad_norm: float = field(default=1.0,
        metadata={"help": "Max gradient norm for gradient clipping"})
    warmup_steps: int = field(default=0,
        metadata={"help": "Number of learning rate warmup steps"})
    lr_scheduler_type: str = field(default="constant",
        metadata={"help": "LR scheduler type: constant, linear, cosine"})

    # Train backend
    train_backend: str = field(default="fsdp",
        metadata={"help": "Training backend name (fsdp/veomni built-in; megatron scaffold requires actor_class_path in train_backend_kwargs_json); or custom via train_backend_path"})
    train_backend_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to custom TrainBackend class (overrides built-in backend selection)"})
    train_backend_kwargs_json: str = field(default="",
        metadata={"help": "JSON object string forwarded to the selected train backend"})

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
        metadata={"help": "FSDP sharding: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD"})
    fsdp_cpu_offload: bool = field(default=False,
        metadata={"help": "Offload FSDP parameters and gradients to CPU"})
    fsdp_backward_prefetch: str = field(default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy: BACKWARD_PRE, BACKWARD_POST"})

    # Memory optimization
    use_gradient_checkpointing: bool = field(default=False,
        metadata={"help": "Enable gradient checkpointing to save memory at the cost of compute"})

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.num_inner_epochs < 1:
            raise ValueError("num_inner_epochs must be >= 1.")
        backend = str(self.train_backend or "fsdp").strip().lower()
        supported = {"fsdp", "megatron", "veomni"}
        if backend not in supported and not self.train_backend_path:
            raise ValueError(
                f"Unsupported train_backend={self.train_backend!r}. "
                f"Expected one of {sorted(supported)} or provide --train-backend-path."
            )
        raw_backend_kwargs = self.train_backend_kwargs_json
        if raw_backend_kwargs is None:
            return
        if not isinstance(raw_backend_kwargs, str):
            raise ValueError(
                f"train_backend_kwargs_json must be a JSON object string, got: {type(raw_backend_kwargs).__name__}"
            )
        text = raw_backend_kwargs.strip()
        if not text:
            return
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid train_backend_kwargs_json: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("train_backend_kwargs_json must decode to a JSON object.")


@dataclass
class RolloutLoggingConfig:
    """Rollout loop/buffer, checkpoint/eval, and logging controls."""

    # Optional custom rollout pipeline function (slime-style pluggable rollout)
    rollout_pipeline_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to custom rollout pipeline function (prompts/engine/reward_fn style)"})

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
        metadata={"help": "Group samples by prompt in the rollout buffer"})
    rollout_buffer_group_size: Optional[int] = field(default=None,
        metadata={"help": "Samples per group (defaults to num_samples_per_prompt)"})
    rollout_buffer_dispatch_groups: int = field(default=0,
        metadata={"help": "Number of prompt-groups merged into one training batch (0 = prompts_per_batch)"})
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
        metadata={"help": "Total number of rollout iterations (training steps)"})
    start_rollout_id: int = field(default=0,
        metadata={"help": "Starting rollout ID (for resuming training)"})
    rollouts_per_epoch: Optional[int] = field(default=None,
        metadata={"help": "Number of rollouts per data epoch (None = single pass)"})
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
        metadata={"help": "Save a checkpoint every N training steps"})
    resume_from_checkpoint: Optional[str] = field(default=None,
        metadata={"help": "Path to checkpoint directory to resume training from"})

    # Evaluation
    eval_steps: int = field(default=100,
        metadata={"help": "Run evaluation every N training steps"})
    eval_batch_size: int = field(default=4,
        metadata={"help": "Batch size for evaluation"})

    # Logging
    logging_steps: int = field(default=10,
        metadata={"help": "Log metrics every N training steps"})
    logging_dir: Optional[str] = field(default=None,
        metadata={"help": "Directory for TensorBoard/WandB logs (defaults to output_dir/logs)"})
    report_to: str = field(default="tensorboard",
        metadata={"help": "Logging backend: tensorboard, wandb, or none"})
    project_name: str = field(default="diffusionrl",
        metadata={"help": "Project name for WandB/TensorBoard"})
    run_name: Optional[str] = field(default=None,
        metadata={"help": "Run name for WandB/TensorBoard (auto-generated if None)"})

    def validate(self) -> None:
        if self.num_rollout < 1:
            raise ValueError("num_rollout must be >= 1.")
        if self.update_weights_interval < 1:
            raise ValueError("update_weights_interval must be >= 1.")


_GROUP_CONFIG_TYPES = {
    "model": ModelConfig,
    "sampling": SamplingConfig,
    "reward": RewardConfig,
    "ray": RayConfig,
    "algorithm": AlgorithmConfig,
    "training": TrainingConfig,
    "rollout": RolloutLoggingConfig,
}
_GROUP_CONFIG_NAMES = set(_GROUP_CONFIG_TYPES.keys())


# Names of sub-dataclass fields within group configs (currently only "window")
_SUB_CONFIG_NAMES: set[str] = set()


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
                _SUB_CONFIG_NAMES.add(config_field.name)
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


def is_training_actor_direct_sampling_mode(args: Any) -> bool:
    """Return whether training actors should directly handle sampling."""
    return bool(getattr(args, "training_actor_direct_sampling", False))

@dataclass
class TrainingArguments:
    """All configuration parameters for GRPO training."""

    # ========== Paths (Dynamic Loading) ==========
    data_source_path: str = field(default="diffusionrl.data.DefaultDataSource",
        metadata={"help": "Python dotpath to DataSource class for loading training prompts"})

    # ========== Grouped Configuration ==========
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutLoggingConfig = field(default_factory=RolloutLoggingConfig)

    # ========== Data Configuration ==========
    data_path: Optional[str] = field(default="data/samples/prompts_toy.json",
        metadata={"help": "Path to training prompt data file (JSON, JSONL, or TXT)"})
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

    # ========== Misc ==========
    debug: bool = field(default=False,
        metadata={"help": "Enable debug mode (extra logging and assertions)"})

    def __getattr__(self, name: str) -> Any:
        path = _FLAT_FIELD_PATH_INDEX.get(name)
        if path is not None:
            group_name, sub_name = path
            group = object.__getattribute__(self, group_name)
            if sub_name is not None:
                return getattr(getattr(group, sub_name), name)
            return getattr(group, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        path = _FLAT_FIELD_PATH_INDEX.get(name)
        if path is not None and name not in _GROUP_CONFIG_NAMES and name not in _SUB_CONFIG_NAMES:
            group_name, sub_name = path
            if group_name in self.__dict__:
                group = self.__dict__[group_name]
                if sub_name is not None:
                    setattr(getattr(group, sub_name), name, value)
                else:
                    setattr(group, name, value)
                return
        super().__setattr__(name, value)

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
}


def _load_yaml_mapping(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return a flat dict of key-value pairs."""
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
) -> None:
    """Apply YAML values to raw_args for keys the user did NOT explicitly set on CLI."""
    all_known_keys = set(defaults.keys())
    for key, value in yaml_data.items():
        cli_key = key.replace("-", "_")
        if cli_key not in all_known_keys:
            warnings.warn(
                f"Unknown key '{key}' in YAML config (no matching CLI argument). Ignoring.",
                stacklevel=3,
            )
            continue
        # Only apply YAML value if user did not explicitly set via CLI
        if cli_key in explicit_cli_keys:
            continue
        if raw_args.get(cli_key) == defaults.get(cli_key):
            raw_args[cli_key] = value


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


def parse_args(argv: Optional[List[str]] = None) -> TrainingArguments:
    """Parse command line arguments and return TrainingArguments.

    Supports ``--config path/to/config.yaml`` for YAML-based configuration.
    CLI arguments override YAML values when both are provided.
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

    # Build argument groups for organized --help output
    _arg_groups: Dict[str, argparse._ArgumentGroup] = {}
    for field_name, field_type, default, help_text, group_key in _collect_cli_field_specs():
        # Get or create the argument group
        if group_key not in _arg_groups:
            display_name = _GROUP_DISPLAY_NAMES.get(group_key, group_key)
            _arg_groups[group_key] = parser.add_argument_group(display_name)
        group = _arg_groups[group_key]

        # Convert field name to argument name
        arg_name = f"--{field_name.replace('_', '-')}"

        if field_type == bool:
            group.add_argument(
                arg_name,
                type=_parse_cli_bool,
                default=default,
                help=help_text,
            )
        elif field_type == int:
            group.add_argument(arg_name, type=int, default=default, help=help_text)
        elif field_type == float:
            group.add_argument(arg_name, type=float, default=default, help=help_text)
        elif field_type is dict or get_origin(field_type) is dict:
            group.add_argument(arg_name, type=_parse_cli_json_object, default=default, help=help_text)
        else:
            group.add_argument(arg_name, type=str, default=default, help=help_text)

    cli_argv = list(argv) if argv is not None else sys.argv[1:]
    parsed_args = parser.parse_args(cli_argv)

    raw_args = vars(parsed_args)

    # YAML config merging: CLI values take precedence over YAML
    if raw_args.get("config"):
        defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
        yaml_data = _load_yaml_mapping(raw_args["config"])
        explicit_cli_keys = _collect_explicit_cli_destinations(cli_argv, parser)
        _merge_yaml_overrides(raw_args, yaml_data, defaults, explicit_cli_keys)

    # Remove --config from raw_args (not a TrainingArguments field)
    raw_args.pop("config", None)

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
    args = validate_args(args)

    return args

def _resolve_model_runtime_contract(
    args: TrainingArguments,
    *,
    explicit_sampler_path: bool,
    explicit_sampler_engine_type: bool,
) -> Any:
    """Resolve model path/type and model-declared runtime defaults."""
    if args.model_path == DEFAULT_MODEL_PATH:
        resolved_model_path = resolve_model_bundle_path(args.model_type)
        if not resolved_model_path:
            raise ValueError(
                f"Unknown model_type={args.model_type!r}. "
                f"Discovered model types: {list_model_types()}. "
                "Provide --model-path explicitly for custom models."
            )
        args.model_path = resolved_model_path
        logger.info(
            "Auto-resolved model_type=%s to model_path=%s",
            args.model_type,
            args.model_path,
        )

    validate_dotpath(args.model_path, label="model")
    model_cls = load_function(args.model_path)

    declared_model_type_fn = getattr(model_cls, "declared_model_type", None)
    if callable(declared_model_type_fn):
        declared_model_type = declared_model_type_fn()
        if isinstance(declared_model_type, str) and declared_model_type.strip():
            normalized_declared = declared_model_type.strip().lower()
            if str(args.model_type).strip().lower() != normalized_declared:
                logger.info(
                    "Aligning model_type=%s to declared model_type=%s from model_path=%s",
                    args.model_type,
                    normalized_declared,
                    args.model_path,
                )
            args.model_type = normalized_declared

    model_defaults: Dict[str, Optional[str]] = {"sampler_path": None, "sampler_engine_type": None}
    sampler_path_fn = getattr(model_cls, "default_sampler_path", None)
    if callable(sampler_path_fn):
        model_defaults["sampler_path"] = sampler_path_fn()
    engine_type_fn = getattr(model_cls, "default_sampler_engine", None)
    if callable(engine_type_fn):
        model_defaults["sampler_engine_type"] = engine_type_fn()

    if (
        args.sampler_path == DEFAULT_SAMPLER_PATH
        and not explicit_sampler_path
        and not explicit_sampler_engine_type
    ):
        model_sampler_path = model_defaults.get("sampler_path")
        if model_sampler_path:
            args.sampler_path = model_sampler_path
            logger.info(
                "Auto-mapped model_path=%s to sampler_path=%s",
                args.model_path,
                args.sampler_path,
            )
        else:
            logger.warning(
                "Model %s does not declare default_sampler_path(); keeping sampler_path=%s.",
                args.model_path,
                args.sampler_path,
            )

    if args.sampler_engine_type is None:
        model_engine_type = model_defaults.get("sampler_engine_type")
        if model_engine_type:
            args.sampler_engine_type = model_engine_type
            logger.info(
                "Auto-selected sampler_engine_type=%s from model_path=%s",
                args.sampler_engine_type,
                args.model_path,
            )
        else:
            raise ValueError(
                f"Model {args.model_path} does not declare default_sampler_engine(). "
                "Provide --sampler-engine-type explicitly."
            )

    return model_cls


def _normalize_sampling_basics(args: TrainingArguments) -> tuple[bool, bool, str]:
    """Normalize direct-sampling mode, engine kwargs, and sglang mode."""
    if not isinstance(args.engine_kwargs, dict):
        logger.warning("engine_kwargs is not a dict. Resetting to empty dict.")
        args.engine_kwargs = {}

    direct_sampling = bool(getattr(args, "training_actor_direct_sampling", False))
    args.training_actor_direct_sampling = direct_sampling

    is_sglang_engine = str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    sglang_logprob_mode = str(getattr(args, "sglang_logprob_mode", "replay") or "replay").strip().lower()
    args.sglang_logprob_mode = sglang_logprob_mode
    if is_sglang_engine:
        args.engine_kwargs["sglang_logprob_mode"] = sglang_logprob_mode
    return direct_sampling, is_sglang_engine, sglang_logprob_mode


def _apply_training_actor_direct_sampling_overrides(
    args: TrainingArguments,
    *,
    training_actor_direct_sampling: bool,
) -> None:
    """Apply training-actor direct-sampling-only constraints and overrides."""
    if not training_actor_direct_sampling:
        return

    logger.warning(
        "training_actor_direct_sampling=true is an experimental path. "
        "Default production path remains dedicated rollout actors."
    )
    if args.sampler_engine_type != "fsdp":
        raise ValueError(
            "training_actor_direct_sampling=true requires sampler_engine_type='fsdp'. "
            f"Got sampler_engine_type={args.sampler_engine_type}."
        )
    backend_name = str(getattr(args, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_path = str(getattr(args, "train_backend_path", "") or "").strip().lower()
    veomni_like_backend = backend_name == "veomni" or ("veomni" in backend_path)
    if backend_name != "fsdp" and not veomni_like_backend:
        raise ValueError(
            "training_actor_direct_sampling=true currently supports train_backend='fsdp' "
            "or VeOmni-native backends. "
            f"Got train_backend={getattr(args, 'train_backend', None)!r}, "
            f"train_backend_path={getattr(args, 'train_backend_path', None)!r}."
        )

    if args.rollout_num_nodes != 0 or args.rollout_num_gpus_per_node != 0:
        logger.warning(
            "training_actor_direct_sampling=true: disabling rollout placement groups "
            f"(rollout_num_nodes={args.rollout_num_nodes}, "
            f"rollout_num_gpus_per_node={args.rollout_num_gpus_per_node})."
        )
        args.rollout_num_nodes = 0
        args.rollout_num_gpus_per_node = 0

    # Training-actor sampling is an implicit colocate mode:
    # rollout actors are disabled and training actors serve generation.
    if not args.colocate_rollout_training:
        logger.info(
            "training_actor_direct_sampling=true: forcing colocate_rollout_training=true "
            "(training actors directly handle sampling)."
        )
    args.colocate_rollout_training = True

    if args.offload or args.offload_train or args.offload_rollout:
        logger.warning(
            "training_actor_direct_sampling=true: disabling offload_train/offload_rollout."
        )
    args.offload = False
    args.offload_train = False
    args.offload_rollout = False


def _normalize_replay_mode(
    args: TrainingArguments,
    *,
    training_actor_direct_sampling: bool,
    is_sglang_engine: bool,
    sglang_logprob_mode: str,
) -> tuple[bool, bool, str]:
    """Normalize replay flags for sglang/non-sglang engines."""
    replay_enabled = bool(getattr(args, "replay_log_probs", False))
    replay_guard = (not training_actor_direct_sampling) and getattr(args, "loss_type", "grpo") == "grpo"

    if (
        args.model_type == "mochi"
        and is_sglang_engine
        and sglang_logprob_mode == "replay"
        and not getattr(args, "replay_sampler_path", None)
    ):
        logger.warning(
            "Mochi + SGLang no longer uses FastVideo replay fallback. "
            "Auto-switching sglang_logprob_mode to 'native'."
        )
        sglang_logprob_mode = "native"
        args.sglang_logprob_mode = sglang_logprob_mode
        args.engine_kwargs["sglang_logprob_mode"] = sglang_logprob_mode

    if is_sglang_engine and replay_guard:
        if sglang_logprob_mode == "replay":
            if not replay_enabled:
                logger.info(
                    "SGLang log_prob mode='replay': auto-enabled replay_log_probs "
                    "for sampler_engine_type='sglang' + loss_type='grpo'."
                )
            replay_enabled = True
            args.replay_log_probs = True
        else:
            if replay_enabled:
                logger.warning(
                    "SGLang log_prob mode='native' overrides replay_log_probs flags; disabling replay path."
                )
            replay_enabled = False
            args.replay_log_probs = False
            logger.info(
                "SGLang log_prob mode='native': expecting rollout/native log_probs from inference engine."
            )

    if replay_enabled and not replay_guard:
        logger.warning(
            "replay_log_probs=true is only valid for "
            "training_actor_direct_sampling=false + loss_type='grpo'. "
            "Disabling replay_log_probs."
        )
        args.replay_log_probs = False
    else:
        args.replay_log_probs = replay_enabled

    return replay_guard, replay_enabled, sglang_logprob_mode


def _apply_colocate_and_offload_rules(
    args: TrainingArguments,
    *,
    training_actor_direct_sampling: bool,
) -> None:
    """Normalize offload and colocate flags."""
    if training_actor_direct_sampling:
        # Training-actor sampling has no separate rollout actors; keep offload disabled.
        args.offload = False
        args.offload_train = False
        args.offload_rollout = False
        return

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True

    if args.colocate_rollout_training:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        validate_colocate_fractions(args)

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False


def _normalize_training_misc(args: TrainingArguments) -> None:
    """Normalize misc training knobs that affect downstream components."""
    if args.advantage_type == "per_prompt":
        if args.per_prompt_mode == "running" and not args.use_per_prompt_stat_tracker:
            args.use_per_prompt_stat_tracker = True
            logger.info("Auto-enabled use_per_prompt_stat_tracker for advantage_type='per_prompt'")
        elif args.per_prompt_mode != "running" and args.use_per_prompt_stat_tracker:
            logger.info("per_prompt_mode != 'running' will ignore per_prompt_stat_tracker")

    if args.use_global_std:
        args.use_running_stats = True

    if isinstance(args.lora_target_modules, str):
        stripped = args.lora_target_modules.strip()
        if stripped:
            args.lora_target_modules = [s.strip() for s in stripped.split(",") if s.strip()]
        else:
            args.lora_target_modules = None

    if isinstance(args.gradient_accumulation_steps, int):
        args.gradient_accumulation_steps = str(args.gradient_accumulation_steps)

    if args.num_inner_epochs < 1:
        raise ValueError(f"num_inner_epochs must be >= 1, got: {args.num_inner_epochs}")


def _normalize_train_backend_config(args: TrainingArguments) -> None:
    """Normalize train-backend selection and backend kwargs JSON payload."""
    backend = str(getattr(args, "train_backend", "fsdp") or "fsdp").strip().lower()
    args.train_backend = backend

    backend_path = getattr(args, "train_backend_path", None)
    supported = {"fsdp", "megatron", "veomni"}
    if backend not in supported and not backend_path:
        raise ValueError(
            f"Unsupported train_backend={backend!r}. "
            f"Expected one of {sorted(supported)} or provide --train-backend-path."
        )
    if backend in {"megatron"} and not backend_path:
        logger.warning(
            "train_backend=%s is currently a scaffold backend: launch/topology interfaces are wired, "
            "but runtime training flow is not fully implemented yet. "
            "Use train_backend_kwargs_json.actor_class_path to provide a Megatron-dedicated actor.",
            backend,
        )

    raw = getattr(args, "train_backend_kwargs_json", "")
    if raw is None:
        args.train_backend_kwargs_json = ""
    elif not isinstance(raw, str):
        raise ValueError(
            f"train_backend_kwargs_json must be a JSON object string, got: {type(raw).__name__}"
        )
    else:
        text = raw.strip()
        if not text:
            args.train_backend_kwargs_json = ""
            if backend == "megatron" and not backend_path:
                logger.warning(
                    "train_backend=%s without actor_class_path will fail at runtime. "
                    "Set train_backend_kwargs_json with actor_class_path.",
                    backend,
                )
        else:
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Invalid train_backend_kwargs_json: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("train_backend_kwargs_json must decode to a JSON object.")
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
                    "train_backend=%s requires actor_class_path in train_backend_kwargs_json "
                    "to launch a Megatron-specific training actor.",
                    backend,
                )
            args.train_backend_kwargs_json = json.dumps(parsed)

def _normalize_weight_sync(args: TrainingArguments, *, training_actor_direct_sampling: bool) -> None:
    weight_sync_mode = getattr(args, "weight_sync_mode", "auto")
    train_backend = str(getattr(args, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_path = str(getattr(args, "train_backend_path", "") or "").strip().lower()
    veomni_like_backend = train_backend == "veomni" or ("veomni" in backend_path)
    if weight_sync_mode == "auto":
        if (
            not training_actor_direct_sampling
            and (
                args.sampler_engine_type in ("fsdp", "sglang")
                or veomni_like_backend
            )
        ):
            args.weight_sync_mode = "checkpoint_path"
        else:
            args.weight_sync_mode = "object_ref"

    if (
        not training_actor_direct_sampling
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
        and args.weight_sync_mode != "checkpoint_path"
    ):
        logger.warning(
            "sampler_engine_type='sglang' does not support object_ref tensor push; "
            "forcing weight_sync_mode=checkpoint_path."
        )
        args.weight_sync_mode = "checkpoint_path"

    if (
        args.weight_sync_mode == "checkpoint_path"
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    ):
        root = repo_root(env_repo_root=ENV_REPO_ROOT)
        default_sync_dir = resolve_repo_relative_path("outputs/weight_sync", root)
        if os.path.realpath(args.weight_sync_dir) == os.path.realpath(default_sync_dir):
            args.weight_sync_dir = "/dev/shm/diffusionrl_weight_sync"
            logger.info(
                "Auto-switched weight_sync_dir to %s for sglang checkpoint sync performance.",
                args.weight_sync_dir,
            )

    if (
        args.weight_sync_mode == "checkpoint_path"
        and (
            int(getattr(args, "rollout_num_nodes", 1)) > 1
            or int(getattr(args, "training_num_nodes", 1)) > 1
            or int(getattr(args, "reward_dedicated_num_nodes", 0)) > 1
        )
        and is_probably_local_weight_sync_dir(
            args.weight_sync_dir,
            root=repo_root(env_repo_root=ENV_REPO_ROOT),
        )
    ):
        raise ValueError(
            "weight_sync_mode=checkpoint_path in multi-node mode requires a shared filesystem path. "
            f"Got local-only weight_sync_dir={args.weight_sync_dir}. "
            "Use a shared mount (e.g. /mnt/shared/... or NFS path)."
        )

def validate_args(args: TrainingArguments) -> TrainingArguments:
    """
    Validate and normalize arguments for colocate/offload logic.

    Args:
        args: TrainingArguments instance to validate

    Returns:
        Validated and normalized TrainingArguments
    """
    explicit_sampler_path = args.sampler_path != DEFAULT_SAMPLER_PATH
    explicit_sampler_engine_type = getattr(args, "sampler_engine_type", None) is not None

    validate_grouped_configs(args)

    normalize_repo_relative_paths(
        args,
        env_repo_root=ENV_REPO_ROOT,
        env_data_root=ENV_DATA_ROOT,
        env_model_root=ENV_MODEL_ROOT,
        local_to_hf_fallback=_LOCAL_TO_HF_FALLBACK,
    )

    model_cls = _resolve_model_runtime_contract(
        args,
        explicit_sampler_path=explicit_sampler_path,
        explicit_sampler_engine_type=explicit_sampler_engine_type,
    )
    training_actor_direct_sampling, is_sglang_engine, sglang_logprob_mode = _normalize_sampling_basics(args)
    validate_algorithm_kwargs_json(args)
    validate_loss_kwargs_json(args)
    _normalize_train_backend_config(args)
    validate_dynamic_dotpaths(args)
    _apply_training_actor_direct_sampling_overrides(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
    )
    replay_guard, _, sglang_logprob_mode = _normalize_replay_mode(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
        is_sglang_engine=is_sglang_engine,
        sglang_logprob_mode=sglang_logprob_mode,
    )
    _apply_colocate_and_offload_rules(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
    )

    if (
        str(getattr(args, "train_backend", "fsdp")).lower() == "fsdp"
        and args.training_num_nodes > 1
        and args.fsdp_sharding_strategy == "FULL_SHARD"
    ):
        logger.warning(
            f"Multi-node training detected ({args.training_num_nodes} nodes). "
            f"Consider using --fsdp-sharding-strategy HYBRID_SHARD for better performance."
        )

    validate_reward_and_rollout_buffer_config(args)
    validate_rollout_layout(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
    )
    _normalize_training_misc(args)
    validate_model_specific_logic(args, model_cls=model_cls)
    validate_algorithm_loss_consistency(args)
    validate_resolved_engine_loss_contract(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
        is_sglang_engine=is_sglang_engine,
        replay_guard=replay_guard,
        sglang_logprob_mode=sglang_logprob_mode,
    )

    if args.sampling_adapter:
        args.engine_kwargs.setdefault("sampling_adapter", args.sampling_adapter)
    _normalize_weight_sync(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
    )
    validate_runtime_mode_constraints(
        args,
        training_actor_direct_sampling=training_actor_direct_sampling,
    )

    return args


def get_default_args() -> TrainingArguments:
    """Get default arguments without parsing command line."""
    return TrainingArguments()
