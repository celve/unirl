"""
diffusionrl Arguments - Configuration parameters for training.

Resolution order:
1) Explicit CLI flags
2) YAML values from --config
3) Dataclass field defaults
4) validate_args() checks the explicit config and runtime builders derive
   resolved values without mutating TrainingArguments

New contributors: start from parse_args() -> validate_args().
"""
import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_args, get_origin

from diffusionrl.config.paths import (
    is_probably_local_weight_sync_dir,
    repo_root,
)
from diffusionrl.config.resolution import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SAMPLER_PATH,
    resolve_algorithm_kwargs,
    resolve_algorithm_path,
    resolve_debug_mode,
    resolve_logprob_source,
    resolve_lora_target_modules,
    resolve_model_runtime,
    resolve_nominal_local_training_batch_size,
    resolve_num_updates_per_local_batch,
    resolve_local_micro_batch_size,
    resolve_local_update_batch_size,
    resolve_prompts_per_rollout,
    resolve_global_rollout_batch_size,
    resolve_rollout_topology,
    resolve_sync_protocol,
    resolve_train_backend_kwargs,
    resolve_train_backend_name,
    resolve_training_dp_size,
    resolve_training_plan,
    resolve_training_topology,
)
from diffusionrl.config.rollout_topology import (
    DIRECT_ROLLOUT_MODE,
    ROLLOUT_MODES,
    normalize_rollout_mode,
    normalize_rollout_service_engine,
    rollout_mode_is_colocated,
    rollout_mode_uses_service,
)
from diffusionrl.config.validation import (
    apply_model_config_hook,
    validate_colocate_fractions,
    validate_dotpath,
    validate_dynamic_dotpaths,
    validate_grouped_configs,
    validate_model_runtime_contract,
    validate_nft_sampling_contract,
    validate_resolved_engine_algorithm_contract,
    validate_reward_and_rollout_buffer_config,
    validate_rollout_layout,
    validate_runtime_mode_constraints,
)
from diffusionrl.types.engine import uses_dedicated_rollout_engine

logger = logging.getLogger(__name__)

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"

def _validate_precision_name(value: Any, *, field_name: str) -> None:
    valid = {
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "half",
        "fp32",
        "float32",
        "float",
    }
    key = str(value or "").strip().lower()
    if key not in valid:
        raise ValueError(
            f"{field_name} must be one of bf16/fp16/fp32, got: {value!r}"
        )


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
    max_samples_per_request: Optional[int] = field(default=None,
        metadata={"help": "Generated-sample cap per training-actor direct-sampling request; rollout_total_samples stays prompts_per_rollout*samples_per_prompt"})
    logprob_source: str = field(default="replay",
        metadata={"help": "SGLang log-prob mode: replay (training-side) or native (engine-side)"})
    replay_log_probs: bool = field(default=False,
        metadata={"help": "Replay old log-probs on training actor (for SGLang replay mode)"})
    replay_sampler_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to replay sampler, if different from sampler_path"})
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Explicit sampler constructor kwargs for direct sampling and replay sampler instantiation."})
    num_inference_steps: int = field(default=50,
        metadata={"help": "Number of denoising steps during sampling"})
    eta: float = field(default=1.0,
        metadata={"help": "SDE noise coefficient (eta=0 is ODE, eta=1 is full SDE)"})
    sde_type: str = field(default="flow",
        metadata={"help": "Transition rule. Supported: flow, cps, dance, dpm2"})
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

    def validate(self) -> None:
        mode = str(self.logprob_source).strip().lower()
        if mode not in ("replay", "native"):
            raise ValueError(
                f"logprob_source must be one of replay/native, got: {self.logprob_source}"
            )
        if self.max_samples_per_request is not None and int(self.max_samples_per_request) < 1:
            raise ValueError("max_samples_per_request must be >= 1 when set.")
        if not isinstance(self.sampler_kwargs, dict):
            raise ValueError("sampling.sampler_kwargs must be a dict.")
        if not self.sampler_path:
            raise ValueError(
                "sampler_path must be set. It is usually auto-resolved from model_type. "
                "Set --model.model-type or provide --sampling.sampler-path explicitly."
            )


@dataclass
class RewardConfig:
    """Reward path, reward model, and reward pool controls."""

    reward_path: Optional[str] = field(default="diffusionrl.reward.local.LocalRewardScorer",
        metadata={"help": "Python dotpath to reward scorer class"})
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
        metadata={"help": "Device for local in-process reward scorers: cpu, auto, or cuda"})
    allow_local_reward_cuda_contention: bool = field(default=False,
        metadata={"help": "Allow local_reward_device=cuda without dedicated reward GPUs (may contend with rollout/training GPUs)"})

    def validate(self) -> None:
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
                "reward_path must be set for local/ray reward scoring. "
                "Available: diffusionrl.reward.local.LocalRewardScorer, "
                "or provide a custom reward scorer dotpath."
            )


@dataclass
class RayConfig:
    """Ray resource layout, colocate/offload, and scheduling controls."""

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

    def validate(self) -> None:
        """Keep RayConfig validation intentionally thin.

        Topology-dependent and cross-field rollout/Ray checks happen later after
        explicit rollout validation in validate_args()/config.validation.
        """
        for attr_name in (
            "rollout_num_nodes",
            "rollout_num_gpus_per_node",
            "training_num_nodes",
            "training_num_gpus_per_node",
        ):
            if int(getattr(self, attr_name)) < 0:
                raise ValueError(f"ray.{attr_name} must be >= 0.")

        strategy = str(self.placement_strategy or "PACK").strip().upper()
        valid_strategies = {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}
        if strategy not in valid_strategies:
            raise ValueError(
                "ray.placement_strategy must be one of "
                f"{sorted(valid_strategies)}, got: {self.placement_strategy!r}"
            )


@dataclass
class SyncConfig:
    """Train->rollout weight synchronization controls."""

    protocol: str = field(default="auto",
        metadata={"help": "Weight sync mode: auto, disabled, tensor_payload, nccl_broadcast, checkpoint_path"})
    dir: str = field(default="outputs/weight_sync",
        metadata={"help": "Directory for checkpoint-based weight sync (use shared FS for multi-node)"})
    bucket_mb: int = field(default=256,
        metadata={"help": "Weight sync tensor bucket size (MB) for tensor/distributed strategies"})
    flush_cache: bool = field(default=True,
        metadata={"help": "Whether rollout side flushes runtime cache after each weight sync bucket"})
    target_modules: Optional[List[str]] = field(default=None,
        metadata={"help": "Rollout-side modules that receive weight updates (defaults to ['transformer'])."})

    def validate(self) -> None:
        _valid_modes = (
            "auto", "disabled", "tensor_payload", "nccl_broadcast", "checkpoint_path",
        )
        if self.protocol not in _valid_modes:
            raise ValueError(
                f"sync.protocol must be one of {'/'.join(_valid_modes)}, "
                f"got: {self.protocol!r}. Use 'auto' (recommended) to let diffusionrl choose the best mode."
            )
        if int(self.bucket_mb) < 1:
            raise ValueError("sync.bucket_mb must be >= 1.")
        if self.target_modules is not None:
            if not isinstance(self.target_modules, list):
                raise ValueError("sync.target_modules must be a list of module names.")
            for module_name in self.target_modules:
                if not str(module_name).strip():
                    raise ValueError("sync.target_modules cannot contain empty names.")


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
    """Algorithm controls and shared algorithm-construction surface."""

    # Algorithm selection
    algorithm_type: str = field(default="grpo",
        metadata={"help": "Built-in algorithm family: grpo, nft, or mix_grpo"})
    algorithm_path: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to Algorithm class (auto-resolved from algorithm_type when omitted)"})
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Canonical extension surface for algorithm-specific kwargs. Both rollout and training instantiate algorithms from the same algorithm_kwargs payload."})

    # Advantage and policy objective
    clip_range: float = field(default=1e-4,
        metadata={"help": "PPO clipping range for policy ratio. Smaller = more conservative"})
    clip_schedule: str = field(default="constant",
        metadata={"help": "Clip range schedule: constant, linear_decay, cosine_decay"})
    kl_coef: float = field(default=0.01,
        metadata={"help": "KL divergence penalty coefficient"})
    use_kl_penalty: bool = field(default=True,
        metadata={"help": "Add KL penalty term to the loss"})
    component_mix_stage: str = field(default="reward",
        metadata={"help": "Where multi-component reward mixing happens: reward or advantage"})
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
        if self.prompts_per_rollout is not None and self.prompts_per_rollout < 1:
            raise ValueError("prompts_per_rollout must be >= 1.")
        if self.component_mix_stage not in ("reward", "advantage"):
            raise ValueError(
                f"component_mix_stage must be one of reward/advantage, got: {self.component_mix_stage}"
            )
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
    local_update_batch_size: Optional[int] = field(default=None,
        metadata={"help": "Local optimizer-update batch size in samples. This is the primary training geometry owner when set."})
    local_micro_batch_size: Optional[int] = field(default=None,
        metadata={"help": "Local micro-batch size for one forward/backward pass. Defaults to local_update_batch_size when omitted."})
    num_updates_per_local_batch: Optional[int] = field(default=None,
        metadata={"help": "Number of optimizer updates performed from one local batch. Defaults to 1 when local_update_batch_size is set."})
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

    fsdp_cpu_offload: bool = field(default=False,
        metadata={"help": "Offload FSDP parameters and gradients to CPU"})

    # Memory optimization
    use_gradient_checkpointing: bool = field(default=False,
        metadata={"help": "Enable gradient checkpointing to save memory at the cost of compute"})

    def validate(self) -> None:
        explicit_geometry_fields = {
            "local_update_batch_size": self.local_update_batch_size,
            "local_micro_batch_size": self.local_micro_batch_size,
            "num_updates_per_local_batch": self.num_updates_per_local_batch,
        }
        for field_name, value in explicit_geometry_fields.items():
            if value is not None and int(value) < 1:
                raise ValueError(f"{field_name} must be >= 1 when set.")

        if (
            self.num_updates_per_local_batch is not None
            and self.local_update_batch_size is None
        ):
            raise ValueError(
                "training.num_updates_per_local_batch requires "
                "training.local_update_batch_size."
            )
        backend = str(self.train_backend or "fsdp").strip().lower()
        supported = {"fsdp", "megatron", "veomni"}
        if backend not in supported and not self.train_backend_path:
            raise ValueError(
                f"Unsupported train_backend={self.train_backend!r}. "
                f"Expected one of {sorted(supported)} or provide --training.train-backend-path."
            )


@dataclass
class PrecisionConfig:
    """Precision controls for model load, FSDP wrapping, and rollout tensors."""

    model_precision: str = field(default="bf16",
        metadata={"help": "Precision used to load model weights/components (default: bf16)"})
    fsdp_precision: str = field(default="fp32",
        metadata={"help": "FSDP param precision on the training side (default: fp32)"})
    autocast_precision: str = field(default="bf16",
        metadata={"help": "Autocast precision for sampler/model forward passes (default: bf16)"})
    trajectory_precision: str = field(default="fp16",
        metadata={"help": "Precision used to store rollout trajectory latents (default: fp16)"})
    logprob_precision: str = field(default="fp32",
        metadata={"help": "Precision used to store rollout log-prob tensors (default: fp32)"})

    def validate(self) -> None:
        _validate_precision_name(self.model_precision, field_name="precision.model_precision")
        _validate_precision_name(self.fsdp_precision, field_name="precision.fsdp_precision")
        _validate_precision_name(self.autocast_precision, field_name="precision.autocast_precision")
        _validate_precision_name(self.trajectory_precision, field_name="precision.trajectory_precision")
        _validate_precision_name(self.logprob_precision, field_name="precision.logprob_precision")


@dataclass
class RolloutLoggingConfig:
    """Rollout loop/buffer, checkpoint/eval, and logging controls."""

    mode: Optional[str] = field(default=None,
        metadata={"help": "Canonical rollout topology: direct_rollout, separate_rollout, or colocate_rollout"})
    service_engine: Optional[str] = field(default=None,
        metadata={"help": "Canonical rollout engine selector: fsdp for direct_rollout, sglang for separate_rollout or colocate_rollout."})
    service_num_gpus: Optional[int] = field(default=None,
        metadata={"help": "Dedicated rollout service GPUs per actor/engine. Required for separate_rollout and colocate_rollout."})
    engine_tp_size: Optional[int] = field(default=None,
        metadata={"help": "Dedicated rollout service tensor parallel hint. Does not determine actor GPU ownership."})
    engine_sp_size: Optional[int] = field(default=None,
        metadata={"help": "Dedicated rollout service sequence/spatial parallel hint. Does not determine actor GPU ownership."})
    service_require_memory_api: Optional[bool] = field(default=None,
        metadata={"help": "Whether dedicated rollout service requires concrete memory API handlers."})
    service_transport_dtype: Optional[str] = field(default=None,
        metadata={"help": "Dedicated rollout transport payload dtype override."})
    service_transport_drop_decoded_videos: Optional[bool] = field(default=None,
        metadata={"help": "Whether rollout transport drops decoded video payloads after reward handling."})
    service_transport_log_payload_bytes: Optional[bool] = field(default=None,
        metadata={"help": "Whether rollout transport logs serialized payload sizes for debugging."})
    sglang_local_mode: Optional[bool] = field(default=None,
        metadata={"help": "Whether SGLang rollout uses in-actor local generator mode."})
    sglang_verify_weight_checksum: Optional[bool] = field(default=None,
        metadata={"help": "Whether SGLang verifies weight checksum after rollout-side updates."})
    sglang_prompt_encoder_device: Optional[str] = field(default=None,
        metadata={"help": "Device for SGLang-side prompt encoder construction."})
    sglang_prompt_encoder_dtype: Optional[str] = field(default=None,
        metadata={"help": "Prompt encoder dtype for SGLang rollout (auto/fp16/bf16/fp32)."})
    sglang_prompt_encoder_max_length: Optional[int] = field(default=None,
        metadata={"help": "Prompt encoder max sequence length for SGLang rollout."})
    sglang_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Engine-scoped SGLang rollout kwargs. ServerArgs-compatible keys are forwarded to the SGLang rollout runtime."})

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
    rollout_buffer_reassemble_by_group: bool = field(default=False,
        metadata={"help": "Reassemble outgoing training batches in the rollout buffer by explicit group_ids"})
    rollout_buffer_group_size: Optional[int] = field(default=None,
        metadata={"help": "Explicit samples per logical group. Required when rollout_buffer_reassemble_by_group=true"})
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
        normalized_mode = normalize_rollout_mode(self.mode)
        if normalized_mode and normalized_mode not in ROLLOUT_MODES:
            raise ValueError(
                "rollout.mode must be one of "
                f"{sorted(ROLLOUT_MODES)}, got: {self.mode!r}"
            )
        for attr_name in (
            "service_num_gpus",
            "engine_tp_size",
            "engine_sp_size",
            "sglang_prompt_encoder_max_length",
        ):
            value = getattr(self, attr_name)
            if value is not None and int(value) < 1:
                raise ValueError(f"rollout.{attr_name} must be >= 1 when set.")
        if not isinstance(self.sglang_kwargs, dict):
            raise ValueError("rollout.sglang_kwargs must be a dict.")
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
    "sync": SyncConfig,
    "algorithm": AlgorithmConfig,
    "training": TrainingConfig,
    "precision": PrecisionConfig,
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
    rollout_mode = normalize_rollout_mode(getattr(args.rollout, "mode", None))
    if not rollout_mode:
        raise ValueError(
            "rollout.mode must be set explicitly before calling "
            "is_training_actor_sampling_mode(). Run validate_args()/parse_args() first."
        )
    return rollout_mode == DIRECT_ROLLOUT_MODE

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
    sync: SyncConfig = field(default_factory=SyncConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
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
    "sync": "Weight Sync",
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
            source_path = ".".join(parts)
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
                    _assign(leaf_dest, value, source_path=source_path)
                    continue
                if _is_yaml_container_path(parts):
                    _walk(value, parts)
                    continue
                _assign(".".join(parts), value, source_path=source_path)
                continue

            leaf_dest = _resolve_yaml_leaf_dest(parts)
            if leaf_dest is not None:
                _assign(leaf_dest, value, source_path=source_path)
            else:
                _assign(".".join(parts), value, source_path=source_path)

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


def _build_resolved_config_view(args: TrainingArguments) -> Dict[str, Any]:
    """Build a read-only resolved config view for debugging/inspection."""
    resolved = args.to_flat_dict()
    explicit_sampler_path = args.sampling.sampler_path != DEFAULT_SAMPLER_PATH
    model_runtime = resolve_model_runtime(
        args,
        explicit_sampler_path=explicit_sampler_path,
    )
    rollout_topology = resolve_rollout_topology(args)
    resolved["model.model_path"] = model_runtime.model_path
    resolved["model.model_type"] = model_runtime.model_type
    resolved["sampling.sampler_path"] = model_runtime.sampler_path
    resolved["algorithm.algorithm_path"] = resolve_algorithm_path(args)
    resolved["training.train_backend"] = resolve_train_backend_name(args)
    resolved["rollout.mode"] = rollout_topology.mode
    resolved["rollout.service_engine"] = rollout_topology.service_engine
    resolved["sync.protocol"] = resolve_sync_protocol(
        args,
        training_actor_sampling_mode=rollout_topology.training_actor_sampling_mode,
        rollout_service_engine=rollout_topology.service_engine,
    )
    train_topology = resolve_training_topology(args)
    train_plan = resolve_training_plan(args)
    resolved["runtime.training_actor_count"] = train_topology.actor_count
    resolved["runtime.training_world_size"] = train_topology.world_size
    resolved["runtime.training_dp_size"] = train_topology.dp_size
    resolved["runtime.training_tp_size"] = train_topology.tp_size
    resolved["runtime.training_pp_size"] = train_topology.pp_size
    resolved["runtime.training_sp_size"] = train_topology.sp_size
    resolved["runtime.training_ep_size"] = train_topology.ep_size
    resolved["runtime.prompts_per_rollout"] = resolve_prompts_per_rollout(args)
    resolved["runtime.training_global_batch_size"] = train_plan.global_batch_size
    resolved["runtime.training_local_batch_size"] = train_plan.local_batch_size
    resolved["runtime.training_local_update_batch_size"] = (
        train_plan.local_update_batch_size
    )
    resolved["runtime.training_local_micro_batch_size"] = (
        train_plan.local_micro_batch_size
    )
    resolved["runtime.training_num_updates_per_local_batch"] = (
        train_plan.num_updates_per_local_batch
    )
    return resolved


def _print_config_views(
    *,
    args: TrainingArguments,
    print_resolved_config: bool,
) -> None:
    if not print_resolved_config:
        return

    use_color = _supports_color_output()

    if print_resolved_config:
        print(_color("[Resolved Config] final runtime values", "bold", enabled=use_color))
        resolved_flat = _build_resolved_config_view(args)
        for key in sorted(resolved_flat.keys()):
            value_text = _format_config_value(resolved_flat.get(key))
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
        help="Print resolved runtime config after validation.",
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

    # Validate explicit config. Runtime builders derive resolved values separately.
    args = validate_args(args)
    _print_config_views(
        args=args,
        print_resolved_config=print_resolved_config,
    )

    return args

def _resolve_model_runtime_contract(
    args: TrainingArguments,
    *,
    explicit_sampler_path: bool,
) -> tuple[Any, Dict[str, Optional[str]]]:
    """Resolve model path/type and model-declared runtime defaults without mutating args."""
    resolved = resolve_model_runtime(
        args,
        explicit_sampler_path=explicit_sampler_path,
    )
    model_defaults: Dict[str, Optional[str]] = {
        "sampler_path": resolved.sampler_path,
        "sampler_engine_type": resolved.model_default_engine_type,
    }
    return resolved.model_cls, model_defaults


def _validate_rollout_topology(
    args: TrainingArguments,
    *,
    model_default_engine_type: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Validate explicit rollout topology without mutating args."""
    del model_default_engine_type
    resolved = resolve_rollout_topology(args)
    return resolved.training_actor_sampling_mode, resolved.service_engine


def _validate_sampling_basics(
    args: TrainingArguments,
    *,
    rollout_service_engine: Optional[str],
) -> tuple[bool, str]:
    """Validate sampling/runtime basics without mutating args."""
    if not isinstance(args.sampling.sampler_kwargs, dict):
        raise ValueError("sampling.sampler_kwargs must be a dict.")

    if args.sampling.sampler_kwargs:
        logger.info("sampling.sampler_kwargs configured with explicit sampler constructor overrides.")

    is_sglang_engine = normalize_rollout_service_engine(rollout_service_engine) == "sglang"
    logprob_source = resolve_logprob_source(args)
    return is_sglang_engine, logprob_source


def _apply_training_actor_sampling_overrides(
    args: TrainingArguments,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate direct-sampling runtime compatibility without mutating args."""
    if not training_actor_sampling_mode:
        return

    rollout_mode = normalize_rollout_mode(args.rollout.mode)
    rollout_service_engine = normalize_rollout_service_engine(args.rollout.service_engine)
    if rollout_mode_uses_service(rollout_mode) or uses_dedicated_rollout_engine(rollout_service_engine):
        raise ValueError(
            "Dedicated rollout service engines cannot use direct_rollout mode. "
            f"Got rollout.mode={rollout_mode!r}, rollout.service_engine={rollout_service_engine!r}."
        )

    backend_name = str(getattr(args.training, "train_backend", "fsdp") or "fsdp").strip().lower()
    backend_path = str(getattr(args.training, "train_backend_path", "") or "").strip()
    from diffusionrl.runtime.training.backends import resolve_train_backend_capabilities

    backend_caps = resolve_train_backend_capabilities(
        backend_name,
        backend_path=backend_path or None,
    )
    supports_direct_sampling = bool(backend_caps.supports_training_actor_sampling)

    if not supports_direct_sampling:
        raise ValueError(
            "rollout.mode=%r resolves to direct training-actor sampling, "
            "but train_backend=%r does not declare supports_training_actor_sampling=true."
            % (rollout_mode, backend_name)
        )


def _validate_replay_mode(
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
                "rollout.service_engine='sglang' with logprob_source='replay' requires "
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
            "dedicated rollout services + algorithm_type='grpo'. "
            "Either disable replay_log_probs or adjust your config."
        )

    return replay_guard, replay_enabled, logprob_source


def _apply_colocate_and_offload_rules(
    args: TrainingArguments,
    *,
    training_actor_sampling_mode: bool,
) -> None:
    """Validate offload/colocate flags without mutating args."""
    if training_actor_sampling_mode:
        if bool(args.ray.offload) or bool(args.ray.offload_train) or bool(args.ray.offload_rollout):
            raise ValueError(
                "direct_rollout uses training actors for sampling and cannot be combined with "
                "ray.offload / ray.offload_train / ray.offload_rollout."
            )
        return

    if rollout_mode_is_colocated(args.rollout.mode):
        validate_colocate_fractions(args)


def _validate_training_misc(args: TrainingArguments) -> None:
    """Validate misc training knobs that affect downstream components."""
    if args.algorithm.adv_normalization not in {"global", "group"}:
        raise ValueError(
            "algorithm.adv_normalization must be 'global' or 'group'. "
            f"Got: {args.algorithm.adv_normalization!r}."
        )

    resolve_lora_target_modules(args.training.lora_target_modules)

    if bool(getattr(args.rollout, "rollout_buffer_reassemble_by_group", False)):
        configured_group_size = getattr(args.rollout, "rollout_buffer_group_size", None)
        if configured_group_size is None:
            raise ValueError(
                "rollout.rollout_buffer_reassemble_by_group=true requires rollout.rollout_buffer_group_size "
                "to be set explicitly. Implicit binding to algorithm.samples_per_prompt was removed."
            )

    _validate_training_batch_geometry(args)


def _resolve_training_dp_size(args: TrainingArguments) -> int:
    return resolve_training_dp_size(args)


def _validate_training_batch_geometry(args: TrainingArguments) -> None:
    """Validate training batch-geometry knobs without mutating args."""
    update_batch_size = resolve_local_update_batch_size(args)
    micro_batch_size = resolve_local_micro_batch_size(args)
    num_updates_per_local_batch = resolve_num_updates_per_local_batch(args)
    local_batch_size = resolve_nominal_local_training_batch_size(args)
    global_batch_size = resolve_global_rollout_batch_size(args)
    resolved_prompts_per_rollout = resolve_prompts_per_rollout(args)

    if micro_batch_size > update_batch_size:
        raise ValueError(
            "training.local_micro_batch_size must be <= "
            "training.local_update_batch_size. "
            f"Got micro_batch_size={micro_batch_size}, "
            f"update_batch_size={update_batch_size}."
        )
    if local_batch_size != update_batch_size * num_updates_per_local_batch:
        raise ValueError(
            "Resolved local training batch size does not match the training geometry. "
            f"Got local_batch_size={local_batch_size}, "
            f"local_update_batch_size={update_batch_size}, "
            f"num_updates_per_local_batch={num_updates_per_local_batch}."
        )
    if global_batch_size != local_batch_size * _resolve_training_dp_size(args):
        raise ValueError(
            "Resolved global rollout batch size does not match local_batch_size * dp_size. "
            f"Got global_batch_size={global_batch_size}, local_batch_size={local_batch_size}, "
            f"dp_size={_resolve_training_dp_size(args)}."
        )
    if resolved_prompts_per_rollout < 1:
        raise ValueError(
            "Resolved prompts_per_rollout must be >= 1. "
            f"Got: {resolved_prompts_per_rollout}."
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
            "sampling runs directly on training actors."
        )

    max_samples_per_request = int(max_samples_per_request)
    prompts_per_rollout = int(resolve_prompts_per_rollout(args))
    total_samples = prompts_per_rollout * int(args.algorithm.samples_per_prompt)
    if max_samples_per_request < 1:
        raise ValueError("sampling.max_samples_per_request must be >= 1.")
    if prompts_per_rollout < 1:
        raise ValueError("Resolved prompts_per_rollout must be >= 1.")
def _validate_algorithm_kwargs_payload(args: TrainingArguments) -> None:
    """Validate algorithm_kwargs payload without mutating args."""
    resolve_algorithm_kwargs(args)


def _validate_algorithm_path(args: TrainingArguments) -> None:
    """Resolve algorithm.algorithm_path for validation without mutating args."""
    resolve_algorithm_path(args)


def _validate_train_backend_config(args: TrainingArguments) -> None:
    """Validate train-backend selection and kwargs without mutating args."""
    backend = resolve_train_backend_name(args)
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

    parsed = resolve_train_backend_kwargs(args)

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
    if "num_actors" in parsed:
        raise ValueError(
            "training.train_backend_kwargs.num_actors is not supported. "
            "Training actor count is owned by ray.training_num_nodes × "
            "ray.training_num_gpus_per_node."
        )
    if backend == "megatron" and not backend_path and not str(parsed.get("actor_class_path", "")).strip():
        logger.warning(
            "train_backend=%s requires actor_class_path in train_backend_kwargs "
            "to launch a Megatron-specific training actor.",
            backend,
        )

def _validate_weight_sync(args: TrainingArguments, *, training_actor_sampling_mode: bool) -> None:
    rollout_service_engine = normalize_rollout_service_engine(
        getattr(args.rollout, "service_engine", None)
    )
    resolved_mode = resolve_sync_protocol(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
        rollout_service_engine=rollout_service_engine,
    )
    is_multi_node = (
        int(getattr(args.ray, "rollout_num_nodes", 1)) > 1
        or int(getattr(args.ray, "training_num_nodes", 1)) > 1
        or int(getattr(args.reward, "reward_dedicated_num_nodes", 0)) > 1
    )
    if (
        resolved_mode in {"tensor_payload", "nccl_broadcast"}
        and rollout_service_engine != "sglang"
    ):
        raise ValueError(
            "sync.protocol in {tensor_payload,nccl_broadcast} currently requires "
            "rollout.service_engine='sglang'. "
            f"Got rollout.service_engine={rollout_service_engine!r}."
        )

    if (
        resolved_mode == "checkpoint_path"
        and is_multi_node
        and is_probably_local_weight_sync_dir(
            args.sync.dir,
            root=repo_root(env_repo_root=ENV_REPO_ROOT),
        )
    ):
        raise ValueError(
            "sync.protocol=checkpoint_path in multi-node mode requires a shared filesystem path. "
            f"Got local-only sync.dir={args.sync.dir}. "
            "Use a shared mount (e.g. /mnt/shared/... or NFS path)."
        )


def _validate_debug_config(args: TrainingArguments) -> str:
    """Validate debug mode and enforce mode-specific constraints."""
    debug_mode = resolve_debug_mode(args)

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
) -> TrainingArguments:
    """
    Validate arguments without mutating the original config values.

    Args:
        args: TrainingArguments instance to validate

    Returns:
        Validated TrainingArguments
    """
    explicit_sampler_path = args.sampling.sampler_path != DEFAULT_SAMPLER_PATH

    validate_grouped_configs(args)
    debug_mode = _validate_debug_config(args)

    model_cls, model_defaults = _resolve_model_runtime_contract(
        args,
        explicit_sampler_path=explicit_sampler_path,
    )
    training_actor_sampling_mode, rollout_service_engine = _validate_rollout_topology(
        args,
        model_default_engine_type=model_defaults.get("sampler_engine_type"),
    )
    is_sglang_engine, logprob_source = _validate_sampling_basics(
        args,
        rollout_service_engine=rollout_service_engine,
    )
    _validate_algorithm_kwargs_payload(args)
    _validate_algorithm_path(args)
    _validate_train_backend_config(args)
    if debug_mode == "train_only":
        # Keep train_only focused on train-side imports only. Rollout/reward/data
        # extensions are not exercised on this path and should not block replay.
        resolved_model = resolve_model_runtime(
            args,
            explicit_sampler_path=explicit_sampler_path,
        )
        validate_dotpath(resolved_model.model_path, label="model")
        validate_dotpath(resolved_model.sampler_path, label="sampler")
        validate_dotpath(resolve_algorithm_path(args), label="algorithm")
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
    replay_guard, _, logprob_source = _validate_replay_mode(
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
    _validate_training_misc(args)
    _validate_direct_sampling_batch_geometry(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
    )
    apply_model_config_hook(args, model_cls=model_cls)
    validate_model_runtime_contract(args)
    validate_nft_sampling_contract(args)
    if debug_mode != "train_only":
        validate_resolved_engine_algorithm_contract(
            args,
            training_actor_sampling_mode=training_actor_sampling_mode,
            is_sglang_engine=is_sglang_engine,
            replay_guard=replay_guard,
            logprob_source=logprob_source,
        )

    _validate_weight_sync(
        args,
        training_actor_sampling_mode=training_actor_sampling_mode,
    )
    if debug_mode != "train_only":
        validate_runtime_mode_constraints(
            args,
            training_actor_sampling_mode=training_actor_sampling_mode,
            model_cls=model_cls,
        )

    return args


def get_default_args() -> TrainingArguments:
    """Get default arguments without parsing command line."""
    return TrainingArguments()
