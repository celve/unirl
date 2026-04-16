"""
diffusionrl cmdline schema - dataclass-backed argument groups for training.

Resolution order:
1) Explicit CLI flags
2) YAML values from --config
3) Dataclass field defaults
4) public config entrypoints validate the explicit config and derived helpers compute
   derived values without mutating TrainingArguments

Schema and public cmdline entrypoints stay in this file. Parser mechanics live
in ``diffusionrl.cmdline.argument_parsing`` and generic config derivation lives
in ``resolution.py``. The CLI entry point ``parse_args`` lives in
``diffusionrl.cmdline.parse_args``.
"""

import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
from diffusionrl.cmdline.resolution import (
    attach_training_plan,
    derive_config,
)
from diffusionrl.cmdline.validation import (
    validate_algorithm_kwargs_payload,
    validate_dynamic_dotpaths,
    validate_grouped_configs,
    validate_nft_sampling_contract,
    validate_rollout_mode,
    validate_rollout_mode_constraints,
)
from diffusionrl.config.assembly import DerivedConfig
from diffusionrl.config.resolution import DEFAULT_MODEL_PATH
from diffusionrl.config.validation import (
    validate_engine_algorithm_contract,
    validate_precision_type,
    validate_reward_config,
    validate_rollout_layout,
    validate_train_backend_config,
    validate_training_batch_geometry,
)
from diffusionrl.reward.config import RewardSpec
from diffusionrl.sde.rules import SUPPORTED_USER_SDE_TYPES, supported_sde_type_text

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Model Configuration:
    contains model type and checkpoint paths.
    """

    model_type: str = field(
        default="hunyuan",
        metadata={
            "help": "Model architecture type (hunyuan, flux, sd3, mochi, wan2.1, bagel)"
        },
    )
    model_dotpath: str = field(
        default=DEFAULT_MODEL_PATH,
        metadata={
            "help": "Python dotpath to ModelBundle class. Auto-derived from model_type"
        },
    )
    pretrained_model_ckpt_path: str = field(
        default="",
        metadata={
            "help": "Path to pretrained model weights (local path or HuggingFace ID)"
        },
    )
    vae_ckpt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to separate VAE checkpoint, if not bundled with the model"
        },
    )
    text_encoder_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to separate text encoder checkpoint, if not bundled"},
    )

    def validate(self) -> None:
        if not self.model_dotpath:
            raise ValueError(
                "model_dotpath must be set. It is usually auto-derived from model_type. "
                "Set --model.model-type (hunyuan, flux, sd3, mochi) or provide "
                "--model.model-dotpath explicitly."
            )


@dataclass
class SamplingConfig:
    """
    Sampling Engine Configuration:
    contains sampler type, logprob source, and denoising controls.
    """

    sampler_dotpath: str = field(
        default="",
        metadata={
            "help": "Optional Python dotpath to Sampler class; omit to auto-derive from model_type"
        },
    )
    logprob_source: str = field(
        default="replay",
        metadata={
            "help": "SGLang log-prob mode: replay (training-side replay path) or native (engine-side log_probs)",
            "choices": ["replay", "native"],
        },
    )
    replay_sampler_dotpath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Python dotpath to replay sampler, if different from sampler_dotpath"
        },
    )
    sampler_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Explicit sampler constructor kwargs for direct sampling and replay sampler instantiation."
        },
    )
    num_inference_steps: int = field(
        default=50, metadata={"help": "Number of denoising steps during sampling"}
    )
    eta: float = field(
        default=1.0,
        metadata={"help": "SDE noise coefficient (eta=0 is ODE, eta=1 is full SDE)"},
    )
    sde_type: str = field(
        default="flow",
        metadata={
            "help": "Transition rule. Supported: flow, cps, dance, dpm2",
            "choices": ["flow", "cps", "dance", "dpm2"],
        },
    )
    max_samples_per_request: Optional[int] = field(
        default=None,
        metadata={
            "help": "Generated-sample cap per training-actor direct-sampling request; rollout_total_samples stays prompts_per_rollout*samples_per_prompt"
        },
    )
    shift: float = field(
        default=3.0,
        metadata={
            "help": "Shift parameter for the sampling timestep schedule (model-specific)"
        },
    )
    guidance_scale: float = field(
        default=7.5,
        metadata={"help": "Classifier-free guidance scale (0.0 = no guidance)"},
    )
    sampling_adapter: Optional[str] = field(
        default=None,
        metadata={
            "help": "Sampling adapter type for special modes (e.g. 'old' for NFT)"
        },
    )
    init_same_noise: bool = field(
        default=False,
        metadata={
            "help": "Use identical initial noise for all samples of the same prompt"
        },
    )

    # Generated media dimensions and frame rate
    height: int = field(
        default=256, metadata={"help": "Generated image/video height in pixels"}
    )
    width: int = field(
        default=256, metadata={"help": "Generated image/video width in pixels"}
    )
    num_frames: int = field(
        default=16,
        metadata={"help": "Number of video frames to generate (video models only)"},
    )
    fps: int = field(
        default=8, metadata={"help": "Video frame rate (video models only)"}
    )

    @property
    def replay_enabled(self) -> bool:
        return self.logprob_source == "replay"

    def validate(self) -> None:
        if (
            self.max_samples_per_request is not None
            and self.max_samples_per_request < 1
        ):
            raise ValueError("max_samples_per_request must be >= 1 when set.")
        raw_sde_type = str(self.sde_type or "").strip().lower()
        if raw_sde_type not in SUPPORTED_USER_SDE_TYPES:
            raise ValueError(
                f"Unknown sampling.sde_type={self.sde_type!r}. "
                f"Supported values: {supported_sde_type_text()}."
            )
        if not isinstance(self.sampler_kwargs, dict):
            raise ValueError("sampling.sampler_kwargs must be a dict.")

        _precision_keys = {
            "autocast_precision",
            "trajectory_precision",
            "logprob_precision",
        }
        _leaked = _precision_keys & set(self.sampler_kwargs)
        if _leaked:
            raise ValueError(
                f"sampling.sampler_kwargs must not contain precision keys {sorted(_leaked)}; "
                "use precision.* instead."
            )


@dataclass
class RewardConfig:
    """
    Reward Configuration:
    contains reward backend, reward provider configs and multi-reward related fields.
    """

    reward_backend: str = field(
        default="local",
        metadata={
            "help": "Reward backend: local or http",
            "choices": ["local", "http"],
        },
    )

    # http reward service related fields
    reward_service_urls: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "HTTP reward service URL(s). Single URL string or a list for load balancing"
        },
    )

    # local reward model related fields
    reward_dotpath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional Python dotpath to a custom reward scorer class; omit to use built-in reward_components"
        },
    )
    reward_model_ckpt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to reward model weights (local path or HuggingFace ID)"
        },
    )
    reward_components: Optional[List[str]] = field(
        default_factory=lambda: ["hpsv2"],
        metadata={
            "help": "Reward component name(s). Single scorer name or a list such as [hpsv2, pickscore, clip, ocr]"
        },
    )
    reward_batch_size: int = field(
        default=8, metadata={"help": "Batch size for reward model inference"}
    )
    local_reward_device: str = field(
        default="cpu",
        metadata={
            "help": "Device for local in-process reward scorers: cpu, auto, or cuda",
            "choices": ["auto", "cpu", "cuda"],
        },
    )

    # multi-reward related fields
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward component in multi-reward aggregation"
        },
    )
    reward_aggregation_method: str = field(
        default="weighted_sum",
        metadata={
            "help": "Multi-reward aggregation method: weighted_sum, mean, min, max, concat"
        },
    )

    def __post_init__(self):
        if isinstance(self.reward_service_urls, str):
            self.reward_service_urls = (
                [self.reward_service_urls] if self.reward_service_urls.strip() else None
            )
        if isinstance(self.reward_components, str):
            self.reward_components = (
                [self.reward_components] if self.reward_components.strip() else None
            )

    @property
    def has_http_reward_urls(self) -> bool:
        return bool(self.reward_service_urls)

    @property
    def has_http_reward(self) -> bool:
        return str(self.reward_backend or "local").strip().lower() == "http"

    @property
    def has_builtin_reward(self) -> bool:
        if not isinstance(self.reward_components, list):
            return False
        return any(str(name or "").strip() for name in self.reward_components)

    def validate(self) -> None:
        reward_backend = self.reward_backend
        if reward_backend not in ("local", "http"):
            raise ValueError(
                "reward_backend must be one of local/http, "
                f"got: {self.reward_backend}"
            )
        if self.local_reward_device not in ("cpu", "auto", "cuda"):
            raise ValueError(
                "local_reward_device must be one of cpu/auto/cuda, "
                f"got: {self.local_reward_device}"
            )
        if reward_backend == "http" and not self.has_http_reward_urls:
            raise ValueError("reward_backend='http' requires reward_service_urls.")
        if reward_backend != "http" and self.has_http_reward_urls:
            raise ValueError(
                "reward_service_urls is only valid when reward_backend='http'."
            )
        if (
            not self.has_http_reward
            and not self.reward_dotpath
            and not self.has_builtin_reward
        ):
            raise ValueError(
                "Reward scoring requires either reward_components for built-ins, "
                "or reward_dotpath for a custom scorer."
            )


@dataclass
class RayConfig:
    """Ray resource layout, colocate/offload, and scheduling controls."""

    ray_address: Optional[str] = field(
        default=None,
        metadata={"help": "Ray cluster address (None = auto-detect or start local)"},
    )
    rollout_num_nodes: int = field(
        default=1, metadata={"help": "Number of nodes for rollout actors"}
    )
    rollout_num_gpus_per_node: int = field(
        default=4, metadata={"help": "GPUs per node for rollout actors"}
    )
    training_num_nodes: int = field(
        default=1, metadata={"help": "Number of nodes for training actors"}
    )
    training_num_gpus_per_node: int = field(
        default=4, metadata={"help": "GPUs per node for training actors"}
    )
    placement_strategy: str = field(
        default="PACK",
        metadata={
            "help": "Ray placement group strategy: PACK or SPREAD",
            "choices": ["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"],
        },
    )
    colocate_training_gpu_fraction: float = field(
        default=0.4,
        metadata={"help": "GPU memory fraction for training when colocated"},
    )
    colocate_rollout_gpu_fraction: float = field(
        default=0.4, metadata={"help": "GPU memory fraction for rollout when colocated"}
    )
    allow_noset_multi_gpu_inference: bool = field(
        default=False,
        metadata={"help": "Allow multi-GPU rollout actors (experimental NOSET layout)"},
    )
    offload_train: bool = field(
        default=False, metadata={"help": "Enable model offload for training actors"}
    )
    offload_rollout: bool = field(
        default=False, metadata={"help": "Enable model offload for rollout actors"}
    )

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
            if getattr(self, attr_name) < 0:
                raise ValueError(f"ray.{attr_name} must be >= 0.")


@dataclass
class SyncConfig:
    """
    Train->rollout weight synchronization Configuration:
    contains cadence, protocol, directory, bucket size, flush cache, and target modules.
    """

    rollout_update_interval: int = field(
        default=1,
        metadata={
            "help": (
                "Run train→rollout weight sync every N outer rollout steps (rollout_id); "
                "used by separate/async training loops."
            )
        },
    )
    protocol: str = field(
        default=None,
        metadata={
            "help": "Explicit weight sync mode. Must be one of: disabled, tensor_payload, nccl_broadcast, checkpoint_path",
            "choices": [
                "disabled",
                "tensor_payload",
                "nccl_broadcast",
                "checkpoint_path",
            ],
        },
    )
    dir: str = field(
        default="outputs/weight_sync",
        metadata={
            "help": "Directory for checkpoint-based weight sync (use shared FS for multi-node)"
        },
    )
    bucket_size: int = field(
        default=256,
        metadata={
            "help": "Weight sync tensor bucket size (MB) for tensor/distributed strategies"
        },
    )
    flush_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether the rollout side flushes inference-engine caches after each weight sync bucket"
        },
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Rollout-side modules that receive weight updates (defaults to ['transformer'])."
        },
    )

    def validate(self) -> None:
        normalized_protocol = str(self.protocol or "").strip().lower()
        if not normalized_protocol:
            raise ValueError(
                "sync.protocol must be set explicitly. "
                "Choose one of disabled/tensor_payload/nccl_broadcast/checkpoint_path."
            )
        if int(self.rollout_update_interval) < 1:
            raise ValueError(
                f"sync.rollout_update_interval must be >= 1, got: {self.rollout_update_interval}"
            )
        if self.bucket_size < 1:
            raise ValueError("sync.bucket_size must be >= 1.")
        if self.target_modules is not None:
            if not isinstance(self.target_modules, list):
                raise ValueError("sync.target_modules must be a list of module names.")
            for module_name in self.target_modules:
                if not str(module_name).strip():
                    raise ValueError("sync.target_modules cannot contain empty names.")


@dataclass
class SchedulerConfig:
    """
    Stateless index-scheduler config for rollout or training timestep selection.
    It is a sub-config in AlgorithmConfig.
    """

    timestep_strategy: str = field(
        default="all",
        metadata={
            "help": "Index scheduler type: all or window",
            "choices": ["all", "window"],
        },
    )
    timestep_fraction: Any = field(
        default=1.0,
        metadata={
            "help": "Fraction of total timesteps to select. Single float x means [0, x); tuple (x, y) means [x, y)."
        },
    )
    num_sde_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Randomly sample this many SDE timestep indices from the timestep_fraction range per step. None means use all indices in range."
        },
    )
    window_strategy: str = field(
        default="progressive",
        metadata={
            "help": "Window progression: progressive or random",
            "choices": ["progressive", "random"],
        },
    )
    window_size: int = field(
        default=4, metadata={"help": "Number of timestep indices in each window"}
    )
    iters_per_window: int = field(
        default=25,
        metadata={"help": "Number of training steps before advancing the window"},
    )
    window_init_timestep: int = field(
        default=0, metadata={"help": "Initial left boundary for the window scheduler"}
    )
    max_iters_per_window: Optional[int] = field(
        default=10, metadata={"help": "Reserved for decay-style window schedulers"}
    )
    min_iters_per_window: Optional[int] = field(
        default=1, metadata={"help": "Reserved for decay-style window schedulers"}
    )
    overlap_size: int = field(
        default=0,
        metadata={
            "help": "Number of overlapping timesteps between adjacent windows (0 = no overlap)"
        },
    )
    roll_back: bool = field(
        default=False,
        metadata={"help": "Wrap back to the beginning after reaching the last window"},
    )


@dataclass
class AlgorithmConfig:
    """Algorithm Configuration:
    contains algorithm type, dotpath, kwargs, rollout geometry and window scheduler configuration.

    Public/common surface:
    - algorithm_type / algorithm_dotpath
    - algorithm_kwargs
    - samples_per_prompt / prompts_per_rollout
    - rollout/training scheduler sub-configs
    """

    # Algorithm selection
    algorithm_type: str = field(
        default="grpo",
        metadata={"help": "Built-in algorithm family: grpo, nft, or mix_grpo"},
    )
    algorithm_dotpath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Python dotpath to Algorithm class (auto-derive from algorithm_type when omitted)"
        },
    )
    algorithm_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "YAML-only extension surface for algorithm-specific kwargs. Shared framework fields have dedicated algorithm.* entries. On CLI, use repeated --algorithm.kwarg KEY=VALUE only for true algorithm-specific extension keys."
        },
    )

    # Framework-owned rollout geometry
    samples_per_prompt: int = field(
        default=4,
        metadata={
            "help": "Number of generated samples per prompt. This remains framework-owned because rollout geometry depends on it."
        },
    )
    prompts_per_rollout: int = field(
        default=1,
        metadata={
            "help": "Number of unique prompts per rollout step. Rollout geometry is defined by prompts_per_rollout * samples_per_prompt."
        },
    )

    # Shared algorithm surface
    component_mix_stage: str = field(
        default="reward",
        metadata={
            "help": "Stage that applies multi-component reward mixing: reward or advantage",
            "choices": ["reward", "advantage"],
        },
    )
    adv_normalization_scope: str = field(
        default="group",
        metadata={
            "help": "Advantage normalization scope: group or global",
            "choices": ["group", "global"],
        },
    )
    adv_norm_eps: float = field(
        default=1e-8, metadata={"help": "Numerical epsilon for advantage normalization"}
    )
    clip_max: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional absolute clip for normalized advantages (None means no clipping)"
        },
    )
    use_global_std: bool = field(
        default=False,
        metadata={
            "help": "Use global std instead of per-group std during grouped normalization"
        },
    )
    trim_outliers_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of outliers to trim from each side for grouped reward normalization"
        },
    )
    eval_ema_decay: float = field(
        default=0.9, metadata={"help": "Eval EMA decay shared by built-in algorithms"}
    )
    eval_ema_update_interval: int = field(
        default=1, metadata={"help": "Eval EMA update interval in optimizer steps"}
    )
    shuffle_samples: bool = field(
        default=True,
        metadata={"help": "Shuffle training samples before local update execution"},
    )
    shuffle_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional deterministic shuffle seed for training sample order"
        },
    )
    training_share_rollout_indices: bool = field(
        default=True,
        metadata={
            "help": "Reuse rollout index scheduler for training timestep selection unless disabled"
        },
    )

    # Sub-configuration
    rollout_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def global_batch_size(self) -> int:
        return self.prompts_per_rollout * self.samples_per_prompt

    @property
    def window(self) -> SchedulerConfig:
        """Deprecated alias for older internal callers."""
        return self.rollout_scheduler

    def validate(self) -> None:
        if not self.algorithm_type and not self.algorithm_dotpath:
            raise ValueError(
                "algorithm_type or algorithm_dotpath must be set. "
                "Available built-ins: grpo, nft, mix_grpo."
            )
        if self.samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be >= 1.")
        if self.prompts_per_rollout < 1:
            raise ValueError("prompts_per_rollout must be >= 1.")
        if not isinstance(self.algorithm_kwargs, dict):
            raise ValueError("algorithm.algorithm_kwargs must be a dict.")
        if float(self.adv_norm_eps) <= 0:
            raise ValueError(
                "algorithm.adv_norm_eps must be > 0. " f"Got: {self.adv_norm_eps!r}"
            )
        if self.clip_max is not None and float(self.clip_max) <= 0:
            raise ValueError(
                "algorithm.clip_max must be > 0 when set. " f"Got: {self.clip_max!r}"
            )
        if not (0.0 <= float(self.trim_outliers_ratio) < 0.5):
            raise ValueError(
                "algorithm.trim_outliers_ratio must be in [0.0, 0.5). "
                f"Got: {self.trim_outliers_ratio!r}"
            )
        if float(self.eval_ema_decay) < 0:
            raise ValueError("algorithm.eval_ema_decay must be >= 0.")
        if int(self.eval_ema_update_interval) < 1:
            raise ValueError("algorithm.eval_ema_update_interval must be >= 1.")
        window_cfg = self.window
        if (
            window_cfg.max_iters_per_window is not None
            and window_cfg.min_iters_per_window is not None
            and window_cfg.min_iters_per_window > window_cfg.max_iters_per_window
        ):
            raise ValueError("min_iters_per_window must be <= max_iters_per_window.")


@dataclass
class TrainingConfig:
    """
    Training Configuration:
    contains optimizer, update schedule, train backend, LoRA, and core training controls.
    """

    # Optimizer and update schedule

    # Local Batch Size -> Mini Batch Size (per update) -> Micro Batch Size (per forward/backward pass)
    micro_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Micro-batch size per GPU for one forward/backward pass."},
    )
    num_updates_per_batch: int = field(
        default=1,
        metadata={
            "help": "Number of optimizer updates performed from one local batch. literally equals local_batch_size / mini_batch_size"
        },
    )
    learning_rate: float = field(
        default=1e-6, metadata={"help": "Peak learning rate for the optimizer"}
    )
    optimizer_type: str = field(
        default="adamw",
        metadata={
            "help": "Optimizer type: adamw, adam, sgd",
            "choices": ["adamw", "adam", "sgd"],
        },
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Adam optimizer beta1 (first moment decay)"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Adam optimizer beta2 (second moment decay)"}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam optimizer epsilon for numerical stability"},
    )
    weight_decay: float = field(
        default=1e-4, metadata={"help": "Weight decay (L2 regularization) coefficient"}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm for gradient clipping"}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Number of learning rate warmup steps"}
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "LR scheduler type: constant, linear, cosine",
            "choices": ["constant", "linear", "cosine"],
        },
    )

    # Train backend
    train_backend: str = field(
        default="fsdp",
        metadata={
            "help": "Training backend name (fsdp/veomni built-in; megatron scaffold requires actor_class_path in train_backend_kwargs); or custom via train_backend_dotpath"
        },
    )
    train_backend_dotpath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Python dotpath to custom TrainBackend class (overrides built-in backend selection)"
        },
    )
    train_backend_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra kwargs for selected train backend; accepts JSON string (CLI) or mapping (YAML)"
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing training checkpoint.pt to resume from (after actors init)"
        },
    )

    # LoRA
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Enable LoRA (Low-Rank Adaptation) for parameter-efficient training"
        },
    )
    lora_rank: int = field(
        default=16,
        metadata={
            "help": "LoRA rank (lower = fewer parameters, higher = more expressive)"
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor (effective scale = alpha/rank)"},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "LoRA target modules (e.g. [to_q, to_k, to_v])"}
    )

    # FSDP sharding
    fsdp_mode: str = field(
        default="full",
        metadata={
            "help": "FSDP sharding mode: 'full' = shard across all ranks; "
            "'hybrid' = HSDP, shard within node (8 GPUs), replicate across nodes"
        },
    )
    reshard_after_forward: bool = field(
        default=True,
        metadata={
            "help": "Whether FSDP reshards (frees) full params after forward pass. "
            "True=save memory, False=save backward allgather (uses more memory)"
        },
    )

    # Offload
    fsdp_cpu_offload: bool = field(
        default=False, metadata={"help": "Offload FSDP parameters and gradients to CPU"}
    )

    # Memory optimization
    use_gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient checkpointing to save memory at the cost of compute"
        },
    )

    def __post_init__(self):
        if isinstance(self.lora_target_modules, str):
            self.lora_target_modules = (
                [self.lora_target_modules] if self.lora_target_modules.strip() else None
            )

    def validate(self) -> None:
        if self.micro_batch_size is not None and self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1 when set.")
        if self.num_updates_per_batch < 1:
            raise ValueError("num_updates_per_batch must be >= 1.")
        if self.lora_target_modules is not None and not isinstance(
            self.lora_target_modules, list
        ):
            raise ValueError(
                "training.lora_target_modules must be a list of strings or null. "
                f"Got: {type(self.lora_target_modules).__name__}"
            )
        if not isinstance(self.train_backend_kwargs, dict):
            raise ValueError("training.train_backend_kwargs must be a dict.")


@dataclass
class PrecisionConfig:
    """Precision controls for training and rollout."""

    model_precision: str = field(
        default="bf16",
        metadata={"help": "Training model/component load precision (default: bf16)"},
    )
    fsdp_precision: str = field(
        default="fp32",
        metadata={"help": "Training-side FSDP param precision (default: fp32)"},
    )

    # Training Precision
    training_autocast_precision: str = field(
        default="bf16",
        metadata={
            "help": "Training-side autocast precision for loss/model forwards (default: bf16)"
        },
    )

    # Rollout Precision
    rollout_autocast_precision: str = field(
        default="bf16",
        metadata={
            "help": "Rollout-side autocast precision for sampler/replay forwards (default: bf16)"
        },
    )
    trajectory_precision: str = field(
        default="fp16",
        metadata={
            "help": "Precision used to store rollout trajectory latents (default: fp16)"
        },
    )
    logprob_precision: str = field(
        default="fp32",
        metadata={
            "help": "Precision used to store rollout log-prob tensors (default: fp32)"
        },
    )

    def validate(self) -> None:
        validate_precision_type(
            self.model_precision,
            field_name="precision.model_precision",
        )
        validate_precision_type(
            self.fsdp_precision,
            field_name="precision.fsdp_precision",
        )
        validate_precision_type(
            self.training_autocast_precision,
            field_name="precision.training_autocast_precision",
        )
        validate_precision_type(
            self.rollout_autocast_precision,
            field_name="precision.rollout_autocast_precision",
        )
        validate_precision_type(
            self.trajectory_precision,
            field_name="precision.trajectory_precision",
        )
        validate_precision_type(
            self.logprob_precision,
            field_name="precision.logprob_precision",
        )


@dataclass
class RolloutConfig:
    """Rollout topology, buffer, control, and artifact configuration."""

    # --- Topology ---
    mode: str = field(
        default=None,
        metadata={
            "help": "Canonical rollout topology: direct_sampling, separate, or colocate",
            "choices": ["direct_sampling", "separate", "colocate"],
        },
    )
    rollout_engine: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dedicated rollout engine selector for separate or colocate. Must be unset in direct_sampling."
        },
    )
    rollout_batch_size: int = field(
        default=1,
        metadata={
            "help": "Max prompts per rollout-engine generate() call before actor-side sub-batching."
        },
    )
    num_gpus_per_actor: Optional[int] = field(
        default=None,
        metadata={
            "help": "Dedicated rollout service GPUs per actor/engine. Required for separate and colocate."
        },
    )
    tp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Dedicated rollout service tensor parallel hint."},
    )
    sp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Dedicated rollout service sequence/spatial parallel hint."},
    )
    transport_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Cast rollout transport payloads to this dtype (fp16/bf16) to reduce transfer size."
        },
    )
    transport_drop_decoded_videos: bool = field(
        default=True,
        metadata={
            "help": "Drop decoded video tensors from rollout transport payloads after reward handling."
        },
    )
    sglang_local_mode: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether SGLang rollout uses in-actor local generator mode."},
    )
    sglang_verify_weight_checksum: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether SGLang verifies weight checksum after rollout-side updates."
        },
    )
    sglang_disable_autocast: Optional[bool] = field(
        default=None,
        metadata={"help": "Disable torch.autocast inside SGLang rollout engine."},
    )
    sglang_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Engine-scoped SGLang rollout kwargs."}
    )

    # --- Buffer ---
    max_queue_size: int = field(
        default=0, metadata={"help": "Max rollout buffer queue size (0 = unbounded)"}
    )
    drop_invalid: bool = field(
        default=True, metadata={"help": "Drop samples that fail validation checks"}
    )
    reward_min: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum reward threshold for sample filtering (None = no filter)"
        },
    )
    reward_max: Optional[float] = field(
        default=None,
        metadata={
            "help": "Maximum reward threshold for sample filtering (None = no filter)"
        },
    )
    min_samples: int = field(
        default=1,
        metadata={"help": "Minimum samples required before dispatching a batch"},
    )
    plugin_dotpaths: List[str] = field(
        default_factory=list,
        metadata={"help": "Rollout buffer filter plugin class dotpath(s)."},
    )

    # --- Control ---
    num_rollout: int = field(
        default=1000,
        metadata={"help": "Total number of rollout iterations (outer-loop steps)"},
    )
    start_rollout_id: int = field(
        default=0, metadata={"help": "Starting rollout step/ID for resuming training"}
    )
    max_inflight_rollouts: int = field(
        default=1,
        metadata={
            "help": "Max concurrent in-flight rollouts for the async training runner"
        },
    )

    # --- Artifacts ---
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "Output directory for checkpoints, logs, and generated samples"
        },
    )
    save_steps: int = field(
        default=100,
        metadata={
            "help": "Save a checkpoint every N training steps (0 disables periodic saves)"
        },
    )

    def __post_init__(self):
        if self.rollout_engine is not None:
            val = self.rollout_engine.strip().lower()
            self.rollout_engine = val if val else None

    def set_start_rollout_id(self, rollout_id: int) -> None:
        """Update the rollout control cursor on the canonical config owner."""
        rollout_id = int(rollout_id)
        if rollout_id < 0:
            raise ValueError("start_rollout_id must be >= 0.")
        self.start_rollout_id = rollout_id

    def validate(self) -> None:
        if not str(self.mode or "").strip():
            raise ValueError(
                "rollout.mode must be set explicitly. "
                "Implicit rollout topology derivation has been removed."
            )
        for attr_name in (
            "num_gpus_per_actor",
            "tp_size",
            "sp_size",
            "rollout_batch_size",
        ):
            value = getattr(self, attr_name)
            if value is not None and int(value) < 1:
                raise ValueError(f"rollout.{attr_name} must be >= 1 when set.")
        if self.transport_dtype is not None:
            validate_precision_type(
                self.transport_dtype,
                field_name="rollout.transport_dtype",
                allow_disable_aliases=True,
            )
        if not isinstance(self.sglang_kwargs, dict):
            raise ValueError("rollout.sglang_kwargs must be a dict.")
        forbidden_sglang_precision_keys = {
            "prompt_encoder_dtype",
            "sglang_prompt_encoder_dtype",
        }
        precision_override_keys = sorted(
            key
            for key in self.sglang_kwargs.keys()
            if str(key) in forbidden_sglang_precision_keys
        )
        if precision_override_keys:
            raise ValueError(
                "SGLang prompt-encoder precision is controlled by "
                "precision.rollout_autocast_precision; remove the following "
                f"engine-specific override(s) from rollout.sglang_kwargs: "
                f"{precision_override_keys}"
            )
        if self.max_queue_size < 0:
            raise ValueError(
                f"rollout.max_queue_size must be >= 0, got: {self.max_queue_size}"
            )
        if self.min_samples < 1:
            raise ValueError(
                f"rollout.min_samples must be >= 1, got: {self.min_samples}"
            )
        if (
            self.reward_min is not None
            and self.reward_max is not None
            and self.reward_min > self.reward_max
        ):
            raise ValueError(
                f"rollout.reward_min must be <= rollout.reward_max, "
                f"got min={self.reward_min}, max={self.reward_max}"
            )
        for i, path in enumerate(self.plugin_dotpaths):
            if not str(path).strip():
                raise ValueError(
                    f"rollout.plugin_dotpaths[{i}] must be a non-empty string"
                )
        if self.num_rollout < 1:
            raise ValueError("num_rollout must be >= 1.")
        if self.start_rollout_id < 0:
            raise ValueError("start_rollout_id must be >= 0.")
        if self.max_inflight_rollouts < 1:
            raise ValueError("max_inflight_rollouts must be >= 1.")
        if int(self.save_steps) < 0:
            raise ValueError("save_steps must be >= 0 (0 disables periodic saves).")


@dataclass
class EvaluationConfig:
    """Evaluation cadence and batch sizing configuration."""

    eval_steps: int = field(
        default=100,
        metadata={
            "help": "Run evaluation every N training steps (0 disables periodic eval)"
        },
    )
    eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size for evaluation"}
    )
    num_inference_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional eval-only denoising step override. If unset, eval reuses sampling.num_inference_steps."
        },
    )
    sampling_adapter: Optional[str] = field(
        default=None, metadata={"help": "Optional eval-only LoRA adapter override."}
    )
    sde_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional eval-only sampler transition override. If unset, eval reuses sampling.sde_type."
        },
    )
    eta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional eval-only eta override. If unset, eval reuses sampling.eta."
        },
    )

    def validate(self) -> None:
        if int(self.eval_steps) < 0:
            raise ValueError("eval_steps must be >= 0 (0 disables periodic eval).")
        if int(self.eval_batch_size) < 1:
            raise ValueError("eval_batch_size must be >= 1.")
        if self.num_inference_steps is not None and int(self.num_inference_steps) < 1:
            raise ValueError("evaluation.num_inference_steps must be >= 1 when set.")
        if self.eta is not None and float(self.eta) < 0:
            raise ValueError("evaluation.eta must be >= 0 when set.")


@dataclass
class LoggingConfig:
    """Experiment-logging and reporting configuration."""

    logging_steps: int = field(
        default=10,
        metadata={
            "help": "Log metrics every N training steps (0 disables periodic step logging)"
        },
    )
    logging_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory for WandB logs/artifacts (defaults to output_dir/logs)"
        },
    )
    report_to_wandb: bool = field(
        default=False, metadata={"help": "Enable WandB reporting (True/False)"}
    )
    project_name: str = field(
        default="diffusionrl", metadata={"help": "Project name for WandB"}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Run name for WandB (auto-generated if None)"}
    )
    log_media: bool = field(
        default=False,
        metadata={"help": "Log generated media previews to WandB (true/false)"},
    )
    media_max_items: int = field(
        default=8,
        metadata={
            "help": "Maximum generated media items to log per rollout when log_media=true"
        },
    )
    tags: Optional[List[str]] = field(
        default=None, metadata={"help": "Tags for WandB run (e.g. [exp1, baseline])."}
    )
    entity: Optional[str] = field(
        default=None, metadata={"help": "WandB entity (team or username)."}
    )
    transport_log_payload_bytes: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether rollout transport logs serialized payload sizes for debugging."
        },
    )

    def __post_init__(self):
        if isinstance(self.tags, str):
            self.tags = [self.tags] if self.tags.strip() else None

    def validate(self) -> None:
        if self.logging_steps < 0:
            raise ValueError(
                "logging_steps must be >= 0 (0 disables periodic step logging)."
            )
        if self.media_max_items < 1:
            raise ValueError("media_max_items must be >= 1.")


@dataclass
class DebugConfig:
    """Debug mode and intermediate artifact controls."""

    mode: str = field(
        default="none",
        metadata={
            "help": "Debug mode: none or train_only",
            "choices": ["none", "train_only"],
        },
    )
    save_dir: str = field(
        default="outputs/debug",
        metadata={"help": "Directory for debug artifacts and saved rollout payloads"},
    )
    save_intermediates: bool = field(
        default=False,
        metadata={
            "help": "Save rollout debug payloads during the normal training loop"
        },
    )
    load_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to debug payload/training batch for train_only mode"},
    )
    num_rollouts: int = field(
        default=1, metadata={"help": "Number of train_only iterations to run"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to dump per-step SDE tensors for train-inference consistency debugging. "
            "Sampling tensors saved to <dir>/sampling/, training tensors to <dir>/training/."
        },
    )

    def validate(self) -> None:
        if int(self.num_rollouts) < 1:
            raise ValueError("debug num_rollouts must be >= 1.")
        if self.mode == "train_only" and not self.load_path:
            logger.info(
                "debug mode=train_only without --debug.load-path: "
                "will generate synthetic training data."
            )


@dataclass
class TrainingArguments:
    """All configuration parameters for DiffusionRL training."""

    # ========== Dotpaths For Custom Modules (Dynamic Loading) ==========
    data_source_dotpath: str = field(
        default="diffusionrl.data.DefaultDataSource",
        metadata={
            "help": "Python dotpath to DataSource class for loading training/eval prompt streams"
        },
    )
    rollout_function_dotpath: str = field(
        default="diffusionrl.rollout.default_rollout.generate_rollout",
        metadata={
            "help": "Python dotpath to the rollout function invoked by the rollout service. This is the main rollout extension seam."
        },
    )
    eval_function_dotpath: str = field(
        default="diffusionrl.rollout.default_rollout.evaluate_rollout",
        metadata={
            "help": "Python dotpath to the evaluation function invoked by the rollout service."
        },
    )
    reward_hook_dotpath: str = field(
        default="diffusionrl.rollout.default_rollout.score_rewards_hook",
        metadata={
            "help": "Python dotpath to the reward hook used by rollout/eval functions. This makes reward a first-class rollout hook."
        },
    )

    # ========== Grouped Configuration ==========
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ========== Data Configuration ==========
    data_path: Optional[str] = field(
        default="data/samples/prompts_toy.json",
        metadata={
            "help": "Path to training prompt data file (JSON, JSONL, or TXT). JSON items should provide text via 'prompt' or 'caption'."
        },
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to evaluation prompt data file. If unset, eval uses data_path with deterministic ordering."
        },
    )

    # ========== Seed ==========
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    def to_dotted_dict(self) -> Dict[str, Any]:
        """Export config using namespace-preserving dotted keys."""
        dotted: Dict[str, Any] = {}
        _populate_dotted_config_dict(dotted, value=self)
        return dotted


GROUP_CONFIG_TYPES = {
    info.name: info.type
    for info in fields(TrainingArguments)
    if isinstance(info.type, type) and is_dataclass(info.type)
}
GROUP_CONFIG_NAMES = set(GROUP_CONFIG_TYPES)
TOP_LEVEL_FIELD_NAMES = {
    info.name
    for info in fields(TrainingArguments)
    if info.name not in GROUP_CONFIG_NAMES
}
GROUP_SUBCONFIG_NAMES: Dict[str, set[str]] = {
    _group_name: {
        info.name
        for info in fields(_group_type)
        if isinstance(info.type, type) and is_dataclass(info.type)
    }
    for _group_name, _group_type in GROUP_CONFIG_TYPES.items()
}

_ANSI_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
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


def _copy_config_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _populate_dotted_config_dict(
    target: Dict[str, Any],
    *,
    value: Any,
    prefix: Optional[str] = None,
) -> None:
    if not is_dataclass(value):
        if prefix is None:
            raise ValueError(
                "Dotted config export requires a key prefix for leaf values."
            )
        target[prefix] = _copy_config_value(value)
        return

    for info in fields(type(value)):
        child = getattr(value, info.name)
        child_prefix = info.name if prefix is None else f"{prefix}.{info.name}"
        if is_dataclass(child):
            _populate_dotted_config_dict(target, value=child, prefix=child_prefix)
        else:
            target[child_prefix] = _copy_config_value(child)


_DERIVED_VIEW_KEYS = (
    "derived.training.actor_count",
    "derived.training.world_size",
    "derived.training.dp_size",
    "derived.training.tp_size",
    "derived.training.pp_size",
    "derived.training.sp_size",
    "derived.training.ep_size",
    "derived.rollout.prompts_per_rollout",
    "derived.training.global_batch_size",
    "derived.training.local_batch_size",
    "derived.training.local_mini_batch_size",
    "derived.training.micro_batch_size",
    "derived.training.num_updates_per_batch",
)


def _record_derivation_error(
    derived_view: Dict[str, Any],
    *,
    scope: str,
    exc: Exception,
) -> None:
    derived_view[f"derived.errors.{scope}"] = f"{type(exc).__name__}: {exc}"


def _apply_derived_config(
    view: Dict[str, Any],
    *,
    base_config: Dict[str, Any],
    derived_config: DerivedConfig,
) -> None:
    """Overlay pre-derived values onto a dotted config view."""
    model_spec = derived_config.model_spec
    rollout_info = derived_config.rollout_info
    train_backend_config = derived_config.train_backend_config
    train_topology = derived_config.training_topology
    train_plan = derived_config.require_training_plan()

    view["algorithm.algorithm_dotpath"] = derived_config.algorithm_dotpath
    view["model.model_dotpath"] = model_spec.model_dotpath
    view["model.model_type"] = model_spec.model_type
    view["sampling.sampler_dotpath"] = model_spec.sampler_dotpath
    view["training.train_backend"] = train_backend_config.name
    view["rollout.mode"] = rollout_info.mode
    view["rollout.rollout_engine"] = rollout_info.rollout_engine

    view["derived.training.actor_count"] = train_topology.actor_count
    view["derived.training.world_size"] = train_topology.world_size
    view["derived.training.dp_size"] = train_topology.dp_size
    view["derived.training.tp_size"] = train_topology.tp_size
    view["derived.training.pp_size"] = train_topology.pp_size
    view["derived.training.sp_size"] = train_topology.sp_size
    view["derived.training.ep_size"] = train_topology.ep_size
    view["derived.rollout.prompts_per_rollout"] = int(
        base_config["algorithm.prompts_per_rollout"]
    )

    view["derived.training.global_batch_size"] = train_plan.global_batch_size
    view["derived.training.local_batch_size"] = train_plan.local_batch_size
    view["derived.training.local_mini_batch_size"] = train_plan.local_mini_batch_size
    view["derived.training.micro_batch_size"] = train_plan.micro_batch_size
    view["derived.training.num_updates_per_batch"] = train_plan.num_updates_per_batch


def build_derived_config_view(
    dotted_config: Dict[str, Any],
    *,
    derived_config: DerivedConfig,
) -> Dict[str, Any]:
    """Build a dotted config view from base config plus pre-derived values."""
    derived_view = {
        key: _copy_config_value(value) for key, value in dotted_config.items()
    }

    try:
        _apply_derived_config(
            derived_view,
            base_config=dotted_config,
            derived_config=derived_config,
        )
    except Exception as exc:
        _record_derivation_error(derived_view, scope="config", exc=exc)
        for key in _DERIVED_VIEW_KEYS:
            derived_view.setdefault(key, None)
        return derived_view

    return derived_view


def print_config_views(
    *,
    args: TrainingArguments,
    print_derived_config: bool,
    derived_config: DerivedConfig,
) -> None:
    if not print_derived_config:
        return

    use_color = _supports_color_output()
    print(_color("[Derived Config] final derived values", "bold", enabled=use_color))
    derived_dotted = build_derived_config_view(
        args.to_dotted_dict(),
        derived_config=derived_config,
    )
    for key in sorted(derived_dotted.keys()):
        value_text = _format_config_value(derived_dotted.get(key))
        print(f"  {_color(key, 'cyan', enabled=use_color)}: {value_text}")


def validate_and_derive_config(
    args: TrainingArguments,
    *,
    derived_config: Optional[DerivedConfig] = None,
) -> Tuple[TrainingArguments, DerivedConfig]:
    """Return validated args plus reusable derived config."""
    if isinstance(derived_config, DerivedConfig):
        refreshed_derived_config = derive_config(args, existing=derived_config)
        return args, attach_training_plan(
            args,
            derived_config=refreshed_derived_config,
        )

    validate_grouped_configs(args)
    validation_derived_config = derive_config(args)
    args = validate_args(
        args,
        derived_config=validation_derived_config,
        skip_grouped_validation=True,
    )
    derived_config = attach_training_plan(
        args,
        derived_config=validation_derived_config,
    )
    return args, derived_config


def validate_args(
    args: TrainingArguments,
    *,
    derived_config: Optional[DerivedConfig] = None,
    skip_grouped_validation: bool = False,
) -> TrainingArguments:
    """Validate arguments. Grouped validation may normalize fields such as ``rollout.plugin_dotpaths``."""
    if not skip_grouped_validation:
        validate_grouped_configs(args)
    debug_mode = args.debug.mode

    derived = derive_config(
        args,
        existing=derived_config,
    )
    derived_model = derived.model_spec
    backend_config = derived.train_backend_config
    backend_name = backend_config.name
    rollout_info = derived.rollout_info
    validate_algorithm_kwargs_payload(args)
    validate_train_backend_config(
        train_backend_config=backend_config,
    )
    backend_capabilities = derived.require_train_backend_capabilities().as_dict()
    validate_rollout_mode(
        args,
        rollout_info=rollout_info,
        backend_capabilities=backend_capabilities,
        backend_name=backend_name,
    )
    if debug_mode == "train_only":
        validate_dynamic_dotpaths(
            args,
            resolved_model=derived_model,
            algorithm_dotpath=derived.algorithm_dotpath,
            include_data_source=False,
            include_rollout_buffer_plugins=False,
        )
    else:
        validate_dynamic_dotpaths(
            args,
            resolved_model=derived_model,
            algorithm_dotpath=derived.algorithm_dotpath,
        )
        validate_reward_config(RewardSpec.from_args(args))
        validate_rollout_layout(
            rollout_info=rollout_info,
            rollout_num_nodes=args.ray.rollout_num_nodes,
            rollout_num_gpus_per_node=args.ray.rollout_num_gpus_per_node,
            training_num_nodes=args.ray.training_num_nodes,
            training_num_gpus_per_node=args.ray.training_num_gpus_per_node,
            rollout_num_gpus_per_actor=int(args.rollout.num_gpus_per_actor or 0),
            allow_noset_multi_gpu_inference=bool(args.ray.allow_noset_multi_gpu_inference),
        )

    validate_training_batch_geometry(
        prompts_per_rollout=args.algorithm.prompts_per_rollout,
        samples_per_prompt=args.algorithm.samples_per_prompt,
        global_batch_size=args.algorithm.global_batch_size,
        num_updates_per_batch=args.training.num_updates_per_batch,
        micro_batch_size=args.training.micro_batch_size,
        topology=derived.training_topology,
    )
    validate_nft_sampling_contract(args)

    if debug_mode != "train_only":
        validation_algorithm = create_algorithm_from_init_payload(
            build_algorithm_init_payload_from_args(
                args,
                sampling_spec=derived.sampling_spec,
            )
        )
        sampling_requirements = validation_algorithm.get_sampling_requirements()
        validate_engine_algorithm_contract(
            algorithm_type=args.algorithm.algorithm_type,
            rollout_info=rollout_info,
            effective_engine_capabilities=rollout_info.effective_engine_capabilities,
            sampling_requirements=sampling_requirements,
        )
        validate_rollout_mode_constraints(
            rollout_info=rollout_info,
            model_cls=derived_model.model_cls,
        )

    return args


def get_default_args() -> TrainingArguments:
    """Get default arguments without parsing command line."""
    return TrainingArguments()
