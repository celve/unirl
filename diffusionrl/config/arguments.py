"""
diffusionrl Arguments - Configuration parameters for training.

Resolution order:
1) Explicit CLI flags
2) YAML values from --config
3) Dataclass field defaults
4) public config entrypoints validate the explicit config and derived helpers compute
   resolved values without mutating TrainingArguments

Schema and public config entrypoints stay in this file. Parser mechanics live
in ``argument_parsing.py`` and generic config derivation lives in
``resolution.py``.
"""
import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional

from diffusionrl.algorithms.construction import (
    instantiate_algorithm_from_config,
    resolve_algorithm_dotpath,
    build_algorithm_config,
)
from diffusionrl.config.argument_parsing import (
    GROUP_DISPLAY_NAMES,
    build_add_argument_kwargs,
    build_cli_option_strings,
    collect_cli_field_specs,
    collect_explicit_cli_destinations,
    load_yaml_mapping,
    merge_yaml_overrides,
    parse_cli_key_value,
)
from diffusionrl.config.resolution import (
    DEFAULT_MODEL_PATH,
    ROLLOUT_MODES,
    ConfigBundle,
    collect_sampling_requirements,
    normalize_train_backend_name,
    resolve_config,
)
from diffusionrl.config.validation import (
    apply_model_config_hook,
    validate_algorithm_kwargs_payload,
    validate_algorithm_dotpath,
    validate_dynamic_dotpaths,
    validate_grouped_configs,
    validate_model_sampling_contract,
    validate_nft_sampling_contract,
    validate_resolved_engine_algorithm_contract,
    validate_reward_and_rollout_buffer_config,
    validate_rollout_layout,
    validate_rollout_mode,
    validate_rollout_mode_constraints,
    validate_precision_type,
    validate_train_backend_config,
    validate_training_batch_geometry,
    validate_training_misc,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Model Configuration:
    contains model type and checkpoint paths.
    """

    model_type: str = field(default="hunyuan",
        metadata={"help": "Model architecture type (hunyuan, flux, sd3, mochi, wan2.1, bagel)"})
    model_dotpath: str = field(default=DEFAULT_MODEL_PATH,
        metadata={"help": "Python dotpath to ModelBundle class. Auto-resolved from model_type"})
    pretrained_model_ckpt_path: str = field(default="",
        metadata={"help": "Path to pretrained model weights (local path or HuggingFace ID)"})
    vae_ckpt_path: Optional[str] = field(default=None,
        metadata={"help": "Path to separate VAE checkpoint, if not bundled with the model"})
    text_encoder_ckpt_path: Optional[str] = field(default=None,
        metadata={"help": "Path to separate text encoder checkpoint, if not bundled"})

    def validate(self) -> None:
        if not self.model_dotpath:
            raise ValueError(
                "model_dotpath must be set. It is usually auto-resolved from model_type. "
                "Set --model.model-type (hunyuan, flux, sd3, mochi) or provide "
                "--model.model-dotpath explicitly."
            )

@dataclass
class SamplingConfig:
    """
    Sampling Engine Configuration:
    contains sampler type, logprob source, and denoising controls.
    """

    sampler_dotpath: str = field(default="",
        metadata={"help": "Optional Python dotpath to Sampler class; omit to auto-resolve from model_type"})
    logprob_source: str = field(default="replay",
        metadata={
            "help": "SGLang log-prob mode: replay (training-side replay path) or native (engine-side log_probs)",
            "choices": ["replay", "native"],
        },
    )
    replay_sampler_dotpath: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to replay sampler, if different from sampler_dotpath"})
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Explicit sampler constructor kwargs for direct sampling and replay sampler instantiation."})
    num_inference_steps: int = field(default=50,
        metadata={"help": "Number of denoising steps during sampling"})
    eta: float = field(default=1.0,
        metadata={"help": "SDE noise coefficient (eta=0 is ODE, eta=1 is full SDE)"})
    sde_type: str = field(default="flow",
        metadata={
            "help": "Transition rule. Supported: flow, cps, dance, dpm2",
            "choices": ["flow", "cps", "dance", "dpm2"],
        },
    )
    max_samples_per_request: Optional[int] = field(default=None,
        metadata={"help": "Generated-sample cap per training-actor direct-sampling request; rollout_total_samples stays prompts_per_rollout*samples_per_prompt"})
    shift: float = field(default=3.0,
        metadata={"help": "Shift parameter for the sampling timestep schedule (model-specific)"})
    guidance_scale: float = field(default=7.5,
        metadata={"help": "Classifier-free guidance scale (0.0 = no guidance)"})
    sampling_adapter: Optional[str] = field(default=None,
        metadata={"help": "Sampling adapter type for special modes (e.g. 'old' for NFT)"})
    init_same_noise: bool = field(default=False,
        metadata={"help": "Use identical initial noise for all samples of the same prompt"})

    # Generated media dimensions and frame rate
    height: int = field(default=256,
        metadata={"help": "Generated image/video height in pixels"})
    width: int = field(default=256,
        metadata={"help": "Generated image/video width in pixels"})
    num_frames: int = field(default=16,
        metadata={"help": "Number of video frames to generate (video models only)"})
    fps: int = field(default=8,
        metadata={"help": "Video frame rate (video models only)"})

    def validate(self) -> None:
        if self.logprob_source not in ("replay", "native"):
            raise ValueError(f"logprob_source must be one of replay/native, got: {self.logprob_source}")
        if self.max_samples_per_request is not None and self.max_samples_per_request < 1:
            raise ValueError("max_samples_per_request must be >= 1 when set.")
        if not isinstance(self.sampler_kwargs, dict):
            raise ValueError("sampling.sampler_kwargs must be a dict.")

        _precision_keys = {"autocast_precision", "trajectory_precision", "logprob_precision"}
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

    reward_backend: str = field(default="local",
        metadata={
            "help": "Reward backend: local, http, ray_pool",
            "choices": ["local", "http", "ray_pool"],
        },
    )

    # http reward service related fields
    reward_service_urls: Optional[List[str]] = field(default=None,
        metadata={"help": "HTTP reward service URL(s). Accepts a single URL or a list/comma-separated list for load balancing"})

    # local reward model related fields
    reward_dotpath: Optional[str] = field(default=None,
        metadata={"help": "Optional Python dotpath to a custom reward scorer class; omit to use built-in reward_components"})
    reward_model_ckpt_path: Optional[str] = field(default=None,
        metadata={"help": "Path to reward model weights (local path or HuggingFace ID)"})
    reward_components: Optional[List[str]] = field(default_factory=lambda: ["hpsv2"],
        metadata={"help": "Reward component name(s). Accepts a single built-in scorer name or a list/comma-separated list such as hpsv2, pickscore, clip, ocr"})
    reward_batch_size: int = field(default=8,
        metadata={"help": "Batch size for reward model inference"})
    local_reward_device: str = field(default="cpu",
        metadata={"help": "Device for local in-process reward scorers: cpu, auto, or cuda"})

    # ray_pool reward pool related fields    
    reward_dedicated_num_nodes: int = field(default=0,
        metadata={"help": "Number of nodes for dedicated reward actors (mutually exclusive with num_gpus)"})
    reward_dedicated_num_gpus: int = field(default=0,
        metadata={"help": "Total GPUs for dedicated reward actors (0 = CPU reward)"})
    reward_dedicated_num_gpus_per_node: int = field(default=0,
        metadata={"help": "GPUs per node for dedicated reward actors"})
    reward_dedicated_gpus_per_actor: int = field(default=1,
        metadata={"help": "GPUs per individual reward actor"})
    reward_location: str = field(default="auto",
        metadata={"help": "Where default reward scoring runs: auto, driver, or sampling_actor"})

   
    # multi-reward related fields
    reward_weights: Optional[List[float]] = field(default=None,
        metadata={"help": "Weights for each reward component in multi-reward aggregation"})
    reward_aggregation_method: str = field(default="weighted_sum",
        metadata={"help": "Multi-reward aggregation method: weighted_sum, mean, min, max, concat"})

    @property
    def has_http_reward_urls(self) -> bool:
        return bool(self.reward_service_urls)

    @property
    def has_http_reward(self) -> bool:
        return str(self.reward_backend or "local").strip().lower() == "http"

    @property
    def has_builtin_reward(self) -> bool:
        if isinstance(self.reward_components, str):
            return bool(str(self.reward_components).strip())
        if not isinstance(self.reward_components, list):
            return False
        return any(str(name or "").strip() for name in self.reward_components)

    @property
    def has_dedicated_reward_pool(self) -> bool:
        return bool(
            self.reward_dedicated_num_gpus > 0 or self.reward_dedicated_num_nodes > 0
        )

    def validate(self) -> None:
        reward_backend = str(self.reward_backend or "local").strip().lower()
        if reward_backend not in ("local", "http", "ray_pool"):
            raise ValueError(
                "reward_backend must be one of local/http/ray_pool, "
                f"got: {self.reward_backend}"
            )
        reward_location = str(self.reward_location or "auto").strip().lower()
        if reward_location not in ("driver", "sampling_actor", "auto"):
            raise ValueError(
                "reward_location must be one of driver/sampling_actor/auto, "
                f"got: {self.reward_location}"
            )
        local_reward_device = str(self.local_reward_device or "cpu").strip().lower()
        if local_reward_device not in ("cpu", "auto", "cuda"):
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
        if reward_backend == "ray_pool" and not self.has_dedicated_reward_pool:
            raise ValueError(
                "reward_backend='ray_pool' requires reward_dedicated_num_gpus "
                "or reward_dedicated_num_nodes."
            )
        if reward_backend != "ray_pool" and self.has_dedicated_reward_pool:
            raise ValueError(
                "reward_dedicated_* settings are only valid when "
                "reward_backend='ray_pool'."
            )
        if not self.has_http_reward and not self.reward_dotpath and not self.has_builtin_reward:
            raise ValueError(
                "Reward scoring requires either reward_components for built-ins, "
                "or reward_dotpath for a custom scorer."
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
        metadata={
            "help": "Ray placement group strategy: PACK or SPREAD",
            "choices": ["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"],
        },
    )
    colocate_training_gpu_fraction: float = field(default=0.4,
        metadata={"help": "GPU memory fraction for training when colocated"})
    colocate_rollout_gpu_fraction: float = field(default=0.4,
        metadata={"help": "GPU memory fraction for rollout when colocated"})
    allow_noset_multi_gpu_inference: bool = field(default=False,
        metadata={"help": "Allow multi-GPU rollout actors (experimental NOSET layout)"})
    offload_train: bool = field(default=False,
        metadata={"help": "Enable model offload for training actors"})
    offload_rollout: bool = field(default=False,
        metadata={"help": "Enable model offload for rollout actors"})

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

        _valid_strategies = ("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD")
        strategy = self.placement_strategy.strip().upper()
        if strategy not in _valid_strategies:
            raise ValueError(
                "ray.placement_strategy must be one of "
                f"{sorted(_valid_strategies)}, got: {self.placement_strategy!r}"
            )

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
    protocol: str = field(default=None,
        metadata={
            "help": "Explicit weight sync mode. Must be one of: disabled, tensor_payload, nccl_broadcast, checkpoint_path",
            "choices": ["disabled", "tensor_payload", "nccl_broadcast", "checkpoint_path"],
        },
    )
    dir: str = field(default="outputs/weight_sync",
        metadata={"help": "Directory for checkpoint-based weight sync (use shared FS for multi-node)"})
    bucket_size: int = field(default=256,
        metadata={"help": "Weight sync tensor bucket size (MB) for tensor/distributed strategies"})
    flush_cache: bool = field(default=True,
        metadata={"help": "Whether the rollout side flushes inference-engine caches after each weight sync bucket"})
    target_modules: Optional[List[str]] = field(default=None,
        metadata={"help": "Rollout-side modules that receive weight updates (defaults to ['transformer'])."})

    def validate(self) -> None:
        _valid_protocols = ("disabled", "tensor_payload", "nccl_broadcast", "checkpoint_path")
        normalized_protocol = self.protocol.strip().lower()
        if not normalized_protocol:
            raise ValueError(
                "sync.protocol must be set explicitly. "
                "Choose one of disabled/tensor_payload/nccl_broadcast/checkpoint_path."
            )
        if normalized_protocol not in _valid_protocols:
            raise ValueError(
                f"sync.protocol must be one of {'/'.join(_valid_protocols)}, "
                f"got: {self.protocol!r}."
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
    """Stateless index-scheduler config for rollout or training timestep selection."""

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
    num_sde_steps: Optional[int] = field(default=None,
        metadata={"help": "Randomly sample this many SDE timestep indices from the timestep_fraction range per step. None means use all indices in range."})
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
    overlap_size: int = field(default=0,
        metadata={"help": "Number of overlapping timesteps between adjacent windows (0 = no overlap)"})
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
    algorithm_type: str = field(default="grpo",
        metadata={"help": "Built-in algorithm family: grpo, nft, or mix_grpo"})
    algorithm_dotpath: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to Algorithm class (auto-resolved from algorithm_type when omitted)"})
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "YAML-only extension surface for algorithm-specific kwargs. Shared framework fields have dedicated algorithm.* entries. On CLI, use repeated --algorithm.kwarg KEY=VALUE only for true algorithm-specific extension keys."})

    # Framework-owned rollout geometry
    samples_per_prompt: int = field(default=4,
        metadata={"help": "Number of generated samples per prompt. This remains framework-owned because rollout geometry depends on it."})
    prompts_per_rollout: int = field(default=1,
        metadata={"help": "Number of unique prompts per rollout step. Rollout geometry is defined by prompts_per_rollout * samples_per_prompt."})

    # Shared algorithm surface
    component_mix_stage: str = field(default="reward",
        metadata={
            "help": "Stage that applies multi-component reward mixing: reward or advantage",
            "choices": ["reward", "advantage"],
        },
    )
    adv_normalization_scope: str = field(default="group",
        metadata={
            "help": "Advantage normalization scope: group or global",
            "choices": ["group", "global"],
        },
    )
    adv_norm_eps: float = field(default=1e-8,
        metadata={"help": "Numerical epsilon for advantage normalization"})
    clip_max: Optional[float] = field(default=None,
        metadata={"help": "Optional absolute clip for normalized advantages (None means no clipping)"})
    use_global_std: bool = field(default=False,
        metadata={"help": "Use global std instead of per-group std during grouped normalization"})
    trim_outliers_ratio: float = field(default=0.0,
        metadata={"help": "Fraction of outliers to trim from each side for grouped reward normalization"})
    eval_ema_decay: float = field(default=0.9,
        metadata={"help": "Eval EMA decay shared by built-in algorithms"})
    eval_ema_update_interval: int = field(default=1,
        metadata={"help": "Eval EMA update interval in optimizer steps"})
    shuffle_samples: bool = field(default=True,
        metadata={"help": "Shuffle training samples before local update execution"})
    shuffle_seed: Optional[int] = field(default=None,
        metadata={"help": "Optional deterministic shuffle seed for training sample order"})
    training_share_rollout_indices: bool = field(default=True,
        metadata={"help": "Reuse rollout index scheduler for training timestep selection unless disabled"})

    # Sub-configuration
    rollout_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

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
        _valid_mix_stages = ("reward", "advantage")
        component_mix_stage = self.component_mix_stage.strip().lower()
        if component_mix_stage not in _valid_mix_stages:
            raise ValueError(
                "algorithm.component_mix_stage must be one of "
                f"{sorted(_valid_mix_stages)}, "
                f"Got: {self.component_mix_stage!r}"
            )
        _valid_adv_scopes = ("group", "global")
        adv_normalization_scope = self.adv_normalization_scope.strip().lower()
        if adv_normalization_scope not in _valid_adv_scopes:
            raise ValueError(
                "algorithm.adv_normalization_scope must be one of "
                f"{sorted(_valid_adv_scopes)}, "
                f"Got: {self.adv_normalization_scope!r}"
            )
        if float(self.adv_norm_eps) <= 0:
            raise ValueError(
                "algorithm.adv_norm_eps must be > 0. "
                f"Got: {self.adv_norm_eps!r}"
            )
        if self.clip_max is not None and float(self.clip_max) <= 0:
            raise ValueError(
                "algorithm.clip_max must be > 0 when set. "
                f"Got: {self.clip_max!r}"
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
            and window_cfg.min_iters_per_window
            > window_cfg.max_iters_per_window
        ):
            raise ValueError(
                "min_iters_per_window must be <= max_iters_per_window."
            )

@dataclass
class TrainingConfig:
    """
    Training Configuration:
    contains optimizer, update schedule, train backend, LoRA, and core training controls.
    """

    # Optimizer and update schedule

    # Local Batch Size -> Mini Batch Size (per update) -> Micro Batch Size (per forward/backward pass)
    micro_batch_size: Optional[int] = field(default=None,
        metadata={"help": "Micro-batch size per GPU for one forward/backward pass."})
    num_updates_per_batch: Optional[int] = field(default=None,
        metadata={"help": "Number of optimizer updates performed from one local batch. literally equals local_batch_size / mini_batch_size"})
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
        metadata={
            "help": "LR scheduler type: constant, linear, cosine",
            "choices": ["constant", "linear", "cosine"],
        },
    )

    # Train backend
    train_backend: str = field(default="fsdp",
        metadata={"help": "Training backend name (fsdp/veomni built-in; megatron scaffold requires actor_class_path in train_backend_kwargs); or custom via train_backend_dotpath"})
    train_backend_dotpath: Optional[str] = field(default=None,
        metadata={"help": "Python dotpath to custom TrainBackend class (overrides built-in backend selection)"})
    train_backend_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Extra kwargs for selected train backend; accepts JSON string (CLI) or mapping (YAML)"})
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing training checkpoint.pt to resume from (after actors init)"},
    )

    # LoRA
    use_lora: bool = field(default=False,
        metadata={"help": "Enable LoRA (Low-Rank Adaptation) for parameter-efficient training"})
    lora_rank: int = field(default=16,
        metadata={"help": "LoRA rank (lower = fewer parameters, higher = more expressive)"})
    lora_alpha: int = field(default=16,
        metadata={"help": "LoRA alpha scaling factor (effective scale = alpha/rank)"})
    lora_target_modules: Optional[str] = field(default=None,
        metadata={"help": "Comma-separated LoRA target modules (e.g. 'to_q,to_k,to_v')"})

    #Offload
    fsdp_cpu_offload: bool = field(default=False,
        metadata={"help": "Offload FSDP parameters and gradients to CPU"})

    # Memory optimization
    use_gradient_checkpointing: bool = field(default=False,
        metadata={"help": "Enable gradient checkpointing to save memory at the cost of compute"})

    def validate(self) -> None:
        explicit_geometry_fields = {
            "micro_batch_size": self.micro_batch_size,
            "num_updates_per_batch": self.num_updates_per_batch,        }
        for field_name, value in explicit_geometry_fields.items():
            if value is not None and int(value) < 1:
                raise ValueError(f"{field_name} must be >= 1 when set.")
        if not isinstance(self.train_backend_kwargs, dict):
            raise ValueError("training.train_backend_kwargs must be a dict.")

@dataclass
class PrecisionConfig:
    """Precision controls for training and rollout."""

    model_precision: str = field(default="bf16",
        metadata={"help": "Training model/component load precision (default: bf16)"})
    fsdp_precision: str = field(default="fp32",
        metadata={"help": "Training-side FSDP param precision (default: fp32)"})

    # Training Precision
    training_autocast_precision: str = field(default="bf16",
        metadata={"help": "Training-side autocast precision for loss/model forwards (default: bf16)"})
        
    # Rollout Precision
    rollout_autocast_precision: str = field(default="bf16",
        metadata={"help": "Rollout-side autocast precision for sampler/replay forwards (default: bf16)"})
    trajectory_precision: str = field(default="fp16",
        metadata={"help": "Precision used to store rollout trajectory latents (default: fp16)"})
    logprob_precision: str = field(default="fp32",
        metadata={"help": "Precision used to store rollout log-prob tensors (default: fp32)"})

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
    mode: str = field(default=None,
        metadata={
            "help": "Canonical rollout topology: direct_sampling, separate, or colocate",
            "choices": ["direct_sampling", "separate", "colocate"],
        },
    )
    rollout_engine: Optional[str] = field(default=None,
        metadata={"help": "Dedicated rollout engine selector for separate or colocate. Must be unset in direct_sampling."})
    rollout_batch_size: int = field(default=1,
        metadata={"help": "Max prompts per rollout-engine generate() call before actor-side sub-batching."})
    num_gpus_per_actor: Optional[int] = field(default=None,
        metadata={"help": "Dedicated rollout service GPUs per actor/engine. Required for separate and colocate."})
    tp_size: Optional[int] = field(default=None,
        metadata={"help": "Dedicated rollout service tensor parallel hint."})
    sp_size: Optional[int] = field(default=None,
        metadata={"help": "Dedicated rollout service sequence/spatial parallel hint."})
    transport_dtype: Optional[str] = field(default=None,
        metadata={"help": "Cast rollout transport payloads to this dtype (fp16/bf16) to reduce transfer size."})
    transport_drop_decoded_videos: bool = field(default=True,
        metadata={"help": "Drop decoded video tensors from rollout transport payloads after reward handling."})
    sglang_local_mode: Optional[bool] = field(default=None,
        metadata={"help": "Whether SGLang rollout uses in-actor local generator mode."})
    sglang_verify_weight_checksum: Optional[bool] = field(default=None,
        metadata={"help": "Whether SGLang verifies weight checksum after rollout-side updates."})
    sglang_disable_autocast: Optional[bool] = field(default=None,
        metadata={"help": "Disable torch.autocast inside SGLang rollout engine."})
    sglang_kwargs: Dict[str, Any] = field(default_factory=dict,
        metadata={"help": "Engine-scoped SGLang rollout kwargs."})

    # --- Buffer ---
    max_queue_size: int = field(default=0,
        metadata={"help": "Max rollout buffer queue size (0 = unbounded)"})
    drop_invalid: bool = field(default=True,
        metadata={"help": "Drop samples that fail validation checks"})
    reward_min: Optional[float] = field(default=None,
        metadata={"help": "Minimum reward threshold for sample filtering (None = no filter)"})
    reward_max: Optional[float] = field(default=None,
        metadata={"help": "Maximum reward threshold for sample filtering (None = no filter)"})
    min_samples: int = field(default=1,
        metadata={"help": "Minimum samples required before dispatching a batch"})
    group_size: Optional[int] = field(default=None,
        metadata={"help": "Reassemble outgoing training batches by explicit group_ids (None = passthrough)"})
    group_ttl_seconds: float = field(default=0.0,
        metadata={"help": "Time-to-live for incomplete groups in seconds (0 = no timeout)"})
    max_pending_samples: int = field(default=0,
        metadata={"help": "Max pending samples in buffer before blocking rollout (0 = unbounded)"})
    plugin_dotpaths: List[str] = field(default_factory=list,
        metadata={"help": "Rollout buffer filter plugin class dotpath(s)."})

    # --- Control ---
    num_rollout: int = field(default=1000,
        metadata={"help": "Total number of rollout iterations (outer-loop steps)"})
    start_rollout_id: int = field(default=0,
        metadata={"help": "Starting rollout step/ID for resuming training"})
    max_inflight_rollouts: int = field(default=1,
        metadata={"help": "Max concurrent in-flight rollouts for the async training runner"})

    # --- Artifacts ---
    output_dir: str = field(default="outputs",
        metadata={"help": "Output directory for checkpoints, logs, and generated samples"})
    save_steps: int = field(default=100,
        metadata={"help": "Save a checkpoint every N training steps (0 disables periodic saves)"})

    def set_start_rollout_id(self, rollout_id: int) -> None:
        """Update the rollout control cursor on the canonical config owner."""
        rollout_id = int(rollout_id)
        if rollout_id < 0:
            raise ValueError("start_rollout_id must be >= 0.")
        self.start_rollout_id = rollout_id

    def validate(self) -> None:
        if self.mode is not None and self.mode not in ROLLOUT_MODES:
            raise ValueError(
                f"rollout.mode must be one of {sorted(ROLLOUT_MODES)}, got: {self.mode!r}"
            )
        for attr_name in ("num_gpus_per_actor", "tp_size", "sp_size", "rollout_batch_size"):
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
        forbidden_sglang_precision_keys = {"prompt_encoder_dtype", "sglang_prompt_encoder_dtype"}
        precision_override_keys = sorted(
            key for key in self.sglang_kwargs.keys()
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
            raise ValueError(f"rollout.max_queue_size must be >= 0, got: {self.max_queue_size}")
        if self.min_samples < 1:
            raise ValueError(f"rollout.min_samples must be >= 1, got: {self.min_samples}")
        if self.reward_min is not None and self.reward_max is not None and self.reward_min > self.reward_max:
            raise ValueError(
                f"rollout.reward_min must be <= rollout.reward_max, "
                f"got min={self.reward_min}, max={self.reward_max}"
            )
        if self.group_size is not None and self.group_size < 1:
            raise ValueError(f"rollout.group_size must be >= 1 when provided, got: {self.group_size}")
        if float(self.group_ttl_seconds) < 0:
            raise ValueError(f"rollout.group_ttl_seconds must be >= 0, got: {self.group_ttl_seconds}")
        if int(self.max_pending_samples) < 0:
            raise ValueError(f"rollout.max_pending_samples must be >= 0, got: {self.max_pending_samples}")
        for i, path in enumerate(self.plugin_dotpaths):
            if not str(path).strip():
                raise ValueError(f"rollout.plugin_dotpaths[{i}] must be a non-empty string")
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

    eval_steps: int = field(default=100,
        metadata={"help": "Run evaluation every N training steps (0 disables periodic eval)"})
    eval_batch_size: int = field(default=4,
        metadata={"help": "Batch size for evaluation"})
    num_inference_steps: Optional[int] = field(default=None,
        metadata={"help": "Optional eval-only denoising step override. If unset, eval reuses sampling.num_inference_steps."})
    sampling_adapter: Optional[str] = field(default=None,
        metadata={"help": "Optional eval-only LoRA adapter override."})
    sde_type: Optional[str] = field(default=None,
        metadata={"help": "Optional eval-only sampler transition override. If unset, eval reuses sampling.sde_type."})
    eta: Optional[float] = field(default=None,
        metadata={"help": "Optional eval-only eta override. If unset, eval reuses sampling.eta."})

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

    logging_steps: int = field(default=10,
        metadata={"help": "Log metrics every N training steps (0 disables periodic step logging)"})
    logging_dir: Optional[str] = field(default=None,
        metadata={"help": "Directory for WandB logs/artifacts (defaults to output_dir/logs)"})
    report_to_wandb: bool = field(default=False,
        metadata={"help": "Enable WandB reporting (True/False)"})
    project_name: str = field(default="diffusionrl",
        metadata={"help": "Project name for WandB"})
    run_name: Optional[str] = field(default=None,
        metadata={"help": "Run name for WandB (auto-generated if None)"})
    log_media: bool = field(default=False,
        metadata={"help": "Log generated media previews to WandB (true/false)"})
    media_max_items: int = field(default=8,
        metadata={"help": "Maximum generated media items to log per rollout when log_media=true"})
    tags: Optional[str] = field(default=None,
        metadata={"help": "Comma-separated tags for WandB run (e.g. 'exp1,baseline')."})
    entity: Optional[str] = field(default=None,
        metadata={"help": "WandB entity (team or username)."})
    transport_log_payload_bytes: Optional[bool] = field(default=None,
        metadata={"help": "Whether rollout transport logs serialized payload sizes for debugging."})

    def validate(self) -> None:
        if self.logging_steps < 0:
            raise ValueError("logging_steps must be >= 0 (0 disables periodic step logging).")
        if self.media_max_items < 1:
            raise ValueError("media_max_items must be >= 1.")

@dataclass
class DebugConfig:
    """Debug mode and intermediate artifact controls."""

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
        mode = self.debug_mode
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

@dataclass
class TrainingArguments:
    """All configuration parameters for DiffusionRL training."""

    # ========== Dotpaths For Custom Modules (Dynamic Loading) ==========
    data_source_dotpath: str = field(default="diffusionrl.data.DefaultDataSource",
        metadata={"help": "Python dotpath to DataSource class for loading training/eval prompt streams"})
    rollout_function_dotpath: str = field(default="diffusionrl.rollout.default_rollout.generate_rollout",
        metadata={"help": "Python dotpath to the rollout function invoked by the rollout service. This is the main rollout extension seam."})
    eval_function_dotpath: str = field(default="diffusionrl.rollout.default_rollout.evaluate_rollout",
        metadata={"help": "Python dotpath to the evaluation function invoked by the rollout service."})
    reward_hook_dotpath: str = field(default="diffusionrl.rollout.default_rollout.score_rewards_hook",
        metadata={"help": "Python dotpath to the reward hook used by rollout/eval functions. This makes reward a first-class rollout hook."})

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
    data_path: Optional[str] = field(default="data/samples/prompts_toy.json",
        metadata={"help": "Path to training prompt data file (JSON, JSONL, or TXT). JSON items should provide text via 'prompt' or 'caption'."})
    eval_data_path: Optional[str] = field(default=None,
        metadata={"help": "Optional path to evaluation prompt data file. If unset, eval uses data_path with deterministic ordering."})

    # ========== Seed ==========
    seed: int = field(default=42,
        metadata={"help": "Random seed for reproducibility"})

    def to_dotted_dict(self) -> Dict[str, Any]:
        """Export config using namespace-preserving dotted keys."""
        dotted: Dict[str, Any] = {}
        _populate_dotted_config_dict(dotted, value=self)
        return dotted


_GROUP_CONFIG_TYPES = {
    info.name: info.type
    for info in fields(TrainingArguments)
    if isinstance(info.type, type) and is_dataclass(info.type)
}
_GROUP_CONFIG_NAMES = set(_GROUP_CONFIG_TYPES)
_TOP_LEVEL_FIELD_NAMES = {
    info.name for info in fields(TrainingArguments) if info.name not in _GROUP_CONFIG_NAMES
}
_GROUP_SUBCONFIG_NAMES: Dict[str, set[str]] = {
    _group_name: {
        info.name
        for info in fields(_group_type)
        if isinstance(info.type, type) and is_dataclass(info.type)
    }
    for _group_name, _group_type in _GROUP_CONFIG_TYPES.items()
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
            raise ValueError("Dotted config export requires a key prefix for leaf values.")
        target[prefix] = _copy_config_value(value)
        return

    for info in fields(type(value)):
        child = getattr(value, info.name)
        child_prefix = info.name if prefix is None else f"{prefix}.{info.name}"
        if is_dataclass(child):
            _populate_dotted_config_dict(target, value=child, prefix=child_prefix)
        else:
            target[child_prefix] = _copy_config_value(child)


def _record_resolution_error(
    resolved: Dict[str, Any],
    *,
    scope: str,
    exc: Exception,
) -> None:
    resolved[f"resolved.errors.{scope}"] = f"{type(exc).__name__}: {exc}"


def build_resolved_config_view(args: TrainingArguments) -> Dict[str, Any]:
    """Build a dotted canonical config view with resolved derived values."""
    resolved = args.to_dotted_dict()
    for key in (
        "resolved.training.actor_count",
        "resolved.training.world_size",
        "resolved.training.dp_size",
        "resolved.training.tp_size",
        "resolved.training.pp_size",
        "resolved.training.sp_size",
        "resolved.training.ep_size",
        "resolved.rollout.prompts_per_rollout",
        "resolved.training.global_batch_size",
        "resolved.training.local_batch_size",
        "resolved.training.local_mini_batch_size",
        "resolved.training.micro_batch_size",
        "resolved.training.num_updates_per_batch",
    ):
        resolved.setdefault(key, None)

    resolved["training.train_backend"] = normalize_train_backend_name(args)
    resolved["rollout.mode"] = args.rollout.mode
    resolved["rollout.rollout_engine"] = args.rollout.rollout_engine
    resolved["sync.protocol"] = str(args.sync.protocol).strip().lower()

    try:
        resolved["algorithm.algorithm_dotpath"] = resolve_algorithm_dotpath(
            algorithm_type=args.algorithm.algorithm_type,
            algorithm_dotpath=args.algorithm.algorithm_dotpath,
        )
    except Exception as exc:
        _record_resolution_error(resolved, scope="algorithm_dotpath", exc=exc)

    try:
        resolved_config = resolve_config(args, include_training_plan=True)
    except Exception as exc:
        _record_resolution_error(resolved, scope="config", exc=exc)
        return resolved

    model_spec = resolved_config.model_spec
    rollout_topology = resolved_config.rollout_mode_info.rollout_topology
    train_topology = resolved_config.training_topology
    train_plan = resolved_config.training_plan

    resolved["model.model_dotpath"] = model_spec.model_dotpath
    resolved["model.model_type"] = model_spec.model_type
    resolved["sampling.sampler_dotpath"] = model_spec.sampler_dotpath
    resolved["training.train_backend"] = resolved_config.train_backend_config.name
    resolved["rollout.mode"] = rollout_topology.mode
    resolved["rollout.rollout_engine"] = rollout_topology.rollout_engine
    resolved["sync.protocol"] = resolved_config.rollout_mode_info.sync_protocol
    resolved["resolved.training.actor_count"] = train_topology.actor_count
    resolved["resolved.training.world_size"] = train_topology.world_size
    resolved["resolved.training.dp_size"] = train_topology.dp_size
    resolved["resolved.training.tp_size"] = train_topology.tp_size
    resolved["resolved.training.pp_size"] = train_topology.pp_size
    resolved["resolved.training.sp_size"] = train_topology.sp_size
    resolved["resolved.training.ep_size"] = train_topology.ep_size
    resolved["resolved.rollout.prompts_per_rollout"] = int(args.algorithm.prompts_per_rollout)
    if train_plan is not None:
        resolved["resolved.training.global_batch_size"] = train_plan.global_batch_size
        resolved["resolved.training.local_batch_size"] = train_plan.local_batch_size
        resolved["resolved.training.local_mini_batch_size"] = train_plan.local_mini_batch_size
        resolved["resolved.training.micro_batch_size"] = train_plan.micro_batch_size
        resolved["resolved.training.num_updates_per_batch"] = (
            train_plan.num_updates_per_batch
        )
    return resolved


def print_config_views(*, args: TrainingArguments, print_resolved_config: bool) -> None:
    if not print_resolved_config:
        return

    use_color = _supports_color_output()
    print(_color("[Resolved Config] final derived values", "bold", enabled=use_color))
    resolved_dotted = build_resolved_config_view(args)
    for key in sorted(resolved_dotted.keys()):
        value_text = _format_config_value(resolved_dotted.get(key))
        print(f"  {_color(key, 'cyan', enabled=use_color)}: {value_text}")


def _merge_algorithm_kwarg_overrides(raw_args: Dict[str, Any]) -> None:
    """Overlay repeated algorithm-specific ``--algorithm.kwarg key=value`` items."""
    overrides = raw_args.pop("_algorithm_kwarg_overrides", None) or []
    if not overrides:
        return

    dest = "algorithm.algorithm_kwargs"
    merged = dict(raw_args.get(dest) or {})
    for item in overrides:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "Internal error: parsed --algorithm.kwarg item must be a (key, value) pair. "
                f"Got: {item!r}"
            )
        key, value = item
        merged[str(key)] = value
    raw_args[dest] = merged


def parse_args(argv: Optional[List[str]] = None) -> TrainingArguments:
    """Parse command line arguments and return ``TrainingArguments``."""
    parser = argparse.ArgumentParser(
        description="diffusionrl training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override YAML values.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print resolved config after validation.",
    )
    parser.add_argument(
        "--allow-unknown-config-keys",
        action="store_true",
        help="Allow unknown keys in --config YAML (default is fail-fast).",
    )

    cli_field_specs = collect_cli_field_specs(
        training_args_type=TrainingArguments,
        group_config_names=_GROUP_CONFIG_NAMES,
        group_config_types=_GROUP_CONFIG_TYPES,
    )

    hidden_cli_destinations = {
        "algorithm.algorithm_kwargs",
    }

    arg_groups: Dict[str, argparse._ArgumentGroup] = {}
    for field_name, field_type, default, help_text, group_key in cli_field_specs:
        dest = f"{group_key}.{field_name}" if group_key else field_name
        if dest in hidden_cli_destinations:
            continue
        if group_key not in arg_groups:
            display_name = GROUP_DISPLAY_NAMES.get(group_key, group_key)
            arg_groups[group_key] = parser.add_argument_group(display_name)
        group = arg_groups[group_key]

        option_strings = build_cli_option_strings(field_name, group_key)
        add_kwargs = build_add_argument_kwargs(
            field_type,
            default,
            help_text,
            field_name=field_name,
        )
        add_kwargs["dest"] = dest
        group.add_argument(*option_strings, **add_kwargs)

    algorithm_group = arg_groups.get("algorithm")
    if algorithm_group is None:
        algorithm_group = parser.add_argument_group(
            GROUP_DISPLAY_NAMES.get("algorithm", "Algorithm & Advantage")
        )
        arg_groups["algorithm"] = algorithm_group
    algorithm_group.add_argument(
        "--algorithm.kwarg",
        dest="_algorithm_kwarg_overrides",
        action="append",
        default=[],
        type=parse_cli_key_value,
        metavar="KEY=VALUE",
        help=(
            "Append one algorithm-specific algorithm.algorithm_kwargs override. "
            "Shared framework-owned keys must use dedicated --algorithm.* flags. "
            "Repeat this flag to set multiple extension keys."
        ),
    )

    cli_argv = list(argv) if argv is not None else sys.argv[1:]
    parsed_args = parser.parse_args(cli_argv)
    explicit_cli_keys = collect_explicit_cli_destinations(cli_argv, parser)
    action_by_dest = {
        action.dest: action for action in parser._actions if getattr(action, "dest", None)
    }

    raw_args = vars(parsed_args)
    print_resolved_config = bool(raw_args.get("print_resolved_config", False))
    allow_unknown_config_keys = bool(raw_args.get("allow_unknown_config_keys", False))

    if raw_args.get("config"):
        defaults: Dict[str, Any] = {}
        for action in parser._actions:
            dest = getattr(action, "dest", None)
            if not dest or dest == "help" or dest in defaults:
                continue
            defaults[dest] = action.default
        defaults.setdefault("algorithm.algorithm_kwargs", {})
        yaml_data = load_yaml_mapping(raw_args["config"])
        merge_yaml_overrides(
            raw_args,
            yaml_data=yaml_data,
            defaults=defaults,
            explicit_cli_keys=explicit_cli_keys,
            action_by_dest=action_by_dest,
            allow_unknown_config_keys=allow_unknown_config_keys,
            top_level_field_names=_TOP_LEVEL_FIELD_NAMES,
            group_config_names=_GROUP_CONFIG_NAMES,
            group_subconfig_names=_GROUP_SUBCONFIG_NAMES,
        )

    raw_args.pop("config", None)
    raw_args.pop("print_resolved_config", None)
    raw_args.pop("allow_unknown_config_keys", None)
    _merge_algorithm_kwarg_overrides(raw_args)

    grouped_kwargs: Dict[str, Dict[str, Any]] = {name: {} for name in _GROUP_CONFIG_TYPES}
    sub_kwargs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    top_level_kwargs: Dict[str, Any] = {}
    for key, value in raw_args.items():
        if "." in key:
            parts = key.split(".")
            group_name = parts[0]
            if len(parts) == 2 and group_name in _GROUP_CONFIG_TYPES:
                grouped_kwargs[group_name][parts[1]] = value
            elif (
                len(parts) == 3
                and group_name in _GROUP_CONFIG_TYPES
                and parts[1] in _GROUP_SUBCONFIG_NAMES.get(group_name, set())
            ):
                sub_kwargs.setdefault(group_name, {}).setdefault(parts[1], {})[parts[2]] = value
            else:
                top_level_kwargs[key] = value
        else:
            top_level_kwargs[key] = value

    for group_name, group_type in _GROUP_CONFIG_TYPES.items():
        kwargs = dict(grouped_kwargs[group_name])
        for info in fields(group_type):
            ft = info.type
            if isinstance(ft, type) and is_dataclass(ft):
                sub_data = sub_kwargs.get(group_name, {}).get(info.name, {})
                kwargs[info.name] = ft(**sub_data)
        top_level_kwargs[group_name] = group_type(**kwargs)

    args = TrainingArguments(**top_level_kwargs)
    args = validate_args(args)
    print_config_views(args=args, print_resolved_config=print_resolved_config)
    return args


def validate_args(
    args: TrainingArguments,
    *,
    resolved: Optional[ConfigBundle] = None,
) -> TrainingArguments:
    """Validate arguments. Grouped validation may normalize fields such as ``rollout.plugin_dotpaths``."""
    validate_grouped_configs(args)
    debug_mode = args.debug.debug_mode

    resolved_config = resolved if resolved is not None else resolve_config(args)
    resolved_model = resolved_config.model_spec
    model_cls = resolved_model.model_cls
    backend_config = resolved_config.train_backend_config
    backend_name = backend_config.name
    rollout_mode_info = resolved_config.rollout_mode_info
    validate_algorithm_kwargs_payload(args)
    validate_algorithm_dotpath(args)
    validate_train_backend_config(
        backend_name=backend_name,
        backend_kwargs=backend_config.kwargs,
        backend_dotpath=backend_config.backend_dotpath,
    )
    backend_capabilities = resolved_config.train_backend_capabilities.as_dict()
    validate_rollout_mode(
        args,
        rollout_mode_info=rollout_mode_info,
        backend_capabilities=backend_capabilities,
        backend_name=backend_name,
    )
    rollout_topology = rollout_mode_info.rollout_topology
    training_actor_sampling_mode = rollout_mode_info.training_actor_sampling_mode
    if debug_mode == "train_only":
        validate_dynamic_dotpaths(
            args,
            resolved_model=resolved_model,
            include_data_source=False,
            include_rollout_buffer_plugins=False,
        )
    else:
        validate_dynamic_dotpaths(args, resolved_model=resolved_model)

    if debug_mode != "train_only":
        validate_reward_and_rollout_buffer_config(args)
        validate_rollout_layout(
            args,
            rollout_mode_info=rollout_mode_info,
        )
    validate_training_batch_geometry(
        args,
        topology=resolved_config.training_topology,
    )
    validate_training_misc(args)
    apply_model_config_hook(args, model_cls=model_cls)
    validate_model_sampling_contract(args)
    validate_nft_sampling_contract(args)
    if debug_mode != "train_only":
        validation_algorithm = instantiate_algorithm_from_config(
            build_algorithm_config(
                args,
                sampling_spec=resolved_config.sampling_spec,
            )
        )
        sampling_requirements = collect_sampling_requirements(algorithm=validation_algorithm)
        validate_resolved_engine_algorithm_contract(
            args,
            rollout_mode_info=rollout_mode_info,
            sampling_requirements=sampling_requirements,
        )

    if debug_mode != "train_only":
        validate_rollout_mode_constraints(
            args,
            training_actor_sampling_mode=training_actor_sampling_mode,
            model_cls=model_cls,
            rollout_topology=rollout_topology,
        )

    return args


def get_default_args() -> TrainingArguments:
    """Get default arguments without parsing command line."""
    return TrainingArguments()
