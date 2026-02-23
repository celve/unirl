"""
diffusionrl Arguments - Configuration parameters for training.

Reference: slime/utils/arguments.py
"""
import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from diffusionrl.plugins import validate_plugin_target_path
from diffusionrl.utils.misc import load_function
from diffusionrl.config.validation import (
    get_inference_gpus_per_actor,
    is_probably_local_weight_sync_dir,
    normalize_repo_relative_paths,
    repo_root,
    resolve_repo_relative_path,
    resolve_loss_type_requirements,
    validate_colocate_fractions,
    validate_engine_loss_compatibility,
    validate_reward_config,
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

# Model type to model path mapping for automatic configuration
MODEL_TYPE_TO_PATH = {
    "flux": "diffusionrl.models.flux.FluxModelBundle",
    "sd3": "diffusionrl.models.sd3.SD3ModelBundle",
    "hunyuan": "diffusionrl.models.hunyuan.HunyuanModelBundle",
    "mochi": "diffusionrl.models.mochi.MochiModelBundle",
}

# Model type to sampler engine type mapping (single source of truth)
# This mapping determines which engine backend to use for sampling
MODEL_TYPE_TO_SAMPLER_ENGINE = {
    "flux": "fsdp",
    "sd3": "fsdp",
    "hunyuan": "fsdp",
    "mochi": "fastvideo",
}

# Default samplers for each model type
# FSDP Engine: Native PyTorch (DanceGRPO-aligned)
# FastVideo Engine: FastVideo framework
MODEL_TYPE_TO_SAMPLER = {
    # Image models - use FSDP engine
    "flux": "diffusionrl.samplers.fsdp.flux_sampler.FluxSampler",
    "sd3": "diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler",
    # Hunyuan default path uses FSDP sampler to satisfy GRPO data requirements by default.
    "hunyuan": "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler",
    # Mochi still defaults to FastVideo until a requirement-complete FSDP sampler is available.
    "mochi": "diffusionrl.samplers.fastvideo.fastvideo_sampler.FastVideoSampler",
}

# FSDP samplers for video models (DanceGRPO-aligned, use via --sampler-path)
FSDP_SAMPLERS = {
    "hunyuan": "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler",
    # "mochi": "diffusionrl.samplers.fsdp.mochi_sampler.FSDPMochiSampler",  # Future
}


ENGINE_CAPABILITY_REQUIREMENTS: Dict[str, Dict[str, bool]] = {
    "fsdp": {
        "requires_trajectory": True,
        "requires_log_prob": True,
        "requires_embeddings": True,
    },
    "fastvideo": {
        "requires_trajectory": True,
        "requires_log_prob": False,
        "requires_embeddings": False,
    },
    "sglang": {
        "requires_trajectory": True,
        "requires_log_prob": False,
        "requires_embeddings": False,
    },
}

LOSS_TYPE_REQUIREMENTS: Dict[str, Dict[str, bool]] = {
    "grpo": {
        "requires_trajectory": True,
        "requires_log_prob": True,
        "requires_embeddings": True,
    },
    "nft": {
        "requires_trajectory": False,
        "requires_log_prob": False,
        "requires_embeddings": True,
    },
}


def _resolve_model_runtime_defaults(model_path: str) -> Dict[str, Optional[str]]:
    """
    Best-effort model self-description lookup.

    Returns model-declared defaults when available; falls back to None values.
    """
    defaults: Dict[str, Optional[str]] = {
        "sampler_path": None,
        "sampler_engine_type": None,
    }

    try:
        model_cls = load_function(model_path)
    except Exception:
        return defaults

    sampler_path_fn = getattr(model_cls, "default_sampler_path", None)
    if callable(sampler_path_fn):
        try:
            defaults["sampler_path"] = sampler_path_fn()
        except Exception:
            defaults["sampler_path"] = None

    engine_type_fn = getattr(model_cls, "default_sampler_engine", None)
    if callable(engine_type_fn):
        try:
            defaults["sampler_engine_type"] = engine_type_fn()
        except Exception:
            defaults["sampler_engine_type"] = None

    return defaults

@dataclass
class TrainingArguments:
    """All configuration parameters for GRPO training."""

    # ========== Paths (Dynamic Loading) ==========
    # Algorithm path (e.g., "diffusionrl.algorithms.grpo.GRPOAlgorithm")
    algorithm_path: str = "diffusionrl.algorithms.grpo.GRPOAlgorithm"
    # Sampler path (e.g., "diffusionrl.samplers.fsdp.flux_sampler.FluxSampler")
    sampler_path: str = "diffusionrl.samplers.fastvideo.fastvideo_sampler.FastVideoSampler"
    # Reward path (e.g., "diffusionrl.rewards.hpsv2.HPSv2Reward")
    reward_path: str = "diffusionrl.reward.local.LocalRewardWorker"
    # Model path (e.g., "diffusionrl.models.hunyuan.HunyuanModelBundle")
    model_path: str = "diffusionrl.models.hunyuan.HunyuanModelBundle"
    # Data source path
    data_source_path: str = "diffusionrl.data.DefaultDataSource"
    # Optional custom rollout pipeline function (slime-style pluggable rollout)
    rollout_pipeline_path: Optional[str] = None

    # ========== Model Configuration ==========
    pretrained_model_saved_path: str = ""
    model_type: str = "hunyuan"  # hunyuan, mochi, flux, sd3
    vae_saved_path: Optional[str] = None
    text_encoder_path: Optional[str] = None

    # ========== Reward Configuration ==========
    reward_model_saved_path: Optional[str] = None
    reward_model_name: str = "hpsv2"  # hpsv2, pickscore, clip, aesthetic
    reward_batch_size: int = 8
    reward_timeout: float = 60.0
    use_http_reward: bool = False
    reward_service_url: Optional[str] = None

    # Multi-reward configuration
    reward_models: Optional[List[str]] = None  # e.g., ["pickscore", "hpsv2"]
    reward_weights: Optional[List[float]] = None  # e.g., [0.3, 0.7]
    reward_aggregation: str = "weighted_sum"  # weighted_sum, mean, min, max, concat
    reward_mix_mode: str = "reward_aggr"  # reward_aggr (default), advantage_aggr

    # Dedicated reward GPU pool configuration
    reward_dedicated_gpus_per_actor: int = 1  # GPUs per dedicated reward actor (for large models)

    # Remote reward service configuration
    reward_service_urls: Optional[List[str]] = None  # Multiple services
    reward_remote_concurrency: int = 8  # Max concurrent HTTP requests

    reward_dedicated_num_gpus: int = 0  # Total GPUs for dedicated reward actors
    reward_dedicated_num_nodes: int = 0  # Dedicated reward nodes count
    reward_dedicated_num_gpus_per_node: int = 0  # GPUs per dedicated reward node
    reward_placement_strategy: str = "PACK"  # Reward PG placement strategy (PACK or SPREAD)

    # ========== Algorithm Configuration ==========
    clip_range: float = 1e-4
    clip_range_mode: str = "constant"  # constant, linear_decay, cosine_decay
    kl_coef: float = 0.01
    use_kl_penalty: bool = True
    advantage_type: str = "group"  # global, group, per_prompt
    num_samples_per_prompt: int = 4
    prompts_per_batch: int = 1  # unique prompts per rollout step
    advantage_epsilon: float = 1e-8
    advantage_clip_max: Optional[float] = None
    # Per-prompt tracker configuration (for advantage_type="per_prompt")
    # flow_grpo uses full history; set to large value (e.g., 100000) to approximate
    use_per_prompt_stat_tracker: bool = False  # Enable cross-batch per-prompt statistics tracking
    per_prompt_mode: str = "running"  # running (tracker) or batch (per-batch stats)
    per_prompt_buffer_size: int = 16  # Buffer size for per-prompt statistics
    per_prompt_min_count: int = 2  # Min samples before using per-prompt stats
    use_global_std: bool = False  # Global running mean/std for rewards
    cross_rank_shuffle: bool = False  # (reserved) shuffle prompts across ranks in sampler

    # Running statistics configuration (for cross-batch global normalization, DanceGRPO)
    use_running_stats: bool = False  # Enable cross-batch running mean/std for global normalization
    running_stats_warmup: int = 0  # Warmup batches before using running stats

    # ========== Sampling Configuration ==========
    num_inference_steps: int = 50
    eta: float = 1.0  # SDE noise coefficient
    sde_type: str = "sde"  # sde, cps, dance, flux_dance, flux_flow, dpm2
    shift: float = 3.0  # Time shift for sigma schedule
    guidance_scale: float = 7.5
    mixed_sampling: bool = False
    sde_ratio: float = 1.0  # Ratio of SDE steps for MixGRPO
    timestep_fraction: float = 1.0  # Fraction of timesteps to train (DanceGRPO: 0.6)
    sampling_adapter: Optional[str] = None  # LoRA adapter name for sampling (e.g., "old" for NFT)
    sampling_backend: str = "inference"  # inference (default) or training
    # Experimental FastVideo path: replay old log_probs on training actors when
    # inference engine does not return them.
    replay_log_probs: bool = False
    fastvideo_replay_log_probs: bool = False
    # Optional explicit sampler path used only by replay_log_probs path.
    # Useful when rollout sampler/engine does not implement compute_log_prob_for_training.
    replay_sampler_path: Optional[str] = None

    # Shared noise configuration (DanceGRPO, MixGRPO)
    # When enabled, K samples for the same prompt share the same initial noise
    init_same_noise: bool = False

    # ========== Loss Configuration ==========
    loss_type: str = "grpo"  # grpo, nft
    loss_path: Optional[str] = None  # Custom loss path for dynamic loading

    # MixGRPO-specific loss configuration
    ignore_last: bool = False  # Skip last timestep (t->0) in loss - unstable log_prob
    frozen_init_timesteps: int = 0  # Skip first N timesteps in loss computation

    # ========== NFT-specific Configuration ==========
    nft_beta: float = 0.1  # Interpolation weight for positive/negative predictions
    nft_adv_clip_max: float = 5.0  # Maximum advantage clipping
    nft_adv_mode: str = "raw"  # raw, sign, binary, one_only, all, per_timestep
    nft_use_adaptive_weight: bool = True  # Adaptive loss weighting
    nft_timestep_mode: str = "random"  # random, all
    nft_shuffle_timesteps: bool = True  # Shuffle timestep order in "all" mode
    nft_apply_shift: bool = False  # Apply shift to provided timesteps in "all" mode

    # ========== EMA Configuration (for NFT) ==========
    use_ema: bool = False  # Enable EMA for dual adapter mechanism
    ema_decay: float = 0.001  # EMA decay rate (for constant mode)
    ema_decay_type: str = "constant"  # constant, linear, warmup (matches DiffusionNFT)
    ema_flat_steps: int = 0  # For warmup mode: steps before decay starts
    ema_uprate: float = 0.001  # For dynamic modes: decay increase per step
    ema_uphold: float = 0.5  # Maximum decay value for dynamic modes

    # ========== Timestep Scheduler Configuration (for MixGRPO) ==========
    timestep_strategy: str = "all"  # all, window
    window_strategy: str = "progressive"  # progressive, random, decay, exp_decay
    window_group_size: int = 4  # Number of timesteps per window
    window_iters_per_group: int = 25  # Iterations before sliding window
    window_overlap: bool = False  # Enable overlapping windows
    window_overlap_step: int = 1  # Overlap stride
    window_roll_back: bool = False  # Roll back to start when reaching end

    # Window training configuration (MixGRPO)
    # When enabled, only train on SDE window timesteps (not all timesteps)
    window_training: bool = False

    # ========== Training Configuration ==========
    batch_size: int = 4
    gradient_accumulation_steps: str = "1"  # int or "auto"
    gradient_steps_per_epoch: int = 1  # used when grad_accum=auto
    num_inner_epochs: int = 1  # Number of repeated update passes per rollout batch
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    lr_scheduler_type: str = "constant"  # constant, linear, cosine

    # ========== LoRA Configuration ==========
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: Optional[str] = None  # Comma-separated, e.g., "to_q,to_k,to_v"

    # ========== FSDP Configuration ==========
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: str = "BACKWARD_PRE"

    # ========== Memory Optimization ==========
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing to reduce memory

    # ========== Ray Resource Configuration ==========
    # Ray cluster connection
    ray_address: Optional[str] = None  # Ray cluster address, e.g., "auto" or "ip:port"
    # Inference resources
    inference_num_nodes: int = 1
    inference_num_gpus_per_node: int = 4
    # Training resources
    training_num_nodes: int = 1
    training_num_gpus_per_node: int = 4
    # Reward resources
    # NOTE: reward stays outside InferenceActor boundary (via RewardService).
    # Deployment strategy
    colocate_inference_training: bool = False
    # Deprecated/unsupported: reward must not be colocated inside InferenceActor.
    colocate_reward: bool = False
    placement_strategy: str = "PACK"  # PACK or SPREAD
    # Expert switch: allow colocate mode for sglang when using branches that
    # already expose release/resume-memory handlers.
    allow_sglang_colocate_experimental: bool = False

    # Colocate GPU fractions (used when colocate_inference_training=True)
    colocate_training_gpu_fraction: float = 0.4
    colocate_inference_gpu_fraction: float = 0.4
    # Enable NOSET/fractional multi-GPU inference layout explicitly.
    # Default path keeps integer single-GPU actors only.
    allow_noset_multi_gpu_inference: bool = False

    # Rollout data partitioning (reduce object store pressure)
    partition_train_data: bool = True
    # Rollout buffer actor (data-centric handoff with validation/filter plugins)
    rollout_buffer_enabled: bool = True
    rollout_buffer_max_queue_size: int = 0  # 0 means unbounded
    rollout_buffer_drop_invalid: bool = True
    rollout_buffer_reward_min: Optional[float] = None
    rollout_buffer_reward_max: Optional[float] = None
    rollout_buffer_min_samples: int = 1
    rollout_buffer_grouped: bool = False
    # Defaults to num_samples_per_prompt when None.
    rollout_buffer_group_size: Optional[int] = None
    # Number of prompt-groups merged into one dispatch batch (0 => prompts_per_batch).
    rollout_buffer_dispatch_groups: int = 0
    # Allow fallback dispatch of incomplete prompt groups to avoid pipeline stalls.
    rollout_buffer_allow_partial_group: bool = True
    rollout_buffer_group_ttl_seconds: float = 0.0
    rollout_buffer_max_pending_samples: int = 0
    # Comma-separated plugin dotpaths loaded in RolloutBufferActor.
    rollout_buffer_plugin_paths: str = ""

    # ========== Engine Configuration ==========
    # Sampler engine type: fsdp (native PyTorch), fastvideo, sglang (future)
    sampler_engine_type: Optional[str] = None  # Auto-select based on model_type if None
    # Engine-specific kwargs (passed to engine initialization)
    engine_kwargs: dict = field(default_factory=dict)

    # ========== FSDP Engine Configuration ==========
    # GPUs per FSDP inference actor (1 = single GPU, >1 = multi-GPU with FSDP)
    fsdp_num_gpus: int = 1
    # Sharding strategy for FSDP inference (NO_SHARD recommended for inference)
    fsdp_inference_sharding_strategy: str = "NO_SHARD"  # NO_SHARD, FULL_SHARD, SHARD_GRAD_OP

    # ========== FastVideo Engine Configuration ==========
    # Sequence parallelism size (for FastVideo)
    # Each inference actor will use sp_size GPUs
    sp_size: int = 1
    # Tensor parallelism size (for FastVideo/SGLang)
    tp_size: int = 1
    # Total GPUs per FastVideo instance (usually = sp_size)
    fastvideo_num_gpus: Optional[int] = None  # Defaults to sp_size if None

    # ========== Offload Configuration ==========
    offload: bool = False  # Shortcut for both offload_train and offload_rollout
    offload_train: Optional[bool] = None  # Offload training actors during rollout
    offload_rollout: Optional[bool] = None  # Offload rollout actors during training

    # ========== Rollout Configuration ==========
    num_rollout: int = 1000
    start_rollout_id: int = 0
    rollouts_per_epoch: Optional[int] = None
    async_pipeline: bool = False  # Enable rollout/train overlap (separate mode only)
    async_max_inflight: int = 1  # Max in-flight rollout futures in async pipeline
    update_weights_interval: int = 1  # Sync inference weights every N rollouts

    # ========== Checkpointing ==========
    output_dir: str = "outputs"
    save_steps: int = 100
    save_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    # ========== Evaluation ==========
    eval_steps: int = 100
    eval_batch_size: int = 4
    num_eval_samples: int = 16

    # ========== Logging ==========
    logging_steps: int = 10
    logging_dir: Optional[str] = None
    report_to: str = "tensorboard"  # tensorboard, wandb, none
    project_name: str = "diffusionrl"
    run_name: Optional[str] = None

    # ========== Data Configuration ==========
    data_path: Optional[str] = "data/samples/prompts_toy.json"
    prompt_column: str = "prompt"
    max_prompt_length: int = 256

    # ========== Video/Image Configuration ==========
    height: int = 256
    width: int = 256
    num_frames: int = 16  # For video models
    fps: int = 8

    # ========== Seed ==========
    seed: int = 42

    # ========== Misc ==========
    debug: bool = False
    profile: bool = False
    weight_sync_mode: str = "auto"  # auto, object_ref, checkpoint_path
    weight_sync_dir: str = "outputs/weight_sync"


def parse_args(argv: Optional[List[str]] = None) -> TrainingArguments:
    """Parse command line arguments and return TrainingArguments."""
    parser = argparse.ArgumentParser(description="diffusionrl training")

    # Add all arguments from TrainingArguments dataclass
    args_class = TrainingArguments

    for field_name, field_info in args_class.__dataclass_fields__.items():
        field_type = field_info.type
        default = field_info.default if field_info.default is not field_info.default_factory else None

        # Handle Optional types
        if hasattr(field_type, "__origin__"):
            if field_type.__origin__ is type(None) or str(field_type).startswith("typing.Optional"):
                # Extract inner type from Optional
                inner_types = [t for t in field_type.__args__ if t is not type(None)]
                field_type = inner_types[0] if inner_types else str

        # Convert field name to argument name
        arg_name = f"--{field_name.replace('_', '-')}"

        if field_type == bool:
            parser.add_argument(
                arg_name,
                type=lambda x: x.lower() in ('true', '1', 'yes'),
                default=default,
                help=f"{field_name} (default: {default})"
            )
        elif field_type == int:
            parser.add_argument(arg_name, type=int, default=default, help=f"{field_name} (default: {default})")
        elif field_type == float:
            parser.add_argument(arg_name, type=float, default=default, help=f"{field_name} (default: {default})")
        else:
            parser.add_argument(arg_name, type=str, default=default, help=f"{field_name} (default: {default})")

    parsed_args = parser.parse_args(argv)

    # Convert to TrainingArguments
    args_dict = vars(parsed_args)
    # Convert hyphens back to underscores
    args_dict = {k.replace('-', '_'): v for k, v in args_dict.items()}

    args = TrainingArguments(**args_dict)

    # Validate and normalize arguments
    args = validate_args(args)

    return args


def validate_args(args: TrainingArguments) -> TrainingArguments:
    """
    Validate and normalize arguments for colocate/offload logic.

    Args:
        args: TrainingArguments instance to validate

    Returns:
        Validated and normalized TrainingArguments
    """
    explicit_offload_shortcut = bool(getattr(args, "offload", False))
    explicit_offload_train = getattr(args, "offload_train", None) is not None
    explicit_offload_rollout = getattr(args, "offload_rollout", None) is not None

    # Path normalization first so later checks use canonical values.
    normalize_repo_relative_paths(
        args,
        env_repo_root=ENV_REPO_ROOT,
        env_data_root=ENV_DATA_ROOT,
        env_model_root=ENV_MODEL_ROOT,
        local_to_hf_fallback=_LOCAL_TO_HF_FALLBACK,
    )

    # ========== Model type to path mapping ==========
    # Keep default path resolution explicit and local to built-in model types.

    # Auto-map model_type to model_path if user hasn't explicitly specified a custom path.
    default_model_path = "diffusionrl.models.hunyuan.HunyuanModelBundle"
    if args.model_path == default_model_path:
        if args.model_type in MODEL_TYPE_TO_PATH:
            args.model_path = MODEL_TYPE_TO_PATH[args.model_type]
            logger.info(f"Auto-mapped model_type={args.model_type} to model_path={args.model_path}")

    # Resolve model self-declared defaults (sampler/engine) when available.
    model_defaults = _resolve_model_runtime_defaults(args.model_path)

    # Auto-map model_type to sampler_path if user hasn't explicitly specified a custom sampler
    default_sampler_path = "diffusionrl.samplers.fastvideo.fastvideo_sampler.FastVideoSampler"
    if args.sampler_path == default_sampler_path:
        model_sampler_path = model_defaults.get("sampler_path")
        if model_sampler_path:
            args.sampler_path = model_sampler_path
            logger.info(
                "Auto-mapped model_path=%s to sampler_path=%s",
                args.model_path,
                args.sampler_path,
            )
        elif args.model_type in MODEL_TYPE_TO_SAMPLER:
            args.sampler_path = MODEL_TYPE_TO_SAMPLER[args.model_type]
            logger.info(f"Auto-mapped model_type={args.model_type} to sampler_path={args.sampler_path}")

    # Auto-select sampler_engine_type based on model_type if not specified
    if args.sampler_engine_type is None:
        model_engine_type = model_defaults.get("sampler_engine_type")
        if model_engine_type:
            args.sampler_engine_type = model_engine_type
            logger.info(
                "Auto-selected sampler_engine_type=%s from model_path=%s",
                args.sampler_engine_type,
                args.model_path,
            )
        elif args.model_type in MODEL_TYPE_TO_SAMPLER_ENGINE:
            args.sampler_engine_type = MODEL_TYPE_TO_SAMPLER_ENGINE[args.model_type]
            logger.info(f"Auto-selected sampler_engine_type={args.sampler_engine_type} for model_type={args.model_type}")

    # Normalize engine_kwargs early (argparse default for default_factory may be MISSING)
    if not isinstance(args.engine_kwargs, dict):
        logger.warning("engine_kwargs is not a dict. Resetting to empty dict.")
        args.engine_kwargs = {}

    # Explicit plugin path + spec-kind validation.
    validate_plugin_target_path(target_path=args.model_path, kind="model")
    validate_plugin_target_path(target_path=args.sampler_path, kind="sampler")
    if getattr(args, "replay_sampler_path", None):
        validate_plugin_target_path(target_path=args.replay_sampler_path, kind="sampler")
    validate_plugin_target_path(target_path=args.algorithm_path, kind="algorithm")
    validate_plugin_target_path(target_path=args.reward_path, kind="reward")
    if getattr(args, "rollout_pipeline_path", None):
        validate_plugin_target_path(
            target_path=args.rollout_pipeline_path,
            kind="rollout",
        )
    rollout_buffer_plugin_paths = getattr(args, "rollout_buffer_plugin_paths", "") or ""
    if isinstance(rollout_buffer_plugin_paths, str):
        for plugin_path in [part.strip() for part in rollout_buffer_plugin_paths.split(",") if part.strip()]:
            validate_plugin_target_path(
                target_path=plugin_path,
                kind="rollout",
            )

    # ========== Sampling backend validation ==========
    sampling_backend = getattr(args, "sampling_backend", "inference")
    if sampling_backend not in ("inference", "training"):
        raise ValueError(
            f"sampling_backend must be 'inference' or 'training', got: {sampling_backend}"
        )

    if sampling_backend == "training":
        logger.warning(
            "sampling_backend='training' is an experimental path. "
            "Default production path remains sampling_backend='inference'."
        )
        if args.sampler_engine_type != "fsdp":
            raise ValueError(
                "sampling_backend='training' requires sampler_engine_type='fsdp'. "
                f"Got sampler_engine_type={args.sampler_engine_type}."
            )

        if args.inference_num_nodes != 0 or args.inference_num_gpus_per_node != 0:
            logger.warning(
                "sampling_backend='training': disabling inference placement groups "
                f"(inference_num_nodes={args.inference_num_nodes}, "
                f"inference_num_gpus_per_node={args.inference_num_gpus_per_node})."
            )
            args.inference_num_nodes = 0
            args.inference_num_gpus_per_node = 0

        if args.colocate_inference_training:
            logger.warning(
                "sampling_backend='training': disabling colocate_inference_training."
            )
            args.colocate_inference_training = False

        if args.offload or args.offload_train or args.offload_rollout:
            logger.warning(
                "sampling_backend='training': disabling offload_train/offload_rollout."
            )
        args.offload = False
        args.offload_train = False
        args.offload_rollout = False

    replay_enabled = bool(
        getattr(args, "replay_log_probs", False)
        or getattr(args, "fastvideo_replay_log_probs", False)
    )
    replay_guard = (
        sampling_backend == "inference"
        and getattr(args, "loss_type", "grpo") == "grpo"
    )
    if (
        not replay_enabled
        and replay_guard
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    ):
        replay_enabled = True
        args.replay_log_probs = True
        args.fastvideo_replay_log_probs = True
        logger.info("Auto-enabled replay_log_probs for sampler_engine_type='sglang' + loss_type='grpo'.")
    if replay_enabled and not replay_guard:
        logger.warning(
            "replay_log_probs=true is only valid for "
            "sampling_backend='inference' + loss_type='grpo'. "
            "Disabling replay_log_probs/fastvideo_replay_log_probs."
        )
        args.replay_log_probs = False
        args.fastvideo_replay_log_probs = False
    else:
        # Keep legacy flag as backward-compatible alias.
        args.replay_log_probs = replay_enabled
        if replay_enabled:
            args.fastvideo_replay_log_probs = True

    # sglang-diffusion colocate mode still needs upstream release/resume handlers.
    if (
        sampling_backend == "inference"
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
        and bool(getattr(args, "colocate_inference_training", False))
    ):
        if not bool(getattr(args, "allow_sglang_colocate_experimental", False)):
            raise ValueError(
                "sampler_engine_type='sglang' currently defaults to separate mode only "
                "(colocate_inference_training=False). "
                "To force colocate testing on a branch that implements release/resume-memory "
                "handlers, set --allow-sglang-colocate-experimental=true."
            )
        logger.warning(
            "allow_sglang_colocate_experimental=true: enabling colocate_inference_training "
            "for sglang in experimental mode."
        )

    # ========== Handle offload shortcut ==========
    if args.offload:
        args.offload_train = True
        args.offload_rollout = True

    # Colocate mode requires offload for GPU memory management
    if args.colocate_inference_training:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True

        validate_colocate_fractions(args)

    # Set defaults for non-colocate mode
    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    # FastVideo colocate defaults to non-offload unless user explicitly opts in.
    # This keeps the default control flow simple and avoids slow offload loops.
    if (
        sampling_backend != "training"
        and args.colocate_inference_training
        and args.sampler_engine_type == "fastvideo"
        and not explicit_offload_shortcut
    ):
        changed = False
        if not explicit_offload_train and args.offload_train:
            args.offload_train = False
            changed = True
        if not explicit_offload_rollout and args.offload_rollout:
            args.offload_rollout = False
            changed = True
        if changed:
            logger.warning(
                "colocate_inference_training + fastvideo defaults to offload_train=false/offload_rollout=false "
                "for throughput. If memory is insufficient, explicitly enable offload via "
                "--offload=true or --offload-train/--offload-rollout."
            )

    # ========== Validate model-specific defaults ==========
    # NOTE: We no longer auto-override shift for FLUX models.
    # DanceGRPO and MixGRPO explicitly use shift=3.0 for FLUX, while the original
    # FLUX paper uses shift=1.0. Users should explicitly set --shift based on
    # their experiment requirements.
    # - FLUX default (original paper): shift=1.0
    # - DanceGRPO/MixGRPO with FLUX: shift=3.0

    # ========== Multi-node FSDP strategy recommendation ==========
    # Recommend HYBRID_SHARD for multi-node training to reduce cross-node communication
    if args.training_num_nodes > 1 and args.fsdp_sharding_strategy == "FULL_SHARD":
        logger.warning(
            f"Multi-node training detected ({args.training_num_nodes} nodes). "
            f"Consider using --fsdp-sharding-strategy HYBRID_SHARD for better performance."
        )

    # ========== Validate reward configuration ==========
    validate_reward_config(args)

    reward_mix_mode = getattr(args, "reward_mix_mode", "reward_aggr")
    if reward_mix_mode not in ("reward_aggr", "advantage_aggr"):
        raise ValueError(
            f"reward_mix_mode must be one of reward_aggr/advantage_aggr, got: {reward_mix_mode}"
        )
    if int(getattr(args, "rollout_buffer_max_queue_size", 0)) < 0:
        raise ValueError(
            f"rollout_buffer_max_queue_size must be >= 0, got: {args.rollout_buffer_max_queue_size}"
        )
    if int(getattr(args, "rollout_buffer_min_samples", 1)) < 1:
        raise ValueError(
            f"rollout_buffer_min_samples must be >= 1, got: {args.rollout_buffer_min_samples}"
        )
    reward_min = getattr(args, "rollout_buffer_reward_min", None)
    reward_max = getattr(args, "rollout_buffer_reward_max", None)
    if reward_min is not None and reward_max is not None and float(reward_min) > float(reward_max):
        raise ValueError(
            "rollout_buffer_reward_min must be <= rollout_buffer_reward_max, "
            f"got min={reward_min}, max={reward_max}"
        )
    group_size = getattr(args, "rollout_buffer_group_size", None)
    if group_size is not None and int(group_size) < 1:
        raise ValueError(
            f"rollout_buffer_group_size must be >= 1 when provided, got: {group_size}"
        )
    if int(getattr(args, "rollout_buffer_dispatch_groups", 0)) < 0:
        raise ValueError(
            "rollout_buffer_dispatch_groups must be >= 0 "
            f"(0 means prompts_per_batch), got: {args.rollout_buffer_dispatch_groups}"
        )
    if float(getattr(args, "rollout_buffer_group_ttl_seconds", 0.0)) < 0:
        raise ValueError(
            "rollout_buffer_group_ttl_seconds must be >= 0, "
            f"got: {args.rollout_buffer_group_ttl_seconds}"
        )
    if int(getattr(args, "rollout_buffer_max_pending_samples", 0)) < 0:
        raise ValueError(
            "rollout_buffer_max_pending_samples must be >= 0, "
            f"got: {args.rollout_buffer_max_pending_samples}"
        )

    # ========== Inference actor GPU constraints ==========
    if sampling_backend != "training":
        inference_gpus = get_inference_gpus_per_actor(
            args,
            model_type_to_sampler_engine=MODEL_TYPE_TO_SAMPLER_ENGINE,
        )
        if inference_gpus > 1 and args.colocate_inference_training:
            raise ValueError(
                "colocate_inference_training=True does not support multi-GPU inference actors."
            )
        if inference_gpus > 1 and not bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
            raise ValueError(
                "multi-GPU inference actor layout requires --allow-noset-multi-gpu-inference=true. "
                "Default layout keeps integer single-GPU actors."
            )
        if inference_gpus > 1 and bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
            logger.warning(
                "allow_noset_multi_gpu_inference=true enabled. "
                "This is an experimental actor layout and is not part of the default path."
            )

    # ========== Per-prompt tracker auto-enable (Flow-GRPO) ==========
    # When using per_prompt advantage type with running mode, automatically enable the tracker
    if args.advantage_type == "per_prompt":
        if args.per_prompt_mode == "running" and not args.use_per_prompt_stat_tracker:
            args.use_per_prompt_stat_tracker = True
            logger.info("Auto-enabled use_per_prompt_stat_tracker for advantage_type='per_prompt'")
        elif args.per_prompt_mode != "running" and args.use_per_prompt_stat_tracker:
            logger.info("per_prompt_mode != 'running' will ignore per_prompt_stat_tracker")

    # Sync use_global_std flag to running stats (global normalization)
    if args.use_global_std:
        args.use_running_stats = True

    # Normalize LoRA target modules (comma-separated string -> list)
    if isinstance(args.lora_target_modules, str):
        stripped = args.lora_target_modules.strip()
        if stripped:
            args.lora_target_modules = [s.strip() for s in stripped.split(",") if s.strip()]
        else:
            args.lora_target_modules = None

    # ========== Model-specific runtime validation ==========
    from diffusionrl.config.defaults import MODEL_VALIDATORS

    if args.model_type != "flux" and args.sde_type.startswith("flux_"):
        raise ValueError(
            f"sde_type '{args.sde_type}' is only valid for model_type='flux'"
        )
    validator = MODEL_VALIDATORS.get(args.model_type)
    if validator is not None:
        validator(args)

    # ========== NFT default sampling adapter ==========
    if args.loss_type == "nft" and not args.sampling_adapter:
        args.sampling_adapter = "old"
        logger.info("NFT: default sampling_adapter set to 'old'")

    # ========== NFT default deterministic solver ==========
    if args.loss_type == "nft" and args.sde_type == "sde":
        args.sde_type = "dpm2"
        logger.info("NFT: default sde_type set to 'dpm2' for deterministic sampling")

    # ========== Algorithm-Loss consistency validation ==========
    # Enforce that algorithm_path matches loss_type to prevent silent misconfiguration
    ALGORITHM_LOSS_MAP = {
        "diffusionrl.algorithms.grpo.GRPOAlgorithm": "grpo",
        "diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm": "grpo",
        "diffusionrl.algorithms.nft.NFTAlgorithm": "nft",
    }
    expected_loss_type = ALGORITHM_LOSS_MAP.get(args.algorithm_path)
    if expected_loss_type and args.loss_type != expected_loss_type:
        raise ValueError(
            f"algorithm_path={args.algorithm_path} requires loss_type={expected_loss_type}, "
            f"but got loss_type={args.loss_type}. "
            f"Please either change --loss-type to {expected_loss_type} or use a compatible algorithm."
        )

    validate_engine_loss_compatibility(
        args=args,
        sampling_backend=sampling_backend,
        model_type_to_sampler_engine=MODEL_TYPE_TO_SAMPLER_ENGINE,
        engine_capability_requirements=ENGINE_CAPABILITY_REQUIREMENTS,
        loss_type_requirements=LOSS_TYPE_REQUIREMENTS,
    )
    validate_plugin_target_path(
        target_path=args.sampler_path,
        kind="sampler",
        required_capabilities=resolve_loss_type_requirements(
            loss_type=args.loss_type,
            loss_type_requirements=LOSS_TYPE_REQUIREMENTS,
        ),
    )

    # Ensure sampling_adapter is propagated to engine_kwargs
    if args.sampling_adapter:
        args.engine_kwargs.setdefault("sampling_adapter", args.sampling_adapter)

    # Normalize gradient_accumulation_steps to string for auto handling
    if isinstance(args.gradient_accumulation_steps, int):
        args.gradient_accumulation_steps = str(args.gradient_accumulation_steps)

    if args.num_inner_epochs < 1:
        raise ValueError(f"num_inner_epochs must be >= 1, got: {args.num_inner_epochs}")

    # Weight sync mode normalization
    weight_sync_mode = getattr(args, "weight_sync_mode", "auto")
    if weight_sync_mode not in ("auto", "object_ref", "checkpoint_path"):
        raise ValueError(
            f"weight_sync_mode must be one of auto/object_ref/checkpoint_path, got: {weight_sync_mode}"
        )
    if weight_sync_mode == "auto":
        # Prefer path-based sync for local inference engines to avoid driver relay.
        if args.sampler_engine_type in ("fsdp", "fastvideo", "sglang") and sampling_backend != "training":
            args.weight_sync_mode = "checkpoint_path"
        else:
            args.weight_sync_mode = "object_ref"
    if (
        sampling_backend == "inference"
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
        and args.weight_sync_mode != "checkpoint_path"
    ):
        logger.warning(
            "sampler_engine_type='sglang' does not support object_ref tensor push; "
            "forcing weight_sync_mode=checkpoint_path."
        )
        args.weight_sync_mode = "checkpoint_path"

    # SGLang path sync benefits significantly from tmpfs.
    # Only rewrite when user keeps the default weight_sync_dir.
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

    # Single-PG mode currently ignores reward_placement_strategy; keep only PACK to reduce control-plane branches.
    if getattr(args, "reward_placement_strategy", "PACK") != "PACK":
        logger.warning(
            "reward_placement_strategy is deprecated in single-PG mode; forcing PACK."
        )
        args.reward_placement_strategy = "PACK"

    # checkpoint_path sync needs a shared filesystem in multi-node deployments.
    if (
        args.weight_sync_mode == "checkpoint_path"
        and (
            int(getattr(args, "inference_num_nodes", 1)) > 1
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

    if (
        sampling_backend == "inference"
        and str(getattr(args, "sampler_engine_type", "")).lower() == "sglang"
    ):
        supported_sglang_prompt_models = {"hunyuan", "flux", "mochi", "sd3"}
        model_type = str(getattr(args, "model_type", "")).lower()
        if model_type not in supported_sglang_prompt_models:
            raise ValueError(
                "sampler_engine_type='sglang' now uses prompt-only rollout input mode "
                "(no prompt-embedding input path). "
                f"Unsupported model_type={args.model_type!r}. "
                f"Supported model types: {sorted(supported_sglang_prompt_models)}."
            )

    if getattr(args, "async_pipeline", False):
        if args.colocate_inference_training:
            raise ValueError("async_pipeline requires separate mode (colocate_inference_training=False).")
        if sampling_backend == "training":
            raise ValueError("async_pipeline currently requires sampling_backend='inference'.")
        if int(getattr(args, "async_max_inflight", 1)) < 1:
            raise ValueError("async_max_inflight must be >= 1.")
        if args.update_weights_interval <= 0:
            raise ValueError("update_weights_interval must be > 0.")
        if args.offload_train or args.offload_rollout:
            logger.warning("async_pipeline: disabling offload_train/offload_rollout for stable overlap.")
            args.offload_train = False
            args.offload_rollout = False

    return args


def get_default_args() -> TrainingArguments:
    """Get default arguments without parsing command line."""
    return TrainingArguments()
