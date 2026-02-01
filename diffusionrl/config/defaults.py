"""
diffusionrl Default Configurations.

Provides default configurations for different use cases:
- Model-specific defaults (HunyuanVideo, FLUX, SD3, Mochi)
- Algorithm-specific defaults (GRPO, MixGRPO, NFT, AWM)
- Hardware-specific presets (single GPU, multi-GPU, multi-node)
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


# ============================================================
# Model-Specific Defaults
# ============================================================


@dataclass
class HunyuanVideoDefaults:
    """Default configuration for HunyuanVideo training."""

    # Model
    model_path: str = "diffusionrl.models.hunyuan.HunyuanModelBundle"

    # Sampling
    num_inference_steps: int = 50
    eta: float = 0.7
    guidance_scale: float = 7.5
    shift: float = 3.0

    # Video dimensions
    height: int = 480
    width: int = 848
    num_frames: int = 45

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 2
    gradient_accumulation_steps: int = 12

    # Loss
    clip_range: float = 1e-4
    kl_coef: float = 0.01
    sde_type: str = "dance"

    # FSDP
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "num_inference_steps": self.num_inference_steps,
            "eta": self.eta,
            "guidance_scale": self.guidance_scale,
            "shift": self.shift,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "sde_type": self.sde_type,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
        }

    @staticmethod
    def validate(args: Any) -> None:
        """Validate/normalize Hunyuan-specific runtime constraints."""
        fixed_guidance = 6018.0
        if getattr(args, "sampler_engine_type", None) != "fsdp":
            return
        # If user did not override the framework default (7.5), align automatically.
        if abs(float(args.guidance_scale) - 7.5) <= 1e-6:
            args.guidance_scale = fixed_guidance
        if abs(float(args.guidance_scale) - fixed_guidance) > 1e-6:
            raise ValueError(
                f"FSDP Hunyuan sampler uses fixed guidance_scale={fixed_guidance}. "
                f"Got guidance_scale={args.guidance_scale}."
            )


@dataclass
class FluxDefaults:
    """Default configuration for FLUX training."""

    # Model
    model_path: str = "diffusionrl.models.flux.FluxModelBundle"

    # Sampling
    num_inference_steps: int = 28
    eta: float = 1.0
    guidance_scale: float = 3.5
    shift: float = 1.0

    # Image dimensions
    height: int = 1024
    width: int = 1024

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Loss
    clip_range: float = 1e-4
    kl_coef: float = 0.01
    sde_type: str = "flux_dance"

    # FSDP
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "num_inference_steps": self.num_inference_steps,
            "eta": self.eta,
            "guidance_scale": self.guidance_scale,
            "shift": self.shift,
            "height": self.height,
            "width": self.width,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "sde_type": self.sde_type,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
        }

    @staticmethod
    def validate(args: Any) -> None:
        """Validate/normalize FLUX-specific runtime constraints."""
        if args.sde_type in ("dance", "sde"):
            args.sde_type = "flux_dance"
        elif args.sde_type == "flow":
            args.sde_type = "flux_flow"
        elif args.sde_type in ("flux_dance", "flux_flow"):
            pass
        elif args.sde_type.startswith("flux_"):
            raise ValueError(f"Unknown FLUX sde_type: {args.sde_type}")


@dataclass
class SD3Defaults:
    """Default configuration for SD3 training."""

    # Model
    model_path: str = "diffusionrl.models.sd3.SD3ModelBundle"

    # Sampling
    num_inference_steps: int = 28
    eta: float = 1.0
    guidance_scale: float = 7.0
    shift: float = 3.0

    # Image dimensions
    height: int = 1024
    width: int = 1024

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Loss
    clip_range: float = 1e-4
    kl_coef: float = 0.01
    sde_type: str = "sde"

    # FSDP
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "num_inference_steps": self.num_inference_steps,
            "eta": self.eta,
            "guidance_scale": self.guidance_scale,
            "shift": self.shift,
            "height": self.height,
            "width": self.width,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "sde_type": self.sde_type,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
        }


@dataclass
class MochiDefaults:
    """Default configuration for Mochi training."""

    # Model
    model_path: str = "diffusionrl.models.mochi.MochiModelBundle"

    # Sampling
    num_inference_steps: int = 50
    eta: float = 0.7
    guidance_scale: float = 4.5
    shift: float = 4.0

    # Video dimensions
    height: int = 480
    width: int = 848
    num_frames: int = 25

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 2
    gradient_accumulation_steps: int = 16

    # Loss
    clip_range: float = 1e-4
    kl_coef: float = 0.01
    sde_type: str = "sde"

    # FSDP
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "num_inference_steps": self.num_inference_steps,
            "eta": self.eta,
            "guidance_scale": self.guidance_scale,
            "shift": self.shift,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "sde_type": self.sde_type,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
        }


# ============================================================
# Algorithm-Specific Defaults
# ============================================================


@dataclass
class GRPODefaults:
    """Default configuration for GRPO algorithm."""

    algorithm_path: str = "diffusionrl.algorithms.grpo.GRPOAlgorithm"
    advantage_type: str = "group"
    clip_range: float = 1e-4
    kl_coef: float = 0.01
    num_samples_per_prompt: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_path": self.algorithm_path,
            "advantage_type": self.advantage_type,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "num_samples_per_prompt": self.num_samples_per_prompt,
        }


@dataclass
class MixGRPODefaults:
    """Default configuration for MixGRPO algorithm.

    MixGRPO uses a sliding window timestep scheduler to determine
    which timesteps use SDE (stochastic) vs ODE (deterministic) sampling.
    """

    algorithm_path: str = "diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm"
    advantage_type: str = "group"
    clip_range: float = 1e-4
    kl_coef: float = 0.01
    num_samples_per_prompt: int = 4
    sde_ratio: float = 0.5  # Mix 50% SDE, 50% ODE

    # Window scheduler configuration
    timestep_strategy: str = "window"
    window_strategy: str = "progressive"  # progressive, random, decay, exp_decay
    window_group_size: int = 4  # Timesteps per window
    window_iters_per_group: int = 25  # Iterations before sliding
    window_overlap: bool = False  # Overlapping windows
    window_overlap_step: int = 1  # Overlap stride
    window_roll_back: bool = False  # Roll back when reaching end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_path": self.algorithm_path,
            "advantage_type": self.advantage_type,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "num_samples_per_prompt": self.num_samples_per_prompt,
            "sde_ratio": self.sde_ratio,
            "timestep_strategy": self.timestep_strategy,
            "window_strategy": self.window_strategy,
            "window_group_size": self.window_group_size,
            "window_iters_per_group": self.window_iters_per_group,
            "window_overlap": self.window_overlap,
            "window_overlap_step": self.window_overlap_step,
            "window_roll_back": self.window_roll_back,
        }


@dataclass
class NFTDefaults:
    """Default configuration for NFT (Noise-Free Training) algorithm.

    NFT uses forward process optimization with dual adapter mechanism.
    Key differences from GRPO:
    - Uses NFTLoss instead of GRPOLoss
    - Requires EMA for dual adapter updates
    - No trajectory storage needed (only clean latents)
    """

    algorithm_path: str = "diffusionrl.algorithms.nft.NFTAlgorithm"
    advantage_type: str = "group"
    clip_range: float = 1e-4
    kl_coef: float = 0.0  # No KL penalty for NFT

    # NFT-specific parameters
    loss_type: str = "nft"
    nft_beta: float = 0.1  # Interpolation weight
    nft_adv_clip_max: float = 5.0
    nft_adv_mode: str = "raw"
    nft_use_adaptive_weight: bool = True

    # EMA for dual adapter
    use_ema: bool = True
    ema_decay: float = 0.001

    num_samples_per_prompt: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_path": self.algorithm_path,
            "advantage_type": self.advantage_type,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "loss_type": self.loss_type,
            "nft_beta": self.nft_beta,
            "nft_adv_clip_max": self.nft_adv_clip_max,
            "nft_adv_mode": self.nft_adv_mode,
            "nft_use_adaptive_weight": self.nft_use_adaptive_weight,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "num_samples_per_prompt": self.num_samples_per_prompt,
        }



# ============================================================
# Hardware Presets
# ============================================================


@dataclass
class SingleGPUPreset:
    """Configuration for single GPU training."""

    inference_num_nodes: int = 1
    inference_num_gpus_per_node: int = 1
    training_num_nodes: int = 1
    training_num_gpus_per_node: int = 1

    batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # Aggressive memory optimization
    fsdp_cpu_offload: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"
    use_gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_num_nodes": self.inference_num_nodes,
            "inference_num_gpus_per_node": self.inference_num_gpus_per_node,
            "training_num_nodes": self.training_num_nodes,
            "training_num_gpus_per_node": self.training_num_gpus_per_node,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }


@dataclass
class MultiGPUPreset:
    """Configuration for multi-GPU single node training (4-8 GPUs)."""

    inference_num_nodes: int = 1
    inference_num_gpus_per_node: int = 4
    training_num_nodes: int = 1
    training_num_gpus_per_node: int = 4

    batch_size: int = 2
    gradient_accumulation_steps: int = 12

    fsdp_cpu_offload: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    use_gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_num_nodes": self.inference_num_nodes,
            "inference_num_gpus_per_node": self.inference_num_gpus_per_node,
            "training_num_nodes": self.training_num_nodes,
            "training_num_gpus_per_node": self.training_num_gpus_per_node,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }


@dataclass
class MultiNodePreset:
    """Configuration for multi-node training."""

    inference_num_nodes: int = 2
    inference_num_gpus_per_node: int = 8
    training_num_nodes: int = 2
    training_num_gpus_per_node: int = 8

    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    fsdp_cpu_offload: bool = False
    fsdp_sharding_strategy: str = "HYBRID_SHARD"  # Better for multi-node
    use_gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_num_nodes": self.inference_num_nodes,
            "inference_num_gpus_per_node": self.inference_num_gpus_per_node,
            "training_num_nodes": self.training_num_nodes,
            "training_num_gpus_per_node": self.training_num_gpus_per_node,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }


# ============================================================
# Reward Model Defaults
# ============================================================


@dataclass
class HPSv2RewardDefaults:
    """Defaults for HPSv2 reward model."""

    reward_path: str = "diffusionrl.workers.reward.local.LocalRewardWorker"
    reward_model_name: str = "hpsv2"
    reward_batch_size: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_path": self.reward_path,
            "reward_model_name": self.reward_model_name,
            "reward_batch_size": self.reward_batch_size,
        }


@dataclass
class PickScoreRewardDefaults:
    """Defaults for PickScore reward model."""

    reward_path: str = "diffusionrl.workers.reward.local.LocalRewardWorker"
    reward_model_name: str = "pickscore"
    reward_batch_size: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_path": self.reward_path,
            "reward_model_name": self.reward_model_name,
            "reward_batch_size": self.reward_batch_size,
        }


@dataclass
class VideoAlignRewardDefaults:
    """Defaults for VideoAlign HTTP reward service."""

    reward_path: str = "diffusionrl.workers.reward.http.HTTPRewardWorker"
    reward_http_url: str = "http://localhost:8000/score"
    reward_timeout: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_path": self.reward_path,
            "reward_http_url": self.reward_http_url,
            "reward_timeout": self.reward_timeout,
        }


# ============================================================
# Preset Combinations
# ============================================================


def get_hunyuan_grpo_defaults() -> Dict[str, Any]:
    """Get full defaults for HunyuanVideo + GRPO."""
    config = {}
    config.update(HunyuanVideoDefaults().to_dict())
    config.update(GRPODefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_flux_grpo_defaults() -> Dict[str, Any]:
    """Get full defaults for FLUX + GRPO."""
    config = {}
    config.update(FluxDefaults().to_dict())
    config.update(GRPODefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_mochi_grpo_defaults() -> Dict[str, Any]:
    """Get full defaults for Mochi + GRPO."""
    config = {}
    config.update(MochiDefaults().to_dict())
    config.update(GRPODefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_flux_mixgrpo_defaults() -> Dict[str, Any]:
    """Get full defaults for FLUX + MixGRPO (sliding window scheduler)."""
    config = {}
    config.update(FluxDefaults().to_dict())
    config.update(MixGRPODefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_flux_nft_defaults() -> Dict[str, Any]:
    """Get full defaults for FLUX + NFT (forward process optimization)."""
    config = {}
    config.update(FluxDefaults().to_dict())
    config.update(NFTDefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_sd3_grpo_defaults() -> Dict[str, Any]:
    """Get full defaults for SD3 + GRPO."""
    config = {}
    config.update(SD3Defaults().to_dict())
    config.update(GRPODefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_sd3_nft_defaults() -> Dict[str, Any]:
    """Get full defaults for SD3 + NFT."""
    config = {}
    config.update(SD3Defaults().to_dict())
    config.update(NFTDefaults().to_dict())
    config.update(HPSv2RewardDefaults().to_dict())
    config.update(MultiGPUPreset().to_dict())
    return config


def get_defaults_for_model(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific model type.

    Args:
        model_type: One of "hunyuan", "flux", "sd3", "mochi"

    Returns:
        Default configuration dictionary
    """
    if model_type == "hunyuan":
        return get_hunyuan_grpo_defaults()
    elif model_type == "flux":
        return get_flux_grpo_defaults()
    elif model_type == "sd3":
        return get_sd3_grpo_defaults()
    elif model_type == "mochi":
        return get_mochi_grpo_defaults()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


MODEL_VALIDATORS: Dict[str, Callable[[Any], None]] = {
    "flux": FluxDefaults.validate,
    "hunyuan": HunyuanVideoDefaults.validate,
}


def get_defaults_for_algorithm(algorithm: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific algorithm.

    Args:
        algorithm: One of "grpo", "mixgrpo", "nft"

    Returns:
        Default configuration dictionary
    """
    if algorithm == "grpo":
        return GRPODefaults().to_dict()
    elif algorithm == "mixgrpo":
        return MixGRPODefaults().to_dict()
    elif algorithm == "nft":
        return NFTDefaults().to_dict()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_hardware_preset(preset: str) -> Dict[str, Any]:
    """
    Get hardware preset configuration.

    Args:
        preset: One of "single_gpu", "multi_gpu", "multi_node"

    Returns:
        Hardware configuration dictionary
    """
    if preset == "single_gpu":
        return SingleGPUPreset().to_dict()
    elif preset == "multi_gpu":
        return MultiGPUPreset().to_dict()
    elif preset == "multi_node":
        return MultiNodePreset().to_dict()
    else:
        raise ValueError(f"Unknown hardware preset: {preset}")


# ============================================================
# Configuration Merging
# ============================================================


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


def apply_defaults(user_config: Dict[str, Any], model_type: str = "hunyuan") -> Dict[str, Any]:
    """
    Apply defaults to user configuration.

    User values override defaults.

    Args:
        user_config: User-provided configuration
        model_type: Model type for defaults

    Returns:
        Complete configuration with defaults applied
    """
    defaults = get_defaults_for_model(model_type)
    defaults.update(user_config)
    return defaults
