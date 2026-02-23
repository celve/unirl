"""
WandB Logger for diffusionrl Training.

Provides comprehensive logging for training metrics, rollout statistics,
and image samples. Designed to match the logging behavior of DanceGRPO,
flow_grpo, DiffusionNFT, and MixGRPO for comparison and reproducibility.

Usage:
    from diffusionrl.utils.wandb_logger import init_logger, get_logger

    # Initialize (typically in train.py)
    logger = init_logger(project="diffusionrl", run_name="exp1", config=args)

    # Log training metrics
    logger.log_step(step=100, metrics={"loss": 0.5, "policy_loss": 0.3})

    # Log rollout metrics
    logger.log_rollout(rollout_id=10, metrics={"reward_mean": 0.8})

    # Log images
    logger.log_images(images, prompts, rewards, step=100)
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GRPOCoreWandBLogger:
    """WandB logger for diffusionrl training.

    Logs metrics compatible with DanceGRPO, flow_grpo, DiffusionNFT, and MixGRPO
    for cross-validation and comparison.

    Attributes:
        enabled: Whether logging is enabled
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        image_log_interval: How often to log images (in rollouts)
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        rank: int = 0,
        image_log_interval: int = 10,
        enabled: bool = True,
    ):
        """Initialize WandB logger.

        Args:
            project: WandB project name
            run_name: WandB run name
            config: Training configuration (dict or object with __dict__)
            rank: Process rank (only rank 0 logs)
            image_log_interval: How often to log images (in rollouts)
            enabled: Whether to enable logging
        """
        self.project = project
        self.run_name = run_name
        self.image_log_interval = image_log_interval
        self.rank = rank
        self._initialized = False

        # Only enable on rank 0
        self.enabled = enabled and WANDB_AVAILABLE and rank == 0

        if self.enabled and project:
            self._init_wandb(config)

    def _init_wandb(self, config: Optional[Any] = None):
        """Initialize wandb run."""
        if not WANDB_AVAILABLE:
            return

        try:
            # Convert config to dict if needed
            config_dict = None
            if config is not None:
                if isinstance(config, dict):
                    config_dict = config
                elif hasattr(config, "__dict__"):
                    config_dict = vars(config)

            wandb.init(
                project=self.project,
                name=self.run_name,
                config=config_dict,
            )
            self._initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.enabled = False

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "train/",
    ):
        """Log per-step training metrics.

        Metrics typically include:
        - loss: Total loss
        - policy_loss: Policy gradient loss
        - kl_loss: KL divergence loss
        - approx_kl: Approximate KL divergence
        - clip_fraction: Fraction of ratios clipped
        - ratio_mean/std: Importance sampling ratio stats
        - grad_norm: Gradient norm
        - lr: Learning rate

        Args:
            step: Global step number
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names (default: "train/")
        """
        if not self.enabled or not self._initialized:
            return

        try:
            log_dict = {}
            for key, value in metrics.items():
                # Handle tensors
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.mean().item()

                # Add prefix
                log_key = f"{prefix}{key}" if prefix else key
                log_dict[log_key] = value

            wandb.log(log_dict, step=step)
        except Exception as e:
            print(f"Warning: Failed to log step metrics: {e}")

    def log_rollout(
        self,
        rollout_id: int,
        metrics: Dict[str, Any],
    ):
        """Log per-rollout metrics.

        Metrics typically include:
        - reward_mean: Mean reward across samples
        - reward_std: Reward standard deviation
        - advantage_mean: Mean advantage
        - advantage_std: Advantage standard deviation
        - num_samples: Number of samples in rollout
        - zero_std_ratio: Ratio of prompts with zero reward std

        Args:
            rollout_id: Rollout identifier
            metrics: Dictionary of metrics to log
        """
        if not self.enabled or not self._initialized:
            return

        try:
            log_dict = {"rollout_id": rollout_id}

            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.mean().item()
                log_dict[f"rollout/{key}"] = value

            wandb.log(log_dict, step=rollout_id)
        except Exception as e:
            print(f"Warning: Failed to log rollout metrics: {e}")

    def log_images(
        self,
        images: List[Image.Image],
        prompts: List[str],
        rewards: Optional[Union[List[float], torch.Tensor, Dict[str, Any]]] = None,
        step: int = 0,
        max_images: int = 8,
        key: str = "samples",
    ):
        """Log sample images with captions.

        Format: "{prompt:.100} | reward: {reward:.2f}"

        Args:
            images: List of PIL images
            prompts: List of prompts
            rewards: Rewards (list, tensor, or dict with 'avg' key)
            step: Global step number
            max_images: Maximum number of images to log
            key: Key name for wandb logging
        """
        if not self.enabled or not self._initialized:
            return

        try:
            # Sample images if too many
            num_images = min(len(images), max_images)
            sample_indices = list(range(num_images))

            # Process rewards
            reward_values = None
            if rewards is not None:
                if isinstance(rewards, dict):
                    reward_values = rewards.get("avg", rewards.get("rewards"))
                elif isinstance(rewards, torch.Tensor):
                    reward_values = rewards.cpu().tolist()
                else:
                    reward_values = list(rewards)

            # Create wandb images with captions
            wandb_images = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, idx in enumerate(sample_indices):
                    # Save image to temp file
                    img_path = os.path.join(tmpdir, f"{i}.jpg")
                    images[idx].save(img_path)

                    # Create caption
                    prompt = prompts[idx] if idx < len(prompts) else ""
                    if reward_values and idx < len(reward_values):
                        caption = f"{prompt[:100]} | reward: {reward_values[idx]:.2f}"
                    else:
                        caption = f"{prompt[:100]}"

                    wandb_images.append(wandb.Image(img_path, caption=caption))

            wandb.log({key: wandb_images}, step=step)
        except Exception as e:
            print(f"Warning: Failed to log images: {e}")

    def log_eval(
        self,
        step: int,
        eval_metrics: Dict[str, Any],
    ):
        """Log evaluation metrics.

        Args:
            step: Global step number
            eval_metrics: Dictionary of evaluation metrics
        """
        if not self.enabled or not self._initialized:
            return

        try:
            log_dict = {}
            for key, value in eval_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.mean().item()
                log_dict[f"eval/{key}"] = value

            wandb.log(log_dict, step=step)
        except Exception as e:
            print(f"Warning: Failed to log eval metrics: {e}")

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self._initialized:
            try:
                wandb.finish()
            except Exception:
                pass


# Global logger instance
_global_logger: Optional[GRPOCoreWandBLogger] = None


def get_logger() -> Optional[GRPOCoreWandBLogger]:
    """Get the global wandb logger instance."""
    return _global_logger


def set_logger(logger: GRPOCoreWandBLogger):
    """Set the global wandb logger instance."""
    global _global_logger
    _global_logger = logger


def init_logger(
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Any] = None,
    rank: int = 0,
    **kwargs,
) -> GRPOCoreWandBLogger:
    """Initialize and set the global wandb logger.

    Args:
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        rank: Process rank
        **kwargs: Additional arguments for GRPOCoreWandBLogger

    Returns:
        The initialized logger
    """
    global _global_logger
    _global_logger = GRPOCoreWandBLogger(
        project=project,
        run_name=run_name,
        config=config,
        rank=rank,
        **kwargs,
    )
    return _global_logger


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics from multiple training actors.

    Args:
        metrics_list: List of metric dicts from each actor

    Returns:
        Aggregated metrics (mean of each key)
    """
    if not metrics_list:
        return {}

    aggregated = {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    for key in all_keys:
        values = []
        for m in metrics_list:
            if key in m:
                val = m[key]
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.numel() == 1 else val.mean().item()
                if isinstance(val, bool):
                    values.append(float(val))
                elif isinstance(val, (int, float)):
                    values.append(float(val))
        if values:
            aggregated[key] = sum(values) / len(values)

    return aggregated
