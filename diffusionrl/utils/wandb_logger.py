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
        log_dir: Optional[str] = None,
        rank: int = 0,
        image_log_interval: int = 10,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        entity: Optional[str] = None,
        require_success: bool = False,
    ):
        """Initialize WandB logger.

        Args:
            project: WandB project name
            run_name: WandB run name
            config: Training configuration (dict or object with __dict__)
            log_dir: WandB run directory (if provided)
            rank: Process rank (only rank 0 logs)
            image_log_interval: How often to log images (in rollouts)
            enabled: Whether to enable logging
            tags: List of tags for the WandB run. Defaults to ['diffusionrl-reproduce'] if not provided.
            entity: WandB entity (team or username). If None, uses the default entity.
            require_success: Raise immediately if WandB is unavailable or init fails.
        """
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.log_dir = str(log_dir) if log_dir else None
        self.image_log_interval = image_log_interval
        self.rank = rank
        self.tags = tags if tags is not None else ["diffusionrl"]
        self.require_success = bool(require_success)
        self._initialized = False

        # Only enable on rank 0
        self.enabled = enabled and rank == 0

        if self.enabled and project:
            if not WANDB_AVAILABLE:
                self._handle_init_failure(
                    "wandb package is not installed but WandB reporting was requested"
                )
                return
            self._init_wandb(config)

    @property
    def initialized(self) -> bool:
        """Whether wandb.init completed successfully."""
        return bool(self._initialized)

    def _handle_init_failure(
        self,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Disable the logger or raise immediately when strict mode is enabled."""
        self.enabled = False
        full_message = f"{message}: {exc}" if exc is not None else message
        if self.require_success:
            raise RuntimeError(full_message) from exc
        print(f"Warning: {full_message}")

    def _init_wandb(self, config: Optional[Any] = None):
        """Initialize wandb run."""
        if not WANDB_AVAILABLE:
            self._handle_init_failure(
                "wandb package is not installed but WandB reporting was requested"
            )
            return

        try:
            # Convert config to dict if needed
            config_dict = None
            if config is not None:
                if isinstance(config, dict):
                    config_dict = config
                elif hasattr(config, "__dict__"):
                    config_dict = vars(config)

            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)

            init_kwargs = dict(
                project=self.project,
                name=self.run_name,
                config=config_dict,
                dir=self.log_dir,
                tags=self.tags,
            )
            if self.entity:
                init_kwargs["entity"] = self.entity
            wandb.init(**init_kwargs)
            self._init_metric_axes()
            self._initialized = True
        except Exception as e:
            self._handle_init_failure("Failed to initialize wandb", exc=e)

    def _init_metric_axes(self) -> None:
        """Define metric namespaces and their step axes."""
        if not WANDB_AVAILABLE:
            return
        try:
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            # rollout/step tracks the outer rollout-train loop step.
            # It behaves like a framework-level global step, but is not the same
            # thing as optimizer update count when one rollout yields multiple updates.
            wandb.define_metric("rollout/step")
            wandb.define_metric("rollout/*", step_metric="rollout/step")
            wandb.define_metric("perf/*", step_metric="rollout/step")
            wandb.define_metric("sync/*", step_metric="rollout/step")
            wandb.define_metric("buffer/*", step_metric="rollout/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
        except Exception as e:
            print(f"Warning: Failed to define wandb metrics: {e}")

    @staticmethod
    def _coerce_metric_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.numel() == 0:
                return None
            if tensor.numel() == 1:
                return float(tensor.item())
            return float(tensor.to(dtype=torch.float32).mean().item())
        return None

    @staticmethod
    def _apply_prefix(key: str, prefix: str) -> str:
        if not prefix:
            return key
        return key if key.startswith(prefix) else f"{prefix}{key}"

    def log_with_step(
        self,
        *,
        step_key: str,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Log metrics with an explicit namespace step key."""
        if not self.enabled or not self._initialized:
            return

        try:
            log_dict: Dict[str, Any] = {step_key: int(step)}
            for key, value in metrics.items():
                metric_key = self._apply_prefix(str(key), prefix)
                if metric_key == step_key:
                    continue
                scalar = self._coerce_metric_value(value)
                if scalar is None:
                    continue
                log_dict[metric_key] = scalar
            wandb.log(log_dict)
        except Exception as e:
            print(f"Warning: Failed to log metrics ({step_key}): {e}")

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
        self.log_with_step(
            step_key="train/step",
            step=step,
            metrics=metrics,
            prefix=prefix,
        )

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
            rollout_id: Outer rollout-train loop step. Similar to a global step for
                this framework, but not guaranteed to equal optimizer update count.
            metrics: Dictionary of metrics to log
        """
        self.log_with_step(
            step_key="rollout/step",
            step=rollout_id,
            metrics=metrics,
            prefix="rollout/",
        )

    def log_perf(
        self,
        rollout_id: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Log performance metrics keyed by rollout step."""
        self.log_with_step(
            step_key="rollout/step",
            step=rollout_id,
            metrics=metrics,
            prefix="perf/",
        )

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

                    # wandb_images.append(wandb.Image(img_path, caption=caption)) # Bug with temp file here
                    wandb_images.append(wandb.Image(images[idx], caption=caption)) # Bug with temp file here

            wandb.log(
                {
                    "rollout/step": int(step),
                    key: wandb_images,
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log images: {e}")

    def log_generated_media(
        self,
        rollout_id: int,
        media_preview: Dict[str, Any],
        *,
        key: str = "rollout/generated_media",
    ) -> None:
        """Log rollout media preview payload produced by rollout manager."""
        if not isinstance(media_preview, dict):
            return

        images = media_preview.get("images")
        if not isinstance(images, list) or not images:
            return

        prompts = media_preview.get("prompts")
        if not isinstance(prompts, list):
            prompts = []

        rewards = media_preview.get("rewards")
        self.log_images(
            images=images,
            prompts=prompts,
            rewards=rewards if isinstance(rewards, (list, torch.Tensor, dict)) else None,
            step=int(rollout_id),
            max_images=max(1, len(images)),
            key=key,
        )

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
        self.log_with_step(
            step_key="eval/step",
            step=step,
            metrics=eval_metrics,
            prefix="eval/",
        )

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
    log_dir: Optional[str] = None,
    rank: int = 0,
    tags: Optional[List[str]] = None,
    entity: Optional[str] = None,
    require_success: bool = False,
    **kwargs,
) -> GRPOCoreWandBLogger:
    """Initialize and set the global wandb logger.

    Args:
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        rank: Process rank
        tags: List of tags for the WandB run. Defaults to ['diffusionrl-reproduce'] if not provided.
        entity: WandB entity (team or username). If None, uses the default entity.
        require_success: Raise immediately if WandB init fails.
        **kwargs: Additional arguments for GRPOCoreWandBLogger

    Returns:
        The initialized logger
    """
    global _global_logger
    _global_logger = GRPOCoreWandBLogger(
        project=project,
        run_name=run_name,
        config=config,
        log_dir=log_dir,
        rank=rank,
        tags=tags,
        entity=entity,
        require_success=require_success,
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
