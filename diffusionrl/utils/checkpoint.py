"""
diffusionrl Checkpoint Management Utilities.

Provides utilities for saving and loading model checkpoints, including:
- Full model checkpoints (state dict, optimizer, scheduler)
- FSDP-aware checkpoint saving
- Checkpoint versioning and metadata
- Automatic checkpoint cleanup
"""
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Checkpoint version for compatibility
CHECKPOINT_VERSION = "1.0"


def get_checkpoint_path(output_dir: str, rollout_id: int) -> str:
    """
    Get checkpoint path for a specific rollout.

    Args:
        output_dir: Base output directory
        rollout_id: Rollout iteration number

    Returns:
        Full path to checkpoint directory
    """
    return os.path.join(output_dir, f"checkpoint-{rollout_id}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    output_dir: str = "./checkpoints",
    rollout_id: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    additional_state: Optional[Dict[str, Any]] = None,
    use_fsdp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> str:
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Optional learning rate scheduler
        output_dir: Directory to save checkpoint
        rollout_id: Current rollout iteration
        metrics: Optional training metrics
        config: Optional configuration to save
        additional_state: Optional additional state to save
        use_fsdp: Whether model is wrapped with FSDP
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = get_checkpoint_path(output_dir, rollout_id)

    # Only rank 0 saves the checkpoint
    if rank != 0:
        # Wait for rank 0 to save if distributed
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
        return checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get model state dict
    if use_fsdp:
        model_state_dict = _get_fsdp_state_dict(model)
    else:
        model_state_dict = model.state_dict()

    # Build checkpoint
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "rollout_id": rollout_id,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if additional_state is not None:
        checkpoint["additional_state"] = additional_state

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save config separately as JSON for easy inspection
    if config is not None:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "version": CHECKPOINT_VERSION,
        "rollout_id": rollout_id,
        "timestamp": checkpoint["timestamp"],
        "model_type": getattr(model, "model_type", "unknown"),
        "world_size": world_size,
    }
    if metrics is not None:
        metadata["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Checkpoint saved to {checkpoint_dir}")

    # Barrier after saving
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[Union[str, torch.device]] = None,
    use_fsdp: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        use_fsdp: Whether model is wrapped with FSDP
        strict: Whether to strictly enforce state dict keys matching

    Returns:
        Dictionary with checkpoint data
    """
    # Handle directory or file path
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
    else:
        checkpoint_file = checkpoint_path

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Load checkpoint
    map_location = device if device else "cpu"
    checkpoint = torch.load(checkpoint_file, map_location=map_location)

    # Load model state
    if model is not None and "model_state_dict" in checkpoint:
        if use_fsdp:
            _load_fsdp_state_dict(model, checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("Model state loaded")

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state loaded")

    logger.info(f"Checkpoint loaded from {checkpoint_file}")

    return checkpoint


def _get_fsdp_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get full state dict from FSDP model."""
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            return model.state_dict()
    except ImportError:
        logger.warning("FSDP not available, falling back to regular state dict")
        return model.state_dict()


def _load_fsdp_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load state dict into FSDP model."""
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )

        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(state_dict)
    except ImportError:
        logger.warning("FSDP not available, falling back to regular load")
        model.load_state_dict(state_dict)


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the output directory.

    Args:
        output_dir: Base output directory to search

    Returns:
        Path to latest checkpoint directory, or None if not found
    """
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                rollout_id = int(name.split("-")[1])
                checkpoint_path = os.path.join(output_dir, name)
                if os.path.exists(os.path.join(checkpoint_path, "checkpoint.pt")):
                    checkpoints.append((rollout_id, checkpoint_path))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    # Return checkpoint with highest rollout_id
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def list_checkpoints(output_dir: str) -> List[Dict[str, Any]]:
    """
    List all checkpoints in the output directory.

    Args:
        output_dir: Base output directory to search

    Returns:
        List of checkpoint metadata dictionaries
    """
    if not os.path.exists(output_dir):
        return []

    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            checkpoint_path = os.path.join(output_dir, name)
            metadata_path = os.path.join(checkpoint_path, "metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["path"] = checkpoint_path
                checkpoints.append(metadata)
            elif os.path.exists(os.path.join(checkpoint_path, "checkpoint.pt")):
                # Minimal metadata
                try:
                    rollout_id = int(name.split("-")[1])
                    checkpoints.append({
                        "rollout_id": rollout_id,
                        "path": checkpoint_path,
                    })
                except (ValueError, IndexError):
                    continue

    # Sort by rollout_id
    checkpoints.sort(key=lambda x: x.get("rollout_id", 0))
    return checkpoints


def cleanup_checkpoints(
    output_dir: str,
    keep_last_n: int = 5,
    keep_best: bool = True,
    metric_name: str = "loss",
    metric_mode: str = "min",
) -> List[str]:
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        output_dir: Base output directory
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to also keep the best checkpoint
        metric_name: Metric name to use for determining best
        metric_mode: "min" or "max" for metric comparison

    Returns:
        List of deleted checkpoint paths
    """
    checkpoints = list_checkpoints(output_dir)

    if len(checkpoints) <= keep_last_n:
        return []

    # Identify checkpoints to keep
    keep_paths = set()

    # Keep last N by rollout_id
    sorted_by_id = sorted(checkpoints, key=lambda x: x.get("rollout_id", 0), reverse=True)
    for ckpt in sorted_by_id[:keep_last_n]:
        keep_paths.add(ckpt["path"])

    # Keep best by metric
    if keep_best:
        checkpoints_with_metric = [
            c for c in checkpoints
            if "metrics" in c and metric_name in c.get("metrics", {})
        ]
        if checkpoints_with_metric:
            if metric_mode == "min":
                best = min(checkpoints_with_metric, key=lambda x: x["metrics"][metric_name])
            else:
                best = max(checkpoints_with_metric, key=lambda x: x["metrics"][metric_name])
            keep_paths.add(best["path"])

    # Delete others
    deleted = []
    for ckpt in checkpoints:
        if ckpt["path"] not in keep_paths:
            try:
                shutil.rmtree(ckpt["path"])
                deleted.append(ckpt["path"])
                logger.info(f"Deleted checkpoint: {ckpt['path']}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {ckpt['path']}: {e}")

    return deleted


def save_model_only(
    model: nn.Module,
    output_path: str,
    use_fsdp: bool = False,
) -> None:
    """
    Save only the model weights (without optimizer state).

    Useful for final model export or inference.

    Args:
        model: Model to save
        output_path: Output file path
        use_fsdp: Whether model is wrapped with FSDP
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if use_fsdp:
        state_dict = _get_fsdp_state_dict(model)
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, output_path)
    logger.info(f"Model saved to {output_path}")


def load_model_only(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
    use_fsdp: bool = False,
    strict: bool = True,
) -> None:
    """
    Load only model weights from checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file or directory
        device: Device to load tensors to
        use_fsdp: Whether model is wrapped with FSDP
        strict: Whether to strictly enforce state dict keys matching
    """
    # Handle directory path
    if os.path.isdir(checkpoint_path):
        # Try checkpoint.pt first
        ckpt_file = os.path.join(checkpoint_path, "checkpoint.pt")
        if os.path.exists(ckpt_file):
            checkpoint = torch.load(ckpt_file, map_location=device or "cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            # Try model.pt
            model_file = os.path.join(checkpoint_path, "model.pt")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location=device or "cpu")
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    else:
        loaded = torch.load(checkpoint_path, map_location=device or "cpu")
        state_dict = loaded.get("model_state_dict", loaded)

    if use_fsdp:
        _load_fsdp_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict, strict=strict)

    logger.info(f"Model loaded from {checkpoint_path}")


class CheckpointManager:
    """
    High-level checkpoint manager for training.

    Handles automatic saving, loading, and cleanup of checkpoints.
    """

    def __init__(
        self,
        output_dir: str,
        save_steps: int = 100,
        keep_last_n: int = 5,
        keep_best: bool = True,
        metric_name: str = "loss",
        metric_mode: str = "min",
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Base directory for checkpoints
            save_steps: Save every N steps
            keep_last_n: Number of checkpoints to keep
            keep_best: Whether to track and keep best checkpoint
            metric_name: Metric for determining best checkpoint
            metric_mode: "min" or "max"
        """
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        self.best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self.best_checkpoint_path: Optional[str] = None

        os.makedirs(output_dir, exist_ok=True)

    def should_save(self, step: int) -> bool:
        """Check if we should save at this step."""
        return int(self.save_steps) > 0 and (step + 1) % int(self.save_steps) == 0

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Save checkpoint and update best tracking.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional scheduler
            step: Current step (rollout_id)
            metrics: Optional metrics dict
            **kwargs: Additional arguments for save_checkpoint

        Returns:
            Path to saved checkpoint
        """
        path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=self.output_dir,
            rollout_id=step,
            metrics=metrics,
            **kwargs,
        )

        # Update best tracking
        if self.keep_best and metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            is_better = (
                (self.metric_mode == "min" and metric_value < self.best_metric)
                or (self.metric_mode == "max" and metric_value > self.best_metric)
            )
            if is_better:
                self.best_metric = metric_value
                self.best_checkpoint_path = path
                logger.info(f"New best checkpoint: {path} ({self.metric_name}={metric_value:.4f})")

        # Cleanup old checkpoints
        self.cleanup()

        return path

    def cleanup(self) -> None:
        """Clean up old checkpoints."""
        cleanup_checkpoints(
            self.output_dir,
            keep_last_n=self.keep_last_n,
            keep_best=self.keep_best,
            metric_name=self.metric_name,
            metric_mode=self.metric_mode,
        )
