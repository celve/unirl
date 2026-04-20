"""GPU observability helpers used by diffusionrl Ray actors.

All three functions are no-ops unless ``DIFFUSIONRL_LOG_GPU`` is set to a
truthy value in the environment.
"""
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _gpu_debug_enabled() -> bool:
    return os.getenv("DIFFUSIONRL_LOG_GPU", "0").lower() in ("1", "true", "yes")


def log_resource_ids(tag: str, rank: int) -> None:
    """Log Ray resource IDs for GPU debugging.

    Toggle with ``DIFFUSIONRL_LOG_GPU=1``.
    """
    if not _gpu_debug_enabled():
        return
    try:
        import ray

        ctx = ray.get_runtime_context()
        resources = ctx.get_resource_ids()
        logger.info(f"[GPU_RES] {tag} rank={rank} resources={resources}")
    except Exception as e:
        logger.warning(f"[GPU_RES] {tag} failed: {e}")


def log_gpu_state(
    tag: str,
    rank: int,
    device: Any = None,
    offloaded: Any = None,
) -> None:
    """Log GPU memory state for GPU debugging.

    Toggle with ``DIFFUSIONRL_LOG_GPU=1``.
    """
    if not _gpu_debug_enabled():
        return
    try:
        pid = os.getpid()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device_str = device if device is not None else "none"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
        else:
            allocated = 0.0
            reserved = 0.0
        logger.info(
            "[GPU_STATE] %s rank=%s pid=%s cuda_visible=%s device=%s allocated_gb=%.3f "
            "reserved_gb=%.3f offloaded=%s",
            tag, rank, pid, cuda_visible, device_str, allocated, reserved, offloaded,
        )
    except Exception as e:
        logger.warning(f"[GPU_STATE] {tag} failed: {e}")
