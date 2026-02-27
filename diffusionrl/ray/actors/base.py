"""
diffusionrl Ray Actor Base Classes.

Reference: slime/ray/ray_actor.py + slime/ray/train_actor.py
"""
import abc
import logging
import os
import socket
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ============================================================
# Shared utility functions for all actors (Training & Rollout)
# ============================================================

def log_resource_ids(tag: str, rank: int) -> None:
    """Log Ray resource IDs for GPU debugging. Enable with GRPO_LOG_GPU=1."""
    if os.getenv("GRPO_LOG_GPU", "0").lower() not in ("1", "true", "yes"):
        return
    try:
        import ray
        ctx = ray.get_runtime_context()
        resources = ctx.get_resource_ids()
        logger.warning(f"[GPU_RES] {tag} rank={rank} resources={resources}")
    except Exception as e:
        logger.warning(f"[GPU_RES] {tag} failed: {e}")


def log_gpu_state(tag: str, rank: int, device: Any = None, offloaded: Any = None) -> None:
    """Log GPU memory state for debugging. Enable with GRPO_LOG_GPU=1."""
    if os.getenv("GRPO_LOG_GPU", "0").lower() not in ("1", "true", "yes"):
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
        logger.warning(
            "[GPU_STATE] %s rank=%s pid=%s cuda_visible=%s device=%s allocated_gb=%.3f "
            "reserved_gb=%.3f offloaded=%s",
            tag, rank, pid, cuda_visible, device_str, allocated, reserved, offloaded,
        )
    except Exception as e:
        logger.warning(f"[GPU_STATE] {tag} failed: {e}")


def tensor_to_pil(images: torch.Tensor) -> List[Any]:
    """Convert tensor to PIL images. Handles both image [B,C,H,W] and video [B,C,T,H,W]."""
    from PIL import Image
    import numpy as np

    pil_images = []
    images = images.cpu()

    # Handle video: extract middle frame
    if images.dim() == 5:
        T = images.shape[2]
        images = images[:, :, T // 2]

    for img in images:
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))

    return pil_images


class RayActor:
    """
    Ray Actor base class - provides infrastructure functionality.

    This class provides common utilities for all Ray actors including:
    - Network configuration (getting IP and free ports)
    - Master address/port setup for distributed training
    """

    @staticmethod
    def _get_current_node_ip() -> str:
        """Get the IP address of the current node."""
        return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def _get_free_port(start_port: int = 10000, max_attempts: int = 100) -> int:
        """
        Find a free port starting from start_port.

        Args:
            start_port: Port number to start searching from
            max_attempts: Maximum number of ports to try

        Returns:
            A free port number
        """
        for port in range(start_port, start_port + max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("", port))
                sock.close()
                return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")

    @staticmethod
    def _get_current_node_ip_and_free_port(
        start_port: int = 10000,
        consecutive: int = 1,
    ) -> Tuple[str, int]:
        """
        Get current node IP and a free port.

        Args:
            start_port: Port number to start searching from
            consecutive: Number of consecutive free ports needed

        Returns:
            Tuple of (ip_address, port)
        """
        ip = RayActor._get_current_node_ip()

        if consecutive <= 1:
            port = RayActor._get_free_port(start_port)
            return ip, port

        # Find consecutive free ports
        for port in range(start_port, start_port + 10000):
            all_free = True
            for offset in range(consecutive):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(("", port + offset))
                    sock.close()
                except OSError:
                    all_free = False
                    break
            if all_free:
                return ip, port

        raise RuntimeError(f"Could not find {consecutive} consecutive free ports")


class BaseTrainRayActor(RayActor):
    """
    Training Actor base class.

    Provides distributed training setup and common training operations.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
        """
        Initialize training actor.

        Args:
            world_size: Total number of processes
            rank: This process's rank
            master_addr: Master node address (required for rank > 0)
            master_port: Master node port (required for rank > 0)
        """
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self._is_distributed_initialized = False

    def _setup_distributed_env(self) -> None:
        """
        Set up distributed environment variables.

        This must be called before initializing the process group.
        """
        if self.master_addr is None or self.master_port is None:
            raise ValueError("master_addr and master_port must be set")

        os.environ["MASTER_ADDR"] = str(self.master_addr)
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)

        # Set local rank based on CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            # When using placement groups, CUDA_VISIBLE_DEVICES is typically set to a single GPU
            os.environ["LOCAL_RANK"] = "0"
        else:
            # No CUDA_VISIBLE_DEVICES set - this can happen in colocate mode
            # where actor has num_gpus=0 but still needs to use GPU
            device_count = torch.cuda.device_count()
            if device_count > 0:
                os.environ["LOCAL_RANK"] = str(self.rank % device_count)
            else:
                # No GPUs available, use 0 as default
                os.environ["LOCAL_RANK"] = "0"

        logger.info(
            f"Distributed env setup: rank={self.rank}, world_size={self.world_size}, "
            f"master={self.master_addr}:{self.master_port}"
        )

    def _init_distributed(self, backend: str = "nccl") -> None:
        """
        Initialize the distributed process group.

        Args:
            backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
        """
        if self._is_distributed_initialized:
            return

        self._setup_distributed_env()

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                world_size=self.world_size,
                rank=self.rank,
            )
            self._is_distributed_initialized = True
            logger.info(f"Rank {self.rank}: Distributed process group initialized")

    @abc.abstractmethod
    def init(self, config: dict) -> None:
        """
        Initialize model and optimizer.

        Args:
            config: Configuration dictionary containing model, optimizer, and loss configs
        """
        ...

    @abc.abstractmethod
    def train(self, rollout_id: int, data_ref) -> dict:
        """
        Execute one training step.

        Args:
            rollout_id: Current rollout iteration number
            data_ref: Ray ObjectRef containing training data

        Returns:
            Dictionary of training metrics
        """
        ...

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        ...

    @abc.abstractmethod
    def update_weights(self) -> None:
        """Broadcast weights from rank 0 to all other ranks."""
        ...

    @abc.abstractmethod
    def offload(self) -> None:
        """Offload model to CPU to free GPU memory."""
        ...

    @abc.abstractmethod
    def onload(self) -> None:
        """Load model back to GPU from CPU."""
        ...

    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            from diffusionrl.utils import clear_memory

            clear_memory()

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9

        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
        }

    def health_check(self) -> bool:
        """Check if actor is healthy and ready."""
        return True

    def get_rank(self) -> int:
        """Get this actor's rank."""
        return self.rank

    def get_world_size(self) -> int:
        """Get the total number of actors."""
        return self.world_size
