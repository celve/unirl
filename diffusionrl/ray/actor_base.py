"""
diffusionrl Ray Actor Base Classes.
"""
import abc
import logging
import os
from typing import Optional

import torch

from diffusionrl.ray.utils.net import get_free_port, get_node_ip

logger = logging.getLogger(__name__)


class RayActor:
    """Marker base class for all diffusionrl Ray actors.

    Stateless helpers previously attached here now live in
    ``diffusionrl.ray.utils`` (``net``, ``gpu``, ``node``).
    """


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

    def get_master_info(self, start_port: int = 10000) -> dict:
        """Return this actor's node IP and a free port for distributed master."""
        return {
            "master_addr": get_node_ip(),
            "master_port": int(get_free_port(start_port)),
        }

    def set_master_info(self, master_addr: str, master_port: int) -> None:
        """Override distributed master endpoint before process-group init."""
        self.master_addr = str(master_addr)
        self.master_port = int(master_port)

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
    def train(self, rollout_id: int, training_data_handle) -> dict:
        """
        Execute one training step.

        Args:
            rollout_id: Current rollout iteration number
            training_data_handle: Buffer-owned training-data handle for this actor.
                In the Ray-backed path this is typically an ObjectRef, but
                actor implementations may also receive already-materialized
                training batches.

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
