"""Distributed env + state mixin for Ray actors.

Cooperative-super mixin that owns the distributed scaffolding shared by
``TrainActor`` and ``RolloutActor``: rank / world / master endpoint state,
env-var writers, torch process-group init, and general actor utilities.

Each actor retains its own ``_setup_distributed_env`` method — the decision
of what WORLD_SIZE / RANK / LOCAL_RANK values to write differs between
multi-actor training (one rank per actor) and single-actor multi-GPU
rollout (rank 0 within the actor's own process group). The mixin exposes
``_write_distributed_env`` as the shared primitive both callers compose.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from diffusionrl.ray.utils.net import get_free_port, get_node_ip

logger = logging.getLogger(__name__)


class DistributedMixin:
    """Cooperative-super mixin: distributed state + env helpers + utilities."""

    def __init__(
        self,
        *,
        world_size: int = 1,
        rank: int = 0,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self._is_distributed_initialized = False

    # ------------------------------------------------------------------
    # Master endpoint bridging (used by the bootstrap to negotiate the
    # rank-0 node IP and port before process-group init on other ranks).
    # ------------------------------------------------------------------

    def get_master_info(self, start_port: int = 10000) -> Dict[str, Any]:
        """Return this actor's node IP and a free port for distributed master."""
        return {
            "master_addr": get_node_ip(),
            "master_port": int(get_free_port(start_port)),
        }

    def set_master_info(self, master_addr: str, master_port: int) -> None:
        """Override distributed master endpoint before process-group init."""
        self.master_addr = str(master_addr)
        self.master_port = int(master_port)

    # ------------------------------------------------------------------
    # Distributed env primitive + process-group init.
    # ------------------------------------------------------------------

    def _write_distributed_env(
        self,
        *,
        master_addr: str,
        master_port: int,
        world_size: int,
        rank: int,
        local_rank: int,
    ) -> None:
        """Write the five distributed env vars callers pass to torch."""
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

    def _init_distributed(self, backend: str = "nccl") -> None:
        """Initialize the torch.distributed process group (idempotent).

        Calls ``self._setup_distributed_env()`` first — subclasses provide
        their own implementation to compute the right WORLD_SIZE/RANK values.
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

    # ------------------------------------------------------------------
    # General actor utilities — delegated to by Ray remote calls.
    # ------------------------------------------------------------------

    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            from diffusionrl.utils import clear_memory

            clear_memory()

    def get_memory_info(self) -> Dict[str, float]:
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
