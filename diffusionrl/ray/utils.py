"""
diffusionrl Ray Distributed Utilities.

Provides utilities for Ray distributed operations:
- Lock utilities for distributed synchronization
- Helper functions for Ray operations
- Resource management utilities
"""
import asyncio
import logging
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import ray
from ray.util.placement_group import PlacementGroup

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Environment variable names that prevent Ray from overriding device visibility.
# Setting these to "1" lets the engine manage CUDA_VISIBLE_DEVICES manually.
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


# ============================================================
# Lock Utilities
# ============================================================


@ray.remote(num_cpus=0)
class DistributedLock:
    """
    Distributed lock implemented as a Ray actor.

    Provides mutual exclusion across distributed workers.
    """

    def __init__(self):
        self._locked = False
        self._holder: Optional[str] = None
        self._waiters: List[str] = []

    def acquire(self, holder_id: str, timeout: Optional[float] = None) -> bool:
        """
        Attempt to acquire the lock.

        Args:
            holder_id: Unique identifier for the lock holder
            timeout: Maximum time to wait (None for non-blocking)

        Returns:
            True if lock was acquired, False otherwise
        """
        start_time = time.time()

        while True:
            if not self._locked:
                self._locked = True
                self._holder = holder_id
                logger.debug(f"Lock acquired by {holder_id}")
                return True

            if timeout is None:
                return False

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return False

            # Wait a bit and retry
            time.sleep(0.01)

    def release(self, holder_id: str) -> bool:
        """
        Release the lock.

        Args:
            holder_id: Identifier of the holder releasing the lock

        Returns:
            True if lock was released, False if not held by this holder
        """
        if self._holder != holder_id:
            logger.warning(f"Lock release failed: held by {self._holder}, not {holder_id}")
            return False

        self._locked = False
        self._holder = None
        logger.debug(f"Lock released by {holder_id}")
        return True

    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self._locked

    def get_holder(self) -> Optional[str]:
        """Get the current lock holder."""
        return self._holder


class LockContext:
    """Context manager for distributed lock."""

    def __init__(self, lock_actor: ray.actor.ActorHandle, holder_id: str):
        self.lock_actor = lock_actor
        self.holder_id = holder_id
        self._acquired = False

    def __enter__(self):
        self._acquired = ray.get(self.lock_actor.acquire.remote(self.holder_id, timeout=60.0))
        if not self._acquired:
            raise RuntimeError(f"Failed to acquire lock for {self.holder_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            ray.get(self.lock_actor.release.remote(self.holder_id))


def create_distributed_lock(name: str = "grpo_lock") -> ray.actor.ActorHandle:
    """
    Create a named distributed lock.

    Args:
        name: Name for the lock actor

    Returns:
        Lock actor handle
    """
    return DistributedLock.options(name=name, lifetime="detached").remote()


def get_distributed_lock(name: str = "grpo_lock") -> ray.actor.ActorHandle:
    """
    Get an existing distributed lock by name.

    Args:
        name: Name of the lock actor

    Returns:
        Lock actor handle
    """
    return ray.get_actor(name)


# ============================================================
# Resource Utilities
# ============================================================


@dataclass
class GPUInfo:
    """Information about a GPU."""
    node_ip: str
    gpu_id: int
    device_name: str = ""
    memory_total: int = 0
    memory_free: int = 0


@dataclass
class NodeInfo:
    """Information about a Ray node."""
    node_ip: str
    num_gpus: int
    num_cpus: int
    memory_bytes: int
    gpus: List[GPUInfo] = field(default_factory=list)


def get_node_info() -> NodeInfo:
    """
    Get information about the current node.

    Returns:
        NodeInfo with current node details
    """
    node_ip = get_node_ip()

    # Get GPU info
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_info = torch.cuda.mem_get_info(i)
                gpus.append(GPUInfo(
                    node_ip=node_ip,
                    gpu_id=i,
                    device_name=props.name,
                    memory_total=props.total_memory,
                    memory_free=mem_info[0],
                ))
    except ImportError:
        pass

    # Get CPU count
    num_cpus = os.cpu_count() or 1

    # Get memory (basic)
    memory_bytes = 0
    try:
        import psutil
        memory_bytes = psutil.virtual_memory().total
    except ImportError:
        pass

    return NodeInfo(
        node_ip=node_ip,
        num_gpus=len(gpus),
        num_cpus=num_cpus,
        memory_bytes=memory_bytes,
        gpus=gpus,
    )


def get_node_ip() -> str:
    """
    Get the IP address of the current node.

    Returns:
        Node IP address string
    """
    # Try Ray's method first
    try:
        return ray.util.get_node_ip_address()
    except Exception:
        pass

    # Fallback to socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_free_port(start_port: int = 10000, max_tries: int = 1000) -> int:
    """
    Find a free port starting from start_port.

    Args:
        start_port: Port to start searching from
        max_tries: Maximum number of ports to try

    Returns:
        Free port number
    """
    for port in range(start_port, start_port + max_tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_tries}")


def get_consecutive_free_ports(start_port: int = 10000, count: int = 1) -> List[int]:
    """
    Find consecutive free ports.

    Args:
        start_port: Port to start searching from
        count: Number of consecutive ports needed

    Returns:
        List of consecutive free port numbers
    """
    for base_port in range(start_port, start_port + 10000):
        ports = []
        all_free = True

        for offset in range(count):
            port = base_port + offset
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", port))
                s.close()
                ports.append(port)
            except OSError:
                all_free = False
                break

        if all_free:
            return ports

    raise RuntimeError(f"Could not find {count} consecutive free ports starting from {start_port}")


# ============================================================
# Ray Operation Utilities
# ============================================================


def ray_get_with_retry(
    ref: ray.ObjectRef,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Any:
    """
    Get Ray object with retry on failure.

    Args:
        ref: Ray ObjectRef
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries

    Returns:
        The object value
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return ray.get(ref)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"ray.get failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                time.sleep(retry_delay)
            else:
                raise

    raise last_error


def ray_wait_with_progress(
    refs: List[ray.ObjectRef],
    num_returns: int = 1,
    timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple:
    """
    Wait for Ray objects with progress callback.

    Args:
        refs: List of ObjectRefs to wait for
        num_returns: Number of refs to wait for
        timeout: Timeout in seconds
        progress_callback: Callback(completed, total) for progress updates

    Returns:
        Tuple of (ready, remaining) ObjectRefs
    """
    ready = []
    remaining = list(refs)
    total = len(refs)
    start_time = time.time()

    while len(ready) < num_returns and remaining:
        # Calculate remaining timeout
        elapsed = time.time() - start_time
        wait_timeout = None
        if timeout is not None:
            wait_timeout = max(0, timeout - elapsed)
            if wait_timeout <= 0:
                break

        # Wait for next completion
        newly_ready, remaining = ray.wait(remaining, num_returns=1, timeout=wait_timeout)
        ready.extend(newly_ready)

        # Progress callback
        if progress_callback:
            progress_callback(len(ready), total)

    return ready, remaining


async def ray_get_async(ref: ray.ObjectRef) -> Any:
    """
    Async version of ray.get.

    Args:
        ref: Ray ObjectRef

    Returns:
        The object value
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ray.get, ref)


def batch_ray_get(
    refs: List[ray.ObjectRef],
    batch_size: int = 10,
) -> List[Any]:
    """
    Get Ray objects in batches to avoid memory pressure.

    Args:
        refs: List of ObjectRefs
        batch_size: Number of refs to get at once

    Returns:
        List of object values
    """
    results = []

    for i in range(0, len(refs), batch_size):
        batch = refs[i:i + batch_size]
        batch_results = ray.get(batch)
        results.extend(batch_results)

    return results


# ============================================================
# Placement Group Utilities
# ============================================================


def wait_for_placement_group(
    pg: PlacementGroup,
    timeout: float = 300.0,
) -> bool:
    """
    Wait for placement group to be ready with timeout.

    Args:
        pg: PlacementGroup to wait for
        timeout: Timeout in seconds

    Returns:
        True if ready, False if timeout
    """
    try:
        ray.get(pg.ready(), timeout=timeout)
        return True
    except ray.exceptions.GetTimeoutError:
        return False


def get_placement_group_bundles(pg: PlacementGroup) -> List[Dict[str, float]]:
    """
    Get bundle specifications from a placement group.

    Args:
        pg: PlacementGroup

    Returns:
        List of bundle dicts
    """
    # This is a bit hacky but works
    bundle_specs = pg.bundle_specs
    return bundle_specs


# ============================================================
# Timing Utilities
# ============================================================


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.3f}s")

    def reset(self):
        self.start_time = time.time()
        self.elapsed = 0.0


@contextmanager
def timed(name: str = ""):
    """Context manager for timing code blocks."""
    timer = Timer(name)
    with timer:
        yield timer


# ============================================================
# Memory Utilities
# ============================================================


def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def get_gpu_memory_usage() -> Dict[int, Dict[str, int]]:
    """
    Get GPU memory usage for all devices.

    Returns:
        Dict mapping device_id to {allocated, reserved, total}
    """
    result = {}

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                result[i] = {
                    "allocated": torch.cuda.memory_allocated(i),
                    "reserved": torch.cuda.memory_reserved(i),
                    "total": torch.cuda.get_device_properties(i).total_memory,
                }
    except ImportError:
        pass

    return result


def log_gpu_memory_usage(prefix: str = ""):
    """Log current GPU memory usage."""
    usage = get_gpu_memory_usage()

    for device_id, mem in usage.items():
        allocated_gb = mem["allocated"] / 1e9
        reserved_gb = mem["reserved"] / 1e9
        total_gb = mem["total"] / 1e9
        logger.info(
            f"{prefix}GPU {device_id}: "
            f"allocated={allocated_gb:.2f}GB, "
            f"reserved={reserved_gb:.2f}GB, "
            f"total={total_gb:.2f}GB"
        )


# ============================================================
# Actor Health Check Utilities
# ============================================================


def check_actor_health(actor: ray.actor.ActorHandle) -> bool:
    """
    Check if a Ray actor is healthy.

    Args:
        actor: Ray actor handle

    Returns:
        True if actor is healthy
    """
    try:
        # Try to call a simple method
        if hasattr(actor, "health_check"):
            ray.get(actor.health_check.remote(), timeout=10.0)
        else:
            # Just check if actor is alive
            ray.get(ray.put(1), timeout=10.0)
        return True
    except Exception as e:
        logger.warning(f"Actor health check failed: {e}")
        return False


def wait_for_actors_ready(
    actors: List[ray.actor.ActorHandle],
    timeout: float = 300.0,
) -> bool:
    """
    Wait for all actors to be ready.

    Args:
        actors: List of actor handles
        timeout: Timeout in seconds

    Returns:
        True if all actors are ready
    """
    start_time = time.time()

    for actor in actors:
        remaining = timeout - (time.time() - start_time)
        if remaining <= 0:
            return False

        try:
            if hasattr(actor, "health_check"):
                ray.get(actor.health_check.remote(), timeout=remaining)
            else:
                # Default: just try to get a response
                ray.get(ray.put(1), timeout=remaining)
        except Exception as e:
            logger.warning(f"Actor not ready: {e}")
            return False

    return True
