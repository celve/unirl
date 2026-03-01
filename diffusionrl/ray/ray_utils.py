"""
diffusionrl Ray Utilities.

Merges distributed helpers and training-actor sampling support into a
single module:
- Lock utilities for distributed synchronization
- Helper functions for Ray operations
- Resource management utilities
- Training-actor sampling executor & service
"""
import asyncio
import logging
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import torch
import torch.nn as nn

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


# ============================================================
# Training-actor sampling helpers (merged from actor_sampling.py)
# ============================================================

from diffusionrl.types.sampling import RolloutRequest, RolloutOutput
from diffusionrl.samplers.fsdp import sampler_runner

logger = logging.getLogger(__name__)


@contextmanager
def sampling_eval_context(modules: List[nn.Module]):
    """Temporarily switch modules to eval/no-grad and restore training flags.

    Important: do not mutate ``requires_grad`` flags here. Under FSDP, a no-grad
    forward can replace/rebuild parameter objects, which makes post-hoc restoration
    of saved Parameter references unreliable and can leave trainable params frozen.
    """
    original_modes = [(module, bool(module.training)) for module in modules if isinstance(module, nn.Module)]
    for module, _ in original_modes:
        module.eval()

    try:
        # Use no_grad instead of inference_mode to avoid FSDP grad_fn/AccumulateGrad
        # assertion failures when returning to training.
        with torch.no_grad():
            yield
    finally:
        for module, was_training in original_modes:
            module.train(was_training)


def _tensor_to_pil(images: torch.Tensor) -> List[Any]:
    """Convert tensor to PIL images; video tensors use middle frame."""
    from PIL import Image
    import numpy as np

    pil_images = []
    images = images.cpu()

    if images.dim() == 5:
        frame_count = images.shape[2]
        images = images[:, :, frame_count // 2]

    for img in images:
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))

    return pil_images


class ActorSamplingExecutor:
    """Sampling executor used by TrainingActor RPC boundary."""

    def iter_reflection_modules(
        self,
        obj: Any,
        *,
        include_transformer: bool,
    ) -> List[Tuple[str, nn.Module]]:
        """Collect likely offloadable submodules from arbitrary objects via reflection."""
        return sampler_runner.iter_offloadable_modules(
            obj, include_transformer=include_transformer
        )

    def ensure_sampling_components(self, actor: Any) -> None:
        if actor._sampling_ready:
            return
        if actor.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")

        try:
            actor.model_bundle.load_aux_components()
        except Exception as e:
            logger.warning("Failed to load auxiliary components: %s", e)
            raise

        actor.text_encoder = getattr(actor.model_bundle, "text_encoder", None)
        actor.vae = getattr(actor.model_bundle, "vae", None)
        actor.scheduler = getattr(actor.model_bundle, "scheduler", None)

        sampler_path = actor._sampling_config.get("sampler_path")
        if not sampler_path:
            raise ValueError("sampling_config must provide sampler_path for training-actor sampling")

        sampler_kwargs = dict(actor._sampling_config.get("sampler_kwargs", {}))
        actor._sampler = sampler_runner.create_sampler(
            sampler_path=sampler_path,
            model=actor.model,
            text_encoder=actor.text_encoder,
            vae=actor.vae,
            eta=actor._sampling_config.get("eta", 1.0),
            sde_type=actor._sampling_config.get("sde_type", "sde"),
            shift=actor._sampling_config.get("shift", 3.0),
            model_bundle=actor.model_bundle,
            **sampler_kwargs,
        )

        actor._sampling_ready = True

    def _encode_prompt(self, actor: Any, prompts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        return sampler_runner.encode_prompt(actor.model_bundle, prompts, **kwargs)

    def _decode_latents(self, actor: Any, latents: torch.Tensor) -> torch.Tensor:
        return sampler_runner.decode_latents(actor.vae, latents)

    def _iter_sampling_mode_modules(self, actor: Any) -> List[nn.Module]:
        modules: List[nn.Module] = []
        seen: Set[int] = set()
        for component in (actor.model, actor.text_encoder, actor.vae):
            if isinstance(component, nn.Module):
                ident = id(component)
                if ident not in seen:
                    modules.append(component)
                    seen.add(ident)

        if actor.model_bundle is not None and hasattr(actor.model_bundle, "iter_offloadable_modules"):
            for _name, component in actor.model_bundle.iter_offloadable_modules(include_transformer=True):
                if isinstance(component, nn.Module):
                    ident = id(component)
                    if ident not in seen:
                        modules.append(component)
                        seen.add(ident)
        return modules

    @contextmanager
    def _sampling_eval_context(self, actor: Any):
        modules = self._iter_sampling_mode_modules(actor)
        with sampling_eval_context(modules):
            yield

    def generate(self, actor: Any, request: RolloutRequest) -> RolloutOutput:
        if not actor._is_initialized:
            raise RuntimeError("Actor not initialized. Call init() first.")

        if actor._is_offloaded:
            actor.onload()

        self.ensure_sampling_components(actor)

        # Extract fields and apply config defaults
        prompts = request.prompts
        prompt_embeds = request.prompt_embeds
        pooled_prompt_embeds = request.pooled_prompt_embeds
        encoder_attention_mask = request.encoder_attention_mask
        text_ids = request.text_ids
        kwargs = dict(request.kwargs)

        num_inference_steps = request.num_inference_steps or actor._sampling_config.get("num_inference_steps", 50)
        guidance_scale = request.guidance_scale if request.guidance_scale is not None else actor._sampling_config.get("guidance_scale", 7.5)
        height = request.height or actor._sampling_config.get("height", 256)
        width = request.width or actor._sampling_config.get("width", 256)
        num_frames = request.num_frames or actor._sampling_config.get("num_frames", 16)

        sampling_adapter = request.sampling_adapter
        if sampling_adapter is None:
            sampling_adapter = actor._sampling_config.get("sampling_adapter")

        init_same_noise = kwargs.pop("init_same_noise", actor._sampling_config.get("init_same_noise", False))
        num_samples_per_prompt = kwargs.pop(
            "num_samples_per_prompt",
            actor._sampling_config.get("num_samples_per_prompt", 1),
        )

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=actor._device)
            generator.manual_seed(request.seed)

        with self._sampling_eval_context(actor):
            if prompts is not None and prompt_embeds is None:
                encoded = self._encode_prompt(actor, prompts)
                prompt_embeds = encoded.get("prompt_embeds")
                pooled_prompt_embeds = encoded.get("pooled_prompt_embeds", pooled_prompt_embeds)
                negative_prompt_embeds = encoded.get("negative_prompt_embeds")
                negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")
                if text_ids is None:
                    text_ids = encoded.get("text_ids")
            else:
                negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
                negative_pooled_prompt_embeds = kwargs.pop("negative_pooled_prompt_embeds", None)

            output = sampler_runner.run_sample(
                model=actor.model,
                sampler=actor._sampler,
                sampling_adapter=sampling_adapter,
                prompts=prompts,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                encoder_attention_mask=encoder_attention_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                sde_indices=request.sde_indices,
                text_ids=text_ids,
                init_same_noise=init_same_noise,
                num_samples_per_prompt=num_samples_per_prompt,
                **kwargs,
            )

        if request.decode_for_reward:
            try:
                decoded = self._decode_latents(actor, output.latents)
                decoded_images = _tensor_to_pil(decoded)
                output = RolloutOutput(
                    latents=output.latents,
                    timesteps=output.timesteps,
                    trajectories=output.trajectories,
                    log_probs=output.log_probs,
                    embeddings=output.embeddings,
                    decoded_images=decoded_images,
                    metadata=output.metadata,
                    step_indices=output.step_indices,
                )
            except Exception as e:
                logger.warning("Failed to decode latents: %s", e)

        return output.to_device("cpu")

    def generate_batch(self, actor: Any, requests: List[RolloutRequest]) -> List[RolloutOutput]:
        return [self.generate(actor, req) for req in requests]

class TrainingActorSamplingService:
    """Delegates training-actor sampling RPCs to ActorSamplingExecutor."""

    def __init__(self, executor: Optional[ActorSamplingExecutor] = None) -> None:
        self._executor = executor or ActorSamplingExecutor()

    @property
    def executor(self) -> ActorSamplingExecutor:
        return self._executor

    def generate(self, actor, request: RolloutRequest) -> RolloutOutput:
        return self._executor.generate(actor, request)

    def generate_batch(self, actor, requests: List[RolloutRequest]) -> List[RolloutOutput]:
        return self._executor.generate_batch(actor, requests)
