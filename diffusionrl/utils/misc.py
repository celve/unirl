"""
diffusionrl Utilities - Miscellaneous utility functions.

Reference: slime/utils/misc.py
"""
import importlib
import logging
import os
import random
import gc
from typing import Any, Callable, Optional, Type, TypeVar

import numpy as np
import torch


logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_function(path: str) -> Any:
    """
    Dynamically load a class or function from a module path.

    Args:
        path: Full path to the class/function, e.g., "diffusionrl.algorithms.grpo.GRPOAlgorithm"

    Returns:
        The loaded class or function

    Example:
        >>> algorithm_cls = load_function("diffusionrl.algorithms.grpo.GRPOAlgorithm")
        >>> algorithm = algorithm_cls(clip_range=1e-4)
    """
    if path is None or path == "":
        raise ValueError("Path cannot be None or empty")

    parts = path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid path format: {path}. Expected 'module.path.ClassName'")

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Module '{module_path}' has no attribute '{class_name}'")

    return cls


def load_class(path: str, base_class: Optional[Type[T]] = None) -> Type[T]:
    """
    Load a class from a module path with optional base class validation.

    Args:
        path: Full path to the class
        base_class: Optional base class to validate against

    Returns:
        The loaded class

    Raises:
        TypeError: If loaded class is not a subclass of base_class
    """
    cls = load_function(path)

    if base_class is not None and not issubclass(cls, base_class):
        raise TypeError(f"Loaded class {cls} is not a subclass of {base_class}")

    return cls


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logger(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the training run.

    Args:
        level: Logging level
        format_str: Log format string
        log_file: Optional file path to write logs
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )

    # Reduce verbosity of some libraries
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_rank() -> int:
    """Get the rank of the current process in distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def get_device() -> torch.device:
    """Get the appropriate device for the current process."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_val / (1024 ** 3)


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.

    Returns:
        Dict with total, allocated, cached, and free memory in GB
    """
    if not torch.cuda.is_available():
        return {}

    device = get_device()
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    cached = torch.cuda.memory_reserved(device)
    free = total - allocated

    return {
        "total_gb": bytes_to_gb(total),
        "allocated_gb": bytes_to_gb(allocated),
        "cached_gb": bytes_to_gb(cached),
        "free_gb": bytes_to_gb(free),
    }


def clear_memory() -> None:
    """Clear GPU memory cache with synchronization and GC."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int) -> str:
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


class Timer:
    """Simple timer context manager for profiling."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()

        return self

    def __exit__(self, *args):
        if isinstance(self.start_time, torch.cuda.Event):
            self.end_time.record()
            torch.cuda.synchronize()
            self.elapsed = self.start_time.elapsed_time(self.end_time) / 1000.0  # Convert to seconds
        else:
            import time
            self.elapsed = time.time() - self.start_time

        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.4f}s")


def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely divide two tensors, avoiding division by zero."""
    return a / (b + eps)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_callable_name(fn: Callable) -> str:
    """Get the fully qualified name of a callable."""
    module = getattr(fn, "__module__", "")
    name = getattr(fn, "__name__", str(fn))
    return f"{module}.{name}" if module else name
