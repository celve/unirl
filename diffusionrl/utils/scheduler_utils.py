"""Index schedulers used by GRPO-style algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np

Strategy = Literal["all", "progressive", "random", "decay", "exp_decay"]


@dataclass
class WindowConfig:
    """Configuration for stateless window-based index scheduling."""

    strategy: Strategy = "all"
    group_size: int = 4
    iters_per_group: int = 25
    init_timestep: int = 0
    overlap: bool = False
    overlap_step: int = 1
    roll_back: bool = False
    max_iters_per_group: Optional[int] = None
    min_iters_per_group: Optional[int] = None
    exp_decay_threshold: int = 13
    exp_decay_k: float = 0.1

    def __post_init__(self) -> None:
        if self.strategy == "decay":
            if self.max_iters_per_group is None:
                self.max_iters_per_group = self.iters_per_group
            if self.min_iters_per_group is None:
                self.min_iters_per_group = max(1, self.iters_per_group // 4)


class TimestepScheduler(ABC):
    """Abstract base class for stateless timestep-index schedulers."""

    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """Return the selected indices for the given step."""


def normalize_timestep_fraction(
    timestep_fraction: Union[float, Tuple[float, float], List[float]],
) -> Tuple[float, float]:
    """Normalize timestep_fraction to a ``(start, end)`` tuple."""
    if isinstance(timestep_fraction, (list, tuple)):
        if len(timestep_fraction) != 2:
            raise ValueError(
                f"timestep_fraction tuple must have exactly 2 elements, got {len(timestep_fraction)}"
            )
        start, end = float(timestep_fraction[0]), float(timestep_fraction[1])
    else:
        start, end = 0.0, float(timestep_fraction)
    if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0):
        raise ValueError(
            f"timestep_fraction values must be in [0.0, 1.0], got ({start}, {end})"
        )
    if start > end:
        raise ValueError(f"timestep_fraction start ({start}) must be <= end ({end})")
    return (start, end)


class AllSDEScheduler(TimestepScheduler):
    """Full-range index scheduler with optional range filtering and sparse sampling."""

    def __init__(
        self,
        num_timesteps: int,
        timestep_fraction: Union[float, Tuple[float, float]] = 1.0,
        num_sde_steps: Optional[int] = None,
    ):
        super().__init__(num_timesteps)
        self.timestep_fraction = timestep_fraction
        self.num_sde_steps = num_sde_steps
        self._fraction_start, self._fraction_end = normalize_timestep_fraction(
            timestep_fraction
        )
        self._effective_start = int(num_timesteps * self._fraction_start)
        self._effective_end = int(num_timesteps * self._fraction_end)
        if num_sde_steps is not None:
            pool_size = self._effective_end - self._effective_start
            if num_sde_steps > pool_size:
                raise ValueError(
                    f"num_sde_steps ({num_sde_steps}) exceeds available timesteps "
                    f"in fraction range [{self._effective_start}, {self._effective_end}) "
                    f"(pool_size={pool_size})"
                )
            if num_sde_steps <= 0:
                raise ValueError(f"num_sde_steps must be positive, got {num_sde_steps}")

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        pool = list(range(self._effective_start, self._effective_end))
        if self.num_sde_steps is None or self.num_sde_steps >= len(pool):
            return set(pool)
        seed = 0 if step is None else int(step)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(pool, size=self.num_sde_steps, replace=False)
        return set(int(i) for i in chosen)


class WindowScheduler(TimestepScheduler):
    """Stateless sliding-window index scheduler."""

    WINDOW_STRATEGY_TO_METHOD_NAME = {
        "all": None,
        "progressive": "_resolve_progressive",
        "random": "_resolve_random",
    }

    def __init__(self, num_timesteps: int, config: WindowConfig):
        super().__init__(num_timesteps)
        self.config = config
        if self.config.strategy not in self.WINDOW_STRATEGY_TO_METHOD_NAME:
            raise ValueError(
                f"Bad strategy configuration for WindowScheduler: {self.config.strategy}. "
                f"Available options: {set(self.WINDOW_STRATEGY_TO_METHOD_NAME.keys())}"
            )

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        if self.config.strategy == "all":
            return set(range(self.num_timesteps))
        resolve_method = getattr(
            self,
            self.WINDOW_STRATEGY_TO_METHOD_NAME[self.config.strategy],
        )
        return resolve_method(0 if step is None else int(step))

    def _resolve_progressive(self, step: int) -> Set[int]:
        group_step = step // self.config.iters_per_group
        stride = (
            self.config.overlap_step if self.config.overlap else self.config.group_size
        )
        remaining = (
            self.num_timesteps - self.config.init_timestep - self.config.group_size
        )
        num_one_round_group_steps = max(1, remaining // stride + 1)
        if group_step >= num_one_round_group_steps and not self.config.roll_back:
            group_step = num_one_round_group_steps - 1
            return self._resolve_progressive(group_step * self.config.iters_per_group)

        group_step = group_step % num_one_round_group_steps
        cur_timestep = self.config.init_timestep + group_step * stride
        return set(range(cur_timestep, cur_timestep + self.config.group_size))

    def _resolve_random(self, step: int) -> Set[int]:
        rng = np.random.default_rng(step)
        max_start = max(0, self.num_timesteps - self.config.group_size)
        cur_timestep = int(rng.integers(0, max_start + 1))
        return set(range(cur_timestep, cur_timestep + self.config.group_size))


SCHEDULER_REGISTRY: Dict[str, Type[TimestepScheduler]] = {
    "all": AllSDEScheduler,
    "window": WindowScheduler,
}


def create_indices_scheduler(
    *,
    scheduler_config: Any,
    num_timesteps: int,
) -> TimestepScheduler:
    """Create an index scheduler from a scheduler-config object or dict."""

    if isinstance(scheduler_config, dict):

        def _read(name: str, default: Any) -> Any:
            return scheduler_config.get(name, default)

    else:

        def _read(name: str, default: Any) -> Any:
            return getattr(scheduler_config, name, default)

    scheduler_type = str(_read("timestep_strategy", "all"))
    if scheduler_type == "all":
        return AllSDEScheduler(
            num_timesteps=int(num_timesteps),
            timestep_fraction=_read("timestep_fraction", 1.0),
            num_sde_steps=_read("num_sde_steps", None),
        )
    if scheduler_type == "window":
        return WindowScheduler(
            int(num_timesteps),
            WindowConfig(
                strategy=_read("window_strategy", "progressive"),
                group_size=int(_read("window_group_size", 4)),
                iters_per_group=int(_read("window_iters_per_group", 25)),
                init_timestep=int(_read("window_init_timestep", 0)),
                overlap=bool(_read("window_overlap", False)),
                overlap_step=int(_read("window_overlap_step", 1)),
                roll_back=bool(_read("window_roll_back", False)),
                max_iters_per_group=_read("window_max_iters_per_group", None),
                min_iters_per_group=_read("window_min_iters_per_group", None),
            ),
        )
    raise ValueError(
        f"Unknown scheduler_type: {scheduler_type}. Available: {list(SCHEDULER_REGISTRY.keys())}"
    )
