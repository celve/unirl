"""Audited worker RNG path — one derivation from the run seed to every worker draw.

The recipes' seed fields reach data order and the rollout engine only, so model
materialization and LoRA adapter initialization draw from whatever global RNG state
the process happens to hold. Two runs configured with different seed IDs can
therefore share an initialization, and one run is not reproducible on rerun. This
module makes the worker-side draws a function of (run seed, rank, purpose) and
records what it seeded so a run's archive can show it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MASK63 = (1 << 63) - 1

# What this process seeded, in call order — archived as the run's RNG provenance.
_SEEDING_LOG: List[Dict[str, Any]] = []


def derive_worker_seed(base_seed: int, *, rank: int, purpose: str) -> int:
    """Stable int63 seed from the run seed, worker rank and a purpose label."""
    digest = hashlib.sha256(f"{int(base_seed)}:{int(rank)}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & _MASK63


def current_rank() -> int:
    """This worker's global rank, from torch.distributed when initialized, else the env."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # pragma: no cover - torch.distributed optional at import time
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def seed_worker_rngs(base_seed: Optional[int], *, purpose: str, rank: Optional[int] = None) -> Optional[int]:
    """Seed python/numpy/torch global RNGs for one purpose; returns the derived seed.

    ``base_seed=None`` is the framework's documented "draw from OS entropy" contract,
    so it is recorded and left alone rather than forced to a default.
    """
    resolved_rank = current_rank() if rank is None else int(rank)
    if base_seed is None:
        _SEEDING_LOG.append({"purpose": purpose, "rank": resolved_rank, "base_seed": None, "derived_seed": None})
        logger.info("seed_worker_rngs(%s): base seed is null — OS entropy, run is not reproducible", purpose)
        return None

    seed = derive_worker_seed(int(base_seed), rank=resolved_rank, purpose=purpose)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed & 0xFFFFFFFF)
    except Exception:  # pragma: no cover - numpy is a base dependency, guard is for partial envs
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _SEEDING_LOG.append(
        {"purpose": purpose, "rank": resolved_rank, "base_seed": int(base_seed), "derived_seed": seed}
    )
    logger.info("seed_worker_rngs(%s): rank=%d base=%d derived=%d", purpose, resolved_rank, int(base_seed), seed)
    return seed


def seeding_provenance() -> List[Dict[str, Any]]:
    """Everything this process seeded, in call order, for the run's archive."""
    return list(_SEEDING_LOG)


def reset_seeding_provenance() -> None:
    """Clear the record — for tests that assert on a single seeding sequence."""
    _SEEDING_LOG.clear()


__all__ = [
    "current_rank",
    "derive_worker_seed",
    "reset_seeding_provenance",
    "seed_worker_rngs",
    "seeding_provenance",
]
