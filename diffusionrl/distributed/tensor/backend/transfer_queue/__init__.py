"""TransferQueue subsystem: typed config + driver bootstrap + actor bridge.

Each backend variant registers a dataclass under Hydra group ``transfer_queue``
whose ``_target_`` points at a ``Backend`` subclass; ``TransferQueueRuntime.init``
calls ``build(cfg.transfer_queue)`` to instantiate it. Disabled =
``cfg.transfer_queue`` is absent (no defaults entry).
"""

from diffusionrl.distributed.tensor.backend.transfer_queue.base import Backend
from diffusionrl.distributed.tensor.backend.transfer_queue.mooncake import (
    MooncakeBackend,
    MooncakeBackendConfig,
    MooncakeZeroCopyConfig,
)
from diffusionrl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime
from diffusionrl.distributed.tensor.backend.transfer_queue.simple import (
    SimpleBackend,
    SimpleBackendConfig,
)
from diffusionrl.distributed.tensor.backend.transfer_queue.transport import TQTransport

__all__ = [
    "Backend",
    "MooncakeBackend",
    "MooncakeBackendConfig",
    "MooncakeZeroCopyConfig",
    "SimpleBackend",
    "SimpleBackendConfig",
    "TQTransport",
    "TransferQueueRuntime",
]
