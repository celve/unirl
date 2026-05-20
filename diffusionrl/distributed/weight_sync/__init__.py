"""Weight-sync protocol variants + their handlers.

Each variant registers a dataclass under Hydra group ``sync`` whose
``_target_`` points at the handler class; ``TrainingWeightSyncMixin.setup_weight_sync``
calls ``build(cfg.sync, model=..., rollout_runtime=..., placement_cfg=...)`` to
instantiate the handler.
"""

from diffusionrl.distributed.weight_sync.base import (
    BucketedUpdateWeight,
    UpdateWeight,
)
from diffusionrl.distributed.weight_sync.checkpoint import (
    CheckpointSyncConfig,
    UpdateWeightFromCheckpoint,
)
from diffusionrl.distributed.weight_sync.ipc import (
    IPCBucketedSyncConfig,
    UpdateWeightFromIPC,
)
from diffusionrl.distributed.weight_sync.nccl import (
    NcclBroadcastSyncConfig,
    UpdateWeightFromDistributed,
)
from diffusionrl.distributed.weight_sync.tensor import (
    TensorPayloadSyncConfig,
    UpdateWeightFromTensor,
)

__all__ = [
    "BucketedUpdateWeight",
    "CheckpointSyncConfig",
    "IPCBucketedSyncConfig",
    "NcclBroadcastSyncConfig",
    "TensorPayloadSyncConfig",
    "UpdateWeight",
    "UpdateWeightFromCheckpoint",
    "UpdateWeightFromDistributed",
    "UpdateWeightFromIPC",
    "UpdateWeightFromTensor",
]
