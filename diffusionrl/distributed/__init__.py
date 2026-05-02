"""Distributed coordination helpers.

Contract:

- ``distributed/`` owns distributed coordination semantics and sync protocols
- it may depend on the active transport/runtime boundary when needed
- it must not own Ray actors, group construction, placement, or business workflow
"""

from diffusionrl.distributed.weight_sync import (
    BucketedUpdateWeight,
    CheckpointSyncConfig,
    NcclBroadcastSyncConfig,
    TensorPayloadSyncConfig,
    UpdateWeight,
    UpdateWeightFromCheckpoint,
    UpdateWeightFromDistributed,
    UpdateWeightFromTensor,
)
from diffusionrl.distributed.weight_sync_checkpoint import (
    checkpoint_ready_marker_path,
    cleanup_published_checkpoint,
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
    wait_for_published_checkpoint,
)

__all__ = [
    "BucketedUpdateWeight",
    "CheckpointSyncConfig",
    "NcclBroadcastSyncConfig",
    "TensorPayloadSyncConfig",
    "UpdateWeight",
    "UpdateWeightFromCheckpoint",
    "UpdateWeightFromDistributed",
    "UpdateWeightFromTensor",
    "checkpoint_ready_marker_path",
    "cleanup_published_checkpoint",
    "publish_checkpoint_atomic",
    "publish_sglang_transformer_checkpoint_atomic",
    "wait_for_published_checkpoint",
]
