"""Training execution helpers and backend integration."""

from diffusionrl.training.backends import (
    FSDPBackend,
    FSDPBackendConfig,
    LRSchedulerProtocol,
    OptimizerProtocol,
    TrainBackend,
    TrainBackendConfig,
)
from diffusionrl.training.batch_partition import shard_training_batch_for_rank
from diffusionrl.training.factories import build_lr_scheduler, build_optimizer
from diffusionrl.training.types import (
    BaseTrainBackendConfig,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
    derive_train_backend_capabilities,
    derive_train_backend_launch_spec,
    resolve_train_backend_capabilities,
    supported_train_backends,
)

__all__ = [
    "build_lr_scheduler",
    "build_optimizer",
    "shard_training_batch_for_rank",
    "TrainBackend",
    "TrainBackendConfig",
    "OptimizerProtocol",
    "LRSchedulerProtocol",
    "FSDPBackend",
    "FSDPBackendConfig",
    "BaseTrainBackendConfig",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "resolve_train_backend_capabilities",
    "derive_train_backend_capabilities",
    "derive_train_backend_launch_spec",
    "supported_train_backends",
]
