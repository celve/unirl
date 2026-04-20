"""Training execution helpers and backend integration."""

from diffusionrl.training.backends import (
    FSDPBackend,
    FSDPBackendConfig,
    LRSchedulerProtocol,
    OptimizerProtocol,
    TrainBackend,
    TrainBackendConfig,
    VeOmniBackend,
    VeOmniBackendConfig,
)
from diffusionrl.training.batch_partition import shard_training_batch_for_rank
from diffusionrl.training.factories import build_lr_scheduler, build_optimizer
from diffusionrl.training.train_executor import TrainExecutor, TrainExecutorConfig
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
from diffusionrl.training.update_schedule import (
    TrainingUpdateChunk,
    TrainingUpdateSchedule,
    create_training_update_schedule,
    validate_batch_against_plan,
)
from diffusionrl.training.workflow import TrainingWorkflow

__all__ = [
    "TrainExecutor",
    "TrainExecutorConfig",
    "TrainingWorkflow",
    "build_lr_scheduler",
    "build_optimizer",
    "shard_training_batch_for_rank",
    "TrainingUpdateChunk",
    "TrainingUpdateSchedule",
    "create_training_update_schedule",
    "validate_batch_against_plan",
    "TrainBackend",
    "TrainBackendConfig",
    "OptimizerProtocol",
    "LRSchedulerProtocol",
    "FSDPBackend",
    "FSDPBackendConfig",
    "VeOmniBackend",
    "VeOmniBackendConfig",
    "BaseTrainBackendConfig",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "resolve_train_backend_capabilities",
    "derive_train_backend_capabilities",
    "derive_train_backend_launch_spec",
    "supported_train_backends",
]
