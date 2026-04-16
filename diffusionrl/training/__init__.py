"""Training execution helpers and backend integration."""

from diffusionrl.training.backends import (
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
    create_train_backend_from_init_payload,
    derive_train_backend_capabilities,
    supported_train_backends,
)
from diffusionrl.training.batch_partition import shard_training_batch_for_rank
from diffusionrl.training.train_executor import TrainExecutor, TrainExecutorConfig
from diffusionrl.training.update_schedule import (
    TrainingUpdateChunk,
    TrainingUpdateSchedule,
    create_training_update_schedule,
    validate_batch_against_plan,
    validate_training_plan,
)
from diffusionrl.training.workflow import TrainingWorkflow

__all__ = [
    "TrainExecutor",
    "TrainExecutorConfig",
    "TrainingWorkflow",
    "shard_training_batch_for_rank",
    "TrainingUpdateChunk",
    "TrainingUpdateSchedule",
    "validate_training_plan",
    "create_training_update_schedule",
    "validate_batch_against_plan",
    "TrainBackend",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "create_train_backend_from_init_payload",
    "derive_train_backend_capabilities",
    "supported_train_backends",
]
