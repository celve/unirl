"""Training execution helpers and backend integration."""

from diffusionrl.training.train_executor import (
    TrainExecutor,
    TrainExecutorConfig,
)
from diffusionrl.training.batch_partition import (
    shard_training_batch_for_rank,
)
from diffusionrl.training.update_schedule import (
    TrainingExecutionPlan,
    TrainingUpdateChunk,
    TrainingUpdateSchedule,
    coerce_training_execution_plan,
    create_training_update_schedule,
    validate_batch_against_plan,
)
from diffusionrl.training.backends import (
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
    create_train_backend,
    resolve_train_backend_capabilities,
    resolve_train_backend_capabilities_from_args,
    supported_train_backends,
)

__all__ = [
    "TrainExecutor",
    "TrainExecutorConfig",
    "shard_training_batch_for_rank",
    "TrainingExecutionPlan",
    "TrainingUpdateChunk",
    "TrainingUpdateSchedule",
    "coerce_training_execution_plan",
    "create_training_update_schedule",
    "validate_batch_against_plan",
    "TrainBackend",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "create_train_backend",
    "resolve_train_backend_capabilities",
    "resolve_train_backend_capabilities_from_args",
    "supported_train_backends",
]
