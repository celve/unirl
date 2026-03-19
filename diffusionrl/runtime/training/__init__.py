"""Pure training execution runtime helpers (Ray-agnostic)."""

from diffusionrl.runtime.training.train_executor import (
    TrainExecutor,
    TrainExecutorConfig,
)
from diffusionrl.runtime.training.batch_partition import (
    partition_training_batch,
    shard_training_batch_for_rank,
)
from diffusionrl.runtime.training.update_schedule import (
    TrainingExecutionPlan,
    TrainingUpdateChunk,
    TrainingUpdateSchedule,
    coerce_training_execution_plan,
    create_training_update_schedule,
    validate_batch_against_plan,
)
from diffusionrl.runtime.training.backends import (
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
    create_train_backend,
    resolve_train_backend_capabilities,
    supported_train_backends,
)

__all__ = [
    "TrainExecutor",
    "TrainExecutorConfig",
    "partition_training_batch",
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
    "supported_train_backends",
]
