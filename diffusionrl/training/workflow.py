"""Train-side workflow that stays independent of Ray actor plumbing."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from diffusionrl.types.training_batch import BackwardTrainingBatch, TrainingBatch


class TrainingWorkflow:
    """Own the train-side business chain after actor-side batch materialization.

    The actor keeps process-local concerns such as ObjectRef resolution, device
    state logging, backend initialization, and lifecycle management. This
    workflow owns the train-side business path on a materialized batch:

    - build the train executor
    - prepare/shard/move batch to device
    - skip when the current rank has no local shard
    - optional log-prob replay
    - optional backend-provided train step
    - fallback to the shared TrainExecutor update loop
    """

    def execute(
        self,
        *,
        rollout_id: int,
        batch: TrainingBatch,
        build_executor: Callable[[], Any],
        replay_batch: Optional[Callable[[BackwardTrainingBatch], BackwardTrainingBatch]] = None,
        backend_train_step: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
        on_prepared_batch: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        executor = build_executor()
        current_batch = executor.prepare_batch(batch)
        if current_batch is None:
            return executor.skipped_metrics(rollout_id)

        if on_prepared_batch is not None:
            on_prepared_batch()

        if isinstance(current_batch, BackwardTrainingBatch) and replay_batch is not None:
            current_batch = replay_batch(current_batch)

        if backend_train_step is not None:
            backend_metrics = backend_train_step(
                rollout_id=rollout_id,
                batch=current_batch,
                executor=executor,
            )
            if backend_metrics is not None:
                return backend_metrics

        return executor.execute_prepared_batch(
            rollout_id=rollout_id,
            batch=current_batch,
        )


__all__ = ["TrainingWorkflow"]
