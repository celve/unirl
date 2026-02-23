#!/usr/bin/env python
"""
diffusionrl async training loop (separate mode only).

This mirrors slime's overlap pattern:
- launch rollout N+1 while training on rollout N
- periodically synchronize weights with explicit boundary
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import ray
from diffusionrl.ray.data_buffer import normalize_rollout_result as _normalize_rollout_payload
from diffusionrl.runtime.async_runtime import AsyncPipelineRuntime

logger = logging.getLogger(__name__)


def _normalize_rollout_result(rollout_result: Any) -> Any:
    """Normalize rollout output to an ObjectRef-compatible payload."""
    return _normalize_rollout_payload(rollout_result)


def train_async_loop(
    *,
    args,
    rollout_manager,
    training_group,
    wandb_logger: Optional[Any],
    should_save_fn: Callable[[int, Any], bool],
    should_eval_fn: Callable[[int, Any], bool],
    sync_weights_fn: Callable[..., None],
) -> None:
    """Asynchronous train loop with rollout/train overlap."""
    logger.info("Starting async pipeline loop (separate mode)")
    max_inflight = int(getattr(args, "async_max_inflight", 1))
    update_interval = max(1, int(getattr(args, "update_weights_interval", 1)))
    use_rollout_buffer = bool(getattr(args, "rollout_buffer_enabled", True))
    runtime = AsyncPipelineRuntime(
        max_inflight=max_inflight,
        initial_rollout_id=args.start_rollout_id,
    )
    next_rollout_to_launch = int(args.start_rollout_id)

    def _sync_boundary_for(rollout_id: int) -> int:
        """Largest rollout id allowed before next weight sync boundary."""
        boundary = ((int(rollout_id) // update_interval) + 1) * update_interval - 1
        return min(boundary, int(args.num_rollout) - 1)

    def _launch_rollout(rollout_id: int) -> None:
        if not runtime.can_launch():
            raise RuntimeError(
                f"Cannot launch rollout {rollout_id}: inflight queue is full "
                f"(inflight={runtime.inflight_count}, max_inflight={runtime.max_inflight})"
            )
        if use_rollout_buffer:
            rollout_future = rollout_manager.generate_and_buffer.remote(rollout_id)
        else:
            rollout_future = rollout_manager.generate.remote(
                rollout_id,
                world_size=training_group.num_actors,
            )
        runtime.launch_rollout(
            rollout_id,
            rollout_future,
        )

    def _fill_inflight_window(current_rollout: int) -> None:
        nonlocal next_rollout_to_launch
        if next_rollout_to_launch >= int(args.num_rollout):
            return

        launch_limit = _sync_boundary_for(current_rollout)
        while runtime.can_launch() and next_rollout_to_launch <= launch_limit:
            _launch_rollout(next_rollout_to_launch)
            next_rollout_to_launch += 1

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        _fill_inflight_window(rollout_id)

        # Resolve current rollout payload.
        resolved = runtime.resolve_next_rollout(ray.get)
        if resolved.rollout_id != rollout_id:
            raise RuntimeError(
                f"Async rollout ordering violated: expected rollout_id={rollout_id}, "
                f"got {resolved.rollout_id}"
            )
        if use_rollout_buffer:
            rollout_result = ray.get(
                rollout_manager.pop_training_data.remote(
                    world_size=training_group.num_actors
                )
            )
        else:
            rollout_result = resolved.payload
        rollout_data_ref = _normalize_rollout_result(rollout_result)

        should_sync = (rollout_id + 1) % update_interval == 0

        # Train current rollout.
        metrics = training_group.train(rollout_id, rollout_data_ref)

        if rollout_id % args.logging_steps == 0:
            avg_loss = sum(m.get("loss", 0) for m in metrics) / max(len(metrics), 1)
            logger.info(f"[async] Rollout {rollout_id}: loss={avg_loss:.4f}")
            if wandb_logger is not None:
                from diffusionrl.utils.wandb_logger import aggregate_metrics

                aggregated = aggregate_metrics(metrics)
                aggregated["loss"] = avg_loss
                wandb_logger.log_step(rollout_id, aggregated)

        if should_save_fn(rollout_id, args):
            save_path = f"{args.output_dir}/checkpoint-{rollout_id}"
            training_group.save_model(save_path)
            logger.info(f"[async] Checkpoint saved: {save_path}")

        # Bound updates at a generation boundary to avoid update during active generation.
        if should_sync:
            runtime.assert_no_inflight_for_weight_sync()
            sync_weights_fn(
                rollout_id=rollout_id,
                training_group=training_group,
                rollout_manager=rollout_manager,
            )

        if should_eval_fn(rollout_id, args):
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            logger.info(
                f"[async] Eval at {rollout_id}: "
                f"mean_reward={eval_metrics['mean_reward']:.4f}"
            )
            if wandb_logger is not None:
                wandb_logger.log_eval(rollout_id, eval_metrics)

    ray.get(rollout_manager.dispose.remote())
    training_group.dispose()
    if wandb_logger is not None:
        wandb_logger.finish()
