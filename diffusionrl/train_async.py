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
from diffusionrl.runtime.async_runtime import AsyncPipelineRuntime

logger = logging.getLogger(__name__)


def _normalize_rollout_result(rollout_result: Any) -> Any:
    """Normalize rollout output to an ObjectRef-compatible payload."""
    if isinstance(rollout_result, list):
        return rollout_result
    if isinstance(rollout_result, ray.ObjectRef):
        return rollout_result
    return ray.put(rollout_result)


def train_async_loop(
    *,
    args,
    rollout_manager,
    training_group,
    wandb_logger: Optional[Any],
    should_save_fn: Callable[[int, Any], bool],
    should_eval_fn: Callable[[int, Any], bool],
    sync_weights_fn: Callable[..., int],
    initial_weight_version: int = 0,
) -> None:
    """Asynchronous train loop with rollout/train overlap."""
    logger.info("Starting async pipeline loop (separate mode)")
    max_inflight = int(getattr(args, "async_max_inflight", 1))
    runtime = AsyncPipelineRuntime(
        max_inflight=max_inflight,
        initial_rollout_id=args.start_rollout_id,
        initial_weight_version=int(initial_weight_version),
    )

    def _launch_rollout(rollout_id: int) -> None:
        if not runtime.can_launch():
            raise RuntimeError(
                f"Cannot launch rollout {rollout_id}: inflight queue is full "
                f"(inflight={runtime.inflight_count}, max_inflight={runtime.max_inflight})"
            )
        rollout_future = rollout_manager.generate.remote(
            rollout_id,
            world_size=training_group.num_actors,
        )
        runtime.launch_rollout(
            rollout_id,
            rollout_future,
            weight_version=runtime.expected_weight_version,
        )

    # Initial rollout launch.
    _launch_rollout(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Resolve current rollout payload.
        resolved = runtime.resolve_next_rollout(ray.get)
        if resolved.rollout_id != rollout_id:
            raise RuntimeError(
                f"Async rollout ordering violated: expected rollout_id={rollout_id}, "
                f"got {resolved.rollout_id}"
            )
        runtime.ensure_rollout_version(resolved)
        rollout_data_ref = _normalize_rollout_result(resolved.payload)

        should_sync = (rollout_id + 1) % args.update_weights_interval == 0

        # Launch next rollout before training only when this step does not sync.
        # This keeps overlap while avoiding stale-version rollouts around sync boundaries.
        if rollout_id + 1 < args.num_rollout and not should_sync:
            _launch_rollout(rollout_id + 1)

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
            target_weight_version = runtime.expected_weight_version + 1
            synced_version = sync_weights_fn(
                args,
                rollout_id,
                training_group,
                rollout_manager,
                target_weight_version=target_weight_version,
            )
            if int(synced_version) != int(target_weight_version):
                raise RuntimeError(
                    f"Sync returned unexpected version: expected={target_weight_version}, "
                    f"got={synced_version}"
                )
            new_weight_version = runtime.advance_weight_version()
            if int(new_weight_version) != int(synced_version):
                raise RuntimeError(
                    f"Runtime/rollout version mismatch after sync: runtime={new_weight_version}, "
                    f"synced={synced_version}"
                )
            logger.info(f"[async] Advanced rollout weight version to {new_weight_version}")

            # Relaunch the immediate next rollout after sync with the new weight version.
            if rollout_id + 1 < args.num_rollout and not runtime.has_rollout(rollout_id + 1):
                _launch_rollout(rollout_id + 1)

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
