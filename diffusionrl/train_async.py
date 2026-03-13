#!/usr/bin/env python
"""
diffusionrl async training loop (separate mode only).

This mirrors slime's overlap pattern:
- launch rollout N+1 while training on rollout N
- periodically synchronize weights with explicit boundary
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TYPE_CHECKING

import ray
from diffusionrl.runtime.async_runtime import AsyncPipelineRuntime
from diffusionrl.utils.wandb_logger import aggregate_metrics
from diffusionrl.utils.wandb_metrics import (
    build_buffer_metrics,
    build_sync_metrics,
    compute_rollout_batch_metrics,
)

if TYPE_CHECKING:
    from diffusionrl.utils.weight_sync import WeightSyncProtocol

logger = logging.getLogger(__name__)


def _should_log_rollout(rollout_id: int, args) -> bool:
    interval = int(getattr(args.rollout, "logging_steps", 0))
    return interval > 0 and rollout_id % interval == 0


def train_async_loop(
    *,
    args,
    rollout_manager,
    rollout_buffer,
    training_group,
    wandb_logger: Optional[Any],
    should_save_fn: Callable[[int, Any], bool],
    should_eval_fn: Callable[[int, Any], bool],
    weight_sync: "WeightSyncProtocol",
) -> None:
    """Asynchronous train loop with rollout/train overlap."""
    logger.info("Starting async pipeline loop (separate mode)")
    max_inflight = int(getattr(args.rollout, "async_max_inflight", 1))
    update_interval = max(1, int(getattr(args.rollout, "update_weights_interval", 1)))
    enforce_rollout_alignment = not bool(getattr(args.rollout, "rollout_buffer_grouped", False))
    rollout_on_gpu = True
    buffer_consumer_spec = training_group.get_buffer_consumer_spec()
    runtime = AsyncPipelineRuntime(
        max_inflight=max_inflight,
        initial_rollout_id=args.rollout.start_rollout_id,
    )
    next_rollout_to_launch = int(args.rollout.start_rollout_id)

    def _sync_boundary_for(rollout_id: int) -> int:
        """Largest rollout id allowed before next weight sync boundary."""
        boundary = ((int(rollout_id) // update_interval) + 1) * update_interval - 1
        return min(boundary, int(args.rollout.num_rollout) - 1)

    def _launch_rollout(rollout_id: int) -> None:
        if not runtime.can_launch():
            raise RuntimeError(
                f"Cannot launch rollout {rollout_id}: inflight queue is full "
                f"(inflight={runtime.inflight_count}, max_inflight={runtime.max_inflight})"
            )
        should_log_rollout = _should_log_rollout(rollout_id, args)
        rollout_future = rollout_manager.generate_and_push.remote(
            rollout_id=rollout_id,
            buffer=rollout_buffer,
            collect_media_preview=bool(should_log_rollout and wandb_media_enabled),
            media_max_items=wandb_media_max_items,
        )
        runtime.launch_rollout(
            rollout_id,
            rollout_future,
        )

    def _fill_inflight_window(current_rollout: int) -> None:
        nonlocal next_rollout_to_launch
        if next_rollout_to_launch >= int(args.rollout.num_rollout):
            return

        launch_limit = _sync_boundary_for(current_rollout)
        while runtime.can_launch() and next_rollout_to_launch <= launch_limit:
            _launch_rollout(next_rollout_to_launch)
            next_rollout_to_launch += 1

    def _ensure_rollout_on_gpu() -> None:
        nonlocal rollout_on_gpu
        if bool(getattr(args.ray, "offload_rollout", False)) and not rollout_on_gpu:
            ray.get(rollout_manager.wake_up.remote())
            rollout_on_gpu = True

    num_samples_per_prompt = max(1, int(getattr(args.algorithm, "num_samples_per_prompt", 1)))
    wandb_media_enabled = bool(
        wandb_logger is not None and bool(getattr(args.rollout, "wandb_log_media", False))
    )
    wandb_media_max_items = max(1, int(getattr(args.rollout, "wandb_media_max_items", 8)))

    def _collect_rollout_batch_metrics(batch_ref) -> dict:
        try:
            training_data = ray.get(batch_ref)
        except Exception as exc:
            logger.warning("[async] Failed to materialize training batch for rollout metrics: %s", exc)
            return {}
        return compute_rollout_batch_metrics(
            training_data=training_data,
            num_samples_per_prompt=num_samples_per_prompt,
        )

    # rollout_id is the outer rollout-train loop step; it behaves similarly to
    # a framework-level global step, but may differ from optimizer step count.
    # global_optimizer_step tracks real optimizer step for wandb logging
    global_optimizer_step = 0
    for rollout_id in range(args.rollout.start_rollout_id, args.rollout.num_rollout):
        step_start_t = time.perf_counter()
        sync_result = None
        sync_phase_s = 0.0
        eval_phase_s = 0.0

        rollout_phase_start_t = time.perf_counter()
        _fill_inflight_window(rollout_id)

        # Resolve current rollout payload.
        resolved = runtime.resolve_next_rollout(ray.get)
        if resolved.rollout_id != rollout_id:
            raise RuntimeError(
                f"Async rollout ordering violated: expected rollout_id={rollout_id}, "
                f"got {resolved.rollout_id}"
            )
        rollout_payload = ray.get(
            rollout_buffer.pop_training_data.remote(
                consumer_spec=buffer_consumer_spec,
                expected_rollout_id=rollout_id if enforce_rollout_alignment else None,
            )
        )
        rollout_data_ref = rollout_payload["training_data"]
        rollout_metadata = dict(rollout_payload.get("metadata") or {})
        sample_count = int(rollout_payload.get("sample_count", 0) or 0)
        rollout_phase_s = time.perf_counter() - rollout_phase_start_t

        should_sync = (rollout_id + 1) % update_interval == 0

        # Train current rollout.
        train_phase_start_t = time.perf_counter()
        metrics = training_group.train(rollout_id, rollout_data_ref)
        train_phase_s = time.perf_counter() - train_phase_start_t

        if should_save_fn(rollout_id, args):
            save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
            training_group.save_model(save_path)
            logger.info(f"[async] Checkpoint saved: {save_path}")

        # Bound updates at a generation boundary to avoid update during active generation.
        if should_sync:
            runtime.assert_no_inflight_for_weight_sync()
            sync_phase_start_t = time.perf_counter()
            sync_result = weight_sync.sync(rollout_id=rollout_id)
            sync_phase_s = time.perf_counter() - sync_phase_start_t
            rollout_on_gpu = True  # Protocol internally calls wake_up

        if should_eval_fn(rollout_id, args):
            eval_phase_start_t = time.perf_counter()
            _ensure_rollout_on_gpu()
            # NOTE: Async mode uses separate rollout actors for eval. EMA-based
            # eval would require syncing EMA weights to rollout actors which is
            # not yet implemented. The eval here uses the latest synced weights.
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            eval_phase_s = time.perf_counter() - eval_phase_start_t
            logger.info(
                f"[async] Eval at {rollout_id}: "
                f"mean_reward={eval_metrics['mean_reward']:.4f}"
            )
            if wandb_logger is not None:
                wandb_logger.log_eval(rollout_id, eval_metrics)

        # === Per-optimizer-step wandb logging ===
        if wandb_logger is not None and metrics:
            per_step_list = metrics[0].get("_per_optimizer_step_metrics", [])
            for per_step_m in per_step_list:
                if per_step_m.get("has_backward", False):
                    global_optimizer_step += 1
                    wandb_step_m = {
                        k: v for k, v in per_step_m.items()
                        if k != "has_backward"
                    }
                    wandb_logger.log_step(global_optimizer_step, wandb_step_m)

        should_log = _should_log_rollout(rollout_id, args)
        if should_log:
            avg_loss = sum(m.get("loss", 0) for m in metrics) / max(len(metrics), 1)
            step_time_s = time.perf_counter() - step_start_t
            logger.info(
                "[async] Rollout %s: loss=%.4f rollout=%.3fs train=%.3fs sync=%.3fs eval=%.3fs step=%.3fs",
                rollout_id,
                avg_loss,
                rollout_phase_s,
                train_phase_s,
                sync_phase_s,
                eval_phase_s,
                step_time_s,
            )

            if wandb_logger is not None:
                # === Rollout 级别聚合上报 ===
                aggregated = aggregate_metrics(metrics)
                aggregated["loss"] = avg_loss
                wandb_logger.log_rollout(rollout_id, aggregated)

                rollout_metrics = _collect_rollout_batch_metrics(rollout_data_ref)
                if rollout_metrics:
                    wandb_logger.log_rollout(rollout_id, rollout_metrics)
                media_preview = rollout_metadata.get("wandb_media_preview")
                if media_preview:
                    wandb_logger.log_generated_media(rollout_id, media_preview)

                perf_metrics = {
                    "rollout_phase_s": rollout_phase_s,
                    "train_phase_s": train_phase_s,
                    "sync_phase_s": sync_phase_s,
                    "eval_phase_s": eval_phase_s,
                    "step_time_s": step_time_s,
                    "samples_per_rollout": float(sample_count),
                    "samples_per_s": (
                        float(sample_count) / float(step_time_s)
                        if step_time_s > 0 and sample_count > 0
                        else 0.0
                    ),
                }
                wandb_logger.log_perf(rollout_id, perf_metrics)

                if sync_result is not None:
                    sync_metrics = build_sync_metrics(sync_result)
                    if sync_metrics:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics=sync_metrics,
                        )

                try:
                    buffer_stats = ray.get(rollout_buffer.get_stats.remote())
                except Exception as exc:
                    logger.warning("[async] Failed to fetch rollout buffer stats: %s", exc)
                else:
                    buffer_metrics = build_buffer_metrics(buffer_stats)
                    if buffer_metrics:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics=buffer_metrics,
                        )

    try:
        ray.get(rollout_buffer.dispose.remote())
    finally:
        ray.kill(rollout_buffer)
    ray.get(rollout_manager.dispose.remote())
    training_group.dispose()
    if wandb_logger is not None:
        wandb_logger.finish()
