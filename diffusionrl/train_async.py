#!/usr/bin/env python
"""
diffusionrl async training entrypoint (separate mode only).

Usage:
    python -m diffusionrl.train_async --pretrained-model-saved-path /path/to/model --num-rollout 100

This overlaps rollout and training with explicit synchronization boundaries:
- launch rollout N+1 while training on rollout N
- periodically synchronize weights with explicit boundary
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from diffusionrl.config import parse_args
from diffusionrl.config.arguments import is_training_actor_sampling_mode
from diffusionrl.config.resolution import resolve_rollout_topology
from diffusionrl.config.rollout_topology import runtime_mode_label_for_rollout_mode
from diffusionrl.config.validation import (
    validate_async_training_runner,
    validate_training_actor_sampling_mode,
    validate_weight_sync,
)
from diffusionrl.training.backends import resolve_train_backend_capabilities_from_args
from diffusionrl.utils.train_utils import (
    RolloutSDEController,
    build_control_algorithm,
    collect_rollout_batch_metrics,
    create_rollout_timestep_scheduler,
    should_eval,
    should_log,
    should_save,
)
from diffusionrl.utils.wandb_logger import aggregate_metrics
from diffusionrl.utils.wandb_metrics import (
    build_buffer_metrics,
    build_sync_metrics,
)

if TYPE_CHECKING:
    from diffusionrl.distributed.weight_sync import WeightSyncCoordinator

logger = logging.getLogger(__name__)


# Main control-plane path (async mode):
# parse_args -> create_placement_groups_from_args -> create_rollout_manager
# -> create_training_actor_group -> rollout_manager.produce_training_payload
# -> rollout_buffer.push_payload_ref/pop -> training_group.train -> weight_sync.sync

@dataclass(frozen=True)
class InflightRollout:
    """A rollout launched but not yet consumed."""

    rollout_id: int
    future: Any


@dataclass(frozen=True)
class ResolvedRollout:
    """A rollout-scoped future result resolved from an inflight future."""

    rollout_id: int
    result: Any


class AsyncPipelineRuntime:
    """Minimal producer-consumer state for the async training loop."""

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        initial_rollout_id: int = 0,
    ) -> None:
        del initial_rollout_id
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")

        self.max_inflight = int(max_inflight)
        self._inflight: Dict[int, InflightRollout] = {}

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def can_launch(self) -> bool:
        return self.inflight_count < self.max_inflight

    def launch_rollout(
        self,
        rollout_id: int,
        future: Any,
    ) -> InflightRollout:
        rid = int(rollout_id)
        if not self.can_launch():
            raise RuntimeError(
                f"Async inflight queue full: inflight={self.inflight_count}, max_inflight={self.max_inflight}"
            )
        if rid in self._inflight:
            raise RuntimeError(f"Rollout {rid} is already inflight")

        inflight = InflightRollout(rollout_id=rid, future=future)
        self._inflight[rid] = inflight
        return inflight

    def resolve_next_rollout(self, resolver: Callable[[Any], Any]) -> ResolvedRollout:
        if not self._inflight:
            raise RuntimeError("No inflight rollout to resolve")

        rid = min(self._inflight.keys())
        inflight = self._inflight.pop(rid)
        result = resolver(inflight.future)
        return ResolvedRollout(
            rollout_id=inflight.rollout_id,
            result=result,
        )

    def assert_no_inflight_for_weight_sync(self) -> None:
        if self._inflight:
            pending = sorted(self._inflight.keys())
            raise RuntimeError(
                "Weight sync requires empty inflight queue, but found pending rollouts: "
                f"{pending}"
            )


def train_async_loop(  # [PUBLIC-API → train()] async 核心循环
    *,
    args,
    rollout_manager,
    rollout_buffer,
    training_group,
    training_runtime,
    rollout_runtime,
    rollout_sde_controller,
    wandb_logger: Optional[Any],
    should_save_fn: Callable[[int, Any], bool],
    should_eval_fn: Callable[[int, Any], bool],
    should_log_fn: Callable[[int, Any], bool],
    collect_rollout_batch_metrics_fn: Callable[[Any], dict],
    weight_sync: "WeightSyncCoordinator",
) -> None:
    """Asynchronous train loop with rollout/train overlap."""
    import ray

    logger.info("Starting async pipeline loop (separate mode)")
    max_inflight = int(getattr(args.rollout, "async_max_inflight", 1))
    update_interval = max(1, int(getattr(args.rollout, "update_weights_interval", 1)))
    enforce_rollout_alignment = not bool(getattr(args.rollout, "rollout_buffer_reassemble_by_group", False))
    rollout_on_gpu = True
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
        should_log_rollout = should_log_fn(rollout_id, args)
        rollout_sde_indices = rollout_sde_controller.next_sde_indices()
        payload_ref = rollout_manager.produce_training_payload.remote(
            rollout_id=rollout_id,
            sde_indices=rollout_sde_indices,
            collect_media_preview=bool(should_log_rollout and wandb_media_enabled),
            media_max_items=wandb_media_max_items,
        )
        enqueue_ref = rollout_buffer.push_payload_ref.remote(
            payload_ref=payload_ref,
        )
        runtime.launch_rollout(
            rollout_id,
            enqueue_ref,
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
        if (
            bool(getattr(args.ray, "offload_rollout", False))
            and rollout_runtime is not None
            and not rollout_on_gpu
        ):
            rollout_runtime.wake_up()
            rollout_on_gpu = True

    wandb_media_enabled = bool(
        wandb_logger is not None and bool(getattr(args.rollout, "wandb_log_media", False))
    )
    wandb_media_max_items = max(1, int(getattr(args.rollout, "wandb_media_max_items", 8)))

    global_optimizer_step = 0
    for rollout_id in range(args.rollout.start_rollout_id, args.rollout.num_rollout):
        step_start_t = time.perf_counter()
        sync_result = None
        sync_phase_s = 0.0
        eval_phase_s = 0.0

        rollout_phase_start_t = time.perf_counter()
        _fill_inflight_window(rollout_id)

        resolved = runtime.resolve_next_rollout(ray.get)
        if resolved.rollout_id != rollout_id:
            raise RuntimeError(
                f"Async rollout ordering violated: expected rollout_id={rollout_id}, "
                f"got {resolved.rollout_id}"
            )
        push_result = resolved.result
        if not push_result.get("accepted", False):
            raise RuntimeError(
                f"Rollout buffer rejected rollout_id={push_result.get('payload_rollout_id')}: {push_result.get('error')}"
            )
        rollout_payload = ray.get(
            rollout_buffer.pop_training_data.remote(
                expected_rollout_id=rollout_id if enforce_rollout_alignment else None,
            )
        )
        training_data_handle = rollout_payload.training_data
        rollout_metadata = dict(rollout_payload.metadata or {})
        sample_count = int(rollout_payload.sample_count or 0)
        rollout_phase_s = time.perf_counter() - rollout_phase_start_t

        should_sync = (rollout_id + 1) % update_interval == 0

        train_phase_start_t = time.perf_counter()
        metrics = training_group.train(rollout_id, training_data_handle)
        train_phase_s = time.perf_counter() - train_phase_start_t

        if should_save_fn(rollout_id, args):
            save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
            training_runtime.save_model(save_path)
            logger.info("[async] Checkpoint saved: %s", save_path)

        if should_sync:
            runtime.assert_no_inflight_for_weight_sync()
            sync_phase_start_t = time.perf_counter()
            sync_result = weight_sync.sync(rollout_id=rollout_id)
            sync_phase_s = time.perf_counter() - sync_phase_start_t
            rollout_on_gpu = True

        if should_eval_fn(rollout_id, args):
            eval_phase_start_t = time.perf_counter()
            _ensure_rollout_on_gpu()
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            eval_phase_s = time.perf_counter() - eval_phase_start_t
            logger.info(
                "[async] Eval at %s: mean_reward=%.4f",
                rollout_id,
                eval_metrics["mean_reward"],
            )
            if wandb_logger is not None:
                wandb_logger.log_eval(rollout_id, eval_metrics)

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

        should_log_step = should_log_fn(rollout_id, args)
        if should_log_step:
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
                aggregated = aggregate_metrics(metrics)
                aggregated["loss"] = avg_loss
                wandb_logger.log_rollout(rollout_id, aggregated)

                rollout_metrics = collect_rollout_batch_metrics_fn(training_data_handle)
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


def train(args):  # [PUBLIC-API → main()] async 入口：资源创建 + 异步训练循环
    """Asynchronous training entrypoint."""
    validate_async_training_runner(args)

    import ray

    from diffusionrl.ray.buffer_actor import create_buffer_actor
    from diffusionrl.ray.group_factory import create_rollout_actor_group, create_training_actor_group
    from diffusionrl.ray.group_runtime import RolloutGroupRuntime, TrainingGroupRuntime
    from diffusionrl.ray.placement_group import create_placement_groups_from_args
    from diffusionrl.ray.rollout_manager import create_rollout_manager
    from diffusionrl.distributed.weight_sync import create_weight_sync
    from diffusionrl.utils import configure_logger, set_seed
    from diffusionrl.utils.wandb_logger import init_logger
    from diffusionrl.utils.wandb_metrics import compute_rollout_batch_metrics

    configure_logger()
    set_seed(args.seed)

    debug_mode = str(getattr(args.debug, "debug_mode", "none") or "none").strip().lower()
    training_actor_sampling_mode = is_training_actor_sampling_mode(args)
    algorithm_config, control_algorithm = build_control_algorithm(args)
    rollout_topology = resolve_rollout_topology(args)
    backend_capabilities = resolve_train_backend_capabilities_from_args(args)
    validate_training_actor_sampling_mode(
        args,
        training_actor_sampling_mode=rollout_topology.training_actor_sampling_mode,
        backend_capabilities=backend_capabilities,
    )
    sync_mode = str(args.sync.protocol).strip().lower()
    validate_weight_sync(
        args,
        training_actor_sampling_mode=rollout_topology.training_actor_sampling_mode,
    )
    runtime_mode_label = runtime_mode_label_for_rollout_mode(rollout_topology.mode)
    rollout_sde_controller = RolloutSDEController(
        algorithm=control_algorithm,
        timestep_scheduler=create_rollout_timestep_scheduler(args, algorithm=control_algorithm),
    )

    if training_actor_sampling_mode:
        raise ValueError(
            "train_async.py requires a dedicated rollout engine and does not support training-actor direct sampling."
        )

    logger.info("Starting diffusionRL async training...")
    logger.info("Model: %s", args.model.pretrained_model_saved_path)
    logger.info("Algorithm: %s", algorithm_config["algorithm_path"])
    logger.info("Mode: %s", runtime_mode_label)
    logger.info("Weight sync mode: %s", sync_mode)
    logger.info(
        "Async controls: async_max_inflight=%s update_weights_interval=%s",
        args.rollout.async_max_inflight,
        args.rollout.update_weights_interval,
    )
    logger.info(
        "Periodic controls: save_steps=%s eval_steps=%s logging_steps=%s",
        args.rollout.save_steps,
        args.rollout.eval_steps,
        args.rollout.logging_steps,
    )
    logger.info(
        "Debug flags: mode=%s save_intermediates=%s save_dir=%s",
        debug_mode,
        bool(getattr(args.debug, "debug_save_intermediates", False)),
        getattr(args.debug, "debug_save_dir", ""),
    )

    if not ray.is_initialized():
        if args.ray.ray_address:
            ray.init(address=args.ray.ray_address, ignore_reinit_error=True)
        else:
            ray.init()

    wandb_logger = None
    rollout_manager = None
    rollout_buffer = None
    rollout_group = None
    training_group = None
    weight_sync = create_weight_sync(args, mode=sync_mode)

    try:
        if args.rollout.report_to_wandb and args.rollout.project_name:
            wandb_tags_str = getattr(args.rollout, "wandb_tags", None)
            wandb_tags = (
                [t.strip() for t in wandb_tags_str.split(",") if t.strip()]
                if wandb_tags_str
                else None
            )
            wandb_entity = getattr(args.rollout, "wandb_entity", None) or None
            wandb_logger = init_logger(
                project=args.rollout.project_name,
                run_name=args.rollout.run_name,
                config=args.to_flat_dict() if hasattr(args, "to_flat_dict") else vars(args),
                log_dir=getattr(args.rollout, "logging_dir", None),
                rank=0,
                tags=wandb_tags,
                entity=wandb_entity,
                require_success=True,
            )
            if wandb_logger.initialized:
                logger.info(
                    "WandB initialized: project=%s, run=%s",
                    args.rollout.project_name,
                    args.rollout.run_name,
                )

        pgs = create_placement_groups_from_args(args)
        logger.info("Placement groups created")

        rollout_manager, dataset_step_info = create_rollout_manager(
            args,
            reward_pg_result=pgs.get("reward"),
            algorithm_config=algorithm_config,
        )
        logger.info("Rollout manager created")
        if dataset_step_info.get("num_samples", 0) > 0:
            logger.info(
                "Dataset step info: num_samples=%s prompts_per_rollout=%s "
                "estimated_steps_per_dataset_pass=%s steps_before_reset=%s",
                dataset_step_info.get("num_samples"),
                dataset_step_info.get("prompts_per_rollout"),
                dataset_step_info.get("estimated_steps_per_dataset_pass"),
                dataset_step_info.get("steps_before_reset"),
            )
            if not dataset_step_info.get("exact_dataset_pass_per_cycle", False):
                logger.warning(
                    "Dataset pass is not exact under current data-source batching: "
                    "drop_last=%s remainder_samples=%s. "
                    "One reset cycle will not cover the full dataset exactly once.",
                    dataset_step_info.get("drop_last"),
                    dataset_step_info.get("remainder_samples"),
                )

        rollout_pg_result = pgs.get("rollout")
        if rollout_pg_result is None:
            raise ValueError("Missing rollout placement-group allocation.")
        rollout_group = create_rollout_actor_group(args, rollout_pg_result)
        rollout_runtime = RolloutGroupRuntime.from_group(rollout_group)
        ray.get(rollout_manager.attach_sampling_group.remote(rollout_group))
        logger.info("Rollout actor group created and attached")

        training_pg_result = pgs.get("training")
        if training_pg_result is None:
            raise ValueError("Missing training placement-group allocation.")
        training_group = create_training_actor_group(
            args,
            training_pg_result,
            algorithm_config=algorithm_config,
        )
        training_runtime = TrainingGroupRuntime.from_group(training_group)
        resume_from_checkpoint = getattr(args.rollout, "resume_from_checkpoint", None)
        if resume_from_checkpoint:
            training_runtime.load_checkpoint(resume_from_checkpoint)
            logger.info("Checkpoint loaded: %s", resume_from_checkpoint)
            if int(getattr(args.rollout, "start_rollout_id", 0)) == 0:
                match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(resume_from_checkpoint)))
                if match:
                    args.rollout.start_rollout_id = int(match.group(1)) + 1
                    logger.info(
                        "Auto-set start_rollout_id=%s from checkpoint path.",
                        args.rollout.start_rollout_id,
                    )
        train_backend_info = training_runtime.get_train_backend_info()
        expected_global_batch_size = training_runtime.get_expected_global_batch_size()
        logger.info("Training actor group created")
        if train_backend_info:
            logger.info("Training backend: %s", train_backend_info)

        training_runtime.update_weights()
        logger.info("Initial weights synchronized")

        rollout_buffer = create_buffer_actor(args)
        ray.get(
            rollout_buffer.configure_expected_global_batch_size.remote(
                expected_global_batch_size=expected_global_batch_size,
            )
        )
        logger.info(
            "Rollout buffer actor created and configured with expected_global_batch_size=%s",
            expected_global_batch_size,
        )

        weight_sync.setup(
            training_runtime=training_runtime,
            rollout_runtime=rollout_runtime,
        )

        train_async_loop(
            args=args,
            rollout_manager=rollout_manager,
            rollout_buffer=rollout_buffer,
            training_group=training_group,
            training_runtime=training_runtime,
            rollout_runtime=rollout_runtime,
            rollout_sde_controller=rollout_sde_controller,
            wandb_logger=wandb_logger,
            should_save_fn=should_save,
            should_eval_fn=should_eval,
            should_log_fn=should_log,
            collect_rollout_batch_metrics_fn=lambda batch_ref: collect_rollout_batch_metrics(
                ray_module=ray,
                batch_ref=batch_ref,
                compute_rollout_batch_metrics_fn=compute_rollout_batch_metrics,
            ),
            weight_sync=weight_sync,
        )
    finally:
        weight_sync.teardown()
        if rollout_buffer is not None:
            try:
                ray.get(rollout_buffer.dispose.remote())
            finally:
                ray.kill(rollout_buffer)
        if rollout_manager is not None:
            try:
                ray.get(rollout_manager.dispose.remote())
            finally:
                ray.kill(rollout_manager)
        if rollout_group is not None:
            rollout_group.dispose()
        if training_group is not None:
            training_group.dispose()
        if wandb_logger is not None:
            wandb_logger.finish()

    logger.info("Async training complete!")


def main(argv=None):  # [PUBLIC-API → __main__] async CLI 入口
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
