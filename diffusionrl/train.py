#!/usr/bin/env python
"""
diffusionrl Training Entry Point.

Usage:
    python -m diffusionrl.train --pretrained-model-saved-path /path/to/model --num-rollout 100
"""
import logging
import os
import re
import time
from typing import Any, Optional, Set

from diffusionrl.config import parse_args
from diffusionrl.config.arguments import is_training_actor_sampling_mode
from diffusionrl.config.build_domain_args import build_algorithm_config
from diffusionrl.config.resolution import resolve_rollout_topology, resolve_sync_protocol
from diffusionrl.config.rollout_topology import runtime_mode_label_for_rollout_mode
from diffusionrl.ray.group_factory import create_rollout_actor_group, create_training_actor_group
from diffusionrl.ray.group_runtime import RolloutGroupRuntime, TrainingGroupRuntime
from diffusionrl.ray.placement_group import create_placement_groups_from_args
from diffusionrl.ray.buffer_actor import create_buffer_actor
from diffusionrl.ray.rollout_manager import create_rollout_manager

logger = logging.getLogger(__name__)

# Main control-plane path (sync mode):
# parse_args -> create_placement_groups_from_args -> create_rollout_manager
# -> create_training_actor_group -> rollout_manager.produce_training_payload
# -> rollout_buffer.push/pop -> training_group.train -> weight_sync.sync


def should_save(rollout_id: int, args) -> bool:  # [HELPER] used in train() loop & passed to train_async_loop
    """Check if we should save a checkpoint at this rollout."""
    interval = int(getattr(args.rollout, "save_steps", 0))
    return interval > 0 and (rollout_id + 1) % interval == 0


def should_eval(rollout_id: int, args) -> bool:  # [HELPER] used in train() loop & passed to train_async_loop
    """Check if we should run evaluation at this rollout."""
    interval = int(getattr(args.rollout, "eval_steps", 0))
    return interval > 0 and (rollout_id + 1) % interval == 0


def should_log(rollout_id: int, args) -> bool:  # [HELPER] used in train() loop only
    """Check if we should emit periodic rollout logs at this rollout."""
    interval = int(getattr(args.rollout, "logging_steps", 0))
    return interval > 0 and rollout_id % interval == 0


def _build_control_algorithm(args) -> tuple[dict, Any]:
    """Instantiate the control-plane algorithm visible from train.py."""
    from diffusionrl.utils import load_function

    algorithm_config = build_algorithm_config(args)
    algorithm_cls = load_function(algorithm_config["algorithm_path"])
    from_config = getattr(algorithm_cls, "from_config", None)
    if not callable(from_config):
        raise TypeError(
            f"Algorithm {algorithm_config['algorithm_path']} must implement classmethod from_config(config)."
        )
    return algorithm_config, from_config(dict(algorithm_config))


def _create_rollout_timestep_scheduler(args, *, algorithm: Any):
    """Create the control-plane timestep scheduler for rollout SDE selection."""
    from diffusionrl.runtime.contracts import resolve_sampling_requirements
    from diffusionrl.samplers.schedulers import get_scheduler
    from diffusionrl.samplers.schedulers.timestep_window import _normalize_timestep_fraction

    requirements = resolve_sampling_requirements(args, algorithm=algorithm)
    scheduler_type = getattr(args.algorithm.window, "timestep_strategy", "all")
    num_timesteps = int(args.sampling.num_inference_steps)
    timestep_fraction = getattr(args.sampling, "timestep_fraction", 1.0)

    if scheduler_type == "all" and requirements.sde_ratio < 1.0:
        group_size = max(1, int(num_timesteps * requirements.sde_ratio))
        scheduler_type = "window"
        logger.info("Auto-configured window scheduler from sde_ratio=%s", requirements.sde_ratio)
    else:
        group_size = None

    if scheduler_type == "window":
        explicit_group_size = getattr(args.algorithm.window, "window_group_size", None)
        if explicit_group_size is None and requirements.sde_ratio < 1.0:
            group_size = max(1, int(num_timesteps * requirements.sde_ratio))
        else:
            group_size = explicit_group_size or 4
        scheduler = get_scheduler(
            scheduler_type="window",
            num_timesteps=num_timesteps,
            timestep_fraction=timestep_fraction,
            strategy=getattr(args.algorithm.window, "window_strategy", "progressive"),
            group_size=group_size,
            iters_per_group=getattr(args.algorithm.window, "window_iters_per_group", 25),
            max_iters_per_group=getattr(args.algorithm.window, "window_max_iters_per_group", None),
            min_iters_per_group=getattr(args.algorithm.window, "window_min_iters_per_group", None),
            overlap=getattr(args.algorithm.window, "window_overlap", False),
            overlap_step=getattr(args.algorithm.window, "window_overlap_step", 1),
            roll_back=getattr(args.algorithm.window, "window_roll_back", False),
        )
        logger.info("Control-plane window scheduler initialized: group_size=%s", group_size)
        return scheduler

    scheduler = get_scheduler(
        scheduler_type="all",
        num_timesteps=num_timesteps,
        timestep_fraction=timestep_fraction,
    )
    frac_start, frac_end = _normalize_timestep_fraction(timestep_fraction)
    if frac_start > 0.0 or frac_end < 1.0:
        eff_start = int(num_timesteps * frac_start)
        eff_end = int(num_timesteps * frac_end)
        logger.info(
            "Control-plane all-SDE scheduler initialized; timestep_fraction=%s (SDE on timesteps [%s, %s)/%s)",
            timestep_fraction,
            eff_start,
            eff_end,
            num_timesteps,
        )
    else:
        logger.info("Control-plane all-SDE scheduler initialized")
    return scheduler


class RolloutSDEController:
    """Driver-side control-plane owner of rollout SDE scheduling."""

    def __init__(
        self,
        *,
        algorithm: Any,
        timestep_scheduler: Optional[Any],
    ) -> None:
        self.algorithm = algorithm
        self.timestep_scheduler = timestep_scheduler
        # Keep the same semantics as the old RolloutManager-owned counter:
        # current_step starts at 0 for the first produced rollout of this run.
        self._current_step = 0

    def next_sde_indices(self) -> Optional[Set[int]]:
        """Return the current rollout's SDE indices and advance local scheduler state."""
        sde_indices = self.algorithm.resolve_rollout_sde_indices(
            timestep_scheduler=self.timestep_scheduler,
            current_step=self._current_step,
        )
        normalized = (
            set(int(i) for i in sde_indices)
            if sde_indices is not None
            else None
        )
        self._current_step += 1
        if self.timestep_scheduler is not None:
            self.timestep_scheduler.update(self._current_step)
        return normalized


def train(args):  # [PUBLIC-API → main()] 主入口：资源创建 + sync/async 训练循环
    """Main training loop."""
    debug_mode = str(getattr(args.debug, "debug_mode", "none") or "none").strip().lower()
    if debug_mode == "train_only":
        from diffusionrl.debug import run_debug_train_only

        return run_debug_train_only(args)

    import ray

    from diffusionrl.utils import configure_logger, set_seed
    from diffusionrl.utils.wandb_metrics import (
        build_buffer_metrics,
        build_sync_metrics,
        compute_rollout_batch_metrics,
    )
    from diffusionrl.runtime.weight_sync import create_weight_sync
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger

    configure_logger()
    set_seed(args.seed)
    training_actor_sampling_mode = is_training_actor_sampling_mode(args)
    algorithm_config, control_algorithm = _build_control_algorithm(args)
    rollout_topology = resolve_rollout_topology(args)
    runtime_mode_label = runtime_mode_label_for_rollout_mode(rollout_topology.mode)
    sync_mode = resolve_sync_protocol(
        args,
        training_actor_sampling_mode=rollout_topology.training_actor_sampling_mode,
        rollout_service_engine=rollout_topology.service_engine,
    )
    rollout_sde_controller = RolloutSDEController(
        algorithm=control_algorithm,
        timestep_scheduler=_create_rollout_timestep_scheduler(args, algorithm=control_algorithm),
    )

    logger.info("Starting diffusionRL training...")
    logger.info(f"Model: {args.model.pretrained_model_saved_path}")
    logger.info(f"Algorithm: {algorithm_config['algorithm_path']}")
    logger.info(f"Mode: {runtime_mode_label}")
    logger.info(f"Offload train: {args.ray.offload_train}, Offload rollout: {args.ray.offload_rollout}")
    logger.info(f"Weight sync mode: {sync_mode}, async_pipeline={args.rollout.async_pipeline}")
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

    # Initialize Ray
    if not ray.is_initialized():
        if args.ray.ray_address:
            ray.init(address=args.ray.ray_address, ignore_reinit_error=True)
        else:
            ray.init()

    # Initialize WandB logger (only on rank 0)
    wandb_logger = None
    if args.rollout.report_to_wandb and args.rollout.project_name:
        # Parse wandb tags from config
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

    # 1. Resource allocation
    pgs = create_placement_groups_from_args(args)
    logger.info("Placement groups created")

    rollout_on_gpu = True
    weight_sync = create_weight_sync(args)

    # 2. Create rollout manager.
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

    rollout_group = None
    rollout_runtime = None
    if not training_actor_sampling_mode:
        rollout_pg_result = pgs.get("rollout")
        if rollout_pg_result is None:
            raise ValueError("Missing rollout placement-group allocation.")
        rollout_group = create_rollout_actor_group(args, rollout_pg_result)
        rollout_runtime = RolloutGroupRuntime.from_group(rollout_group)
        ray.get(rollout_manager.attach_sampling_group.remote(rollout_group))
        logger.info("Rollout actor group created and attached")

    # 3. Optional pre-offload for colocate memory pressure relief before training actor init.
    if args.ray.offload_rollout and rollout_runtime is not None:
        rollout_runtime.sleep()
        rollout_on_gpu = False

    # 4. Create training actors.
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
    buffer_consumer_spec = training_runtime.get_buffer_consumer_spec()
    logger.info("Training actor group created")
    if train_backend_info:
        logger.info("Training backend: %s", train_backend_info)

    # 5. Direct-sampling path: training actors serve rollout requests.
    if training_actor_sampling_mode:
        ray.get(rollout_manager.attach_sampling_group.remote(training_group))
        logger.info("Attached training actors as rollout sampling source")

    # 6. Initial weight synchronization for rollout-side actors.
    training_runtime.update_weights()
    logger.info("Initial weights synchronized")

    # 7. Restore rollout side and offload train side as needed after initial sync.
    if args.ray.offload_rollout and args.ray.offload_train and rollout_runtime is not None:
        training_runtime.offload()
        rollout_runtime.wake_up()
        rollout_on_gpu = True
    elif args.ray.offload_rollout and rollout_runtime is not None:
        rollout_runtime.wake_up()
        rollout_on_gpu = True

    # 8. Build rollout buffer.
    rollout_buffer = create_buffer_actor(args)
    ray.get(
        rollout_buffer.bind_runtime.remote(
            rollout_manager=rollout_manager,
            consumer_spec=buffer_consumer_spec,
        )
    )
    logger.info("Rollout buffer actor created and bound to rollout manager")

    # 9. Setup weight-sync coordinator (binds training/rollout runtime facades).
    weight_sync.setup(
        training_runtime=training_runtime,
        rollout_runtime=rollout_runtime,
    )

    debug_save_intermediates = bool(getattr(args.debug, "debug_save_intermediates", False))

    # 10. Core async training loop
    if args.rollout.async_pipeline:
        from diffusionrl.train_async import train_async_loop

        try:
            train_async_loop(
                args=args,
                rollout_manager=rollout_manager,
                rollout_buffer=rollout_buffer,
                training_group=training_group,
                training_runtime=training_runtime,
                rollout_group=rollout_group,
                rollout_runtime=rollout_runtime,
                rollout_sde_controller=rollout_sde_controller,
                wandb_logger=wandb_logger,
                should_save_fn=should_save,
                should_eval_fn=should_eval,
                weight_sync=weight_sync,
            )
        finally:
            weight_sync.teardown()
        # async loop handles cleanup and exits.
        return

    # 11. Core synchronous training loop
    enforce_rollout_alignment = not bool(getattr(args.rollout, "rollout_buffer_reassemble_by_group", False))
    save_rollout_debug_payload = None
    if debug_save_intermediates:
        from diffusionrl.debug.runner import save_rollout_debug_payload as _save_rollout_debug_payload

        save_rollout_debug_payload = _save_rollout_debug_payload

    def offload_train_phase() -> None:
        if args.ray.offload_train:
            training_runtime.offload()
        else:
            training_runtime.clear_memory()

    def ensure_rollout_on_gpu() -> None:
        nonlocal rollout_on_gpu
        if args.ray.offload_rollout and rollout_runtime is not None and not rollout_on_gpu:
            rollout_runtime.wake_up()
            rollout_on_gpu = True

    wandb_media_enabled = bool(
        wandb_logger is not None and bool(getattr(args.rollout, "wandb_log_media", False))
    )
    wandb_media_max_items = max(1, int(getattr(args.rollout, "wandb_media_max_items", 8)))

    def _collect_rollout_batch_metrics(batch_ref) -> dict:
        if batch_ref is None:
            return {}
        try:
            training_data = ray.get(batch_ref)
        except Exception as exc:
            logger.warning("Failed to materialize training batch for rollout metrics: %s", exc)
            return {}
        return compute_rollout_batch_metrics(training_data=training_data)

    # rollout_id is the outer rollout-train loop step; it behaves similarly to
    # a framework-level global step, but may differ from optimizer step count.
    # global_optimizer_step tracks real optimizer step for wandb logging
    global_optimizer_step = 0
    for rollout_id in range(args.rollout.start_rollout_id, args.rollout.num_rollout):
        step_start_t = time.perf_counter()
        sync_result = None
        sync_phase_s = 0.0
        eval_phase_s = 0.0
        should_log_step = should_log(rollout_id, args)
        collect_media_preview = bool(should_log_step and wandb_media_enabled)
        rollout_data_ref = None
        rollout_metadata = {}
        sample_count = 0
        # === PHASE 1: Rollout ===
        rollout_phase_start_t = time.perf_counter()
        ensure_rollout_on_gpu()
        if debug_save_intermediates:
            rollout_sde_indices = rollout_sde_controller.next_sde_indices()
            debug_payload = ray.get(
                rollout_manager.build_training_debug_payload.remote(
                    rollout_id,
                    sde_indices=rollout_sde_indices,
                )
            )
            push_result = ray.get(
                rollout_buffer.push.remote(
                    rollout_id=rollout_id,
                    train_data=debug_payload["training_batch"],
                )
            )
            if not push_result.get("accepted", False):
                raise RuntimeError(
                    f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
                )
            if save_rollout_debug_payload is not None:
                save_rollout_debug_payload(
                    args=args,
                    payload=debug_payload,
                    rollout_id=rollout_id,
                    source="train_loop",
                )
            del debug_payload
        else:
            rollout_sde_indices = rollout_sde_controller.next_sde_indices()
            rollout_payload = ray.get(
                rollout_manager.produce_training_payload.remote(
                    rollout_id=rollout_id,
                    sde_indices=rollout_sde_indices,
                    collect_media_preview=collect_media_preview,
                    media_max_items=wandb_media_max_items,
                )
            )
            push_result = ray.get(
                rollout_buffer.push.remote(
                    rollout_id=rollout_id,
                    train_data=rollout_payload["training_batch"],
                    metadata=rollout_payload.get("metadata"),
                )
            )
            if not push_result.get("accepted", False):
                raise RuntimeError(
                    f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
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

        if args.ray.offload_rollout and rollout_runtime is not None:
            rollout_runtime.sleep()
            rollout_on_gpu = False

        # === PHASE 2: Training ===
        train_phase_start_t = time.perf_counter()
        if args.ray.offload_train:
            training_runtime.onload()

        metrics = training_group.train(rollout_id, rollout_data_ref)
        train_phase_s = time.perf_counter() - train_phase_start_t

        # Periodic: save (before offload to ensure model is on GPU)
        if should_save(rollout_id, args):
            save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
            training_runtime.save_model(save_path)
            logger.info(f"Checkpoint saved: {save_path}")

        # === PHASE 3: Offload + Weight Sync ===
        offload_train_phase()

        if (
            not training_actor_sampling_mode
            and (rollout_id + 1) % args.rollout.update_weights_interval == 0
        ):
            sync_phase_start_t = time.perf_counter()
            sync_result = weight_sync.sync(rollout_id=rollout_id)
            sync_phase_s = time.perf_counter() - sync_phase_start_t
            rollout_on_gpu = True  # Coordinator internally calls wake_up

        # Periodic: evaluate (after weight sync, rollout actors are on GPU)
        if should_eval(rollout_id, args):
            eval_phase_start_t = time.perf_counter()
            ensure_rollout_on_gpu()
            # Swap in EMA weights for stable evaluation when training actors
            # serve as sampling source (direct-sampling mode).
            if training_actor_sampling_mode:
                training_runtime.apply_ema_for_eval()
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            if training_actor_sampling_mode:
                training_runtime.restore_from_eval()
            eval_phase_s = time.perf_counter() - eval_phase_start_t
            logger.info(f"Eval at {rollout_id}: mean_reward={eval_metrics['mean_reward']:.4f}")

            # Log eval metrics to WandB
            if wandb_logger:
                wandb_logger.log_eval(rollout_id, eval_metrics)

        # === Per-optimizer-step wandb logging ===
        if wandb_logger and metrics:
            per_step_list = metrics[0].get("_per_optimizer_step_metrics", [])
            for per_step_m in per_step_list:
                if per_step_m.get("has_backward", False):
                    global_optimizer_step += 1
                    wandb_step_m = {
                        k: v for k, v in per_step_m.items()
                        if k != "has_backward"
                    }
                    wandb_logger.log_step(global_optimizer_step, wandb_step_m)

        if should_log_step:
            avg_loss = sum(m.get("loss", 0) for m in metrics) / max(len(metrics), 1)
            step_time_s = time.perf_counter() - step_start_t
            logger.info(
                "Rollout %s: loss=%.4E rollout=%.3fs train=%.3fs sync=%.3fs eval=%.3fs step=%.3fs",
                rollout_id,
                avg_loss,
                rollout_phase_s,
                train_phase_s,
                sync_phase_s,
                eval_phase_s,
                step_time_s,
            )

            if wandb_logger:
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

                sync_metrics = {}
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
                    logger.warning("Failed to fetch rollout buffer stats: %s", exc)
                else:
                    buffer_metrics = build_buffer_metrics(buffer_stats)
                    if buffer_metrics:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics=buffer_metrics,
                        )

    # Cleanup
    weight_sync.teardown()
    try:
        ray.get(rollout_buffer.dispose.remote())
    finally:
        ray.kill(rollout_buffer)
    ray.get(rollout_manager.dispose.remote())
    if rollout_group is not None:
        rollout_group.dispose()
    training_group.dispose()

    # Finish WandB logging
    if wandb_logger:
        wandb_logger.finish()

    logger.info("Training complete!")


def main(argv=None):  # [PUBLIC-API → __main__] CLI 入口
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
