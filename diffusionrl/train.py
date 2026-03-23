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

from diffusionrl.config import parse_args
from diffusionrl.config.arguments import is_training_actor_sampling_mode
from diffusionrl.config.resolution import resolve_rollout_topology
from diffusionrl.config.rollout_topology import runtime_mode_label_for_rollout_mode
from diffusionrl.config.validation import (
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

logger = logging.getLogger(__name__)

# Main control-plane path (sync mode):
# parse_args -> create_placement_groups_from_args -> create_rollout_manager
# -> create_training_actor_group -> rollout_manager.produce_training_payload
# -> rollout_buffer.push/pop -> training_group.train -> weight_sync.sync

def _offload_train_phase(*, args, training_runtime) -> None:
    """Offload or clear train-side memory after one update phase."""
    if args.ray.offload_train:
        training_runtime.offload()
    else:
        training_runtime.clear_memory()


def _ensure_rollout_on_gpu(*, args, rollout_runtime, rollout_on_gpu: bool) -> bool:
    """Wake the rollout side only when driver policy requires it."""
    if args.ray.offload_rollout and rollout_runtime is not None and not rollout_on_gpu:
        rollout_runtime.wake_up()
        return True
    return rollout_on_gpu


def _push_rollout_training_batch(*, ray_module, rollout_buffer, rollout_id: int, training_batch, metadata=None) -> None:
    """Push a produced training batch into the rollout buffer or fail fast."""
    push_result = ray_module.get(
        rollout_buffer.push.remote(
            rollout_id=int(rollout_id),
            train_data=training_batch,
            metadata=metadata,
        )
    )
    if not push_result.get("accepted", False):
        raise RuntimeError(
            f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
        )


def _admit_rollout_payload_ref(*, ray_module, rollout_buffer, payload_ref) -> dict:
    """Hand a rollout payload ref to the buffer and return its admission receipt."""
    receipt = ray_module.get(
        rollout_buffer.push_payload_ref.remote(
            payload_ref=payload_ref,
        )
    )
    if not receipt.get("accepted", False):
        raise RuntimeError(
            f"Rollout buffer rejected rollout_id={receipt.get('payload_rollout_id')}: {receipt.get('error')}"
        )
    return receipt


def _produce_and_push_rollout(
    *,
    ray_module,
    args,
    rollout_manager,
    rollout_buffer,
    rollout_id: int,
    rollout_sde_controller,
    debug_save_intermediates: bool,
    collect_media_preview: bool,
    media_max_items: int,
    save_rollout_debug_payload=None,
) -> None:
    """Produce one rollout payload and hand it to the buffer."""
    rollout_sde_indices = rollout_sde_controller.next_sde_indices()
    if debug_save_intermediates:
        debug_payload = ray_module.get(
            rollout_manager.build_training_debug_payload.remote(
                rollout_id,
                sde_indices=rollout_sde_indices,
            )
        )
        _push_rollout_training_batch(
            ray_module=ray_module,
            rollout_buffer=rollout_buffer,
            rollout_id=rollout_id,
            training_batch=debug_payload["training_batch"],
        )
        if save_rollout_debug_payload is not None:
            save_rollout_debug_payload(
                args=args,
                payload=debug_payload,
                rollout_id=rollout_id,
                source="train_loop",
            )
        return

    rollout_payload_ref = rollout_manager.produce_training_payload.remote(
        rollout_id=rollout_id,
        sde_indices=rollout_sde_indices,
        collect_media_preview=collect_media_preview,
        media_max_items=media_max_items,
    )
    _admit_rollout_payload_ref(
        ray_module=ray_module,
        rollout_buffer=rollout_buffer,
        payload_ref=rollout_payload_ref,
    )

def train(args):  # [PUBLIC-API → main()] sync 入口：资源创建 + 同步训练循环
    """Synchronous training entrypoint."""
    debug_mode = str(getattr(args.debug, "debug_mode", "none") or "none").strip().lower()
    if debug_mode == "train_only":
        from diffusionrl.debug import run_debug_train_only

        return run_debug_train_only(args)

    import ray

    from diffusionrl.ray.group_factory import create_rollout_actor_group, create_training_actor_group
    from diffusionrl.ray.group_runtime import RolloutGroupRuntime, TrainingGroupRuntime
    from diffusionrl.ray.placement_group import create_placement_groups_from_args
    from diffusionrl.ray.buffer_actor import create_buffer_actor
    from diffusionrl.ray.rollout_manager import create_rollout_manager
    from diffusionrl.utils import configure_logger, set_seed
    from diffusionrl.utils.wandb_metrics import (
        build_buffer_metrics,
        build_sync_metrics,
        compute_rollout_batch_metrics,
    )
    from diffusionrl.distributed.weight_sync import create_weight_sync
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger

    configure_logger()
    set_seed(args.seed)
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

    logger.info("Starting diffusionRL training...")
    logger.info(f"Model: {args.model.pretrained_model_saved_path}")
    logger.info(f"Algorithm: {algorithm_config['algorithm_path']}")
    logger.info(f"Mode: {runtime_mode_label}")
    logger.info(f"Offload train: {args.ray.offload_train}, Offload rollout: {args.ray.offload_rollout}")
    logger.info("Weight sync mode: %s", sync_mode)
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

    wandb_logger = None
    rollout_manager = None
    rollout_buffer = None
    rollout_group = None
    rollout_runtime = None
    training_group = None
    training_runtime = None
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

        # 1. Resource allocation
        pgs = create_placement_groups_from_args(args)
        logger.info("Placement groups created")

        rollout_on_gpu = True

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
        expected_global_batch_size = training_runtime.get_expected_global_batch_size()
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
            rollout_buffer.configure_expected_global_batch_size.remote(
                expected_global_batch_size=expected_global_batch_size,
            )
        )
        logger.info(
            "Rollout buffer actor created and configured with expected_global_batch_size=%s",
            expected_global_batch_size,
        )

        # 9. Setup weight-sync coordinator (binds training/rollout runtime facades).
        weight_sync.setup(
            training_runtime=training_runtime,
            rollout_runtime=rollout_runtime,
        )

        debug_save_intermediates = bool(getattr(args.debug, "debug_save_intermediates", False))

        # 10. Core synchronous training loop
        enforce_rollout_alignment = not bool(getattr(args.rollout, "rollout_buffer_reassemble_by_group", False))
        save_rollout_debug_payload = None
        if debug_save_intermediates:
            from diffusionrl.debug.runner import save_rollout_debug_payload as _save_rollout_debug_payload

            save_rollout_debug_payload = _save_rollout_debug_payload

        wandb_media_enabled = bool(
            wandb_logger is not None and bool(getattr(args.rollout, "wandb_log_media", False))
        )
        wandb_media_max_items = max(1, int(getattr(args.rollout, "wandb_media_max_items", 8)))

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
            training_data_handle = None
            rollout_metadata = {}
            sample_count = 0
            # === PHASE 1: Rollout ===
            rollout_phase_start_t = time.perf_counter()
            rollout_on_gpu = _ensure_rollout_on_gpu(
                args=args,
                rollout_runtime=rollout_runtime,
                rollout_on_gpu=rollout_on_gpu,
            )
            _produce_and_push_rollout(
                ray_module=ray,
                args=args,
                rollout_manager=rollout_manager,
                rollout_buffer=rollout_buffer,
                rollout_id=rollout_id,
                rollout_sde_controller=rollout_sde_controller,
                debug_save_intermediates=debug_save_intermediates,
                collect_media_preview=collect_media_preview,
                media_max_items=wandb_media_max_items,
                save_rollout_debug_payload=save_rollout_debug_payload,
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

            if args.ray.offload_rollout and rollout_runtime is not None:
                rollout_runtime.sleep()
                rollout_on_gpu = False

            # === PHASE 2: Training ===
            train_phase_start_t = time.perf_counter()
            if args.ray.offload_train:
                training_runtime.onload()

            metrics = training_group.train(rollout_id, training_data_handle)
            train_phase_s = time.perf_counter() - train_phase_start_t

            # Periodic: save (before offload to ensure model is on GPU)
            if should_save(rollout_id, args):
                save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
                training_runtime.save_model(save_path)
                logger.info(f"Checkpoint saved: {save_path}")

            # === PHASE 3: Offload + Weight Sync ===
            _offload_train_phase(args=args, training_runtime=training_runtime)

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
                rollout_on_gpu = _ensure_rollout_on_gpu(
                    args=args,
                    rollout_runtime=rollout_runtime,
                    rollout_on_gpu=rollout_on_gpu,
                )
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

                    rollout_metrics = collect_rollout_batch_metrics(
                        ray_module=ray,
                        batch_ref=training_data_handle,
                        compute_rollout_batch_metrics_fn=compute_rollout_batch_metrics,
                    )
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
        if wandb_logger:
            wandb_logger.finish()

    logger.info("Training complete!")


def main(argv=None):  # [PUBLIC-API → __main__] sync CLI 入口
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
