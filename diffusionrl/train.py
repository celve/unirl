#!/usr/bin/env python
"""
diffusionrl Training Entry Point.

Usage:
    python -m diffusionrl.train --config scripts/example_flux_dancegrpo_direct.yaml
"""
import logging
import time
from typing import Any, Dict

from diffusionrl.config import (
    build_resolved_config_view,
    parse_args,
)
from diffusionrl.config.launch_resolution import resolve_launch_config
from diffusionrl.utils.train_utils import (
    RolloutSDEController,
    build_control_algorithm,
    collect_rollout_batch_metrics,
    create_rollout_timestep_scheduler,
    maybe_restore_start_rollout_id_from_checkpoint,
    should_eval,
    should_log,
    should_save,
)

logger = logging.getLogger(__name__)

# Main control-plane path (sync mode):
# parse_args -> create_placement_groups_from_args -> create_rollout_services
# -> create_training_actor_group -> prepare RolloutRequest(s) -> execute sampling
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

def _produce_and_push_rollout(
    *,
    ray_module,
    args,
    rollout_services,
    rollout_function,
    rollout_function_path: str,
    reward_hook,
    rollout_buffer,
    rollout_id: int,
    rollout_sde_controller,
    debug_save_intermediates: bool,
    collect_media_preview: bool,
    media_max_items: int,
    save_rollout_debug_payload=None,
) -> None:
    """Produce one rollout payload and hand it to the buffer."""
    from diffusionrl.rollout.base_types import RolloutContext, RolloutFunctionResult
    from diffusionrl.rollout.factory import DEFAULT_ROLLOUT_FUNCTION_PATH

    rollout_sde_indices = rollout_sde_controller.next_sde_indices()
    context = RolloutContext(
        rollout_id=int(rollout_id),
        sde_indices=rollout_sde_indices,
        collect_media_preview=bool(collect_media_preview),
        media_max_items=int(media_max_items),
        debug_trace={} if debug_save_intermediates else None,
    )
    if debug_save_intermediates and str(rollout_function_path).strip() != DEFAULT_ROLLOUT_FUNCTION_PATH:
        raise ValueError(
            "debug_save_intermediates currently requires the default request-centric rollout pipeline. "
            f"Got custom rollout_function_path={rollout_function_path!r}."
        )

    rollout_result = rollout_function(
        services=rollout_services,
        reward_hook=reward_hook,
        context=context,
    )
    if not isinstance(rollout_result, RolloutFunctionResult):
        raise TypeError(
            "Rollout function must return RolloutFunctionResult. "
            f"Got type={type(rollout_result)}"
        )
    advantages = rollout_services.compute_advantages(
        rewards=rollout_result.rewards,
        group_ids=rollout_result.request.meta.get("group_ids"),
        reward_components=rollout_result.reward_components,
    )
    training_batch = rollout_services.assemble_training_batch(
        request=rollout_result.request,
        sampler_outputs=rollout_result.sampler_outputs,
        rewards=rollout_result.rewards,
        advantages=advantages,
        sde_indices=context.sde_indices,
    )
    metadata = dict(rollout_result.metadata or {})

    if debug_save_intermediates:
        debug_payload = dict(context.debug_trace or {})
        debug_payload.setdefault("training_batch", training_batch)
        debug_payload.setdefault("metadata", dict(metadata or {}))
        debug_payload.setdefault("request", rollout_result.request)
        debug_payload.setdefault("sampler_outputs", rollout_result.sampler_outputs)
        debug_payload.setdefault("rewards", rollout_result.rewards)
        debug_payload.setdefault("advantages", advantages)
        debug_payload.setdefault(
            "reward_components",
            dict(rollout_result.reward_components or {}),
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

    _push_rollout_training_batch(
        ray_module=ray_module,
        rollout_buffer=rollout_buffer,
        rollout_id=rollout_id,
        training_batch=training_batch,
        metadata=dict(metadata or {}),
    )

def train(args):  # [PUBLIC-API → main()] sync 入口：资源创建 + 同步训练循环
    """Synchronous training entrypoint."""
    debug_mode = str(args.debug.debug_mode or "none").strip().lower()
    if debug_mode == "train_only":
        from diffusionrl.debug import run_debug_train_only

        return run_debug_train_only(args)

    import ray

    from diffusionrl.ray.group_factory import create_rollout_actor_group, create_training_actor_group
    from diffusionrl.ray.group_runtime import RolloutGroupRuntime, TrainingGroupRuntime
    from diffusionrl.ray.placement_group import create_placement_groups_from_launch
    from diffusionrl.ray.buffer_actor import create_buffer_actor
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
    launch_config = resolve_launch_config(args)
    algorithm_config, control_algorithm = build_control_algorithm(launch_config.algorithm_config)
    rollout_mode_info = launch_config.rollout_mode_info
    rollout_topology = rollout_mode_info.rollout_topology
    training_actor_sampling_mode = rollout_mode_info.training_actor_sampling_mode
    sync_mode = rollout_mode_info.sync_protocol
    rollout_mode_name = rollout_topology.mode

    rollout_sde_controller = RolloutSDEController(
        algorithm=control_algorithm,
        timestep_scheduler=create_rollout_timestep_scheduler(args, algorithm=control_algorithm),
    )

    rollout_control = args.rollout.control
    rollout_artifacts = args.rollout.artifacts
    rollout_evaluation = args.rollout.evaluation
    rollout_logging = args.rollout.logging
    rollout_buffer_settings = args.rollout.buffer

    logger.info("Starting diffusionRL training...")
    logger.info(f"Model: {args.model.pretrained_model_saved_path}")
    logger.info(f"Algorithm: {algorithm_config['algorithm_path']}")
    logger.info(f"Mode: {rollout_mode_name}")
    logger.info(f"Offload train: {args.ray.offload_train}, Offload rollout: {args.ray.offload_rollout}")
    logger.info("Weight sync mode: %s", sync_mode)
    logger.info(
        "Periodic controls: save_steps=%s eval_steps=%s logging_steps=%s",
        rollout_artifacts.save_steps,
        rollout_evaluation.eval_steps,
        rollout_logging.logging_steps,
    )
    logger.info(
        "Debug flags: mode=%s save_intermediates=%s save_dir=%s",
        debug_mode,
        bool(args.debug.debug_save_intermediates),
        args.debug.debug_save_dir,
    )

    # Initialize Ray
    if not ray.is_initialized():
        if args.ray.ray_address:
            ray.init(address=args.ray.ray_address, ignore_reinit_error=True)
        else:
            ray.init()

    wandb_logger = None
    rollout_services = None
    rollout_function = None
    rollout_function_path = ""
    eval_function = None
    reward_hook = None
    rollout_buffer = None
    rollout_group = None
    rollout_runtime = None
    training_group = None
    training_runtime = None
    weight_sync = create_weight_sync(args, launch_config, mode=sync_mode)

    try:
        if rollout_logging.report_to_wandb and rollout_logging.project_name:
            wandb_tags_str = rollout_logging.wandb_tags
            wandb_tags = (
                [t.strip() for t in wandb_tags_str.split(",") if t.strip()]
                if wandb_tags_str
                else None
            )
            wandb_entity = rollout_logging.wandb_entity or None
            wandb_logger = init_logger(
                project=rollout_logging.project_name,
                run_name=rollout_logging.run_name,
                config=build_resolved_config_view(args),
                log_dir=rollout_logging.logging_dir,
                rank=0,
                tags=wandb_tags,
                entity=wandb_entity,
                require_success=True,
            )
            if wandb_logger.initialized:
                logger.info(
                    "WandB initialized: project=%s, run=%s",
                    rollout_logging.project_name,
                    rollout_logging.run_name,
                )

        # 1. Resource allocation
        pgs = create_placement_groups_from_launch(launch_config)
        logger.info("Placement groups created")

        rollout_on_gpu = True

        # 2. Create rollout services and load rollout hooks.
        from diffusionrl.rollout.factory import (
            DEFAULT_EVAL_FUNCTION_PATH,
            DEFAULT_REWARD_HOOK_PATH,
            DEFAULT_ROLLOUT_FUNCTION_PATH,
            create_rollout_services,
        )
        from diffusionrl.utils import load_function

        rollout_services, dataset_step_info = create_rollout_services(
            args,
            reward_pg_result=pgs.get("reward"),
            launch_config=launch_config,
        )
        rollout_function_path = str(args.rollout_function_path or DEFAULT_ROLLOUT_FUNCTION_PATH)
        rollout_function = load_function(rollout_function_path)
        eval_function = load_function(str(args.eval_function_path or DEFAULT_EVAL_FUNCTION_PATH))
        reward_hook = load_function(str(args.reward_hook_path or DEFAULT_REWARD_HOOK_PATH))
        logger.info("Rollout services created")
        if dataset_step_info.get("num_prompts", 0) > 0:
            logger.info(
                "Dataset step info: num_prompts=%s prompts_per_rollout=%s "
                "estimated_steps_per_dataset_pass=%s steps_before_reset=%s",
                dataset_step_info.get("num_prompts"),
                dataset_step_info.get("prompts_per_rollout"),
                dataset_step_info.get("estimated_steps_per_dataset_pass"),
                dataset_step_info.get("steps_before_reset"),
            )
            if not dataset_step_info.get("exact_dataset_pass_per_cycle", False):
                logger.warning(
                    "Dataset pass is not exact under current data-source batching: "
                    "drop_last=%s remainder_prompts=%s. "
                    "The data source will drop trailing prompts rather than emit a short rollout batch, "
                    "so one reset cycle will not cover the full dataset exactly once.",
                    dataset_step_info.get("drop_last"),
                    dataset_step_info.get("remainder_prompts"),
                )

        if not training_actor_sampling_mode:
            rollout_pg_result = pgs.get("rollout")
            if rollout_pg_result is None:
                raise ValueError("Missing rollout placement-group allocation.")
            rollout_group = create_rollout_actor_group(launch_config, rollout_pg_result)
            rollout_runtime = RolloutGroupRuntime.from_group(rollout_group)
            rollout_services.attach_sampling_group(rollout_group)
            logger.info("Rollout actor group created and attached to rollout services")

        # 3. Optional pre-offload for colocate memory pressure relief before training actor init.
        if args.ray.offload_rollout and rollout_runtime is not None:
            rollout_runtime.sleep()
            rollout_on_gpu = False

        # 4. Create training actors.
        training_pg_result = pgs.get("training")
        if training_pg_result is None:
            raise ValueError("Missing training placement-group allocation.")
        training_group = create_training_actor_group(
            launch_config,
            training_pg_result,
        )
        training_runtime = TrainingGroupRuntime.from_group(training_group)
        resume_from_checkpoint = rollout_artifacts.resume_from_checkpoint
        if resume_from_checkpoint:
            training_runtime.load_checkpoint(resume_from_checkpoint)
            logger.info("Checkpoint loaded: %s", resume_from_checkpoint)
            restored_rollout_id = maybe_restore_start_rollout_id_from_checkpoint(
                args,
                resume_from_checkpoint,
            )
            if restored_rollout_id is not None:
                logger.info(
                    "Auto-set start_rollout_id=%s from checkpoint path.",
                    restored_rollout_id,
                )
        train_backend_info = training_runtime.get_train_backend_info()
        expected_global_batch_size = training_runtime.get_expected_global_batch_size()
        logger.info("Training actor group created")
        if train_backend_info:
            logger.info("Training backend: %s", train_backend_info)

        # 5. Direct-sampling path: training actors serve rollout requests.
        if training_actor_sampling_mode:
            rollout_services.attach_sampling_group(training_group)
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

        debug_save_intermediates = bool(args.debug.debug_save_intermediates)

        # 10. Core synchronous training loop
        enforce_rollout_alignment = not bool(rollout_buffer_settings.reassemble_by_group)
        save_rollout_debug_payload = None
        if debug_save_intermediates:
            from diffusionrl.debug.runner import save_rollout_debug_payload as _save_rollout_debug_payload

            save_rollout_debug_payload = _save_rollout_debug_payload

        wandb_media_enabled = bool(wandb_logger is not None and bool(rollout_logging.wandb_log_media))
        wandb_media_max_items = max(1, int(rollout_logging.wandb_media_max_items))

        # rollout_id is the outer rollout-train loop step; it behaves similarly to
        # a framework-level global step, but may differ from optimizer step count.
        # global_optimizer_step tracks real optimizer step for wandb logging
        global_optimizer_step = 0
        for rollout_id in range(rollout_control.start_rollout_id, rollout_control.num_rollout):
            step_start_t = time.perf_counter()
            sync_result = None
            sync_phase_s = 0.0
            eval_phase_s = 0.0
            should_log_step = should_log(rollout_id, args)
            collect_media_preview = bool(should_log_step and wandb_media_enabled)
            training_data_handle = None
            rollout_metadata = {}
            sample_count = 0
            # Periodic: evaluate at loop start before rollout/train of this step.
            if should_eval(rollout_id, args):
                eval_phase_start_t = time.perf_counter()
                rollout_on_gpu = _ensure_rollout_on_gpu(
                    args=args,
                    rollout_runtime=rollout_runtime,
                    rollout_on_gpu=rollout_on_gpu,
                )
                if training_actor_sampling_mode:
                    with training_runtime.eval_ema_context():
                        eval_metrics = eval_function(
                            services=rollout_services,
                            reward_hook=reward_hook,
                            rollout_id=int(rollout_id),
                        )
                else:
                    eval_metrics = eval_function(
                        services=rollout_services,
                        reward_hook=reward_hook,
                        rollout_id=int(rollout_id),
                    )
                eval_phase_s = time.perf_counter() - eval_phase_start_t
                logger.info(f"Eval at {rollout_id}: mean_reward={eval_metrics['mean_reward']:.4f}")

                if wandb_logger:
                    wandb_logger.log_eval(rollout_id, eval_metrics)

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
                rollout_services=rollout_services,
                rollout_function=rollout_function,
                rollout_function_path=rollout_function_path,
                reward_hook=reward_hook,
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
                save_path = f"{rollout_artifacts.output_dir}/checkpoint-{rollout_id}"
                training_runtime.save_model(save_path)
                logger.info(f"Checkpoint saved: {save_path}")

            # === PHASE 3: Offload + Weight Sync ===
            _offload_train_phase(args=args, training_runtime=training_runtime)

            if (
                not training_actor_sampling_mode
                and (rollout_id + 1) % rollout_control.update_weights_interval == 0
            ):
                sync_phase_start_t = time.perf_counter()
                sync_result = weight_sync.sync(rollout_id=rollout_id)
                sync_phase_s = time.perf_counter() - sync_phase_start_t
                rollout_on_gpu = True  # Coordinator internally calls wake_up

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
        if rollout_services is not None:
            rollout_services.dispose()
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
