#!/usr/bin/env python
"""
diffusionrl Training Entry Point.

Spawns ``TrainActor`` and ``RolloutActor`` handles directly and drives the
rollout loop via ``RolloutPipeline``.

Usage:
    python -m diffusionrl.train --config scripts/example_flux_dancegrpo_direct.yaml
"""

import logging
import time

logger = logging.getLogger(__name__)

"""
Main control-plane path (sync mode):
    parse_args_with_derived_config -> build_launch_config
    -> validate_launch_config_for_train -> create_placement_groups_from_launch
    -> RolloutActorGroup.bootstrap -> rollout_group
    -> TrainActorGroup.bootstrap -> (train_group, master_addr, master_port)
    -> per rollout: RolloutPipeline.run_once (load_prompts + plan_requests
       + exec_request [actor-side run_rollout_pipeline fuses
       generate_buffered + attach_reward + compute_advantages + get_buffer]
       + aggregate + convert_training_data -> TrainingBatch)
    -> train_group.train(rollout_id, batch) [slices per-rank, dispatches]
    -> train_group.sync_weights_to_rollout
"""


def train(args, *, derived_config):  # [PUBLIC-API -> main()] sync entrypoint
    """Synchronous training entrypoint built on TrainActor / RolloutActor handles."""
    debug_mode = str(args.debug.mode or "none").strip().lower()
    if debug_mode == "train_only":
        from diffusionrl.debug import run_debug_train_only

        return run_debug_train_only(args, derived_config=derived_config)

    import ray

    from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
    from diffusionrl.cmdline.resolution import build_launch_config
    from diffusionrl.cmdline.schema import build_derived_config_view
    from diffusionrl.config.validation import validate_launch_config_for_train
    from diffusionrl.distributed.weight_sync import build_weight_sync_config
    from diffusionrl.ray.group import RolloutActorGroup, TrainActorGroup
    from diffusionrl.ray.placement_group import create_placement_groups_from_launch
    from diffusionrl.rollout.pipeline import RolloutPipeline, build_media_preview
    from diffusionrl.utils import configure_logger, load_function, set_seed
    from diffusionrl.utils.train_utils import (
        collect_rollout_batch_metrics,
        maybe_restore_start_rollout_id_from_checkpoint,
        should_eval,
        should_log,
        should_save,
    )
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger
    from diffusionrl.utils.wandb_metrics import compute_rollout_batch_metrics

    configure_logger()
    set_seed(args.seed)
    launch_config = build_launch_config(args, derived_config=derived_config)
    algorithm_init_payload = launch_config.algorithm_init_payload
    control_algorithm = create_algorithm_from_init_payload(algorithm_init_payload)
    rollout_info = launch_config.rollout_info
    sync_mode = rollout_info.sync_protocol
    rollout_mode_name = rollout_info.mode
    training_actor_sampling_mode = rollout_info.training_actor_sampling_mode

    validate_launch_config_for_train(launch_config=launch_config)

    logger.info("Starting diffusionRL training...")
    logger.info(f"Model: {args.model.pretrained_model_ckpt_path}")
    logger.info("Algorithm: %s", algorithm_init_payload.component_dotpath)
    logger.info(f"Mode: {rollout_mode_name}")
    logger.info(f"Offload train: {args.ray.offload_train}, Offload rollout: {args.ray.offload_rollout}")
    logger.info("Weight sync mode: %s", sync_mode)
    logger.info(
        "Periodic controls: save_steps=%s eval_steps=%s logging_steps=%s",
        args.rollout.save_steps,
        args.evaluation.eval_steps,
        args.logging.logging_steps,
    )
    logger.info(
        "Debug flags: mode=%s save_intermediates=%s save_dir=%s",
        debug_mode,
        bool(args.debug.save_intermediates),
        args.debug.save_dir,
    )

    # Initialize Ray
    if not ray.is_initialized():
        if args.ray.ray_address:
            ray.init(address=args.ray.ray_address, ignore_reinit_error=True)
        else:
            # Cap num_cpus to avoid slow startup on high-core machines (e.g. 384 cores)
            # where Ray pre-starts one worker per CPU, causing connect timeouts.
            import os

            _max_cpus = min(os.cpu_count() or 64, 64)
            ray.init(num_cpus=_max_cpus)

    wandb_logger = None
    rollout_group = None
    train_group = None
    sync_config = None

    try:
        if args.logging.report_to_wandb and args.logging.project_name:
            raw_tags = args.logging.tags
            if raw_tags:
                if isinstance(raw_tags, str):
                    raw_tags = raw_tags.split(",")
                wandb_tags = [t.strip() for t in raw_tags if t.strip()]
            else:
                wandb_tags = None
            wandb_logger = init_logger(
                project=args.logging.project_name,
                run_name=args.logging.run_name,
                config=build_derived_config_view(args.to_dotted_dict(), derived_config=derived_config),
                log_dir=args.logging.logging_dir,
                rank=0,
                tags=wandb_tags,
                entity=args.logging.entity or None,
                require_success=True,
            )
            if wandb_logger.initialized:
                logger.info(
                    "WandB initialized: project=%s, run=%s",
                    args.logging.project_name,
                    args.logging.run_name,
                )

        # 1. Resource allocation
        pgs = create_placement_groups_from_launch(launch_config)
        logger.info("Placement groups created")
        rollout_on_gpu = True

        # 2. Driver-side data source + sampling defaults
        data_source_cls = load_function(args.data_source_dotpath)
        data_source = data_source_cls(args)
        prompt_batch_size = int(getattr(control_algorithm, "prompts_per_rollout", args.algorithm.prompts_per_rollout))
        samples_per_prompt = int(getattr(control_algorithm, "samples_per_prompt", 1))
        sampling_spec = launch_config.sampling_spec
        logger.info(
            "Driver-side rollout config: prompt_batch_size=%s samples_per_prompt=%s",
            prompt_batch_size,
            samples_per_prompt,
        )
        rollout_pipeline = RolloutPipeline()

        # 3. Bootstrap rollout group (spawns actors + wraps in single facade)
        if not training_actor_sampling_mode:
            rollout_pgs = pgs.get("rollout")
            if rollout_pgs is None:
                raise ValueError("Missing rollout placement-group allocation.")
            rollout_group = RolloutActorGroup.bootstrap(
                launch_config=launch_config,
                rollout_pgs=rollout_pgs,
            )

        # 4. Optional pre-offload to relieve memory pressure before train init
        if not training_actor_sampling_mode and args.ray.offload_rollout:
            rollout_group.sleep()
            rollout_on_gpu = False

        # 5. Bootstrap training group (picks rendezvous + spawns train actors)
        training_pgs = pgs.get("training")
        if training_pgs is None:
            raise ValueError("Missing training placement-group allocation.")
        train_group, master_addr, master_port = TrainActorGroup.bootstrap(
            launch_config=launch_config,
            training_pgs=training_pgs,
        )
        logger.info(
            "Resolved training distributed master: addr=%s port=%s world_size=%d",
            master_addr,
            master_port,
            train_group.num_actors,
        )

        if args.training.resume_from_checkpoint:
            train_group.load_checkpoint(args.training.resume_from_checkpoint)
            logger.info("Checkpoint loaded: %s", args.training.resume_from_checkpoint)
            restored_rollout_id = maybe_restore_start_rollout_id_from_checkpoint(
                args,
                args.training.resume_from_checkpoint,
            )
            if restored_rollout_id is not None:
                logger.info(
                    "Auto-set start_rollout_id=%s from checkpoint path.",
                    restored_rollout_id,
                )

        # Driver-side values (replace runtime calls that don't exist on TrainActor)
        training_plan = launch_config.training.actor_init_config.training_plan_config
        expected_global_batch_size = int(training_plan.global_batch_size)
        train_backend_info = {
            "name": launch_config.training.backend_name,
            "topology": launch_config.training.topology.as_dict(),
            "training_plan": training_plan.as_dict(),
        }
        logger.info("Training backend: %s", train_backend_info)
        logger.info("Expected global batch size: %s", expected_global_batch_size)

        # 6. Initial weight broadcast skipped (FSDP2 deterministic per-rank init)
        logger.info("Initial update_weights() skipped — FSDP2 TrainActor inits deterministically per rank")

        # 7. Post-init offload dance
        if not training_actor_sampling_mode and args.ray.offload_rollout:
            if args.ray.offload_train:
                train_group.offload()
            rollout_group.wake_up()
            rollout_on_gpu = True

        # 8. Weight sync setup (no central BufferActor)
        if not training_actor_sampling_mode:
            sync_config = build_weight_sync_config(
                args,
                launch_config,
                mode=sync_mode,
                rollout_runtime=rollout_group,
            )
            if sync_config:
                train_group.setup_weight_sync(sync_config)

        # 8b. Resolve pipeline dispatch targets
        if training_actor_sampling_mode:
            pipeline_actors = train_group.get_actors()
            pipeline_handle_group = train_group.handle
        else:
            pipeline_actors = rollout_group.get_actors()
            pipeline_handle_group = rollout_group.handle

        # 9. Main training loop
        # rollout_id is the outer rollout-train loop step; it behaves similarly to
        # a framework-level global step, but may differ from optimizer step count.
        # global_optimizer_step tracks real optimizer step for wandb logging
        wandb_media_enabled = wandb_logger is not None and args.logging.log_media
        wandb_media_max_items = max(1, int(args.logging.media_max_items))
        global_optimizer_step = 0
        for rollout_id in range(args.rollout.start_rollout_id, args.rollout.num_rollout):
            step_start_t = time.perf_counter()
            sync_phase_s = 0.0
            eval_phase_s = 0.0
            should_log_step = should_log(rollout_id, args)

            # === Periodic Eval (before rollout/train of this step) ===
            if should_eval(rollout_id, args):
                eval_phase_start_t = time.perf_counter()
                if not training_actor_sampling_mode:
                    if args.ray.offload_rollout and not rollout_on_gpu:
                        rollout_group.wake_up()
                        rollout_on_gpu = True

                def _run_eval():
                    return rollout_pipeline.run_eval(
                        rollout_actors=pipeline_actors,
                        rollout_handle_group=pipeline_handle_group,
                        data_source=data_source,
                        prompt_batch_size=prompt_batch_size,
                        samples_per_prompt=samples_per_prompt,
                        sampling_spec=sampling_spec,
                        evaluation_settings=args.evaluation,
                        rollout_id=rollout_id,
                        collect_media_preview=wandb_media_enabled,
                        media_max_items=wandb_media_max_items,
                    )

                if training_actor_sampling_mode:
                    with train_group.use_eval_ema():
                        eval_metrics = _run_eval()
                else:
                    eval_metrics = _run_eval()
                eval_phase_s = time.perf_counter() - eval_phase_start_t
                logger.info(
                    "Eval at %s: mean_reward=%.4f std_reward=%.4f (%.3fs)",
                    rollout_id,
                    eval_metrics.get("mean_reward", 0.0),
                    eval_metrics.get("std_reward", 0.0),
                    eval_phase_s,
                )
                if wandb_logger:
                    # Pop media_preview before log_eval so only scalar metrics
                    # (mean/std reward, num_samples, rollout_id) go through the
                    # ``eval/`` numeric panel; images are logged separately.
                    eval_media_preview = eval_metrics.pop("media_preview", None)
                    wandb_logger.log_eval(rollout_id, eval_metrics)
                    if eval_media_preview:
                        wandb_logger.log_generated_media(
                            rollout_id,
                            eval_media_preview,
                            key="eval/generated_media",
                        )

            # === PHASE A: Rollout ===
            rollout_phase_start_t = time.perf_counter()
            if not training_actor_sampling_mode:
                if args.ray.offload_rollout and not rollout_on_gpu:
                    rollout_group.wake_up()
                    rollout_on_gpu = True

            collect_rollout_preview = bool(should_log_step and wandb_media_enabled)
            training_batch, sample_count, rollout_response = rollout_pipeline.run_once(
                rollout_actors=pipeline_actors,
                rollout_handle_group=pipeline_handle_group,
                data_source=data_source,
                prompt_batch_size=prompt_batch_size,
                samples_per_prompt=samples_per_prompt,
                sampling_spec=sampling_spec,
                control_algorithm=control_algorithm,
                rollout_id=rollout_id,
                collect_media_preview=collect_rollout_preview,
                media_max_items=wandb_media_max_items,
            )
            rollout_phase_s = time.perf_counter() - rollout_phase_start_t

            # Even when no preview was requested, the response is now small
            # (actors drop decoded_images after scoring), so we keep the
            # reference for the logging path below. No eager ``del`` needed.

            if not training_actor_sampling_mode and args.ray.offload_rollout:
                rollout_group.sleep()
                rollout_on_gpu = False

            # === PHASE B: Training ===
            train_phase_start_t = time.perf_counter()
            if args.ray.offload_train:
                train_group.onload()

            batch_ref = ray.put(training_batch)  # for collect_rollout_batch_metrics below
            results = train_group.train(rollout_id, training_batch)
            metrics = [r.to_legacy_metric_dict() for r in results]
            train_phase_s = time.perf_counter() - train_phase_start_t

            # Periodic save (collective; before offload so weights are still on GPU)
            if should_save(rollout_id, args):
                save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
                train_group.save_model(save_path)
                logger.info(f"Checkpoint saved: {save_path}")

            # === PHASE C: Offload + Weight Sync ===
            if args.ray.offload_train:
                train_group.offload()
            else:
                train_group.clear_memory()

            if (
                not training_actor_sampling_mode
                and sync_config
                and (rollout_id + 1) % args.sync.rollout_update_interval == 0
            ):
                sync_phase_start_t = time.perf_counter()
                rollout_group.wake_up()
                train_group.sync_weights_to_rollout()
                sync_phase_s = time.perf_counter() - sync_phase_start_t
                rollout_on_gpu = True

            # === Per-optimizer-step wandb logging ===
            if wandb_logger and metrics:
                per_step_list = metrics[0].get("_per_optimizer_step_metrics", [])
                for per_step_m in per_step_list:
                    if per_step_m.get("has_backward", False):
                        global_optimizer_step += 1
                        wandb_step_m = {k: v for k, v in per_step_m.items() if k != "has_backward"}
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
                        batch_ref=batch_ref,
                        compute_rollout_batch_metrics_fn=compute_rollout_batch_metrics,
                    )
                    if rollout_metrics:
                        wandb_logger.log_rollout(rollout_id, rollout_metrics)

                    if wandb_media_enabled:
                        # Primary path: actors populated ``samples.media_preview``
                        # already (capped in ``RolloutPipeline.aggregate``). Fall
                        # back to ``build_media_preview`` only if a legacy code
                        # path left it unset.
                        media_preview = build_media_preview(
                            rollout_response,
                            max_items=wandb_media_max_items,
                        )
                        if media_preview is not None:
                            wandb_logger.log_generated_media(rollout_id, media_preview)
                        del rollout_response

                    perf_metrics = {
                        "rollout_phase_s": rollout_phase_s,
                        "train_phase_s": train_phase_s,
                        "sync_phase_s": sync_phase_s,
                        "eval_phase_s": eval_phase_s,
                        "step_time_s": step_time_s,
                        "samples_per_rollout": float(sample_count),
                        "samples_per_s": (
                            float(sample_count) / float(step_time_s) if step_time_s > 0 and sample_count > 0 else 0.0
                        ),
                    }
                    wandb_logger.log_perf(rollout_id, perf_metrics)

                    if sync_phase_s > 0:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics={"sync/elapsed_s": sync_phase_s},
                        )
    finally:
        if not training_actor_sampling_mode and sync_config and train_group is not None:
            try:
                train_group.teardown_weight_sync()
            except Exception:
                logger.exception("Weight-sync teardown failed")
        if rollout_group is not None:
            rollout_group.dispose()
        if train_group is not None:
            train_group.dispose()
        if wandb_logger:
            wandb_logger.finish()

    logger.info("Training complete!")


def main(argv=None):  # [PUBLIC-API -> __main__] sync CLI entrypoint
    from diffusionrl.cmdline.parse_args import parse_args_with_derived_config

    args, derived_config = parse_args_with_derived_config(argv)
    train(args, derived_config=derived_config)


if __name__ == "__main__":
    main()
