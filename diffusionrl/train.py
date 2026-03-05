#!/usr/bin/env python
"""
diffusionrl Training Entry Point.

Reference: slime/train.py

Usage:
    python -m diffusionrl.train --pretrained-model-saved-path /path/to/model --num-rollout 100
"""
import logging
import os
import re
import time

from diffusionrl.config import parse_args
from diffusionrl.config.arguments import is_training_actor_direct_sampling_mode
from diffusionrl.ray.group_factory import create_training_actor_group
from diffusionrl.ray.placement_group import create_placement_groups_from_args
from diffusionrl.ray.rollout_buffer import create_rollout_buffer_actor
from diffusionrl.ray.rollout_manager import create_rollout_manager

logger = logging.getLogger(__name__)

# Main control-plane path (sync mode):
# parse_args -> create_placement_groups_from_args -> create_rollout_manager
# -> create_training_actor_group -> rollout_manager.generate_and_push
# -> training_group.train -> weight_sync.sync


def should_save(rollout_id: int, args) -> bool:
    """Check if we should save a checkpoint at this rollout."""
    return (rollout_id + 1) % args.rollout.save_steps == 0


def should_eval(rollout_id: int, args) -> bool:
    """Check if we should run evaluation at this rollout."""
    return (rollout_id + 1) % args.rollout.eval_steps == 0


def train(args):
    """Main training loop."""
    debug_mode = str(getattr(args.debug, "debug_mode", "none") or "none").strip().lower()
    if debug_mode == "rollout_only":
        from diffusionrl.debug import run_debug_rollout_only

        return run_debug_rollout_only(args)
    if debug_mode == "train_only":
        from diffusionrl.debug import run_debug_train_only

        return run_debug_train_only(args)
    if debug_mode == "interactive":
        from diffusionrl.debug import run_debug_interactive

        return run_debug_interactive(args)

    import ray

    from diffusionrl.utils import configure_logger, set_seed
    from diffusionrl.utils.wandb_metrics import (
        build_buffer_metrics,
        build_sync_metrics,
        compute_rollout_batch_metrics,
    )
    from diffusionrl.utils.weight_sync import create_weight_sync_protocol
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger

    configure_logger()
    set_seed(args.seed)

    logger.info("Starting GRPO training...")
    logger.info(f"Model: {args.model.pretrained_model_saved_path}")
    logger.info(f"Algorithm: {args.algorithm.algorithm_path}")
    logger.info(f"Mode: {'colocate' if args.ray.colocate_rollout_training else 'separate'}")
    logger.info(f"Offload train: {args.ray.offload_train}, Offload rollout: {args.ray.offload_rollout}")
    logger.info(f"Weight sync mode: {args.ray.weight_sync_mode}, async_pipeline={args.rollout.async_pipeline}")
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
        wandb_logger = init_logger(
            project=args.rollout.project_name,
            run_name=args.rollout.run_name,
            config=args.to_flat_dict() if hasattr(args, "to_flat_dict") else vars(args),
            log_dir=getattr(args.rollout, "logging_dir", None),
            rank=0,
        )
        logger.info(f"WandB initialized: project={args.rollout.project_name}, run={args.rollout.run_name}")

    # 1. Resource allocation
    pgs = create_placement_groups_from_args(args)
    logger.info("Placement groups created")

    training_actor_direct_sampling = is_training_actor_direct_sampling_mode(args)
    rollout_on_gpu = True
    weight_sync = create_weight_sync_protocol(args)

    # 2. Create rollout manager; direct-sampling mode does not allocate rollout actors.
    rollout_pg_result = None if training_actor_direct_sampling else pgs.get("rollout")
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args,
        pg_result=rollout_pg_result,
        reward_pg_result=pgs.get("reward"),
    )
    logger.info(f"Rollout manager created, {num_rollout_per_epoch} rollouts per epoch")

    # 3. Optional pre-offload for colocate memory pressure relief before training actor init.
    if args.ray.offload_rollout:
        ray.get(rollout_manager.sleep.remote())
        rollout_on_gpu = False

    # 4. Create training actors.
    training_pg_result = pgs.get("training")
    if training_pg_result is None:
        raise ValueError("Missing training placement-group allocation.")
    training_group = create_training_actor_group(args, training_pg_result)
    resume_from_checkpoint = getattr(args.rollout, "resume_from_checkpoint", None)
    if resume_from_checkpoint:
        training_group.load_checkpoint(resume_from_checkpoint)
        logger.info("Checkpoint loaded: %s", resume_from_checkpoint)
        if int(getattr(args.rollout, "start_rollout_id", 0)) == 0:
            match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(resume_from_checkpoint)))
            if match:
                args.rollout.start_rollout_id = int(match.group(1)) + 1
                logger.info(
                    "Auto-set start_rollout_id=%s from checkpoint path.",
                    args.rollout.start_rollout_id,
                )
    train_backend_info = training_group.get_train_backend_info()
    buffer_consumer_spec = training_group.get_buffer_consumer_spec()
    logger.info("Training actor group created")
    if train_backend_info:
        logger.info("Training backend: %s", train_backend_info)

    # 5. FSDP direct-sampling path: training actors serve rollout requests.
    if training_actor_direct_sampling:
        ray.get(rollout_manager.attach_sampling_actors.remote(training_group))
        logger.info("Attached training actors as rollout sampling source")

    # 6. Initial weight synchronization for rollout-side actors.
    training_group.update_weights()
    logger.info("Initial weights synchronized")

    # 7. Restore rollout side and offload train side as needed after initial sync.
    if args.ray.offload_rollout and args.ray.offload_train:
        training_group.offload()
        ray.get(rollout_manager.wake_up.remote())
        rollout_on_gpu = True
    elif args.ray.offload_rollout:
        ray.get(rollout_manager.wake_up.remote())
        rollout_on_gpu = True

    # 8. Build rollout buffer and bind runtime handles.
    rollout_buffer = create_rollout_buffer_actor(args)
    ray.get(
        rollout_buffer.bind_runtime.remote(
            rollout_manager=rollout_manager,
            training_group=training_group,
        )
    )
    logger.info("Rollout buffer actor created and bound to runtime handles")

    # 9. Setup weight sync protocol (binds training_group + rollout_manager).
    weight_sync.setup(training_group=training_group, rollout_manager=rollout_manager)

    # 10. Core async training loop
    if args.rollout.async_pipeline:
        from diffusionrl.train_async import train_async_loop

        try:
            train_async_loop(
                args=args,
                rollout_manager=rollout_manager,
                rollout_buffer=rollout_buffer,
                training_group=training_group,
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
    enforce_rollout_alignment = not bool(getattr(args.rollout, "rollout_buffer_grouped", False))
    debug_save_intermediates = bool(getattr(args.debug, "debug_save_intermediates", False))
    save_rollout_debug_payload = None
    if debug_save_intermediates:
        from diffusionrl.debug.runner import save_rollout_debug_payload as _save_rollout_debug_payload

        save_rollout_debug_payload = _save_rollout_debug_payload

    def offload_train_phase() -> None:
        if args.ray.offload_train:
            training_group.offload()
        else:
            training_group.clear_memory()

    def ensure_rollout_on_gpu() -> None:
        nonlocal rollout_on_gpu
        if args.ray.offload_rollout and not rollout_on_gpu:
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
            logger.warning("Failed to materialize training batch for rollout metrics: %s", exc)
            return {}
        return compute_rollout_batch_metrics(
            training_data=training_data,
            num_samples_per_prompt=num_samples_per_prompt,
        )

    for rollout_id in range(args.rollout.start_rollout_id, args.rollout.num_rollout):
        step_start_t = time.perf_counter()
        sync_result = None
        sync_phase_s = 0.0
        eval_phase_s = 0.0
        should_log = (rollout_id % args.rollout.logging_steps == 0)
        collect_media_preview = bool(should_log and wandb_media_enabled)

        # === PHASE 1: Rollout ===
        rollout_phase_start_t = time.perf_counter()
        ensure_rollout_on_gpu()
        if debug_save_intermediates:
            debug_payload = ray.get(rollout_manager.build_training_debug_payload.remote(rollout_id))
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
            ray.get(
                rollout_manager.generate_and_push.remote(
                    rollout_id=rollout_id,
                    buffer=rollout_buffer,
                    collect_media_preview=collect_media_preview,
                    media_max_items=wandb_media_max_items,
                )
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

        if args.ray.offload_rollout:
            ray.get(rollout_manager.sleep.remote())
            rollout_on_gpu = False

        # === PHASE 2: Training ===
        train_phase_start_t = time.perf_counter()
        if args.ray.offload_train:
            training_group.onload()

        metrics = training_group.train(rollout_id, rollout_data_ref)
        train_phase_s = time.perf_counter() - train_phase_start_t

        # Periodic: save (before offload to ensure model is on GPU)
        if should_save(rollout_id, args):
            save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
            training_group.save_model(save_path)
            logger.info(f"Checkpoint saved: {save_path}")

        # === PHASE 3: Offload + Weight Sync ===
        offload_train_phase()

        if (
            not training_actor_direct_sampling
            and (rollout_id + 1) % args.rollout.update_weights_interval == 0
        ):
            sync_phase_start_t = time.perf_counter()
            sync_result = weight_sync.sync(rollout_id=rollout_id)
            sync_phase_s = time.perf_counter() - sync_phase_start_t
            rollout_on_gpu = True  # Protocol internally calls wake_up

        # Periodic: evaluate (after weight sync, rollout actors are on GPU)
        if should_eval(rollout_id, args):
            eval_phase_start_t = time.perf_counter()
            ensure_rollout_on_gpu()
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            eval_phase_s = time.perf_counter() - eval_phase_start_t
            logger.info(f"Eval at {rollout_id}: mean_reward={eval_metrics['mean_reward']:.4f}")

            # Log eval metrics to WandB
            if wandb_logger:
                wandb_logger.log_eval(rollout_id, eval_metrics)

        if should_log:
            avg_loss = sum(m.get("loss", 0) for m in metrics) / max(len(metrics), 1)
            step_time_s = time.perf_counter() - step_start_t
            logger.info(
                "Rollout %s: loss=%.4f rollout=%.3fs train=%.3fs sync=%.3fs eval=%.3fs step=%.3fs",
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
                wandb_logger.log_step(rollout_id, aggregated)

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
    training_group.dispose()

    # Finish WandB logging
    if wandb_logger:
        wandb_logger.finish()

    logger.info("Training complete!")


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
