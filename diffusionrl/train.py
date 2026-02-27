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

from diffusionrl.config import parse_args
from diffusionrl.config.arguments import is_training_actor_direct_sampling_mode

logger = logging.getLogger(__name__)


def should_save(rollout_id: int, args) -> bool:
    """Check if we should save a checkpoint at this rollout."""
    return (rollout_id + 1) % args.save_steps == 0


def should_eval(rollout_id: int, args) -> bool:
    """Check if we should run evaluation at this rollout."""
    return (rollout_id + 1) % args.eval_steps == 0


def train(args):
    """Main training loop."""
    import ray

    from diffusionrl.ray import (
        create_placement_groups_from_args,
        create_rollout_buffer_actor,
        create_rollout_manager,
        create_training_actor_group,
    )
    from diffusionrl.utils import configure_logger, set_seed
    from diffusionrl.utils.weight_sync import create_weight_sync_strategy
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger

    configure_logger()
    set_seed(args.seed)

    logger.info("Starting GRPO training...")
    logger.info(f"Model: {args.pretrained_model_saved_path}")
    logger.info(f"Algorithm: {args.algorithm_path}")
    logger.info(f"Mode: {'colocate' if args.colocate_rollout_training else 'separate'}")
    logger.info(f"Offload train: {args.offload_train}, Offload rollout: {args.offload_rollout}")
    logger.info(f"Weight sync mode: {args.weight_sync_mode}, async_pipeline={args.async_pipeline}")

    # Initialize Ray
    if not ray.is_initialized():
        if args.ray_address:
            ray.init(address=args.ray_address, ignore_reinit_error=True)
        else:
            ray.init()

    # Initialize WandB logger (only on rank 0)
    wandb_logger = None
    if args.report_to == "wandb" and args.project_name:
        wandb_logger = init_logger(
            project=args.project_name,
            run_name=args.run_name,
            config=args.to_flat_dict() if hasattr(args, "to_flat_dict") else vars(args),
            rank=0,
        )
        logger.info(f"WandB initialized: project={args.project_name}, run={args.run_name}")

    # 1. Resource allocation
    pgs = create_placement_groups_from_args(args)
    logger.info("Placement groups created")

    training_actor_direct_sampling = is_training_actor_direct_sampling_mode(args)
    rollout_on_gpu = True
    weight_sync_strategy = create_weight_sync_strategy(args)

    # 2. Create rollout manager; direct-sampling mode does not allocate rollout actors.
    rollout_pg_result = None if training_actor_direct_sampling else pgs.get("rollout")
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args,
        pg_result=rollout_pg_result,
        reward_pg_result=pgs.get("reward"),
    )
    logger.info(f"Rollout manager created, {num_rollout_per_epoch} rollouts per epoch")

    # 3. Optional pre-offload for colocate memory pressure relief before training actor init.
    if args.offload_rollout:
        ray.get(rollout_manager.sleep.remote())
        rollout_on_gpu = False

    # 4. Create training actors.
    training_pg_result = pgs.get("training")
    if training_pg_result is None:
        raise ValueError("Missing training placement-group allocation.")
    training_group = create_training_actor_group(args, training_pg_result)
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    if resume_from_checkpoint:
        training_group.load_checkpoint(resume_from_checkpoint)
        logger.info("Checkpoint loaded: %s", resume_from_checkpoint)
        if int(getattr(args, "start_rollout_id", 0)) == 0:
            match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(resume_from_checkpoint)))
            if match:
                args.start_rollout_id = int(match.group(1)) + 1
                logger.info(
                    "Auto-set start_rollout_id=%s from checkpoint path.",
                    args.start_rollout_id,
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
    if args.offload_rollout and args.offload_train:
        training_group.offload()
        ray.get(rollout_manager.wake_up.remote())
        rollout_on_gpu = True
    elif args.offload_rollout:
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

    # 9. Core async training loop
    if args.async_pipeline:
        from diffusionrl.train_async import train_async_loop

        train_async_loop(
            args=args,
            rollout_manager=rollout_manager,
            rollout_buffer=rollout_buffer,
            training_group=training_group,
            wandb_logger=wandb_logger,
            should_save_fn=should_save,
            should_eval_fn=should_eval,
            sync_weights_fn=weight_sync_strategy.sync,
        )
        # async loop handles cleanup and exits.
        return

    # 10. Core synchronous training loop
    enforce_rollout_alignment = not bool(getattr(args, "rollout_buffer_grouped", False))

    def offload_train_phase() -> None:
        if args.offload_train:
            training_group.offload()
        else:
            training_group.clear_memory()

    def ensure_rollout_on_gpu() -> None:
        nonlocal rollout_on_gpu
        if args.offload_rollout and not rollout_on_gpu:
            ray.get(rollout_manager.wake_up.remote())
            rollout_on_gpu = True

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # === PHASE 1: Rollout ===
        ensure_rollout_on_gpu()

        ray.get(rollout_buffer.request_rollout.remote(rollout_id=rollout_id))
        rollout_payload = ray.get(
            rollout_buffer.pop_training_data.remote(
                consumer_spec=buffer_consumer_spec,
                expected_rollout_id=rollout_id if enforce_rollout_alignment else None,
            )
        )
        rollout_data_ref = rollout_payload["training_data"]

        if args.offload_rollout:
            ray.get(rollout_manager.sleep.remote())
            rollout_on_gpu = False

        # === PHASE 2: Training ===
        if args.offload_train:
            training_group.onload()

        metrics = training_group.train(rollout_id, rollout_data_ref)

        # Log metrics
        if rollout_id % args.logging_steps == 0:
            avg_loss = sum(m.get("loss", 0) for m in metrics) / len(metrics)
            logger.info(f"Rollout {rollout_id}: loss={avg_loss:.4f}")

            # Log to WandB
            if wandb_logger:
                aggregated = aggregate_metrics(metrics)
                aggregated["loss"] = avg_loss
                wandb_logger.log_step(rollout_id, aggregated)

        # Periodic: save (before offload to ensure model is on GPU)
        if should_save(rollout_id, args):
            save_path = f"{args.output_dir}/checkpoint-{rollout_id}"
            training_group.save_model(save_path)
            logger.info(f"Checkpoint saved: {save_path}")

        # === PHASE 3: Offload + Weight Sync ===
        offload_train_phase()

        if (
            not training_actor_direct_sampling
            and (rollout_id + 1) % args.update_weights_interval == 0
        ):
            ensure_rollout_on_gpu()
            weight_sync_strategy.sync(
                rollout_id=rollout_id,
                training_group=training_group,
                rollout_manager=rollout_manager,
            )

        # Periodic: evaluate (after weight sync, rollout actors are on GPU)
        if should_eval(rollout_id, args):
            ensure_rollout_on_gpu()
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            logger.info(f"Eval at {rollout_id}: mean_reward={eval_metrics['mean_reward']:.4f}")

            # Log eval metrics to WandB
            if wandb_logger:
                wandb_logger.log_eval(rollout_id, eval_metrics)

    # Cleanup
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
