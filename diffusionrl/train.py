#!/usr/bin/env python
"""
diffusionrl Training Entry Point - Minimal Design (<100 lines)

Reference: slime/train.py

Usage:
    python -m diffusionrl.train --pretrained-model-saved-path /path/to/model --num-rollout 100
"""
import logging

from diffusionrl.config import parse_args

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

    from diffusionrl.ray.data_buffer import normalize_rollout_result
    from diffusionrl.ray import create_placement_groups_from_args, create_rollout_manager
    from diffusionrl.ray.sampling_mode import create_sampling_mode_plugin
    from diffusionrl.utils import configure_logger, set_seed
    from diffusionrl.utils.weight_sync import create_weight_sync_strategy
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger

    configure_logger()
    set_seed(args.seed)

    logger.info("Starting GRPO training...")
    logger.info(f"Model: {args.pretrained_model_saved_path}")
    logger.info(f"Algorithm: {args.algorithm_path}")
    logger.info(f"Mode: {'colocate' if args.colocate_inference_training else 'separate'}")
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
            config=vars(args),
            rank=0,
        )
        logger.info(f"WandB initialized: project={args.project_name}, run={args.run_name}")

    # 1. Resource allocation
    pgs = create_placement_groups_from_args(args)
    logger.info("Placement groups created")

    # 2. Build sampling mode plugin (keeps train loop skeleton mode-agnostic)
    sampling_mode = create_sampling_mode_plugin(args)
    weight_sync_strategy = create_weight_sync_strategy(args)

    # 3. Create Rollout manager (internally loads sampler/reward/algorithm)
    rollout_pg_result = sampling_mode.rollout_pg_result(pgs)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args,
        pg_result=rollout_pg_result,
        reward_pg_result=pgs.get("reward"),
    )
    logger.info(f"Rollout manager created, {num_rollout_per_epoch} rollouts per epoch")

    # 4. Create training actors + mode-specific initial transitions
    training_group = sampling_mode.create_training_group(pgs, rollout_manager)

    # 5. Core async training loop
    if args.async_pipeline:
        from diffusionrl.train_async import train_async_loop

        train_async_loop(
            args=args,
            rollout_manager=rollout_manager,
            training_group=training_group,
            wandb_logger=wandb_logger,
            should_save_fn=should_save,
            should_eval_fn=should_eval,
            sync_weights_fn=weight_sync_strategy.sync,
        )
        # async loop handles cleanup and exits.
        return

    # 6. Core synchronous training loop
    use_rollout_buffer = bool(getattr(args, "rollout_buffer_enabled", True))
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        sampling_mode.before_rollout(rollout_manager)

        # === PHASE 1: Rollout ===
        # Buffer-centric path: generate->enqueue and consume in separate calls.
        # Legacy path: generate() returns serialized payload directly.
        if use_rollout_buffer:
            ray.get(rollout_manager.generate_and_buffer.remote(rollout_id))
            rollout_result = ray.get(
                rollout_manager.pop_training_data.remote(
                    world_size=training_group.num_actors
                )
            )
        else:
            # generate() returns ray.put(train_data) - an ObjectRef
            # ray.get() behavior varies: may or may not auto-resolve nested ObjectRefs
            rollout_result = ray.get(
                rollout_manager.generate.remote(
                    rollout_id,
                    world_size=training_group.num_actors,
                )
            )

        rollout_data_ref = normalize_rollout_result(rollout_result)

        sampling_mode.after_rollout(rollout_manager)

        # === PHASE 2: Training ===
        sampling_mode.before_train(training_group)

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

        # === PHASE 3: Weight Sync ===
        sampling_mode.after_train(training_group)
        sampling_mode.maybe_sync_weights(
            rollout_id=rollout_id,
            training_group=training_group,
            rollout_manager=rollout_manager,
            sync_weights_fn=weight_sync_strategy.sync,
        )

        # Periodic: evaluate (after weight sync, rollout actors are on GPU)
        if should_eval(rollout_id, args):
            sampling_mode.before_eval(rollout_manager)
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            logger.info(f"Eval at {rollout_id}: mean_reward={eval_metrics['mean_reward']:.4f}")

            # Log eval metrics to WandB
            if wandb_logger:
                wandb_logger.log_eval(rollout_id, eval_metrics)

    # Cleanup
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
