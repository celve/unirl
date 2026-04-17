"""Debug runners for isolated phase testing.

``run_debug_train_only``  — skips sampling/rollout/reward entirely, only
exercises the training code path (model forward, loss, gradient, optimizer,
EMA update).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from diffusionrl.config.assembly import DerivedConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# train_only
# ---------------------------------------------------------------------------

def run_debug_train_only(args: Any, *, derived_config: "DerivedConfig") -> None:
    """Run only the training phase with synthetic or pre-saved data.

    Setup path:
        Ray init -> placement groups -> training actor group
        (skips rollout manager, reward service, rollout buffer, weight sync)

    Data path (one of):
        1. ``--debug.load-path <path>`` — load a previously saved
           ``ForwardTrainingBatch`` or ``BackwardTrainingBatch`` from disk.
        2. Otherwise — generate a synthetic typed training batch on the
           training actor. The synthetic path is intentionally conservative
           and currently only supported for SD3 train-side debugging.

    The loop runs ``args.debug.num_rollouts`` training iterations and logs
    basic metrics (loss, grad_norm, timing).
    """
    import ray

    from diffusionrl.cmdline.resolution import build_launch_config
    from diffusionrl.ray.group_factory import create_training_actor_group
    from diffusionrl.ray.placement_group import create_placement_groups_from_launch
    from diffusionrl.utils import configure_logger, set_seed

    configure_logger()
    set_seed(args.seed)

    num_rollouts = max(1, int(args.debug.num_rollouts))
    debug_load_path: Optional[str] = args.debug.load_path
    model_type = str(getattr(args.model, "model_type", "") or "").strip().lower()

    logger.info("=== DEBUG: train_only mode ===")
    logger.info("Model: %s", args.model.pretrained_model_ckpt_path)
    logger.info("Num training iterations: %d", num_rollouts)
    if debug_load_path:
        logger.info("Loading training batch from: %s", debug_load_path)
    elif model_type != "sd3":
        raise ValueError(
            "debug.mode=train_only synthetic batches are intentionally limited to "
            "model_type='sd3' for pre-release correctness. Use "
            "--debug.load-path with a saved rollout payload for other models."
        )

    # Align training-side scheduler horizon with the actual debug loop length.
    args.rollout.num_rollout = num_rollouts

    launch_config = build_launch_config(args, derived_config=derived_config)

    # 1. Ray
    if not ray.is_initialized():
        if args.ray.ray_address:
            ray.init(address=args.ray.ray_address, ignore_reinit_error=True)
        else:
            ray.init()

    training_group = None
    try:
        # 2. Placement groups (only training PG is needed)
        pgs = create_placement_groups_from_launch(launch_config)
        training_pg_result = pgs.get("training")
        if training_pg_result is None:
            raise ValueError("Missing training placement-group allocation.")

        # 3. Create training actor group (loads model, LoRA, optimizer, loss)
        training_group = create_training_actor_group(launch_config, training_pg_result)
        logger.info("Training actor group created and weights synced")

        # 4. Prepare training batch
        if debug_load_path:
            batch = _load_debug_batch(debug_load_path)
            batch_ref = ray.put(batch)
            logger.info("Loaded training batch: type=%s", type(batch).__name__)
        else:
            batch_size = int(args.algorithm.prompts_per_rollout) * max(
                1, int(args.algorithm.samples_per_prompt)
            )
            height = int(args.sampling.height)
            width = int(args.sampling.width)
            logger.info(
                "Generating synthetic debug batch: batch_size=%d, resolution=%dx%d, "
                "algorithm=%s",
                batch_size,
                height,
                width,
                args.algorithm.algorithm_type,
            )
            batch = ray.get(
                training_group._actor_handles[0].create_debug_training_batch.remote(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    num_inference_steps=int(args.sampling.num_inference_steps),
                )
            )
            batch_ref = ray.put(batch)
            logger.info("Synthetic batch created: type=%s", type(batch).__name__)

        # 5. Training loop — same batch reused for every iteration
        logger.info(
            "--- Starting debug training loop: %d optimizer steps on the SAME batch ---",
            num_rollouts,
        )
        all_losses = []
        total_train_time = 0.0
        for step in range(num_rollouts):
            t0 = time.perf_counter()
            metrics_list = training_group.train(step, batch_ref)
            elapsed = time.perf_counter() - t0
            total_train_time += elapsed

            rank0_metrics = metrics_list[0] if metrics_list else {}
            loss_val = rank0_metrics.get("loss", float("nan"))
            grad_norm = rank0_metrics.get("grad_norm", float("nan"))
            lr = rank0_metrics.get("lr", float("nan"))
            all_losses.append(loss_val)
            logger.info(
                "[step %d/%d] loss=%.6f  grad_norm=%.4f  lr=%.2e  time=%.2fs",
                step + 1,
                num_rollouts,
                loss_val,
                grad_norm,
                lr,
                elapsed,
            )

        import math
        finite_losses = [v for v in all_losses if math.isfinite(v)]
        if finite_losses:
            logger.info(
                "=== Summary: %d steps, loss %.6f -> %.6f, "
                "mean=%.6f, total_time=%.1fs ===",
                num_rollouts,
                finite_losses[0],
                finite_losses[-1],
                sum(finite_losses) / len(finite_losses),
                total_train_time,
            )
        else:
            logger.info(
                "=== Summary: %d steps, all losses nan, total_time=%.1fs ===",
                num_rollouts,
                total_train_time,
            )

        logger.info("=== DEBUG: train_only finished ===")
    finally:
        if training_group is not None:
            training_group.dispose()
        ray.shutdown()


def _load_debug_batch(path: str) -> Any:
    """Load a training batch from a saved .pt file."""
    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(f"debug.load_path not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(data, "clean_latents") or hasattr(data, "trajectories"):
        return data
    if isinstance(data, dict) and "training_batch" in data:
        return data["training_batch"]
    return data


# ---------------------------------------------------------------------------
# save helper (used by main loop with debug.save_intermediates)
# ---------------------------------------------------------------------------

def save_rollout_debug_payload(
    *,
    args: Any,
    payload: Dict[str, Any],
    rollout_id: int,
    source: str = "",
) -> Optional[str]:
    """Persist a rollout debug payload to disk for later replay."""
    import torch

    save_dir = str(args.debug.save_dir or "outputs/debug")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"rollout_payload_{rollout_id}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(payload, save_path)
    logger.info("Debug payload saved: %s (source=%s)", save_path, source)
    return save_path
