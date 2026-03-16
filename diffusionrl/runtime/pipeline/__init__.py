"""Rollout pipeline package.

All stage helpers live in ``rollout_pipeline.py``.  Import from there
directly, or use the convenience re-exports below::

    from diffusionrl.runtime.pipeline.rollout_pipeline import compute_rewards
"""

from .rollout_pipeline import (  # noqa: F401
    # Sampling stage
    distributed_sample,
    # Reward stage
    extract_images_from_output,
    extract_videos_from_output,
    reward_prefers_video_inputs,
    compute_rewards,
    # Advantage stage
    get_reward_component_weights,
    compute_advantages,
    # Assemble stage
    assemble_forward_training_batch,
    assemble_backward_training_batch,
    # Partition stage
    maybe_partition_training_batch,
)
