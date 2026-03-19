"""Rollout pipeline package.

All stage helpers live in ``rollout_pipeline.py``.  Import from there
directly, or use the convenience re-exports below::

    from diffusionrl.runtime.pipeline.rollout_pipeline import distributed_sample
"""

from .rollout_pipeline import (  # noqa: F401
    # Sampling stage
    distributed_sample,
    # Advantage stage
    get_reward_component_weights,
    compute_advantages,
)
