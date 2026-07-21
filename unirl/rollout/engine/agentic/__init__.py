"""Agentic (multi-turn, tool-use) rollout engine (LIN-522).

A rank-0 coordinator over a DP-replicated slab of per-worker drain thread
pools; ``generate`` returns a flat ``List[Sample]`` of variable-depth
trajectories. :class:`AgenticImageRolloutEngine` (LIN-577) extends it with a
terminal diffusion image generation conditioned on each trajectory's final
answer.
"""

from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine
from unirl.rollout.engine.agentic.image_config import AgenticImageRolloutEngineConfig
from unirl.rollout.engine.agentic.image_engine import AgenticImageRolloutEngine

__all__ = [
    "AgenticRolloutEngine",
    "AgenticRolloutEngineConfig",
    "AgenticImageRolloutEngine",
    "AgenticImageRolloutEngineConfig",
]
