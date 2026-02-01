"""
FastVideo runtime patches.

These patches are applied dynamically from diffusionrl to avoid forking FastVideo.
"""

from . import gpu_worker_patch, executor_patch, video_generator_patch


def apply_all() -> None:
    """Apply all FastVideo patches."""
    gpu_worker_patch.apply()
    executor_patch.apply()
    video_generator_patch.apply()
