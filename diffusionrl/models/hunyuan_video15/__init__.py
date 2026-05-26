"""HunyuanVideo-1.5 pipeline on the new four-tier typed architecture.

Re-expression of ``diffusionrl/models/hunyuan_veido1p5.py`` +
``diffusionrl/samplers/fsdp/hunyuan_veido1p5_sampler.py`` (the PR #101
OLD implementation) against the typed ``Bundle`` / ``Pipeline`` /
``EmbedStage`` / ``DiffusionStage`` / ``DecodeStage`` protocols.
Sibling of :mod:`diffusionrl.models.wan21` (text-to-video peer)
and :mod:`diffusionrl.models.hunyuan_image3` (Hunyuan-family peer).

The ``hunyuan_veido1p5`` typo is dropped here; the package is
named ``hunyuan_video15`` (canonical). OLD will be deleted in a
follow-up PR, so the typo is contained to OLD's death-rattle window.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing
``diffusionrl.models.hunyuan_video15`` is enough to make
``model/hunyuan_video15`` resolvable in the ConfigStore.

Scope (v1)
----------
- **Text-to-Video (T2V)**: full support. Dual text encoders
  (Qwen2.5-VL MLLM + ByT5 glyph). CFG with stacked dual-stream
  classifier-free guidance (standard ``uncond + scale * (cond - uncond)``).
- **Image-to-Video (I2V)**: deferred. The transformer's
  ``cond_latents`` / ``cond_mask`` packing slots are zero-filled in v1;
  when I2V lands it will add a ``vision`` stage producing both an
  ``image_embeds`` SigLIP condition AND an image-latent condition that
  participates in the channel-dim concat inside ``predict_noise``.
"""

from diffusionrl.models.hunyuan_video15.bundle import HunyuanVideo15Bundle
from diffusionrl.models.hunyuan_video15.conditions import HunyuanVideo15Conditions
from diffusionrl.models.hunyuan_video15.config import HunyuanVideo15PipelineConfig
from diffusionrl.models.hunyuan_video15.pipeline import HunyuanVideo15Pipeline

__all__ = [
    "HunyuanVideo15Bundle",
    "HunyuanVideo15Conditions",
    "HunyuanVideo15Pipeline",
    "HunyuanVideo15PipelineConfig",
]
