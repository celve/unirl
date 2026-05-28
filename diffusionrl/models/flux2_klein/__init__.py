"""FLUX.2-klein-9B pipeline on the new four-tier typed architecture.

Re-expression of ``main_flux_bundle/diffusionrl/models/flux2.py``
(Klein branch) + ``main_flux_bundle/diffusionrl/samplers/fsdp/flux2_sampler.py``
against the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Sibling of
:mod:`diffusionrl.models.sd3` and :mod:`diffusionrl.models.qwen_image`.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing
``diffusionrl.models.flux2_klein`` is enough to make
``model/flux2_klein_v2`` resolvable in the ConfigStore.
"""

from diffusionrl.models.flux2_klein.bundle import Flux2KleinBundle
from diffusionrl.models.flux2_klein.conditions import Flux2KleinConditions
from diffusionrl.models.flux2_klein.config import Flux2KleinPipelineConfig
from diffusionrl.models.flux2_klein.diffusion import (
    Flux2KleinDiffusionParams,
    Flux2KleinDiffusionStage,
    Flux2KleinDiffusionStep,
)
from diffusionrl.models.flux2_klein.pipeline import Flux2KleinPipeline
from diffusionrl.models.flux2_klein.schedule import (
    Flux2KleinSchedulePolicy,
    build_flux2_klein_schedule_policy,
)
from diffusionrl.models.flux2_klein.text_embed import Flux2KleinTextEmbedStage
from diffusionrl.models.flux2_klein.vae import Flux2KleinVAEDecodeStage

__all__ = [
    "Flux2KleinBundle",
    "Flux2KleinConditions",
    "Flux2KleinDiffusionParams",
    "Flux2KleinDiffusionStage",
    "Flux2KleinDiffusionStep",
    "Flux2KleinPipeline",
    "Flux2KleinPipelineConfig",
    "Flux2KleinSchedulePolicy",
    "Flux2KleinTextEmbedStage",
    "Flux2KleinVAEDecodeStage",
    "build_flux2_klein_schedule_policy",
]
