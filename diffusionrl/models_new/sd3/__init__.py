"""SD3 pipeline — parallel prototype on the new four-tier architecture.

Re-expression of :class:`diffusionrl.models.sd3.SD3ModelBundle` against the
typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` / ``DiffusionStage`` /
``DecodeStage`` protocols. Legacy SD3 keeps serving production GRPO/NFT;
this package proves the new contracts end-to-end before the algorithm /
training-backend migration lands.

Imports from this package have side effects (Hydra ``register_config`` calls
in ``config.py``), so importing ``diffusionrl.models_new.sd3`` is enough to
make ``model/sd3_v2`` resolvable in the ConfigStore.
"""

from diffusionrl.models_new.sd3.bundle import SD3Bundle
from diffusionrl.models_new.sd3.conditions import SD3Conditions
from diffusionrl.models_new.sd3.config import SD3PipelineConfig
from diffusionrl.models_new.sd3.diffusion import SD3DiffusionParams
from diffusionrl.models_new.sd3.pipeline import SD3Pipeline

__all__ = [
    "SD3Bundle",
    "SD3Conditions",
    "SD3DiffusionParams",
    "SD3Pipeline",
    "SD3PipelineConfig",
]
