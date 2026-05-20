"""HunyuanImage 3.0 pipeline — concrete user of the typed Stage protocols.

Re-expression of HunyuanImage3's t2t / i2t / t2i / it2i task topologies
against the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``EncodeStage`` / ``DecodeStage`` / ``ARStage`` / ``DiffusionStage``
protocols. Mirrors :mod:`diffusionrl.models_new.sd3` 1:1 in layout.

PR 2 (current) ships the bundle skeleton + t2i diffusion path. PR 3 lands
the AR stage (t2t / i2t). PR 4 wires the full t2i with AR multi-pass.
PR 5 adds it2i.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing
``diffusionrl.models_new.hunyuan_image3`` is enough to make
``model/hunyuan_image3_v2`` resolvable in the ConfigStore.
"""

from diffusionrl.models_new.hunyuan_image3.ar import (
    HunyuanImage3ARParams,
    HunyuanImage3ARStage,
    HunyuanImage3ARStep,
)
from diffusionrl.models_new.hunyuan_image3.bundle import HunyuanImage3Bundle
from diffusionrl.models_new.hunyuan_image3.conditions import (
    HunyuanImage3ARConditions,
    HunyuanImage3DiffusionConditions,
    HunyuanImage3FusedMultimodalCondition,
)
from diffusionrl.models_new.hunyuan_image3.config import HunyuanImage3PipelineConfig
from diffusionrl.models_new.hunyuan_image3.diffusion import (
    HunyuanImage3DiffusionParams,
    HunyuanImage3DiffusionStage,
    HunyuanImage3DiffusionStep,
)
from diffusionrl.models_new.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from diffusionrl.models_new.hunyuan_image3.text_embed import (
    HunyuanImage3TextEmbedStage,
)
from diffusionrl.models_new.hunyuan_image3.vae import (
    HunyuanImage3VAEDecodeStage,
    HunyuanImage3VAEEncodeStage,
)
from diffusionrl.models_new.hunyuan_image3.vit_encode import (
    HunyuanImage3VitEncodeStage,
)

__all__ = [
    "HunyuanImage3ARConditions",
    "HunyuanImage3ARParams",
    "HunyuanImage3ARStage",
    "HunyuanImage3ARStep",
    "HunyuanImage3Bundle",
    "HunyuanImage3DiffusionConditions",
    "HunyuanImage3DiffusionParams",
    "HunyuanImage3DiffusionStage",
    "HunyuanImage3DiffusionStep",
    "HunyuanImage3FusedMultimodalCondition",
    "HunyuanImage3Pipeline",
    "HunyuanImage3PipelineConfig",
    "HunyuanImage3TextEmbedStage",
    "HunyuanImage3VAEDecodeStage",
    "HunyuanImage3VAEEncodeStage",
    "HunyuanImage3VitEncodeStage",
]
