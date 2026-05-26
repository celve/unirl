"""WAN 2.1 T2V / I2V pipeline — new four-tier architecture.

Re-expression of :class:`diffusionrl.models.wan21.WAN21ModelBundle`
against the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Legacy WAN 2.1 keeps
serving production GRPO (via ``train.py`` + ``model: wan21``); this
package provides the equivalent pipeline (via ``train.py``
+ ``model: wan21``).

This is the framework direction — ``diffusionrl/models/`` is being
deprecated. The legacy bundle stays in tree for the active recipes that
have not migrated, and to keep the merge surface with ``main`` small.

I2V is wired through two optional ``EncodeStage`` siblings:
:class:`WAN21ImageLatentEncodeStage` (``Images`` → 20-channel mask + VAE
latent payload, channel-concatted onto noise before the transformer)
and :class:`WAN21CLIPVisionEncodeStage` (``Images`` → CLIP penultimate
patch embeddings, forwarded as ``encoder_hidden_states_image``). Both
fire only when the I2V checkpoint declares ``transformer.config.image_dim
> 0``; T2V bundles skip them and the pipeline is unchanged.

Imports from this package have side effects (Hydra ``register_config``
calls in ``config.py``), so importing ``diffusionrl.models.wan21``
is enough to make ``model/wan21`` resolvable in the ConfigStore.
"""

from diffusionrl.models.wan21.bundle import WAN21Bundle
from diffusionrl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from diffusionrl.models.wan21.conditions import WAN21Conditions
from diffusionrl.models.wan21.config import WAN21PipelineConfig
from diffusionrl.models.wan21.diffusion import (
    WAN21DiffusionStage,
    WAN21DiffusionStep,
)
from diffusionrl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from diffusionrl.models.wan21.pipeline import WAN21Pipeline
from diffusionrl.models.wan21.text_embed import WAN21TextEmbedStage
from diffusionrl.models.wan21.vae import WAN21VAEDecodeStage

__all__ = [
    "WAN21Bundle",
    "WAN21CLIPVisionEncodeStage",
    "WAN21Conditions",
    "WAN21DiffusionStage",
    "WAN21DiffusionStep",
    "WAN21ImageLatentEncodeStage",
    "WAN21Pipeline",
    "WAN21PipelineConfig",
    "WAN21TextEmbedStage",
    "WAN21VAEDecodeStage",
]
