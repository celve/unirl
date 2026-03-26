"""
ForwardContext — bundles plugin + guidance_scale + embeddings for model forward calls.

Algorithms pass a ForwardContext through their loss computation chain instead of
threading ``prompt_embeds``, ``pooled_prompt_embeds``, ``text_ids``,
``image_ids``, ``encoder_attention_mask``, ``guidance_scale``, etc. individually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from diffusionrl.types import PromptEmbeddings


@dataclass
class ForwardContext:
    """Model forward-call context carried through the algorithm loss chain.

    Instead of passing 8+ model-specific parameters through every function,
    algorithms create a ``ForwardContext`` once at the top of
    ``compute_loss_and_backward`` and pass it down.  Any method that needs to
    call the model simply does::

        pred = ctx.forward(model, latents, sigma)

    This eliminates ~100 lines of parameter threading per algorithm and makes
    it trivial to add new model-specific parameters (they only need to appear
    in ``PromptEmbeddings`` and the concrete ``ForwardPlugin``).

    Attributes:
        plugin: A ``ModelForwardPlugin`` instance (owns CFG, timestep scaling,
                autocast).
        guidance_scale: Classifier-free guidance scale.
        embeddings: ``PromptEmbeddings`` containing all encoder outputs needed
                    by the plugin.
    """

    plugin: Any  # ModelForwardPlugin (Protocol)
    guidance_scale: float
    embeddings: PromptEmbeddings

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """Execute a model forward pass through the plugin.

        All embedding fields are unpacked automatically from
        ``self.embeddings.to_dict()``; callers never need to pass
        ``prompt_embeds``, ``text_ids``, etc.
        """
        kwargs: Dict[str, Any] = self.embeddings.to_dict()
        kwargs["guidance_scale"] = self.guidance_scale
        kwargs.update(extra_kwargs)
        return self.plugin.forward(
            model=model,
            latents=latents,
            sigma=sigma,
            **kwargs,
        )

    def forward_with_embeddings(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        embeddings_override: PromptEmbeddings,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """Like :meth:`forward` but with a different set of embeddings.

        Useful when slicing a mini-batch produces different embeddings while
        the plugin and guidance_scale remain the same.
        """
        kwargs: Dict[str, Any] = embeddings_override.to_dict()
        kwargs["guidance_scale"] = self.guidance_scale
        kwargs.update(extra_kwargs)
        return self.plugin.forward(
            model=model,
            latents=latents,
            sigma=sigma,
            **kwargs,
        )

    def with_embeddings(self, embeddings: PromptEmbeddings) -> "ForwardContext":
        """Return a shallow copy that shares plugin/guidance_scale but uses *embeddings*."""
        return ForwardContext(
            plugin=self.plugin,
            guidance_scale=self.guidance_scale,
            embeddings=embeddings,
        )
