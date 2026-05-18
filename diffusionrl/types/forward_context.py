"""
ForwardContext — pure-data container for model forward parameters.

Each model architecture registers a concrete subclass whose fields are
annotated with ``concat_field()`` (per-sample, batch-indexed) or
``shared_field()`` (batch-shared scalars / tensors).

Inherits from :class:`Batched` for generic ``concat`` / ``select`` /
``slice`` / ``to_device`` / ``clone`` operations.  Adds two
ForwardContext-specific helpers:

- ``cast_dtype``: cast floating-point concat tensors for transport.
- ``to_dict``: export non-None fields for legacy ``**``-expansion paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, ClassVar, Dict, Optional, Sequence, Type, TypeVar

import torch

from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.utils.batched import (
    FieldKind,
    _field_kind,
    field,
    shared_field,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FC = TypeVar("_FC", bound="ForwardContext")

_FORWARD_CONTEXT_REGISTRY: Dict[str, Type["ForwardContext"]] = {}


def register_forward_context(model_type: str):
    """Decorator: register a ForwardContext subclass for a model type."""

    def decorator(cls: Type[_FC]) -> Type[_FC]:
        _FORWARD_CONTEXT_REGISTRY[model_type] = cls
        cls._registered_model_type = model_type  # type: ignore[attr-defined]
        return cls

    return decorator


def get_forward_context_cls(model_type: str) -> Type["ForwardContext"]:
    """Lookup the registered ForwardContext subclass by model type."""
    cls = _FORWARD_CONTEXT_REGISTRY.get(model_type)
    if cls is None:
        raise KeyError(
            f"No ForwardContext registered for model_type={model_type!r}. "
            f"Available: {sorted(_FORWARD_CONTEXT_REGISTRY)}"
        )
    return cls


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass
class ForwardContext(Transportable):
    """Pure-data container for model forward parameters.

    Stores all parameters needed for model forward / inference **except**
    latents and sigma (which vary per denoising step).

    Subclasses annotate fields with ``concat_field()`` (per-sample tensors
    batched along dim-0) or ``shared_field()`` (scalars / batch-shared
    tensors).  All batch operations are inherited from :class:`Batched`
    (via :class:`Transportable`).

    Per-sample tensor fields (``concat_field``-shaped) are tagged
    ``transport=True`` so they round-trip through TransferQueue when this
    container is nested in a :class:`Transportable` parent
    (``RolloutSamples`` / ``TrainingBatch``). Scalars (``guidance_scale``),
    ``torch.dtype`` (``autocast_dtype``), and dicts (``attention_kwargs``)
    are intentionally untagged — they ride along on the Ray return-value
    pickle path with the rest of the container.
    """

    _registered_model_type: ClassVar[str] = ""

    # ---- ForwardContext-specific methods ------------------------------------

    def cast_dtype(self: _FC, dtype: torch.dtype) -> _FC:
        """Cast floating-point concat tensors to *dtype* for transport.

        Only float-type tensors in concat fields are cast.  Shared fields,
        integer tensors (masks, position IDs), and scalars are unchanged.
        """
        kwargs: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            name = f.name
            val = getattr(self, name)
            if _field_kind(f) is FieldKind.CONCAT and isinstance(val, torch.Tensor) and val.is_floating_point():
                kwargs[name] = val.to(dtype=dtype)
            else:
                kwargs[name] = val
        return type(self)(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Export all non-None fields as a plain dict.

        Legacy helper for call sites that still expand ``ForwardContext``
        fields as keyword arguments. New training-forward dispatch should pass
        ``ctx`` directly to ``ModelBundle.forward_denoiser(...)``.
        """
        result: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val
        return result

    # ---- backward-compat aliases -------------------------------------------

    @classmethod
    def cat(cls: Type[_FC], contexts: Sequence[_FC]) -> _FC:
        """Alias for :meth:`concat` (backward compatibility)."""
        return cls.concat(contexts)

    def reindex(self: _FC, indices: torch.Tensor) -> _FC:
        """Alias for :meth:`select` (backward compatibility)."""
        return self.select(indices)


# ---------------------------------------------------------------------------
# Model-specific subclasses
# ---------------------------------------------------------------------------


@register_forward_context(model_type="flux")
@dataclass
class FluxForwardContext(ForwardContext):
    """ForwardContext for FLUX models.

    Field naming follows the model-bundle-owned training-forward contract:
    ``image_ids`` (not ``img_ids``) so bundle implementations can consume the
    context directly without remapping.
    """

    guidance_scale: float = shared_field(default=3.5)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    pooled_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    text_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    image_ids: Optional[torch.Tensor] = shared_field(default=None)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@register_forward_context(model_type="sd3")
@dataclass
class SD3ForwardContext(ForwardContext):
    """ForwardContext for SD3 models."""

    guidance_scale: float = shared_field(default=7.0)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    pooled_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    encoder_attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@register_forward_context(model_type="hunyuan_video")
@dataclass
class HunyuanVideoForwardContext(ForwardContext):
    """ForwardContext for HunyuanVideo (text-to-video) models.

    Note: this targets HunyuanVideo specifically and does **not** cover
    Hunyuan-Image. The class / registry key are explicitly suffixed with
    ``_video`` to disambiguate.
    """

    guidance_scale: float = shared_field(default=1.0)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    pooled_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    encoder_attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@register_forward_context(model_type="mochi")
@dataclass
class MochiForwardContext(ForwardContext):
    """ForwardContext for Mochi video models."""

    guidance_scale: float = shared_field(default=4.5)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    encoder_attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    attention_kwargs: Optional[Dict[str, Any]] = shared_field(default=None)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@register_forward_context(model_type="wan21")
@dataclass
class WAN21ForwardContext(ForwardContext):
    """ForwardContext for WAN 2.1 multi-task video/image generation."""

    guidance_scale: float = shared_field(default=5.0)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    encoder_hidden_states_image: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_encoder_hidden_states_image: Optional[torch.Tensor] = field(
        kind=FieldKind.CONCAT, default=None, transport=True
    )
    image_conditioning_latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    first_frame_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    attention_kwargs: Optional[Dict[str, Any]] = shared_field(default=None)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@register_forward_context(model_type="hunyuan_veido1p5")
@dataclass
class HunyuanVeido1p5ForwardContext(ForwardContext):
    """ForwardContext for HunyuanVideo-1.5 (Qwen2.5-VL + ByT5 + SigLIP).

    The model exposes two parallel text streams (MLLM + ByT5 glyph) plus an
    optional SigLIP vision stream.  All three are fed through the bundle's
    ``forward_denoiser``; the sampler never assembles model-private kwargs
    directly.

    Field-name choices (per skill rules): we deliberately keep
    ``prompt_embeds_mask`` / ``prompt_embeds_2`` / ``prompt_embeds_mask_2``
    instead of overloading the SD3-style ``encoder_attention_mask``, so the
    upstream pipeline wiring (`encoder_hidden_states_2`, `encoder_attention_mask_2`)
    maps cleanly to the bundle-owned contract.
    """

    guidance_scale: float = shared_field(default=6.0)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    prompt_embeds_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    prompt_embeds_2: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    prompt_embeds_mask_2: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds_2: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds_mask_2: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    image_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    cond_latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    cond_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    attention_kwargs: Optional[Dict[str, Any]] = shared_field(default=None)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


# Also register "default" pointing to base ForwardContext for fallback
@register_forward_context(model_type="default")
@dataclass
class DefaultForwardContext(ForwardContext):
    """Fallback ForwardContext with SD3-like fields."""

    guidance_scale: float = shared_field(default=3.5)
    prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    pooled_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    encoder_attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    text_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    image_ids: Optional[torch.Tensor] = shared_field(default=None)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@register_forward_context(model_type="wan22")
@dataclass
class WAN22ForwardContext(WAN21ForwardContext):
    """ForwardContext for WAN 2.2 dual-transformer video generation.

    Extends WAN21ForwardContext with per-stage guidance scale.
    boundary_ratio is a bundle-level config, not per-call context.
    """

    guidance_scale_2: Optional[float] = shared_field(default=None)


__all__ = [
    "ForwardContext",
    "FluxForwardContext",
    "SD3ForwardContext",
    "HunyuanVideoForwardContext",
    "HunyuanVeido1p5ForwardContext",
    "MochiForwardContext",
    "WAN21ForwardContext",
    "WAN22ForwardContext",
    "DefaultForwardContext",
    "register_forward_context",
    "get_forward_context_cls",
]
