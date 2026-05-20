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


__all__ = [
    "ForwardContext",
    "FluxForwardContext",
    "SD3ForwardContext",
    "MochiForwardContext",
    "WAN21ForwardContext",
    "DefaultForwardContext",
    "register_forward_context",
    "get_forward_context_cls",
]
