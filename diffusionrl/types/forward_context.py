"""
ForwardContext — pure-data container for model forward parameters.

Replaces both ``PromptEmbeddings`` (flat 7-field container) and the old
algorithm-side ``ForwardContext`` (which bundled plugin + data).

Each model architecture registers a concrete subclass that declares its
fields via ``ClassVar`` field-classification sets:

- ``shared_fields``: scalar or batch-shared tensors (e.g. ``guidance_scale``,
  ``img_ids`` in FLUX where it's [H*W, 3] and identical across samples).
- ``batched_tensor_fields``: per-sample tensors that get stacked / sliced
  along the batch dimension (e.g. ``prompt_embeds``).

The base class provides generic ``stack`` / ``slice`` / ``reindex`` /
``to_device`` / ``to_dict`` operations driven by these ClassVars, so
downstream code (assemble, buffer, replay) can operate on *any*
ForwardContext subclass without knowing the concrete model fields.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import torch

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
class ForwardContext:
    """Pure-data container for model forward parameters.

    Stores all parameters needed for model forward / inference **except**
    latents and sigma (which vary per denoising step).

    Subclasses declare field classification via ClassVar:

    - ``shared_fields``: batch-shared scalars or tensors that are NOT sliced
      along dim-0 (e.g. ``guidance_scale``, FLUX ``img_ids``).
    - ``batched_tensor_fields``: per-sample tensors that ARE
      stacked / sliced / reindexed along dim-0 (e.g. ``prompt_embeds``).

    Non-tensor fields that are not in either set are treated as scalars
    and simply copied on ``slice`` / ``reindex``.
    """

    shared_fields: ClassVar[FrozenSet[str]] = frozenset()
    batched_tensor_fields: ClassVar[FrozenSet[str]] = frozenset()
    _registered_model_type: ClassVar[str] = ""

    # ---- generic properties ------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Infer batch size from the first stackable tensor field."""
        for name in self.batched_tensor_fields:
            val = getattr(self, name, None)
            if val is not None and isinstance(val, torch.Tensor):
                return int(val.shape[0])
        return 0

    # ---- batch operations --------------------------------------------------

    @classmethod
    def cat(cls, contexts: List["ForwardContext"]) -> "ForwardContext":
        """Concatenate a list of contexts along the batch dimension.

        Analogous to ``torch.cat``: stackable tensors are concatenated
        along dim-0.  Shared fields are taken from the first element.
        Other (per-sample list) fields are flattened via extend.
        """
        if not contexts:
            raise ValueError("Cannot cat empty context list.")

        first = contexts[0]
        actual_cls = type(first)

        if len(contexts) == 1:
            return first

        kwargs: Dict[str, Any] = {}
        for f in dataclass_fields(first):
            name = f.name
            if name in actual_cls.batched_tensor_fields:
                tensors: List[torch.Tensor] = []
                any_none = False
                for ctx in contexts:
                    val = getattr(ctx, name)
                    if val is None:
                        any_none = True
                    else:
                        tensors.append(val)
                if any_none and tensors:
                    raise ValueError(
                        f"ForwardContext.cat: mixed None and non-None values for "
                        f"batched tensor field {name!r}; all contexts must agree "
                        f"(all None or all non-None)."
                    )
                kwargs[name] = torch.cat(tensors, dim=0) if tensors else None
            elif name in actual_cls.shared_fields:
                kwargs[name] = getattr(first, name)
            else:
                collected: List[Any] = []
                for ctx in contexts:
                    val = getattr(ctx, name)
                    assert isinstance(val, list), (
                        f"ForwardContext.cat: field {name!r} is neither shared "
                        f"nor stackable_tensor; expected list, got {type(val).__name__}"
                    )
                    collected.extend(val)
                kwargs[name] = collected

        return actual_cls(**kwargs)

    def slice(self, start: int, end: int) -> "ForwardContext":
        """Slice stackable tensors along dim-0; shared fields are kept as-is."""
        actual_cls = type(self)
        kwargs: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            name = f.name
            val = getattr(self, name)
            if name in actual_cls.batched_tensor_fields:
                if val is not None and isinstance(val, torch.Tensor):
                    kwargs[name] = val[start:end].clone()
                else:
                    kwargs[name] = val
            elif name in actual_cls.shared_fields:
                kwargs[name] = val
            else:
                if isinstance(val, list):
                    kwargs[name] = val[start:end]
                else:
                    kwargs[name] = val
        return actual_cls(**kwargs)

    def reindex(self, indices: torch.Tensor) -> "ForwardContext":
        """Reindex stackable tensors along dim-0 using a permutation tensor."""
        actual_cls = type(self)
        kwargs: Dict[str, Any] = {}
        idx_list = indices.tolist()
        for f in dataclass_fields(self):
            name = f.name
            val = getattr(self, name)
            if name in actual_cls.batched_tensor_fields:
                if val is not None and isinstance(val, torch.Tensor):
                    kwargs[name] = val[indices]
                else:
                    kwargs[name] = val
            elif name in actual_cls.shared_fields:
                kwargs[name] = val
            else:
                if isinstance(val, list):
                    kwargs[name] = [val[i] for i in idx_list]
                else:
                    kwargs[name] = val
        return actual_cls(**kwargs)

    def to_device(self, device: Union[str, torch.device]) -> "ForwardContext":
        """Move all tensor fields to *device*."""
        actual_cls = type(self)
        kwargs: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            name = f.name
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                kwargs[name] = val.to(device)
            else:
                kwargs[name] = val
        return actual_cls(**kwargs)

    def cast_dtype(self, dtype: torch.dtype) -> "ForwardContext":
        """Cast stackable float tensors to *dtype* for transport optimization.

        Only float-type tensors in ``batched_tensor_fields`` are cast.
        Shared fields, integer tensors (masks, position IDs), and scalars
        are kept unchanged.
        """
        actual_cls = type(self)
        kwargs: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            name = f.name
            val = getattr(self, name)
            if (
                name in actual_cls.batched_tensor_fields
                and isinstance(val, torch.Tensor)
                and val.is_floating_point()
            ):
                kwargs[name] = val.to(dtype=dtype)
            else:
                kwargs[name] = val
        return actual_cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Export all non-None fields as a plain dict.

        This is the dict that gets ``**``-expanded into
        ``ModelForwardPlugin.forward(model, latents, sigma, **ctx.to_dict())``.
        """
        result: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val
        return result



# ---------------------------------------------------------------------------
# Model-specific subclasses
# ---------------------------------------------------------------------------


@register_forward_context("flux")
@dataclass
class FluxForwardContext(ForwardContext):
    """ForwardContext for FLUX models.

    Field naming follows the ModelForwardPlugin protocol: ``image_ids``
    (not ``img_ids``) so that ``plugin.forward(**ctx.to_dict())`` maps
    directly to the plugin signature without remapping.
    """

    shared_fields: ClassVar[FrozenSet[str]] = frozenset({
        "guidance_scale", "image_ids", "autocast_dtype",
    })
    batched_tensor_fields: ClassVar[FrozenSet[str]] = frozenset({
        "prompt_embeds", "pooled_prompt_embeds", "text_ids",
    })

    guidance_scale: float = 3.5
    prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    image_ids: Optional[torch.Tensor] = None
    autocast_dtype: Optional[torch.dtype] = None


@register_forward_context("sd3")
@dataclass
class SD3ForwardContext(ForwardContext):
    """ForwardContext for SD3 models."""

    shared_fields: ClassVar[FrozenSet[str]] = frozenset({
        "guidance_scale", "autocast_dtype",
    })
    batched_tensor_fields: ClassVar[FrozenSet[str]] = frozenset({
        "prompt_embeds", "pooled_prompt_embeds",
        "negative_prompt_embeds", "negative_pooled_prompt_embeds",
        "encoder_attention_mask",
    })

    guidance_scale: float = 7.0
    prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    autocast_dtype: Optional[torch.dtype] = None


@register_forward_context("hunyuan")
@dataclass
class HunyuanForwardContext(ForwardContext):
    """ForwardContext for HunyuanVideo models."""

    shared_fields: ClassVar[FrozenSet[str]] = frozenset({
        "guidance_scale", "autocast_dtype",
    })
    batched_tensor_fields: ClassVar[FrozenSet[str]] = frozenset({
        "prompt_embeds", "encoder_attention_mask",
    })

    guidance_scale: float = 1.0
    prompt_embeds: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    autocast_dtype: Optional[torch.dtype] = None


@register_forward_context("mochi")
@dataclass
class MochiForwardContext(ForwardContext):
    """ForwardContext for Mochi video models."""

    shared_fields: ClassVar[FrozenSet[str]] = frozenset({
        "guidance_scale", "autocast_dtype",
    })
    batched_tensor_fields: ClassVar[FrozenSet[str]] = frozenset({
        "prompt_embeds", "encoder_attention_mask",
    })

    guidance_scale: float = 4.5
    prompt_embeds: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    autocast_dtype: Optional[torch.dtype] = None


# Also register "default" pointing to base ForwardContext for fallback
@register_forward_context("default")
@dataclass
class DefaultForwardContext(ForwardContext):
    """Fallback ForwardContext with SD3-like fields."""

    shared_fields: ClassVar[FrozenSet[str]] = frozenset({
        "guidance_scale", "autocast_dtype",
    })
    batched_tensor_fields: ClassVar[FrozenSet[str]] = frozenset({
        "prompt_embeds", "pooled_prompt_embeds",
        "negative_prompt_embeds", "negative_pooled_prompt_embeds",
        "encoder_attention_mask",
        "text_ids", "image_ids",
    })

    guidance_scale: float = 3.5
    prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    image_ids: Optional[torch.Tensor] = None
    autocast_dtype: Optional[torch.dtype] = None


__all__ = [
    "ForwardContext",
    "FluxForwardContext",
    "SD3ForwardContext",
    "HunyuanForwardContext",
    "MochiForwardContext",
    "DefaultForwardContext",
    "register_forward_context",
    "get_forward_context_cls",
]
