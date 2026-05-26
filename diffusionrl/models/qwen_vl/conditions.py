from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.types.conditions import Condition, TextTokenCondition
from diffusionrl.utils.batched import FieldKind, field

_PIXEL_VALUES_KEY = "_qwen_vl_pixel_values"
_IMAGE_GRID_THW_KEY = "_qwen_vl_image_grid_thw"
_MM_TOKEN_TYPE_IDS_KEY = "_qwen_vl_mm_token_type_ids"


@dataclass
class _TensorWrapper(Condition):
    data: Optional[torch.Tensor] = field(kind=FieldKind.SHARED, transport=True, default=None)


@dataclass
class QwenVLARConditions(Transportable):
    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, transport=True, default=None)
    pixel_values: Optional[torch.Tensor] = field(kind=FieldKind.SHARED, transport=True, default=None)
    image_grid_thw: Optional[torch.Tensor] = field(kind=FieldKind.SHARED, transport=True, default=None)
    mm_token_type_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, transport=True, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QwenVLARConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"QwenVLARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got "
                f"{type(prompt).__name__ if prompt is not None else 'None'}"
            )

        pixel_values = None
        pv_wrapper = d.get(_PIXEL_VALUES_KEY)
        if isinstance(pv_wrapper, _TensorWrapper):
            pixel_values = pv_wrapper.data
        elif isinstance(pv_wrapper, torch.Tensor):
            pixel_values = pv_wrapper

        image_grid_thw = None
        gt_wrapper = d.get(_IMAGE_GRID_THW_KEY)
        if isinstance(gt_wrapper, _TensorWrapper):
            image_grid_thw = gt_wrapper.data
        elif isinstance(gt_wrapper, torch.Tensor):
            image_grid_thw = gt_wrapper

        mm_token_type_ids = None
        mt_wrapper = d.get(_MM_TOKEN_TYPE_IDS_KEY)
        if isinstance(mt_wrapper, _TensorWrapper):
            mm_token_type_ids = mt_wrapper.data
        elif isinstance(mt_wrapper, torch.Tensor):
            mm_token_type_ids = mt_wrapper

        return cls(
            prompt=prompt,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("QwenVLARConditions.to_dict: prompt field is None")
        out: Dict[str, Any] = {"prompt": self.prompt}
        if self.pixel_values is not None:
            out[_PIXEL_VALUES_KEY] = _TensorWrapper(data=self.pixel_values)
        if self.image_grid_thw is not None:
            out[_IMAGE_GRID_THW_KEY] = _TensorWrapper(data=self.image_grid_thw)
        if self.mm_token_type_ids is not None:
            out[_MM_TOKEN_TYPE_IDS_KEY] = _TensorWrapper(data=self.mm_token_type_ids)
        return out


__all__ = ["QwenVLARConditions"]
