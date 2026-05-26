from __future__ import annotations

from typing import List, Optional

import PIL.Image
import torch

from diffusionrl.types.conditions import TextTokenCondition
from diffusionrl.types.primitives import Texts

from .bundle import QwenVLBundle
from .conditions import QwenVLARConditions


class QwenVLChatTemplateStage:
    def __init__(
        self,
        bundle: QwenVLBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)

    def embed(
        self,
        texts: Texts,
        images: Optional[List[Optional[PIL.Image.Image]]] = None,
    ) -> QwenVLARConditions:
        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(texts.texts)

        per_sample_inputs = []
        for i, text in enumerate(texts.texts):
            content: list = []
            sample_images: list = []
            if images is not None and i < len(images) and images[i] is not None:
                content.append({"type": "image", "image": images[i]})
                sample_images.append(images[i])
            content.append({"type": "text", "text": text})

            messages: list = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": content})

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            per_sample_inputs.append(inputs)

        max_len = min(
            max(inp["input_ids"].shape[-1] for inp in per_sample_inputs),
            self.max_prompt_length,
        )
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "QwenVLChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "QwenVLBundle.from_config sets pad_token=eos_token when absent."
            )

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        mm_token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.int, device=device)

        for i, inp in enumerate(per_sample_inputs):
            ids = inp["input_ids"].squeeze(0)
            L = min(int(ids.shape[0]), max_len)
            input_ids[i, :L] = ids[:L].to(device)
            mask = inp["attention_mask"].squeeze(0)
            attention_mask[i, :L] = mask[:L].to(device)
            if "mm_token_type_ids" in inp and inp["mm_token_type_ids"] is not None:
                mt = inp["mm_token_type_ids"].squeeze(0)
                mm_token_type_ids[i, :L] = mt[:L].to(device)

        pvs = []
        gts = []
        for inp in per_sample_inputs:
            if "pixel_values" in inp and inp["pixel_values"] is not None:
                pvs.append(inp["pixel_values"].to(device=device, dtype=dtype))
            if "image_grid_thw" in inp and inp["image_grid_thw"] is not None:
                gts.append(inp["image_grid_thw"].to(device=device))

        pixel_values = torch.cat(pvs, dim=0) if pvs else None
        image_grid_thw = torch.cat(gts, dim=0) if gts else None

        return QwenVLARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )


__all__ = ["QwenVLChatTemplateStage"]
