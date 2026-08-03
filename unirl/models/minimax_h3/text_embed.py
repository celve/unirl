"""MiniMax-H3 text embedding stage -- Qwen3-VL layer-50 hidden states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .vendor import MINIMAX_H3_TEXT_ENCODER_LAYER, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle


class MiniMaxH3TextEmbedStage:
    """Encode prompts into the conditioning MiniMax-H3 was trained on.

    Three things differ from an ordinary text-encoder stage, and all three are
    checkpoint contracts rather than preferences:

    * The conditioning is ``hidden_states[50]`` of the Qwen3-VL conditioner --
      an intermediate, **unnormalized** hidden state, not ``last_hidden_state``.
      The final layer is post-norm and is not what H3 consumes.
    * There is no chat template. The prompt is tokenized raw with
      ``add_special_tokens=False``.
    * The call goes to ``text_encoder.model``, the decoder submodule, not to
      ``text_encoder``. H3 never uses the language-model head, and skipping it
      avoids a vocabulary-wide projection over every token.
    """

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.hidden_layer = bundle.text_encoder_hidden_layer
        self.dtype = bundle.dtype
        self.device = bundle.device

    def _build_token_stream(self, prompt: str, keyframes: Sequence) -> Tuple[List[int], List[int], Any, Any]:
        """MiniMax-H3's presentation of a request: labelled keyframes, then prompt.

        Each keyframe is announced as ``"<Picture i>: "`` followed by a vision
        block. The LABEL rows are TEXT; the vision-block rows are tagged VIDEO,
        which is why the packed layout cannot assume a uniform text tag once
        keyframes exist. The prompt follows verbatim -- no chat template.
        """
        token_ids: List[int] = []
        token_tags: List[int] = []
        pixel_values = None
        image_grid_thw = None

        if keyframes:
            vision = self.processor.image_processor(images=list(keyframes), return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = self.processor.image_processor.merge_size**2
            for index in range(len(keyframes)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
        return token_ids, token_tags, pixel_values, image_grid_thw

    @torch.no_grad()
    def embed(self, texts: Texts, keyframes: Optional[Sequence] = None) -> Tuple[TextEmbedCondition, torch.Tensor]:
        """Encode prompts (and any keyframes) -> ``(condition, text_token_tags)``.

        ``keyframes`` is the per-request list of prepared keyframe images, in
        packed order, shared across the batch. Returns the per-row modality tags
        alongside the embedding because with a vision block present they are no
        longer derivable from the embedding length.
        """
        prompts: List[str] = list(texts.to_list()) if hasattr(texts, "to_list") else list(texts)
        require(len(prompts) > 0, "MiniMaxH3TextEmbedStage: no prompts to embed")

        num_layers = len(self.text_encoder.model.layers)
        require(
            num_layers > MINIMAX_H3_TEXT_ENCODER_LAYER,
            f"MiniMaxH3TextEmbedStage: MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of "
            f"its Qwen3-VL conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"the loaded conditioner has {num_layers}.",
        )

        embeds = []
        tags_per_prompt = []
        for prompt in prompts:
            token_ids, token_tags, pixel_values, image_grid_thw = self._build_token_stream(prompt, keyframes or ())
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            # Qwen3-VL lays its 3D rotary positions out per modality run, read
            # off the token type ids the processor derives (0 text, 1 image,
            # 2 video).
            mm_token_type_ids = torch.tensor(
                self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=self.device
            )
            outputs = self.text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=None if pixel_values is None else pixel_values.to(self.device, self.text_encoder.dtype),
                image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(self.device),
                use_cache=False,
                output_hidden_states=True,
            )
            embeds.append(outputs.hidden_states[self.hidden_layer].to(device=self.device, dtype=self.dtype))
            tags_per_prompt.append(torch.tensor(token_tags, dtype=torch.long))

        lengths = {int(e.shape[1]) for e in embeds}
        require(
            len(lengths) == 1,
            f"MiniMaxH3TextEmbedStage: prompts tokenized to differing lengths {sorted(lengths)}. The packed sequence "
            f"geometry must be identical across the batch (LatentSegment stores latents in a CONCAT field), so a "
            f"mixed-length batch cannot be packed. Pad or group prompts by token length upstream.",
        )
        return TextEmbedCondition(embeds=torch.cat(embeds, dim=0)), tags_per_prompt[0]


__all__ = ["MiniMaxH3TextEmbedStage"]
