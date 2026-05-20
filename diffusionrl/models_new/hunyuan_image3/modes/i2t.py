"""i2t — image-to-text autoregressive generation.

Reads ``primitives["text"]: Texts`` (the prompt) and
``primitives["image"]: Images`` (the image to caption / answer about),
plus ``stage_params["ar"]: dict`` (optional). Builds chat-templated
``input_ids`` with embedded ``<img>`` markers via the chat-template
wrapper, then runs ``HunyuanImage3ARStage.autoregress`` against the
backbone in ``mode="gen_text"`` -- the unified MM forward scatters
ViT patch embeddings into the prompt's ``<img>`` slots via
``instantiate_vit_image_tokens``.

Conditions on the response carry the chat-templated ``input_ids`` plus
the ``cond_vit_*`` / ``vit_kwargs`` tensors that drove the ViT-tokens
scatter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from diffusionrl.models_new.types.ar import ARSamplingParams
from diffusionrl.types.conditions import ImageEmbedCondition
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

from ..ar import HunyuanImage3ARParams
from ..conditions import HunyuanImage3ARConditions
from .t2t import _resolve_system_prompt, _stop_tokens_for_bot_task

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def generate(pipeline: "HunyuanImage3Pipeline", req: RolloutReq) -> RolloutResp:
    """i2t — AR-stage rollout with image comprehension."""
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (i2t): "
            f"req.primitives['text'] must be Texts, "
            f"got {type(texts).__name__ if texts is not None else 'None'}"
        )
    images = req.primitives.get("image")
    if not isinstance(images, Images):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (i2t): "
            f"req.primitives['image'] must be Images, "
            f"got {type(images).__name__ if images is not None else 'None'}"
        )

    ar_kwargs: Dict[str, Any] = dict(req.stage_params.get("ar") or {})
    ar_params = HunyuanImage3ARParams(**ar_kwargs)
    bot_task = str(ar_params.bot_task)

    system_prompt = _resolve_system_prompt(
        pipeline.bundle, bot_task, ar_params.use_system_prompt, ar_params.system_prompt
    )
    system_prompt_list = [system_prompt] * len(texts.texts) if system_prompt is not None else None

    # vit: {"joint_image_info": [[JointImageInfo]]*B, "cond_vit_images":
    #       list[Tensor [S_b, D]]*B, "vit_kwargs": {"spatial_shapes",
    #       "attention_mask"}}
    vit = pipeline.vit_encode.encode_for_cond_vit(images)

    # Chat template path: pass batch_cond_image_info so the wrapper
    # splices in <img> markers; the resulting ``cond_vit_image_mask``
    # (now on ``fused``) pins which ``input_ids`` positions hold the
    # ViT scatter target.
    mm = pipeline.text_embed.embed_for_ar(
        texts,
        bot_task=bot_task,
        system_prompt=system_prompt_list,
        cot_text=([ar_params.cot_text] * len(texts.texts) if ar_params.cot_text else None),
        batch_cond_image_info=vit["joint_image_info"],
    )

    cond_vit = ImageEmbedCondition(
        embeds=vit["cond_vit_images"],
        attn_mask=vit["vit_kwargs"]["attention_mask"],
        spatial_shapes=vit["vit_kwargs"]["spatial_shapes"],
    )
    ar_conds = HunyuanImage3ARConditions(
        fused=mm["fused"],
        cond_vit=cond_vit,
        tokenizer_output=mm["tokenizer_output"],
    )

    stop_ids: List[int] = list(ar_params.stop_token_ids or [])
    if not stop_ids:
        stop_ids = _stop_tokens_for_bot_task(pipeline.bundle, bot_task)
    sampling_params = ARSamplingParams(
        max_tokens=int(ar_params.max_tokens),
        temperature=float(ar_params.temperature),
        top_p=float(ar_params.top_p),
        top_k=int(ar_params.top_k),
        stop_token_id=stop_ids[0] if stop_ids else None,
    )
    ar_params_with_stops = HunyuanImage3ARParams(**{**ar_kwargs, "stop_token_ids": stop_ids})

    text_seg = pipeline.ar.autoregress(ar_conds, sampling_params=sampling_params, params=ar_params_with_stops)

    decoded_texts = pipeline._detokenize_text_segment(text_seg)

    return RolloutResp(
        sample_ids=list(req.sample_ids),
        group_ids=list(req.group_ids),
        conditions=ar_conds.to_dict(),
        rollout_traces={"text": text_seg},
        decoded={"text": decoded_texts},
    )
