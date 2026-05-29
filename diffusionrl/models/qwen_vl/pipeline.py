from __future__ import annotations

from typing import Any, Dict

from diffusionrl.models.types.ar import ARSamplingParams
from diffusionrl.models.types.pipeline import Pipeline
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.sampling import get_ar_params

from .ar import QwenVLARParams, QwenVLARStage
from .bundle import QwenVLBundle
from .chat_template import QwenVLChatTemplateStage
from .conditions import QwenVLARConditions
from .config import QwenVLPipelineConfig


class QwenVLPipeline(Pipeline):
    def __init__(
        self,
        *,
        bundle: QwenVLBundle,
        chat_template: QwenVLChatTemplateStage,
        ar: QwenVLARStage,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template
        self.ar = ar

    @classmethod
    def from_config(cls, config) -> "QwenVLPipeline":
        if isinstance(config, dict):
            config = QwenVLPipelineConfig(**{k: v for k, v in config.items() if k != "_target_"})
        bundle = QwenVLBundle.from_config(config)
        chat_template = QwenVLChatTemplateStage(
            bundle,
            max_prompt_length=config.max_prompt_length,
        )
        ar = QwenVLARStage(model=bundle)
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    def generate(self, req: RolloutReq) -> RolloutResp:
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"QwenVLPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        pil_images = None
        images_prim = req.primitives.get("image")
        if images_prim is not None and isinstance(images_prim, Images):
            pil_images = [img.to_pil() for img in images_prim.to_list()]

        chat_overrides: Dict[str, Any] = dict(req.stage_config.get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = QwenVLChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
            )
        else:
            chat_stage = self.chat_template

        conds: QwenVLARConditions = chat_stage.embed(texts, images=pil_images)

        ar = get_ar_params(req.sampling_params)
        if ar is not None:
            params = QwenVLARParams(
                max_tokens=ar.max_new_tokens,
                temperature=ar.temperature,
                top_p=ar.top_p,
                top_k=ar.top_k,
            )
        else:
            params = QwenVLARParams()

        sampling_params = ARSamplingParams(
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=None,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)
        decoded = self._detokenize(segment)

        return RolloutResp(
            tracks={
                "text": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conds.to_dict(),
                    segment=segment,
                    decoded=decoded,
                ),
            }
        )

    def _detokenize(self, segment) -> Texts:
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        tokenizer = self.bundle.tokenizer
        out: list = []
        n = len(cu) - 1
        for i in range(n):
            chunk = segment.tokens[cu[i] : cu[i + 1]]
            ids = chunk.tolist() if chunk.numel() > 0 else []
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        return Texts(texts=out)


__all__ = ["QwenVLPipeline"]
