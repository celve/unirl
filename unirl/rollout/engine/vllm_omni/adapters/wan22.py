"""Wan2.2 family: input/output sub-adapters + the ``wan22_t2v`` modality class.

Single diffusion stage, TP=1, dual-expert DiT (high/low-noise transformers
switched at ``boundary_ratio`` inside the worker). The request side derives
from the shared :class:`~.dit.DitInputAdapter` adding the video-only
``num_frames`` knob and pinning ``boundary_ratio`` from the shared bundle
config (one source for engine and trainer); the response side derives from
:class:`~.dit.DitOutputAdapter` packing per-prompt PIL frame groupings into
``Videos`` and the single-stream UMT5 text conditions.

Geometry: UNGROUPED on purpose — ``core_diff_kwargs`` already sends
``num_outputs_per_prompt=1`` (one request per GRPO sample), which is wan22's
native trainside geometry (``forward_batch_size: 1`` ≡ ``micro_batch_size:
1``), so unlike SD3 the parity recipes pay no throughput premium for it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs, grouped_pils_to_videos
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


def _num_frames(req: RolloutReq) -> int:
    return int(getattr(req.sampling_params.get("diffusion"), "num_frames", 1))


class Wan22InputAdapter(DitInputAdapter):
    """SD3-style request side + the video knobs.

    ``num_frames`` is read by upstream ``forward`` off the typed sampling
    params only (not the prompt dict — unlike HV1.5), and defaults to a
    truthy ``1`` there, so it must always be sent explicitly.
    ``boundary_ratio`` rides the same channel so the worker's expert switch
    uses the recipe's value instead of upstream's warned default.
    """

    def __init__(self, modality: str, *, boundary_ratio: Optional[float] = None) -> None:
        super().__init__(modality)
        self.boundary_ratio = boundary_ratio

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        sampling = super().build_sampling(req)
        sampling[0].kwargs["num_frames"] = _num_frames(req)
        if self.boundary_ratio is not None:
            sampling[0].kwargs["boundary_ratio"] = float(self.boundary_ratio)
        return sampling


class Wan22VideoOutputAdapter(DitOutputAdapter):
    """Single-"video"-track response: frame groupings + UMT5 text conditions."""

    track_name = "video"
    final_output_type = "video"

    _MISSING_CAPTURE_MSG = (
        "build_response: wan22_t2v rollout returned no 'text_capture' on "
        "DiffusionOutput.custom_output. Check that RLWan22Pipeline's "
        "encode_prompt hook ran in every DiT worker — verify "
        "custom_pipeline_args.pipeline_class in the stage YAML."
    )

    def build_decoded(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del req
        _, frame_groups, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: grouped_pils_to_videos(frame_groups)}

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Concat the per-request UMT5 ``text_capture`` dicts into conditions.

        Written by ``RLWan22Pipeline`` after intercepting ``encode_prompt``.
        Embeddings are zero-padded to the fixed ``max_sequence_length`` (512)
        upstream, so a plain dim-0 concat suffices. Returns the conditions
        *dict* (keys aligned with ``WAN21Conditions.from_dict``) — the
        trainer runs ``from_dict(track.conditions)`` itself.
        """
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )

        captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
        if any(c is None for c in captures):
            raise RuntimeError(self._MISSING_CAPTURE_MSG)

        embeds = torch.cat([c["prompt_embeds"] for c in captures], dim=0)
        negatives = [c.get("negative_prompt_embeds") for c in captures]
        if any(n is None for n in negatives) and any(n is not None for n in negatives):
            raise RuntimeError(
                "build_response: wan22_t2v captures mixed present/absent negative_prompt_embeds "
                "across a batch — guidance must be uniform per rollout batch."
            )
        negative_embeds = torch.cat(negatives, dim=0) if negatives[0] is not None else None

        if req.sample_ids and int(embeds.shape[0]) != len(req.sample_ids):
            raise RuntimeError(
                f"wan22 text condition batch {int(embeds.shape[0])} != sample count "
                f"{len(req.sample_ids)} (wan22_t2v is ungrouped: one request per sample)."
            )

        cond_dict: Dict[str, Any] = {"text": TextEmbedCondition(embeds=embeds, pooled=None, attn_mask=None)}
        if negative_embeds is not None:
            cond_dict["negative_text"] = TextEmbedCondition(embeds=negative_embeds, pooled=None, attn_mask=None)
        return cond_dict


@register_adapter("wan22_t2v")
class Wan22T2vAdapter(ModelAdapter):
    """Wan2.2 text → video (single diffusion stage, dual-expert DiT, TP=1)."""

    stage_yaml = "wan22_t2v_rl.yaml"
    omni_mode = "text-to-video"
    # Wan's tokenizer lives in the tokenizer/ subfolder; the worker loads it
    # internally and the driver-side translator needs none.
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Wan22InputAdapter(
            self.modality, boundary_ratio=getattr(model_config, "boundary_ratio", None)
        )
        self.output_adapter = Wan22VideoOutputAdapter(self.modality)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = ["Wan22InputAdapter", "Wan22T2vAdapter", "Wan22VideoOutputAdapter"]
