"""PEPipeline — RolloutReq → RolloutResp end-to-end for Prompt Enhancement.

Implements the two-phase composed flow::

    Texts(raw) ──llm_pipeline.generate──▶ Texts(rewritten)
                                              │
                                              ▼
                              ──diffusion_pipeline.generate──▶ Images

PE composes two child :class:`Pipeline` instances at the *pipeline*
layer, not the stage layer. Each child remains a fully self-contained
unit (its bundle, stages, CFG-empty-negative handling, etc.) and is
reusable in non-PE pipelines. PE's job is request slicing, sequencing,
and response merging.

Invocation pattern
------------------
``PEPipeline.from_config(cfg: DictConfig)`` takes the raw, un-materialized
``cfg.model`` DictConfig. This is a deliberate deviation from the
``SD3Pipeline.from_config(SD3PipelineConfig)`` style: PE needs DictConfig
access for the nested children so it can dispatch each child's own
``_target_`` via :func:`diffusionrl.config.instantiate.build`, but
``build()`` materializes its input before dispatch — collapsing
``Any``-typed nested fields to plain Python dicts and losing the
DictConfig form. Canonical pattern: see
:meth:`diffusionrl.reward.service.RewardService.from_configs`.

Concretely: callers should invoke ``PEPipeline.from_config(cfg.model)``
directly (NOT ``build(cfg.model)``).

σ schedule contract
-------------------
Forwarded verbatim to the diffusion child. The LLM child never reads
``req.sigmas`` (see :class:`Qwen3Pipeline.generate`). The hosting engine
adapter pins ``req.sigmas`` on the parent PE request via
:func:`diffusionrl.sde.runtime.ensure_req_sigmas` before calling
``pe_pipeline.generate``; PE then passes that schedule through to the
diffusion sub-request unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from omegaconf import DictConfig

from diffusionrl.config.instantiate import build
from diffusionrl.models.types.pipeline import Pipeline
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.sampling import get_ar_params, get_diffusion_params

from .bundle import PEBundle

if TYPE_CHECKING:
    from diffusionrl.types.rollout_req import PrimitiveValue


class PEPipeline(Pipeline):
    """PE generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — raw user prompts, fed to the LLM.
    - ``primitives[<other>]`` — forwarded to the diffusion child (e.g.
      ``"negative_text"``, ``"image"``).
    - ``sampling_params: ComposedSamplingParams`` — decomposed into
      ``ARSamplingParams`` for the LLM child and
      ``DiffusionSamplingParams`` for the diffusion child.
    - ``stage_config["chat"]: dict`` (optional) — forwarded to the LLM
      chat-template stage as a per-request system-instruction override.
    - ``sigmas: Tensor[T+1]`` — engine-pinned; forwarded to the diffusion
      child only.
    - ``request_conditions: Dict[str, Condition]`` — forwarded to the
      diffusion child (e.g. ``"initial_latents"``).

    Writes to ``RolloutResp`` (track-dict union of both child responses;
    no sample-axis shifting because both children operate on the *same*
    sample set):

    - ``tracks["text"]: RolloutTrack`` — from the LLM:
      ``segment=TextSegment``, ``decoded=Texts`` (LLM-rewritten prompts),
      ``conditions={"prompt": TextTokenCondition, ...}``.
    - ``tracks["image"]: RolloutTrack`` — from the diffusion:
      ``segment=LatentSegment``, ``decoded=Images`` (final generated
      images), ``conditions={"text": TextEmbedCondition, ...}``.
    """

    def __init__(
        self,
        *,
        diffusion_pipeline: Pipeline,
        llm_pipeline: Pipeline,
    ) -> None:
        super().__init__()
        self.diffusion_pipeline = diffusion_pipeline
        self.llm_pipeline = llm_pipeline
        # Surfaces the composed bundle so downstream code (training-side
        # weight policies, eval introspection, ...) can reach
        # pe_pipeline.bundle.{diffusion, llm} without duplicating the
        # child-pipeline reference.
        self.bundle = PEBundle(
            diffusion=diffusion_pipeline.bundle,
            llm=llm_pipeline.bundle,
        )

    @classmethod
    def from_config(cls, config: DictConfig) -> "PEPipeline":
        """Build PEPipeline from the raw ``cfg.model`` DictConfig.

        ``config.diffusion`` and ``config.llm`` each carry their own
        ``_target_`` pointing at a registered child Pipeline's
        ``from_config`` classmethod; we dispatch each via
        :func:`diffusionrl.config.instantiate.build`.

        Note: this method is NOT invocable via ``build(cfg.model)`` —
        ``build`` materializes the cfg before dispatch, which collapses
        ``Any``-typed nested children to plain Python dicts and loses the
        DictConfig form required to call ``build()`` recursively on
        each child. Call this directly (mirroring
        :meth:`RewardService.from_configs`).
        """
        return cls(
            diffusion_pipeline=build(config.diffusion),
            llm_pipeline=build(config.llm),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        # Phase 1: LLM expand — runs without sigmas / request_conditions.
        llm_req = self._build_llm_req(req)
        llm_resp = self.llm_pipeline.generate(llm_req)

        # The rewritten prompt lives on the LLM track's ``decoded`` field
        # as a single :class:`Texts` primitive (flat post-Step-9 / 8f8d19f).
        # Both Qwen3Pipeline and any future AR LLM following the new
        # pipeline contract emit a track named ``"text"`` with
        # ``decoded=Texts(...)``.
        llm_track = llm_resp.tracks.get("text")
        rewritten_obj = llm_track.decoded if llm_track is not None else None
        if not isinstance(rewritten_obj, Texts):
            raise RuntimeError(
                "PEPipeline.generate: LLM child returned tracks['text'].decoded of "
                f"type {type(rewritten_obj).__name__ if rewritten_obj is not None else 'None'}; "
                "expected Texts. The LLM pipeline must produce a Texts primitive on "
                "tracks['text'].decoded so the diffusion child can consume it as "
                "primitives['text']."
            )
        if len(rewritten_obj.texts) != len(req.sample_ids):
            raise RuntimeError(
                f"PEPipeline.generate: LLM child returned {len(rewritten_obj.texts)} "
                f"rewritten text(s) but the parent request has {len(req.sample_ids)} "
                "sample(s). PE preserves a 1:1 sample mapping across children; the LLM "
                "pipeline must not expand or drop samples."
            )

        # Phase 2: diffusion generate, with the rewritten prompt swapped into primitives["text"].
        diff_req = self._build_diffusion_req(req, rewritten_obj)
        diff_resp = self.diffusion_pipeline.generate(diff_req)

        # Phase 3: track-dict union. Both children operate on the same
        # sample set (sample_ids unchanged), so we do NOT call
        # RolloutResp.concat — that's for sample-axis stacking of multiple
        # shards and would double the sample count. Here we union by track
        # name across the parallel responses. The LLM child contributes
        # the ``"text"`` track; the diffusion child contributes
        # ``"image"`` (and any future per-modality tracks).
        return RolloutResp(tracks={**llm_resp.tracks, **diff_resp.tracks})

    # ------------------------------------------------------------------
    # Child-request construction
    # ------------------------------------------------------------------

    def _build_llm_req(self, req: RolloutReq) -> RolloutReq:
        """Construct the LLM-side child RolloutReq.

        Forwards ``primitives["text"]`` and the AR sampling params.
        Drops sigmas, request_conditions, non-text primitives, and
        diffusion sampling params.
        """
        text = req.primitives.get("text")
        if not isinstance(text, Texts):
            raise TypeError(
                "PEPipeline.generate: req.primitives['text'] must be a Texts primitive; "
                f"got {type(text).__name__ if text is not None else 'None'}. "
                "The LLM child requires the raw user prompt at primitives['text']."
            )
        return RolloutReq(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            primitives={"text": text},
            request_conditions={},
            sampling_params=get_ar_params(req.sampling_params),
            stage_config={k: v for k, v in req.stage_config.items() if k in ("chat",)},
            sigmas=None,
        )

    def _build_diffusion_req(self, req: RolloutReq, rewritten: Texts) -> RolloutReq:
        """Construct the diffusion-side child RolloutReq.

        Replaces ``primitives["text"]`` with ``rewritten``, forwards any
        other primitives (negative_text, image, ...), forwards sigmas
        and request_conditions verbatim, and extracts the diffusion
        sampling params.
        """
        diffusion_primitives: Dict[str, "PrimitiveValue"] = dict(req.primitives)
        diffusion_primitives["text"] = rewritten
        return RolloutReq(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            primitives=diffusion_primitives,
            request_conditions=dict(req.request_conditions),
            sampling_params=get_diffusion_params(req.sampling_params),
            sigmas=req.sigmas,
        )


__all__ = ["PEPipeline"]
