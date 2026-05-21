"""QwenImagePipeline — RolloutReq → RolloutResp end-to-end for Qwen-Image.

Implements the four-tier flow::

    Texts ──text_embed──▶ QwenImageConditions ──diffuse──▶ LatentSegment
                                                              │
                                                              ▼
                                                          vae_decode
                                                              │
                                                              ▼
                                                            Images

Hydra constructs a pipeline via
``QwenImagePipeline.from_config(QwenImagePipelineConfig)`` (see
``config.py``); ``from_config`` loads the :class:`QwenImageBundle` then
constructs the four stages with the precision policy from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins ``req.sigmas`` via
:func:`diffusionrl.sde.runtime.ensure_req_sigmas` BEFORE calling
``generate(req)``; this pipeline reads ``req.sigmas`` and uses it
verbatim. The engine builds the policy with
:meth:`FlowMatchSchedulePolicy.from_pretrained(pretrained_path,
shift=pipeline.shift)`; Qwen-Image's
``scheduler/scheduler_config.json`` carries the dynamic-shift block, so
the policy enables dynamic μ derivation automatically. The pipeline
neither owns a σ builder nor reads model-specific scheduler config.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from diffusionrl.models.types.pipeline import Pipeline
from diffusionrl.sde.kernels import FlowSDEStrategy, StepStrategy
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

from .bundle import QwenImageBundle
from .conditions import QwenImageConditions
from .config import QwenImagePipelineConfig
from .diffusion import (
    QwenImageDiffusionParams,
    QwenImageDiffusionStage,
    QwenImageDiffusionStep,
)
from .text_embed import QwenImageTextEmbedStage
from .vae import QwenImageVAEDecodeStage


class QwenImagePipeline(Pipeline):
    """Qwen-Image generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``primitives["negative_text"]: Texts`` — optional CFG negatives.
    - ``stage_params["diffusion"]: dict`` — kwargs for
      :class:`QwenImageDiffusionParams`.
    - ``sigmas: Tensor[T+1]`` — pinned by the engine adapter (required).

    Writes to ``RolloutResp``:

    - ``conditions["text"]: TextEmbedCondition``; plus
      ``conditions["negative_text"]: TextEmbedCondition`` when negative
      prompts were supplied.
    - ``rollout_traces["image"]: LatentSegment``.
    - ``decoded["image"]: Images``.
    """

    def __init__(
        self,
        *,
        bundle: QwenImageBundle,
        text_embed: QwenImageTextEmbedStage,
        diffusion: QwenImageDiffusionStage,
        vae_decode: QwenImageVAEDecodeStage,
        shift: float = 3.0,
    ) -> None:
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.vae_decode = vae_decode
        # ``shift`` is retained as an attribute so the hosting engine
        # (TrainsideRolloutEngine / VLLMOmniRolloutEngine /
        # SGLangRolloutEngine) can read it when constructing the
        # FlowMatchSchedulePolicy at startup. For Qwen-Image, the
        # checkpoint's scheduler_config.json enables dynamic shifting,
        # so the static ``shift`` value is only used as a fallback when
        # the pretrained path is not a local directory.
        self.shift = shift

    def build_schedule_policy(self):
        """Build the FlowMatchSchedulePolicy for this pipeline.

        Qwen-Image's transformer was trained with **dynamic shift**
        (per the upstream ``QwenImagePipeline`` reference): μ is
        derived per-request from ``image_seq_len`` via
        :func:`calculate_dynamic_mu`. Static shift would silently
        mis-shift σ and drift the GRPO ratio.

        On real-pod runs where ``pretrained_model_ckpt_path`` is a
        local mount, ``from_pretrained`` reads dynamic-shift fields
        from ``scheduler/scheduler_config.json`` automatically.

        On HF-Hub-repo-id paths (e.g. ``Qwen/Qwen-Image``) the path
        isn't local at policy build time. Without ``require_dynamic``,
        ``from_pretrained`` would fall back to ``static_only(shift)``
        — silently wrong. We pass ``require_dynamic=True`` so that path
        either uses the upstream-canonical ``dynamic_overrides`` below
        OR raises loudly.

        Canonical overrides come from upstream QwenImage scheduler
        defaults (``diffusers/src/diffusers/pipelines/qwen_image
        /pipeline_qwen_image.py``).
        """
        from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            getattr(self.bundle, "pretrained_path", None),
            shift=float(self.shift),
            require_dynamic=True,
            dynamic_overrides={
                "base_shift": 0.5,
                "max_shift": 1.15,
                "base_image_seq_len": 256,
                "max_image_seq_len": 4096,
                "time_shift_type": "exponential",
                "vae_scale_factor": 8,
                "patch_size": 2,
            },
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side
        noise pre-computation. Mirrors
        :meth:`QwenImageDiffusionStage._latent_shape`'s arithmetic so the
        driver-provided noise tensor matches what the stage would
        otherwise generate internally.

        Qwen-Image canonical: ``AutoencoderKLQwenImage`` is 16-channel,
        ``QwenImageTransformer2DModel.in_channels=64`` (packed 16×4); the
        post-VAE latent grid is 8× downsampled with an extra patchify-2×2
        rounding (``latent_h = 2 * (H // 16)``).
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8  # AutoencoderKLQwenImage canonical
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: QwenImagePipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "QwenImagePipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`FlowSDEStrategy`; callers running GRPO with a different
        SDE family (Dance / CPS / DPM2) pass an explicit strategy built
        from ``cfg.sampling.sde_strategy``.
        """
        bundle = QwenImageBundle.from_config(config)
        text_embed = QwenImageTextEmbedStage(bundle, max_sequence_length=config.max_sequence_length)
        step = QwenImageDiffusionStep()
        diffusion = QwenImageDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = QwenImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run Qwen-Image t2i end-to-end. Requires ``req.sigmas`` to be
        pinned by the hosting engine adapter."""
        if req.sigmas is None:
            raise ValueError(
                "QwenImagePipeline.generate: req.sigmas is None. The hosting "
                "engine (Trainside / SGLang / VLLMOmni) must call "
                "diffusionrl.sde.runtime.ensure_req_sigmas(req, policy) before "
                "invoking pipeline.generate; see the σ ownership note in "
                "diffusionrl.models.types.pipeline."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"QwenImagePipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"QwenImagePipeline.generate: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )

        # Filter stage_params["diffusion"] down to fields that
        # QwenImageDiffusionParams actually accepts. Upstream drivers
        # (e.g. RolloutPipeline) stash rollout-metadata keys here
        # too — most notably ``num_samples_per_prompt`` — and the
        # dataclass constructor rejects unknown kwargs.
        import dataclasses as _dc

        raw_dict: Dict[str, Any] = dict(req.stage_params.get("diffusion") or {})
        allowed = {f.name for f in _dc.fields(QwenImageDiffusionParams)}
        params_dict = {k: v for k, v in raw_dict.items() if k in allowed}
        params = QwenImageDiffusionParams(**params_dict)

        text_cond = self.text_embed.embed(texts)
        # CFG empty negative: Qwen-Image upstream (diffusers v0.37.1
        # ``QwenImagePipeline`` docstring at ``pipeline_qwenimage.py:509``)
        # recommends a single-space ``" "`` as the canonical empty
        # negative. Empty string ``""`` is unsafe here: ``_get_qwen_prompt_embeds``
        # wraps the prompt in a chat template and strips the first 34
        # tokens of the encoder output (``prompt_template_encode_start_idx``)
        # — an empty user-content slot can degenerate into a near-zero
        # embedding and divide the norm-corrected CFG blend
        # (diffusion.py:248-250) by ~0. Mirrors PR #104 OLD
        # ``encode_inputs`` (``[" "] * batch_size``).
        #
        # Compare WAN21/WAN22 (which default to ``[""] * B``): WAN's
        # text encoder doesn't run a 34-token prefix strip, so ``""``
        # is safe there. The empty-negative value is a per-model
        # property, not a framework knob — hence hardcoded per pipeline.
        if negatives is None and float(params.guidance_scale) > 1.0:
            negatives = Texts(texts=[" "] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        qwen_conds = QwenImageConditions(text=text_cond, negative_text=negative_text_cond)

        schedule = req.sigmas.to(self.bundle.device)

        initial_cond = (req.request_conditions or {}).get("initial_latents")
        initial_latents = getattr(initial_cond, "latents", None) if initial_cond is not None else None

        latent_seg = self.diffusion.diffuse(
            qwen_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        images = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            conditions=qwen_conds.to_dict(),
            rollout_traces={"image": latent_seg},
            decoded={"image": images},
        )


__all__ = ["QwenImagePipeline"]
