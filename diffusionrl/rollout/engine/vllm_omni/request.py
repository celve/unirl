"""``RolloutReq`` → vLLM-Omni request translator.

Single translator, ``_to_omni_per_stage(req, cfg, *, modality, tokenizer)``,
mirrors the official end-to-end inference example at
``vllm-omni/examples/offline_inference/hunyuan_image3/end2end.py:165-194``.

The official example is the canonical reference for the per-prompt dict
shape:

    {"prompt_token_ids": ids,
     "prompt": raw_user_text,
     "use_system_prompt": sys_type,
     "modalities": [...],
     # for it2i / i2t:
     "multi_modal_data": {"image": pil},
     "height": h, "width": w}

Token IDs come from
``vllm_omni.diffusion.models.hunyuan_image3.prompt_utils.build_prompt_tokens``,
which tokenizes segment-by-segment to match HF ``apply_chat_template``
byte-for-byte (single-pass ``tokenizer.encode`` of the assembled string
merges BPE across segment boundaries, shifting token ids vs. the HF
baseline — see ``prompt_utils.py:104-112``).

Modality → upstream task mapping (mirrors upstream ``_TASK_PRESETS``):

    t2i  → ("t2i_think",  "en_unified", ["image"])
    it2i → ("it2i_think", "en_unified", ["image"])
    i2t  → ("i2t",        "en_unified", ["text"])
    t2t  → ("t2t",        "en_unified", ["text"])

The bot_task can be overridden per-request via
``stage_params["bot_task"]`` (e.g. ``"recaption"`` swaps the trigger tag
from ``<think>`` to ``<recaption>``); when omitted, the default for
modality is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from diffusionrl.config.require import require
from diffusionrl.types.primitives import Image, Images, Texts
from diffusionrl.types.rollout_req import RolloutReq

if TYPE_CHECKING:
    from diffusionrl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig


# (default_task_key, default_sys_type, modalities) per modality.
_TASK_DEFAULTS: Dict[str, Tuple[str, str, List[str]]] = {
    "t2i": ("t2i_think", "en_unified", ["image"]),
    "t2i_think_recaption": ("t2i_think", "en_unified", ["image"]),
    "it2i": ("it2i_think", "en_unified", ["image"]),
    "i2t": ("i2t", "en_unified", ["text"]),
    "t2t": ("t2t", "en_unified", ["text"]),
}


def _resolve_task(modality: str, stage_params: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Resolve ``(task_key, sys_type, modalities)`` with optional overrides.

    ``stage_params["bot_task"]`` swaps the trigger tag used by upstream's
    chat template (``think`` / ``recaption``). ``stage_params["sys_type"]``
    overrides the system-prompt key (``en_unified`` / ``en_vanilla``).
    """
    if modality not in _TASK_DEFAULTS:
        raise ValueError(f"_resolve_task: unsupported modality {modality!r}. Choose one of {list(_TASK_DEFAULTS)}.")
    default_task, default_sys, modalities = _TASK_DEFAULTS[modality]

    sys_type = stage_params.get("sys_type") or default_sys

    bot_task = stage_params.get("bot_task")
    if bot_task and modality in ("t2i", "it2i", "t2i_think_recaption"):
        # think / recaption / vanilla — translate to upstream task key.
        if bot_task == "vanilla" and modality == "t2i":
            return "t2i_vanilla", "en_vanilla", modalities
        if bot_task in ("think", "recaption"):
            return f"{modality}_{bot_task}", sys_type, modalities

    return default_task, sys_type, modalities


def _texts_from_req(req: RolloutReq) -> Texts:
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            f"req.primitives['text'] must be Texts, got {type(texts).__name__ if texts is not None else 'None'}"
        )
    if len(texts.texts) != len(req.sample_ids):
        raise ValueError(f"prompt count {len(texts.texts)} != sample_ids count {len(req.sample_ids)}")
    return texts


def _images_from_req(req: RolloutReq, n: int) -> List[Any]:
    """Convert ``req.primitives['image']`` (Images) → list of PIL images.

    Returns an empty list when there's no image primitive. Asserts batch
    alignment when present.
    """
    images = req.primitives.get("image")
    if images is None:
        return []
    if not isinstance(images, Images):
        raise TypeError(f"req.primitives['image'] must be Images when present, got {type(images).__name__}")
    if len(images) != n:
        raise ValueError(f"image batch {len(images)} != prompt count {n}")
    return [Image(pixels=images.pixels[i]).to_pil() for i in range(len(images))]


def _sigmas_list_from_req(req: RolloutReq, num_inference_steps: int) -> Optional[List[float]]:
    """Return ``req.sigmas`` as a plain ``T``-length list[float].

    Worker side (upstream pipeline_sd3 / pipeline_hunyuan_image3) routes
    a non-None ``sampling_params.sigmas`` into the scheduler via
    ``retrieve_timesteps`` → ``set_timesteps(sigmas=...)``. We send the
    schedule the trainer will replay against (``req.sigmas``) so worker
    and replay use identical σ. ``None`` falls back to the worker's
    internal schedule (legacy behavior, kept for engines that bypass
    :func:`diffusionrl.sde.runtime.ensure_req_sigmas`).

    **Shape contract: send ``T`` values, not ``T+1``**. ``req.sigmas``
    is canonically ``T+1`` (terminal 0 included), but diffusers'
    ``set_timesteps(sigmas=...)`` at line 323 of
    ``scheduling_flow_match_euler_discrete.py`` takes ``len(sigmas)`` as
    ``num_inference_steps`` and at line 379 appends a terminal 0 itself.
    If we sent ``T+1``, the worker loop would run ``T+1`` iterations
    (one too many) and ``scheduler.sigmas`` would end up ``T+2``.
    Matches the SGLang adapter (``samplers/sglang/request.py:_to_sglang_kwargs``
    also slices ``[:-1]``).
    """
    if req.sigmas is None:
        return None
    require(
        int(req.sigmas.shape[0]) == num_inference_steps + 1,
        f"req.sigmas length {int(req.sigmas.shape[0])} != "
        f"num_inference_steps+1 ({num_inference_steps + 1}). Engine must "
        f"populate σ for the resolved num_inference_steps.",
    )
    return req.sigmas.detach().to(torch.float32).cpu().tolist()[:-1]


def _to_omni_sd35_t2i(
    req: RolloutReq,
    cfg: "VLLMOmniEngineConfig",
    sampling_params_cls: Any,
) -> Tuple[List[Any], List[Any]]:
    """SD3.5-medium single-stage builder.

    SD3.5 has no AR prelude — the diffusion stage owns the entire
    request. Per-prompt entries are the dict shape that
    ``StableDiffusion3Pipeline.forward`` accepts at
    ``pipeline_sd3.py:632-637`` (``{"prompt": text,
    "negative_prompt": ...}``). Sampling-params list is single-element:
    ``[dit_sampling]``.
    """
    stage_params = req.stage_params or {}
    if req.primitives.get("image") is not None:
        raise ValueError("modality='sd35_t2i' does not accept req.primitives['image']")

    texts = _texts_from_req(req)
    diff_params = stage_params.get("diffusion") or {}

    height = int(diff_params.get("height", cfg.default_height))
    width = int(diff_params.get("width", cfg.default_width))
    negative_prompt = str(diff_params.get("negative_prompt", "") or "")

    prompts: List[Any] = [{"prompt": text, "negative_prompt": negative_prompt} for text in texts.texts]

    num_inference_steps = int(diff_params.get("num_inference_steps", cfg.default_num_inference_steps))
    diff_kwargs: Dict[str, Any] = dict(
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=float(diff_params.get("guidance_scale", cfg.default_guidance_scale)),
        # HI3 upstream gates its use of req.sampling_params.guidance_scale on
        # this flag (vllm-omni/.../pipeline_hunyuan_image3.py:1326-1327);
        # without it the request's guidance_scale is silently ignored and
        # the forward fn default kicks in. SD3 upstream doesn't read this
        # field — setting it here is harmless on that path.
        guidance_scale_provided=True,
        eta=float(diff_params.get("eta", cfg.default_eta)),
        return_trajectory_latents=True,
        return_trajectory_decoded=False,
        num_outputs_per_prompt=1,
    )
    sigmas = _sigmas_list_from_req(req, num_inference_steps)
    if sigmas is not None:
        diff_kwargs["sigmas"] = sigmas
    max_seq_len = diff_params.get("max_sequence_length")
    if max_seq_len is not None:
        diff_kwargs["max_sequence_length"] = int(max_seq_len)
    seed = diff_params.get("seed")
    if seed is not None:
        diff_kwargs["seed"] = int(seed)

    # Pack sparse SDE step indices + the per-sample initial-noise tensor
    # through ``extra_args`` — vllm-omni routes this dict to the worker
    # subprocess as-is (preserving torch.Tensor values; see Flux Kontext
    # for an upstream example of tensor-bearing extra_args). Our
    # ``RLStableDiffusion3Pipeline.forward`` / ``prepare_latents`` read
    # them back out:
    #   - ``sde_indices`` installs the set on
    #     ``FlowMatchSDEDiscreteScheduler._sde_indices_set`` so only those
    #     steps run SDE (the rest degenerate to ODE).
    #   - ``initial_noise_batch`` is a single ``[B, C, H_lat, W_lat]``
    #     tensor; the pipeline's ``prepare_latents`` override slices by
    #     ``int(req.request_id.split('_', 1)[0])`` to pick this request's
    #     row. We source the tensor from ``RolloutReq.request_conditions``
    #     (CONCAT field — sliced correctly under multi-actor sharding;
    #     ``stage_params`` is SHARED and would broadcast the full-batch
    #     tensor to every shard).
    # When neither key is set we omit ``extra_args`` entirely.
    extra_args = dict(diff_kwargs.get("extra_args") or {})
    sde_indices = diff_params.get("sde_indices")
    if sde_indices is not None:
        extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})
    initial_latent_cond = (req.request_conditions or {}).get("initial_latents")
    if initial_latent_cond is not None:
        initial_noise = getattr(initial_latent_cond, "latents", None)
        if initial_noise is None:
            raise RuntimeError(
                "_to_omni_sd35_t2i: request_conditions['initial_latents'] "
                f"has no .latents tensor (got {type(initial_latent_cond).__name__})."
            )
        # Sanity-check batch dim aligns with this shard's prompt count.
        # Mismatch indicates an upstream slicing bug — fail fast here
        # instead of silently mis-slicing inside the worker.
        if int(initial_noise.shape[0]) != len(texts.texts):
            raise RuntimeError(
                f"_to_omni_sd35_t2i: initial_latents.shape[0]={int(initial_noise.shape[0])} "
                f"!= prompt count {len(texts.texts)} after sharding."
            )
        # Tensor stays on whatever device the caller left it (typically CPU);
        # the worker pipeline does the device move right before
        # ``prepare_latents`` returns.
        extra_args["initial_noise_batch"] = initial_noise
    if extra_args:
        diff_kwargs["extra_args"] = extra_args

    dit_sampling = sampling_params_cls(**diff_kwargs)
    return prompts, [dit_sampling]


def _to_omni_per_stage(
    req: RolloutReq,
    cfg: "VLLMOmniEngineConfig",
    *,
    modality: str,
    tokenizer: Any,
) -> Tuple[List[Any], List[Any]]:
    """Translate ``RolloutReq`` to ``(prompts, sampling_params_list)``.

    For HI3 image modalities (t2i/it2i), returns
    ``[ar_sampling, dit_sampling]``. For AR-only modalities (i2t/t2t),
    returns ``[ar_sampling]``. For single-stage diffusion modalities
    (sd35_t2i), returns ``[dit_sampling]`` only — no AR prelude.

    The prompts list is shared across all stages — each entry is a dict
    with the per-prompt fields the official end2end.py builds (see
    module docstring).
    """
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if modality == "sd35_t2i":
        return _to_omni_sd35_t2i(req, cfg, OmniDiffusionSamplingParams)

    from vllm import SamplingParams as VLLMSamplingParams
    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
    )

    stage_params = req.stage_params or {}
    task, sys_type, modalities_field = _resolve_task(modality, stage_params)

    texts = _texts_from_req(req)
    n = len(texts.texts)

    has_image_input = modality in ("it2i", "i2t")
    pil_images = _images_from_req(req, n) if has_image_input else []
    if has_image_input and not pil_images:
        raise ValueError(f"modality={modality!r} requires req.primitives['image']")
    if not has_image_input and req.primitives.get("image") is not None:
        raise ValueError(f"modality={modality!r} does not accept req.primitives['image']")

    diff_params = stage_params.get("diffusion") or {}
    ar_params = stage_params.get("ar") or {}

    height = int(diff_params.get("height", cfg.default_height))
    width = int(diff_params.get("width", cfg.default_width))

    prompts: List[Any] = []
    for i, text in enumerate(texts.texts):
        token_ids = build_prompt_tokens(text, tokenizer, task=task, sys_type=sys_type)
        entry: Dict[str, Any] = {
            "prompt_token_ids": token_ids,
            "prompt": text,
            "use_system_prompt": sys_type,
            "modalities": list(modalities_field),
        }
        if has_image_input:
            pil = pil_images[i]
            entry["multi_modal_data"] = {"image": pil}
            # Upstream HI3 reads height/width off the prompt dict for the
            # it2i path (matches end2end.py:185-187).
            if modality == "it2i":
                entry["height"] = pil.height
                entry["width"] = pil.width
            elif modality == "i2t":
                # Carry h/w for completeness even though i2t doesn't run
                # the DiT; harmless and matches end2end.py.
                entry["height"] = pil.height
                entry["width"] = pil.width
        elif modality in ("t2i", "t2i_think_recaption"):
            entry["height"] = height
            entry["width"] = width

        prompts.append(entry)

    # AR sampling — applies to every modality (Stage 0 is always AR).
    # ``logprobs=1`` makes vLLM emit per-token logp on the sampled token
    # (read by ``ar_capture.extract_ar_segment``).
    ar_sampling = VLLMSamplingParams(
        temperature=float(ar_params.get("temperature", cfg.default_ar_temperature)),
        top_p=float(ar_params.get("top_p", cfg.default_ar_top_p)),
        top_k=int(ar_params.get("top_k", cfg.default_ar_top_k)),
        max_tokens=int(ar_params.get("max_tokens", cfg.default_ar_max_tokens)),
        logprobs=1,
    )

    if modality in ("i2t", "t2t"):
        return prompts, [ar_sampling]

    # Image modalities — DiT sampling. ``eta`` is a typed first-class
    # field on OmniDiffusionSamplingParams (data.py:252); our
    # RLHunyuanImage3Pipeline.forward reads it directly off
    # req.sampling_params.eta for the scheduler swap.
    num_inference_steps = int(diff_params.get("num_inference_steps", cfg.default_num_inference_steps))
    diff_kwargs: Dict[str, Any] = dict(
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=float(diff_params.get("guidance_scale", cfg.default_guidance_scale)),
        # HI3 upstream gates its use of req.sampling_params.guidance_scale on
        # this flag (vllm-omni/.../pipeline_hunyuan_image3.py:1326-1327);
        # without it the request's guidance_scale is silently ignored.
        guidance_scale_provided=True,
        eta=float(diff_params.get("eta", cfg.default_eta)),
        return_trajectory_latents=True,
        return_trajectory_decoded=False,
        num_outputs_per_prompt=1,
    )
    sigmas = _sigmas_list_from_req(req, num_inference_steps)
    if sigmas is not None:
        diff_kwargs["sigmas"] = sigmas
    seed = diff_params.get("seed")
    if seed is not None:
        diff_kwargs["seed"] = int(seed)

    # See _to_omni_sd35_t2i for the rationale on this extra_args plumbing.
    extra_args = dict(diff_kwargs.get("extra_args") or {})
    sde_indices = diff_params.get("sde_indices")
    if sde_indices is not None:
        extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})

    # HI3 does NOT support driver-supplied initial latents today: the
    # latent shape on the DiT stage depends on the AR-emitted token count
    # (only known after stage 0 finishes), and ``RLHunyuanImage3Pipeline``
    # does not override ``prepare_latents`` to consume an injected x_T.
    # If we silently pass the tensor through, the worker would draw its
    # own noise via upstream RNG anyway — which is the exact "set on
    # rollout, ignored on rollout" silent-fallback class of bug. Fail
    # fast at the translator boundary instead.
    if (req.request_conditions or {}).get("initial_latents") is not None:
        raise NotImplementedError(
            f"_to_omni_per_stage: modality={modality!r} does not currently "
            f"consume request_conditions['initial_latents']. To enable "
            f"driver-side x_T injection on HI3, add a ``prepare_latents`` "
            f"override on RLHunyuanImage3Pipeline (mirroring the SD3 "
            f"override at rollout/engine/vllm_omni/sd3/pipeline.py) and "
            f"teach compute_initial_noise_for_request the HI3 latent shape."
        )

    if extra_args:
        diff_kwargs["extra_args"] = extra_args

    dit_sampling = OmniDiffusionSamplingParams(**diff_kwargs)

    return prompts, [ar_sampling, dit_sampling]


__all__ = ["_to_omni_per_stage"]
