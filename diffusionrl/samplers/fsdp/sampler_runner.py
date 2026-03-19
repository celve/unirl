"""Shared sampling execution core for FSDP-based samplers.

This module extracts the common logic used by training-actor direct sampling
and any future native in-process rollout hosts.

These paths ultimately do the same thing: create a sampler, call
sampler.sample() with optional adapter switching, encode prompts,
and decode latents. This module owns the shared core.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from diffusionrl.types import RolloutRequest
from diffusionrl.utils import load_function
from diffusionrl.utils.adapter_utils import switch_adapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler creation
# ---------------------------------------------------------------------------

def create_sampler(
    *,
    sampler_path: str,
    model: nn.Module,
    text_encoder: Any,
    vae: Any,
    eta: float = 1.0,
    sde_type: str = "flow",
    shift: float = 3.0,
    model_bundle: Any = None,
    **sampler_kwargs: Any,
) -> Any:
    """Instantiate a sampler from a dotpath, merging model_bundle extra kwargs.

    Args:
        sampler_path: Fully-qualified dotpath to the sampler class.
        model: Transformer / denoiser module.
        text_encoder: Text encoder (may be None for embedding-only mode).
        vae: VAE decoder (may be None if decoding is not needed).
        eta: SDE noise scale.
        sde_type: Transition rule ("flow", "cps", "dance", "dpm2").
        shift: Time-shift parameter.
        model_bundle: Optional ModelBundle that may provide extra kwargs.
        **sampler_kwargs: Forwarded to sampler constructor.

    Returns:
        A ``BaseSampler`` instance ready for ``sampler.sample()``.
    """
    sampler_cls = load_function(sampler_path)
    if model_bundle is not None and hasattr(model_bundle, "get_sampler_extra_kwargs"):
        extra_kwargs = model_bundle.get_sampler_extra_kwargs() or {}
        for key, value in extra_kwargs.items():
            sampler_kwargs.setdefault(key, value)
    return sampler_cls(
        model=model,
        text_encoder=text_encoder,
        vae=vae,
        eta=eta,
        sde_type=sde_type,
        shift=shift,
        **sampler_kwargs,
    )


# ---------------------------------------------------------------------------
# Core sampling call
# ---------------------------------------------------------------------------

def run_sample(
    *,
    model: nn.Module,
    sampler: Any,
    sampling_adapter: Optional[str] = None,
    **sample_kwargs: Any,
) -> Any:
    """Call ``sampler.sample()`` with optional LoRA adapter switching.

    Args:
        model: The model whose adapter may be switched.
        sampler: A ``BaseSampler`` instance.
        sampling_adapter: If set, temporarily switch to this adapter name
            before sampling (e.g. ``"old"`` for NFT).
        **sample_kwargs: Forwarded verbatim to ``sampler.sample()``.

    Returns:
        ``RolloutOutput`` from the sampler.
    """
    if sampling_adapter and model is not None:
        with switch_adapter(model, sampling_adapter):
            return sampler.sample(**sample_kwargs)
    return sampler.sample(**sample_kwargs)


def generate_prompt_only_rollout(
    *,
    host_label: str,
    request: RolloutRequest,
    model: nn.Module,
    sampler: Any,
    model_bundle: Any,
    device: torch.device,
) -> Any:
    """Run the shared prompt-only FSDP sampling flow for rollout/train actors."""
    prompts = request.prompts
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            f"{host_label} requires non-empty text prompts. "
            "Prompt-embedding-only input is not supported."
        )

    kwargs = dict(request.kwargs)
    unsupported_embedding_kwargs = [
        name
        for name in (
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "text_ids",
            "image_ids",
        )
        if kwargs.get(name) is not None
    ]
    if unsupported_embedding_kwargs:
        raise ValueError(
            f"{host_label} uses prompt-only RolloutRequest input. "
            f"Unsupported embedding kwargs: {unsupported_embedding_kwargs}."
        )

    generator = None
    if request.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(request.seed)

    if request.num_inference_steps is None:
        raise ValueError(f"{host_label} requires RolloutRequest.num_inference_steps to be resolved.")
    if request.guidance_scale is None:
        raise ValueError(f"{host_label} requires RolloutRequest.guidance_scale to be resolved.")
    if request.height is None or request.width is None or request.num_frames is None:
        raise ValueError(
            f"{host_label} requires RolloutRequest geometry to be resolved "
            f"(height={request.height}, width={request.width}, num_frames={request.num_frames})."
        )
    num_inference_steps = int(request.num_inference_steps)
    guidance_scale = float(request.guidance_scale)
    height = int(request.height)
    width = int(request.width)
    num_frames = int(request.num_frames)
    sampling_adapter = request.sampling_adapter

    encoded = encode_prompt(model_bundle, prompts)
    prompt_embeds = encoded.get("prompt_embeds")
    if prompt_embeds is None:
        raise RuntimeError(f"{host_label} prompt encoder returned no prompt_embeds.")

    return run_sample(
        model=model,
        sampler=sampler,
        sampling_adapter=sampling_adapter,
        prompts=prompts,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=encoded.get("pooled_prompt_embeds"),
        negative_prompt_embeds=encoded.get("negative_prompt_embeds"),
        negative_pooled_prompt_embeds=encoded.get("negative_pooled_prompt_embeds"),
        encoder_attention_mask=encoded.get("encoder_attention_mask"),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        latents=request.latents,
        generator=generator,
        sde_indices=request.sde_indices,
        text_ids=encoded.get("text_ids"),
        image_ids=encoded.get("image_ids"),
        init_same_noise=bool(request.init_same_noise),
        samples_per_prompt=int(request.samples_per_prompt),
        noise_group_ids=request.noise_group_ids,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------

def encode_prompt(
    model_bundle: Any,
    prompts: List[str],
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Encode text prompts to embeddings via *model_bundle*.

    Raises:
        RuntimeError: If model_bundle is None or lacks the encoding method.
    """
    if model_bundle is None:
        raise RuntimeError("Model bundle not loaded")
    if not hasattr(model_bundle, "encode_prompt_for_inference"):
        raise RuntimeError("Model bundle does not support inference prompt encoding")
    return model_bundle.encode_prompt_for_inference(prompts, **kwargs)


# ---------------------------------------------------------------------------
# Latent decoding
# ---------------------------------------------------------------------------

def decode_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode latent tensors to pixel space ``[0, 1]`` via VAE.

    Always uses float32 for VAE decoding (bfloat16 is unsupported by most
    VAE implementations).

    Raises:
        RuntimeError: If *vae* is None.
    """
    if vae is None:
        raise RuntimeError("VAE not available for decoding")
    with torch.no_grad():
        if hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
            scaling_factor = vae.config.scaling_factor
        else:
            scaling_factor = 0.18215
        latents_float = latents.to(dtype=torch.float32)
        decoded = vae.to(torch.float32).decode(latents_float / scaling_factor).sample
        return (decoded + 1) / 2  # [-1, 1] -> [0, 1]


# ---------------------------------------------------------------------------
# Module discovery (for offload / eval-context)
# ---------------------------------------------------------------------------

def iter_offloadable_modules(
    obj: Any,
    *,
    include_transformer: bool = True,
) -> List[Tuple[str, nn.Module]]:
    """Discover ``nn.Module`` attributes on *obj* that are likely offloadable.

    Scans ``obj.__dict__`` for attributes whose name matches well-known
    component names (transformer, text_encoder*, vae, image_encoder).

    Args:
        obj: Any object (model_bundle, sampler, …).
        include_transformer: If False, skip attributes containing
            ``"transformer"`` in their name.

    Returns:
        List of ``(attr_name, module)`` pairs.
    """
    if obj is None:
        return []
    known_names = {
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "vae",
        "image_encoder",
    }
    results: List[Tuple[str, nn.Module]] = []
    for name, value in obj.__dict__.items():
        if not isinstance(value, nn.Module):
            continue
        base_name = name.lstrip("_").lower()
        if not include_transformer and "transformer" in base_name:
            continue
        if base_name in known_names or any(
            token in base_name for token in ("encoder", "vae", "transformer")
        ):
            results.append((name, value))
    return results
