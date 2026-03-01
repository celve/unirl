"""Shared sampling execution core for FSDP-based samplers.

This module extracts the common logic used by both:
- ActorSamplingExecutor (scenario a: training actor direct sampling)
- FSDPRolloutEngine (scenario d: standalone FSDP rollout actor)

Both paths ultimately do the same thing: create a sampler, call
sampler.sample() with optional adapter switching, encode prompts,
and decode latents.  The difference is *where the model comes from*
(training actor vs standalone engine) and *lifecycle management*
(eval context vs sleep/wake_up).  This module owns the shared core.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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
    sde_type: str = "sde",
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
        sde_type: SDE variant ("sde", "cps", "dance", …).
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
