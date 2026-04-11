"""Shared sampling execution core for FSDP-based samplers.

This module owns the full sampling lifecycle: sampler creation, prompt
encoding, sample generation with optional adapter switching, latent
decoding, and output post-processing (metadata defaults, decode-for-reward,
transport optimisation, CPU offload).
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.types import RolloutRequest, RolloutSamples
from diffusionrl.utils import load_function
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.media import tensor_to_pil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler creation
# ---------------------------------------------------------------------------

def create_sampler(
    *,
    sampler_dotpath: str,
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
        sampler_dotpath: Fully-qualified dotpath to the sampler class.
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
    sampler_cls = load_function(sampler_dotpath)
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
        ``RolloutSamples`` from the sampler.
    """
    sampler_overrides = sample_kwargs.pop("sampler_overrides", None)
    with temporary_sampler_overrides(sampler, sampler_overrides):
        if sampling_adapter and model is not None:
            with switch_adapter(model, sampling_adapter):
                return sampler.sample(**sample_kwargs)
        return sampler.sample(**sample_kwargs)


@contextmanager
def temporary_sampler_overrides(
    sampler: Any,
    overrides: Optional[Dict[str, Any]],
):
    """Temporarily override mutable sampler attributes for one request."""
    if sampler is None or not isinstance(overrides, dict) or not overrides:
        yield
        return

    original_values: Dict[str, Any] = {}
    try:
        for raw_key, raw_value in overrides.items():
            key = str(raw_key).strip()
            if key not in {"eta", "sde_type"}:
                continue
            if not hasattr(sampler, key):
                continue
            original_values[key] = getattr(sampler, key)
            value = raw_value
            if key == "sde_type":
                value = normalize_sde_type(str(raw_value))
            else:
                value = float(raw_value)
            setattr(sampler, key, value)
        yield
    finally:
        for key, value in original_values.items():
            setattr(sampler, key, value)


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

    kwargs = dict(request.sampling.get("kwargs") or {})

    seed_raw = request.sampling.get("seed")
    base_seed = None if seed_raw is None else int(seed_raw)

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
    sampling_adapter = request.sampling.get("sampling_adapter")
    sde_indices_raw = request.sampling.get("sde_indices")
    sde_indices = None if sde_indices_raw is None else {int(v) for v in sde_indices_raw}

    encoded = encode_prompt(model_bundle, prompts)
    if encoded.get("prompt_embeds") is None:
        raise RuntimeError(f"{host_label} prompt encoder returned no prompt_embeds.")

    return run_sample(
        model=model,
        sampler=sampler,
        sampling_adapter=sampling_adapter,
        prompts=prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        latents=request.inputs.get("latents"),
        base_seed=base_seed,
        sde_indices=sde_indices,
        init_same_noise=bool(request.sampling.get("init_same_noise", False)),
        samples_per_prompt=max(1, int(request.sampling.get("samples_per_prompt", 1))),
        noise_group_ids=request.meta.get("noise_group_ids"),
        **encoded,
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


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------

def _attach_metadata_defaults(
    output: RolloutSamples,
    metadata_defaults: Optional[Dict[str, Any]],
) -> RolloutSamples:
    """Fill missing keys in ``aux['metadata']`` from *metadata_defaults*."""
    if not metadata_defaults:
        return output
    raw_metadata = output.aux.get("metadata")
    metadata = dict(raw_metadata or {})
    changed = False
    for key, value in metadata_defaults.items():
        if key not in metadata:
            metadata[key] = value
            changed = True
    if changed:
        output.aux["metadata"] = metadata
    return output


def _decode_for_reward_if_needed(
    output: RolloutSamples,
    request: RolloutRequest,
    decode_latents_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    host_label: str,
) -> RolloutSamples:
    """VAE-decode final latents when the request asks for reward-ready pixels."""
    if not bool(request.sampling.get("decode_for_reward", False)):
        return output
    raw_metadata = output.aux.get("metadata")
    has_decoded_videos = bool(
        isinstance(raw_metadata, dict) and torch.is_tensor(raw_metadata.get("decoded_videos"))
    )
    if has_decoded_videos or output.aux.get("decoded_images"):
        return output
    if decode_latents_fn is None:
        raise RuntimeError(
            f"decode_for_reward requested but {host_label} does not provide latent decoding."
        )
    try:
        decoded = decode_latents_fn(output.latents)
        decoded_images = tensor_to_pil(decoded)
    except Exception as exc:
        raise RuntimeError(
            f"decode_for_reward requested but {host_label} produced no decoded media "
            f"and latent decoding failed: {exc}"
        ) from exc
    return RolloutSamples(
        latents=output.latents,
        timesteps=output.timesteps,
        aux={**dict(output.aux), "decoded_images": list(decoded_images)},
        meta=output.meta,
    )


def finalize_sampling_output(
    *,
    output: RolloutSamples,
    request: RolloutRequest,
    host_label: str,
    decode_latents_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    metadata_defaults: Optional[Dict[str, Any]] = None,
    local_reward_attach_fn: Optional[Callable[[RolloutSamples], RolloutSamples]] = None,
    transport_optimize_fn: Optional[Callable[[RolloutSamples], RolloutSamples]] = None,
    move_output_to_cpu: bool = True,
) -> RolloutSamples:
    """Apply shared sampling-output post-processing after raw generation.

    Pipeline: metadata defaults → decode-for-reward → local reward →
    transport optimisation → move to CPU.
    """
    output = _attach_metadata_defaults(output, metadata_defaults)
    output = _decode_for_reward_if_needed(output, request, decode_latents_fn, host_label)
    if bool(request.sampling.get("decode_for_reward", False)) and local_reward_attach_fn is not None:
        output = local_reward_attach_fn(output)
    if transport_optimize_fn is not None:
        output = transport_optimize_fn(output)
    if move_output_to_cpu:
        output = output.to_device("cpu")
    return output


__all__ = [
    "create_sampler",
    "run_sample",
    "generate_prompt_only_rollout",
    "encode_prompt",
    "decode_latents",
    "iter_offloadable_modules",
    "finalize_sampling_output",
]
