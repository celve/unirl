"""Shared post-processing helpers for sampling outputs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from diffusionrl.types.sampling import RolloutSamples, RolloutRequest
from diffusionrl.utils.media import tensor_to_pil


def _with_decoded_images(
    *,
    output: RolloutSamples,
    decoded_images: list[Any],
) -> RolloutSamples:
    return RolloutSamples(
        latents=output.latents,
        timesteps=output.timesteps,
        aux={
            **dict(output.aux),
            "decoded_images": list(decoded_images),
        },
        meta=output.meta,
    )


def _attach_metadata_defaults(
    *,
    output: RolloutSamples,
    metadata_defaults: Optional[Dict[str, Any]],
) -> RolloutSamples:
    if not metadata_defaults:
        return output
    raw_metadata = output.aux.get("metadata")
    metadata = dict(raw_metadata or {})
    changed = False
    for key, value in metadata_defaults.items():
        if key not in metadata:
            metadata[key] = value
            changed = True
    if not changed:
        return output
    output.aux["metadata"] = metadata
    return output


def _decode_for_reward_if_needed(
    *,
    output: RolloutSamples,
    request: RolloutRequest,
    decode_latents_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    host_label: str,
) -> RolloutSamples:
    if not bool(request.sampling.get("decode_for_reward", False)):
        return output
    raw_metadata = output.aux.get("metadata")
    has_decoded_videos = bool(
        isinstance(raw_metadata, dict) and torch.is_tensor(raw_metadata.get("decoded_videos"))
    )
    has_decoded_images = bool(output.aux.get("decoded_images"))
    if has_decoded_images or has_decoded_videos:
        return output
    if decode_latents_fn is None:
        raise RuntimeError(
            f"decode_for_reward requested but {host_label} does not provide latent decoding."
        )
    try:
        decoded = decode_latents_fn(output.latents)
        decoded_images = tensor_to_pil(decoded)
        return _with_decoded_images(output=output, decoded_images=decoded_images)
    except Exception as exc:
        raise RuntimeError(
            f"decode_for_reward requested but {host_label} produced no decoded media "
            f"and latent decoding failed: {exc}"
        ) from exc


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
    """Apply shared sampling-output post-processing after raw generation."""
    output = _attach_metadata_defaults(
        output=output,
        metadata_defaults=metadata_defaults,
    )
    output = _decode_for_reward_if_needed(
        output=output,
        request=request,
        decode_latents_fn=decode_latents_fn,
        host_label=host_label,
    )
    if bool(request.sampling.get("decode_for_reward", False)) and local_reward_attach_fn is not None:
        output = local_reward_attach_fn(output)
    if transport_optimize_fn is not None:
        output = transport_optimize_fn(output)
    if move_output_to_cpu:
        output = output.to_device("cpu")
    return output


__all__ = ["finalize_sampling_output"]
