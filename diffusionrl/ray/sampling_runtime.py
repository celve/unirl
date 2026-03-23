"""Shared post-processing helpers for sampling outputs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from diffusionrl.types.sampling import RolloutOutput, RolloutRequest
from diffusionrl.utils.media import tensor_to_pil


def _with_decoded_images(
    *,
    output: RolloutOutput,
    decoded_images: list[Any],
) -> RolloutOutput:
    return RolloutOutput(
        latents=output.latents,
        timesteps=output.timesteps,
        trajectories=output.trajectories,
        log_probs=output.log_probs,
        embeddings=output.embeddings,
        decoded_images=decoded_images,
        metadata=output.metadata,
        step_indices=output.step_indices,
    )


def _attach_metadata_defaults(
    *,
    output: RolloutOutput,
    metadata_defaults: Optional[Dict[str, Any]],
) -> RolloutOutput:
    if not metadata_defaults:
        return output
    metadata = dict(output.metadata or {})
    changed = False
    for key, value in metadata_defaults.items():
        if key not in metadata:
            metadata[key] = value
            changed = True
    if not changed:
        return output
    output.metadata = metadata
    return output


def _decode_for_reward_if_needed(
    *,
    output: RolloutOutput,
    request: RolloutRequest,
    decode_latents_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    host_label: str,
) -> RolloutOutput:
    if not request.decode_for_reward:
        return output
    has_decoded_videos = bool(
        isinstance(output.metadata, dict) and torch.is_tensor(output.metadata.get("decoded_videos"))
    )
    if output.has_decoded_images or has_decoded_videos:
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
    output: RolloutOutput,
    request: RolloutRequest,
    host_label: str,
    decode_latents_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    metadata_defaults: Optional[Dict[str, Any]] = None,
    local_reward_attach_fn: Optional[Callable[[RolloutOutput], RolloutOutput]] = None,
    transport_optimize_fn: Optional[Callable[[RolloutOutput], RolloutOutput]] = None,
    move_output_to_cpu: bool = True,
) -> RolloutOutput:
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
    if request.decode_for_reward and local_reward_attach_fn is not None:
        output = local_reward_attach_fn(output)
    if transport_optimize_fn is not None:
        output = transport_optimize_fn(output)
    if move_output_to_cpu:
        output = output.to_device("cpu")
    return output


__all__ = ["finalize_sampling_output"]
