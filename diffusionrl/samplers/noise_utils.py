"""
Noise utilities for GRPO sampling.

This module provides utilities for generating initial noise in various patterns,
particularly for algorithms that benefit from specific noise initialization strategies.

Used by DanceGRPO and MixGRPO for init_same_noise feature.

Reference:
- DanceGRPO: Shared initial noise across K samples for same prompt
- MixGRPO: Same technique for reduced variance in advantage estimation
"""

from typing import Dict, List, Optional, Tuple, Union
import torch


def generate_shared_noise(
    batch_size: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    noise_group_ids: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Generate initial noise where samples from the same prompt share the same noise.

    This is the core implementation for init_same_noise feature used by
    DanceGRPO and MixGRPO. When K samples are generated for each prompt,
    all K samples start from the same initial noise, which:
    1. Reduces variance in advantage estimation
    2. Makes differences between samples purely due to stochastic sampling path
    3. Improves training stability

    Example:
        # For batch_size=8, samples_per_prompt=4:
        # - prompt_0: samples [0,1,2,3] share noise_0
        # - prompt_1: samples [4,5,6,7] share noise_1
        noise = generate_shared_noise(
            batch_size=8,
            samples_per_prompt=4,
            latent_shape=(16, 64, 64),  # channels, height, width
            device=device,
            dtype=dtype,
        )
        # noise.shape = [8, 16, 64, 64]
        # noise[0] == noise[1] == noise[2] == noise[3]  # Same prompt
        # noise[4] == noise[5] == noise[6] == noise[7]  # Same prompt

    Args:
        batch_size: Total number of samples in batch
        latent_shape: Shape of a single latent (C, H, W) or (C, T, H, W) for video
        device: Device for the tensor
        dtype: Data type for the tensor
        generator: Optional random generator for reproducibility
        noise_group_ids: Explicit per-sample noise sharing groups aligned to the batch

    Returns:
        Noise tensor [batch_size, *latent_shape] with shared noise per explicit group
    """
    if not isinstance(noise_group_ids, list) or len(noise_group_ids) != batch_size:
        raise ValueError(
            "generate_shared_noise requires explicit noise_group_ids aligned to batch_size. "
            f"Got batch_size={batch_size}, noise_group_ids_len="
            f"{len(noise_group_ids) if isinstance(noise_group_ids, list) else None}."
        )

    group_noise: Dict[str, torch.Tensor] = {}
    chunks: List[torch.Tensor] = []
    for raw_group_id in noise_group_ids:
        group_id = str(raw_group_id)
        noise = group_noise.get(group_id)
        if noise is None:
            noise = torch.randn(
                *latent_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            group_noise[group_id] = noise
        chunks.append(noise)
    return torch.stack(chunks, dim=0)


def generate_latents(
    batch_size: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    init_same_noise: bool = False,
    samples_per_prompt: int = 1,
    noise_group_ids: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    High-level function for generating initial latents with optional noise sharing.

    This is the recommended interface for samplers to use when initializing latents.
    It automatically handles the init_same_noise logic.

    Args:
        batch_size: Total number of samples
        latent_shape: Shape of a single latent (C, H, W) or (C, T, H, W)
        device: Device for the tensor
        dtype: Data type for the tensor
        generator: Optional random generator
        init_same_noise: Whether to share noise across samples for same prompt
        samples_per_prompt: Rollout geometry hint kept for sampler API compatibility

    Returns:
        Latent tensor [batch_size, *latent_shape]
    """
    if init_same_noise:
        return generate_shared_noise(
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            noise_group_ids=noise_group_ids,
        )
    return torch.randn(
        batch_size,
        *latent_shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
