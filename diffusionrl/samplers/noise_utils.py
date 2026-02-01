"""
Noise utilities for GRPO sampling.

This module provides utilities for generating initial noise in various patterns,
particularly for algorithms that benefit from specific noise initialization strategies.

Used by DanceGRPO and MixGRPO for init_same_noise feature.

Reference:
- DanceGRPO: Shared initial noise across K samples for same prompt
- MixGRPO: Same technique for reduced variance in advantage estimation
"""

from typing import Optional, Tuple, Union
import torch


def generate_shared_noise(
    batch_size: int,
    num_samples_per_prompt: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
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
        # For batch_size=8, num_samples_per_prompt=4:
        # - prompt_0: samples [0,1,2,3] share noise_0
        # - prompt_1: samples [4,5,6,7] share noise_1
        noise = generate_shared_noise(
            batch_size=8,
            num_samples_per_prompt=4,
            latent_shape=(16, 64, 64),  # channels, height, width
            device=device,
            dtype=dtype,
        )
        # noise.shape = [8, 16, 64, 64]
        # noise[0] == noise[1] == noise[2] == noise[3]  # Same prompt
        # noise[4] == noise[5] == noise[6] == noise[7]  # Same prompt

    Args:
        batch_size: Total number of samples in batch
        num_samples_per_prompt: Number of samples generated per prompt
        latent_shape: Shape of a single latent (C, H, W) or (C, T, H, W) for video
        device: Device for the tensor
        dtype: Data type for the tensor
        generator: Optional random generator for reproducibility

    Returns:
        Noise tensor [batch_size, *latent_shape] with shared noise per prompt group
    """
    if num_samples_per_prompt <= 1:
        # No sharing needed, generate independent noise
        return torch.randn(
            batch_size,
            *latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    # Calculate number of unique prompts
    num_unique_prompts = batch_size // num_samples_per_prompt

    if batch_size % num_samples_per_prompt != 0:
        # Handle incomplete groups - generate extra noise for remaining samples
        remaining = batch_size % num_samples_per_prompt
    else:
        remaining = 0

    # Generate noise for unique prompts only
    base_noise = torch.randn(
        num_unique_prompts,
        *latent_shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    # Repeat each noise sample num_samples_per_prompt times
    # [num_unique, C, H, W] -> [num_unique * K, C, H, W]
    shared_noise = base_noise.repeat_interleave(num_samples_per_prompt, dim=0)

    # Handle remaining samples if batch doesn't divide evenly
    if remaining > 0:
        extra_noise = torch.randn(
            remaining,
            *latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        shared_noise = torch.cat([shared_noise, extra_noise], dim=0)

    return shared_noise


def generate_latents(
    batch_size: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    init_same_noise: bool = False,
    num_samples_per_prompt: int = 1,
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
        num_samples_per_prompt: Number of samples per prompt (used if init_same_noise=True)

    Returns:
        Latent tensor [batch_size, *latent_shape]
    """
    if init_same_noise and num_samples_per_prompt > 1:
        return generate_shared_noise(
            batch_size=batch_size,
            num_samples_per_prompt=num_samples_per_prompt,
            latent_shape=latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        return torch.randn(
            batch_size,
            *latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )


def expand_noise_for_cfg(
    noise: torch.Tensor,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Expand noise tensor for classifier-free guidance.

    When using CFG, we need to duplicate the noise for unconditional branch.

    Args:
        noise: Initial noise tensor [B, C, H, W]
        do_classifier_free_guidance: Whether to expand for CFG

    Returns:
        Expanded noise [2B, C, H, W] if CFG, else [B, C, H, W]
    """
    if do_classifier_free_guidance:
        return torch.cat([noise] * 2, dim=0)
    return noise
