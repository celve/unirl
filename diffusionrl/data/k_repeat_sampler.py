"""
K-Repeat Sampler for GRPO training.

Samples each prompt K times to generate multiple samples per prompt,
which is required for computing group-based advantages.
"""

import logging
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class KRepeatSampler(Sampler[int]):
    """
    Sampler that repeats each index K times for group-based sampling.

    This is used to generate K samples per prompt, which allows computing
    advantages within groups (per-prompt normalization).

    Example with K=4 and batch_size=8:
        - Each prompt appears 4 times consecutively
        - Batch contains 2 unique prompts, each repeated 4 times
        - This allows computing mean/std within each prompt group

    Args:
        data_source: Dataset to sample from
        k: Number of times to repeat each sample
        shuffle: Whether to shuffle the order of prompts
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_source: Sized,
        k: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize K-repeat sampler.

        Args:
            data_source: Dataset (must have __len__)
            k: Number of repeats per sample
            shuffle: Whether to shuffle prompt order
            seed: Random seed
        """
        self.data_source = data_source
        self.k = k
        self.shuffle = shuffle
        self.seed = seed

        self._num_samples = len(data_source)
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        logger.info(f"KRepeatSampler: {self._num_samples} samples x {k} repeats")

    def __iter__(self) -> Iterator[int]:
        """Yield indices with K repeats."""
        # Generate base indices
        if self.shuffle:
            indices = torch.randperm(
                self._num_samples,
                generator=self._generator
            ).tolist()
        else:
            indices = list(range(self._num_samples))

        # Repeat each index K times
        for idx in indices:
            for _ in range(self.k):
                yield idx

    def __len__(self) -> int:
        """Total number of samples (original * K)."""
        return self._num_samples * self.k

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for reproducible shuffling.

        Args:
            epoch: Current epoch number
        """
        self._generator.manual_seed(self.seed + epoch)
