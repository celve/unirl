"""
Data loading utilities for GRPO training.

Provides data sources and datasets for both text prompts and pre-computed embeddings.
"""

from .data_source import ImageRLDataSource, DefaultDataSource
from .datasets import (
    EmbeddingRLDataset,
    DATASET_REGISTRY,
    register_dataset,
    TextPromptDataset,
    create_rl_dataset,
)
from .k_repeat_sampler import KRepeatSampler

__all__ = [
    # Data sources
    "ImageRLDataSource",
    "DefaultDataSource",
    # Datasets
    "EmbeddingRLDataset",
    "DATASET_REGISTRY",
    "register_dataset",
    "TextPromptDataset",
    "create_rl_dataset",
    # Samplers
    "KRepeatSampler",
]
