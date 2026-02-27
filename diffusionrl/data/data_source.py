"""
Data source implementations for GRPO training.

Provides a unified interface for loading both:
- Pre-computed embeddings (for image models)
- Plain text prompts (for all models)
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import DataLoader

from .datasets import (
    TextPromptDataset,
    create_rl_dataset,
    collate_embeddings,
)

logger = logging.getLogger(__name__)


class ImageRLDataSource:
    """
    Data source for image RL training.

    Automatically detects data format:
    - Pre-computed embeddings: JSON with prompt_embed_path fields
    - Plain text prompts: JSON/TXT with prompts

    Returns batches in format compatible with RolloutManager:
    - Embedding mode: {"prompt_embeds": Tensor, "pooled_prompt_embeds": Tensor,
                       "text_ids": Tensor (optional), "prompts": List[str]}
    - Text mode: {"prompts": List[str]}
    """

    def __init__(self, args):
        """
        Initialize data source from arguments.

        Args:
            args: TrainingArguments instance with:
                - data_path: Path to data file (JSON or TXT)
                - model_path: Model bundle class dotpath for embedding dataset builder
                - batch_size: Batch size
                - seed: Random seed
        """
        self.args = args
        self.data_path = getattr(args, 'data_path', None)
        self.model_type = getattr(args, 'model_type', 'flux')
        self.model_path = getattr(args, "model_path", "")
        self.batch_size = getattr(args, 'batch_size', 4)
        self.seed = getattr(args, 'seed', 42)
        self.prompts_per_batch = getattr(args, 'prompts_per_batch', 1)

        # Detect data format and create dataset
        self.dataset = None
        self.data_mode = "text"  # "text" or "embedding"
        self._dataloader = None
        self._iter: Optional[Iterator] = None

        if self.data_path is not None and os.path.exists(self.data_path):
            self._init_dataset()
        else:
            logger.warning(f"Data path not found: {self.data_path}. Using default prompts.")

    def _init_dataset(self) -> None:
        """Initialize dataset based on data format."""
        if self._is_embedding_dataset(self.data_path):
            # Pre-computed embeddings format
            self.dataset = create_rl_dataset(
                json_path=self.data_path,
                model_path=self.model_path,
                seed=self.seed,
            )
            self.data_mode = "embedding"
            logger.info(f"Loaded embedding dataset with {len(self.dataset)} samples")
        else:
            # Plain text prompts format
            self.dataset = TextPromptDataset(
                file_path=self.data_path,
                seed=self.seed,
            )
            self.data_mode = "text"
            logger.info(f"Loaded text prompt dataset with {len(self.dataset)} samples")

        self._create_dataloader()

    def _is_embedding_dataset(self, path: str) -> bool:
        """Check if the data file contains pre-computed embeddings."""
        if not path.endswith('.json'):
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    return "prompt_embed_path" in first_item or "prompt_embeds" in first_item
        except Exception as e:
            logger.warning(f"Error checking data format: {e}")

        return False

    def _create_dataloader(self) -> None:
        """Create dataloader for prompt-batch sampling."""
        if self.dataset is None:
            return

        # Prompts_per_batch 控制 DataLoader batch_size；不在此重复 k 次
        sampler = None
        batch_size = self.prompts_per_batch if self.prompts_per_batch > 0 else self.batch_size

        # Choose collate function based on data mode
        if self.data_mode == "embedding":
            collate_fn = collate_embeddings
        else:
            collate_fn = self._collate_text

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),  # Only shuffle if not using custom sampler
            num_workers=0,  # Keep simple for Ray
            collate_fn=collate_fn,
            drop_last=True,
        )

        self._iter = iter(self._dataloader)

    def _collate_text(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for text prompt dataset."""
        prompts = [item["prompt"] for item in batch]
        metadata = [item.get("metadata") for item in batch]
        result: Dict[str, Any] = {"prompts": prompts}
        if any(m is not None for m in metadata):
            result["metadata"] = metadata
        return result

    @property
    def num_samples(self) -> int:
        """Total number of samples in dataset."""
        if self.dataset is not None:
            return len(self.dataset)
        return 0

    def get_samples(self, batch_size: int) -> Dict[str, Any]:
        """
        Get next batch of samples.

        Args:
            batch_size: Requested batch size (uses internal batch_size)

        Returns:
            Dict containing either:
            - Embedding mode: prompt_embeds, pooled_prompt_embeds, text_ids (optional), prompts
            - Text mode: prompts (List[str])
        """
        if self._iter is None:
            # Fallback to default prompts
            return self._get_default_batch(batch_size)

        try:
            batch = next(self._iter)
        except StopIteration:
            # Reset iterator
            self._iter = iter(self._dataloader)
            batch = next(self._iter)

        return batch

    def _get_default_batch(self, batch_size: int) -> Dict[str, List[str]]:
        """Return default prompts for testing."""
        default_prompts = [
            "A beautiful sunset over the ocean with golden clouds",
            "A majestic cat sitting on a velvet cushion",
            "A futuristic city skyline at night with neon lights",
            "A serene mountain landscape with a crystal clear lake",
            "A vibrant field of wildflowers in spring",
            "An ancient forest with towering trees and mystical fog",
            "A cozy cabin in the woods during a snowfall",
            "A bustling marketplace in a medieval town",
        ]
        return {"prompts": default_prompts[:batch_size]}

    def get_eval_samples(self, batch_size: int) -> List[str]:
        """Get samples for evaluation (returns prompts only)."""
        batch = self.get_samples(batch_size)
        return batch.get("prompts", [])


class DefaultDataSource:
    """
    Default data source that returns simple prompts.

    Used when no data_path is specified or as fallback.
    """

    def __init__(self, args):
        """
        Initialize default data source.

        Args:
            args: TrainingArguments instance
        """
        self.args = args
        self.batch_size = getattr(args, 'batch_size', 4)

        # Default prompts for different scenarios
        self.prompts = [
            "A beautiful sunset over the ocean",
            "A cat playing with a ball of yarn",
            "A mountain landscape with snow",
            "A futuristic city at night",
            "A garden full of colorful flowers",
            "A cozy cabin in the woods",
            "An astronaut floating in space",
            "A tropical beach with palm trees",
        ]

        self._index = 0

    @property
    def num_samples(self) -> int:
        return len(self.prompts)

    def get_samples(self, batch_size: int) -> Dict[str, List[str]]:
        """Get next batch of prompts."""
        result = []
        for _ in range(batch_size):
            result.append(self.prompts[self._index % len(self.prompts)])
            self._index += 1
        return {"prompts": result}

    def get_eval_samples(self, batch_size: int) -> List[str]:
        """Get evaluation samples."""
        return self.prompts[:batch_size]
