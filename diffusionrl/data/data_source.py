"""
Data source implementations for GRPO training.

The default data-source contract is prompt-only:
- prompts plus optional metadata for rollout/eval input

Runtime prompt embeddings are produced inside rollout engines and training
pipelines, not provided by the external dataset.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import DataLoader

from .datasets import PromptExampleDataset, TextPromptDataset, normalize_prompt_example

logger = logging.getLogger(__name__)


class ImageRLDataSource:
    """
    Prompt-only data source for image/video RL training.

    Accepted user-facing formats:
    - JSON/TXT/JSONL prompt datasets
    - Legacy JSON manifests with ``prompt`` or ``caption`` plus extra fields

    Legacy embedding-path fields are ignored at input time; runtime prompt
    embeddings are derived from text prompts inside the rollout engine.
    """

    def __init__(self, args):
        """
        Initialize data source from arguments.

        Args:
            args: TrainingArguments instance with:
                - data_path: Path to data file (JSON, JSONL, or TXT)
                - batch_size: Batch size
                - seed: Random seed
        """
        self.args = args
        self.data_path = getattr(args, 'data_path', None)
        self.eval_data_path = getattr(args, "eval_data_path", None)
        self.seed = getattr(args, 'seed', 42)
        self.prompts_per_batch = getattr(args.algorithm, "prompts_per_batch", None)
        if self.prompts_per_batch is None:
            raise ValueError("algorithm.prompts_per_batch must be set explicitly.")
        self.drop_last = True

        # Training data and eval data are treated as separate prompt sources.
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None  # Backward-compatible alias for the training dataset.
        self._dataloader = None
        self._iter: Optional[Iterator] = None
        self._eval_dataset_ready = False
        self._warned_legacy_embedding_paths: set[str] = set()

        if self.data_path is not None and os.path.exists(self.data_path):
            self._init_dataset()
        else:
            logger.warning(f"Data path not found: {self.data_path}. Using default prompts.")

    def _init_dataset(self) -> None:
        """Initialize the training dataset and dataloader."""
        self.train_dataset = self._build_dataset(self.data_path, shuffle=True)
        self.dataset = self.train_dataset
        logger.info(
            "Loaded prompt-only training dataset from %s (%d samples)",
            self.data_path,
            len(self.train_dataset),
        )
        self._create_dataloader()

    def _warn_if_legacy_embedding_fields_present(self, path: str) -> None:
        """Warn once when a legacy embedding manifest is used as prompt input."""
        if not isinstance(path, str) or not path.endswith(".json"):
            return
        if path in self._warned_legacy_embedding_paths:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return

        first_item = data[0] if isinstance(data, list) and data else None
        if not isinstance(first_item, dict):
            return
        legacy_embedding_keys = (
            "prompt_embed_path",
            "pooled_embed_path",
            "pooled_prompt_embeds_path",
            "text_ids_path",
            "prompt_embeds",
            "pooled_prompt_embeds",
        )
        if any(key in first_item for key in legacy_embedding_keys):
            logger.warning(
                "Prompt-only data source detected legacy embedding fields in %s. "
                "These fields are ignored; only prompt/caption text and metadata are used.",
                path,
            )
            self._warned_legacy_embedding_paths.add(path)

    def _build_dataset(self, path: str, *, shuffle: bool) -> PromptExampleDataset:
        """Build one prompt dataset instance for either training or evaluation."""
        dataset_seed = self.seed if shuffle else None
        self._warn_if_legacy_embedding_fields_present(path)

        return TextPromptDataset(
            file_path=path,
            seed=dataset_seed,
            shuffle=shuffle,
        )

    def _resolve_eval_path(self) -> Optional[str]:
        """Resolve which path should back evaluation prompt selection."""
        if self.eval_data_path:
            if not os.path.exists(self.eval_data_path):
                raise FileNotFoundError(
                    f"Evaluation data path not found: {self.eval_data_path}"
                )
            return self.eval_data_path
        return self.data_path

    def _ensure_eval_dataset(self) -> None:
        """Lazily build the evaluation dataset with deterministic ordering."""
        if self._eval_dataset_ready:
            return

        eval_path = self._resolve_eval_path()
        if eval_path is None or not os.path.exists(eval_path):
            self.eval_dataset = None
            self._eval_dataset_ready = True
            return

        self.eval_dataset = self._build_dataset(eval_path, shuffle=False)
        self._eval_dataset_ready = True

        source_label = "eval_data_path" if self.eval_data_path else "data_path"
        logger.info(
            "Loaded evaluation prompt source from %s=%s (%d examples, deterministic order)",
            source_label,
            eval_path,
            len(self.eval_dataset),
        )

    def _create_dataloader(self) -> None:
        """Create dataloader for prompt-batch sampling."""
        if self.train_dataset is None:
            return

        # prompts_per_batch determines the DataLoader batch size; do not repeat each prompt k times here
        sampler = None
        if len(self.train_dataset) < self.prompts_per_batch:
            raise ValueError(
                "Training dataset is smaller than prompts_per_batch, which would produce an "
                f"empty DataLoader with drop_last=True (num_samples={len(self.train_dataset)}, "
                f"prompts_per_batch={self.prompts_per_batch})."
            )

        self._dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.prompts_per_batch,
            sampler=sampler,
            shuffle=(sampler is None),  # Only shuffle if not using custom sampler
            num_workers=0,  # Keep simple for Ray
            collate_fn=self._collate_text,
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
        if self.train_dataset is not None:
            return len(self.train_dataset)
        return 0

    def _prompt_examples_to_batch(self, prompt_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert normalized prompt examples into a batch payload."""
        result: Dict[str, Any] = {
            "prompts": [item["prompt"] for item in prompt_examples],
        }
        metadata = [item.get("metadata") for item in prompt_examples]
        if any(m is not None for m in metadata):
            result["metadata"] = metadata
        return result

    def get_samples(self, batch_size: int) -> Dict[str, Any]:
        """
        Get next batch of samples.

        Args:
            batch_size: Requested batch size (uses internal batch_size)

        Returns:
            Dict containing prompt text plus optional metadata.
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

    def get_eval_samples(self, batch_size: int) -> Dict[str, Any]:
        """Get a stable eval batch from the dedicated evaluation prompt source."""
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return {"prompts": []}

        self._ensure_eval_dataset()
        if self.eval_dataset is None:
            return self._get_default_batch(batch_size)

        get_prompt_example = getattr(self.eval_dataset, "get_prompt_example", None)
        if not callable(get_prompt_example):
            raise TypeError(
                f"Evaluation dataset {type(self.eval_dataset).__name__} must implement "
                "get_prompt_example(idx) -> {'prompt': ..., 'metadata': ...}."
            )

        prompt_examples = [
            normalize_prompt_example(get_prompt_example(idx))
            for idx in range(min(batch_size, len(self.eval_dataset)))
        ]
        return self._prompt_examples_to_batch(prompt_examples)


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
        self.drop_last = False

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

    def get_eval_samples(self, batch_size: int) -> Dict[str, List[str]]:
        """Get a stable eval batch."""
        batch_size = max(0, int(batch_size))
        return {"prompts": self.prompts[:batch_size]}
