"""
Dataset implementations for GRPO training.

Supports:
- Pre-computed embeddings (EmbeddingRLDataset)
- Plain text prompts (TextPromptDataset)

Reference:
- unified_grpo/data/datasets.py
- flow_grpo data loading
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


class TextPromptDataset(Dataset):
    """
    Simple dataset for text prompts.

    Supports:
    - JSON file with list of strings
    - JSON file with list of dicts containing 'prompt' key
    - JSONL file with one JSON object per line (JSON Lines format)
    - TXT file with one prompt per line

    Example JSON formats:
        ["prompt 1", "prompt 2", ...]
        [{"prompt": "prompt 1"}, {"prompt": "prompt 2"}, ...]

    Example JSONL format:
        {"prompt": "prompt 1"}
        {"prompt": "prompt 2"}
    """

    def __init__(
        self,
        file_path: str,
        prompt_key: str = "prompt",
        seed: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Initialize text prompt dataset.

        Args:
            file_path: Path to JSON or TXT file containing prompts
            prompt_key: Key for prompt in JSON dicts
            seed: Random seed for shuffling
            shuffle: Whether to shuffle prompts on load
        """
        self.file_path = file_path
        self.prompt_key = prompt_key
        self.samples: List[Dict[str, Any]] = []

        # Load prompts
        self._load_prompts()

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.samples)

        logger.info(f"Loaded {len(self.samples)} prompts from {file_path}")

    def _load_prompts(self) -> None:
        """Load prompts from file."""
        if self.file_path.endswith('.json'):
            with open(self.file_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        self.samples.append({"prompt": item})
                    elif isinstance(item, dict) and self.prompt_key in item:
                        metadata = {k: v for k, v in item.items() if k != self.prompt_key}
                        sample: Dict[str, Any] = {"prompt": item[self.prompt_key]}
                        if metadata:
                            sample["metadata"] = metadata
                        self.samples.append(sample)
                    else:
                        logger.warning(f"Skipping invalid item: {item}")
            elif isinstance(data, dict):
                # Handle dict format with prompts key
                if "prompts" in data:
                    prompts = data["prompts"]
                    if isinstance(prompts, list):
                        for item in prompts:
                            if isinstance(item, str):
                                self.samples.append({"prompt": item})
                            elif isinstance(item, dict) and self.prompt_key in item:
                                metadata = {k: v for k, v in item.items() if k != self.prompt_key}
                                sample: Dict[str, Any] = {"prompt": item[self.prompt_key]}
                                if metadata:
                                    sample["metadata"] = metadata
                                self.samples.append(sample)
                    elif isinstance(prompts, str):
                        self.samples.append({"prompt": prompts})
                elif self.prompt_key in data:
                    prompt_val = data[self.prompt_key]
                    if isinstance(prompt_val, list):
                        self.samples.extend({"prompt": p} for p in prompt_val)
                    elif isinstance(prompt_val, str):
                        self.samples.append({"prompt": prompt_val})

        elif self.file_path.endswith('.jsonl'):
            # JSON Lines format: one JSON object per line
            with open(self.file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, str):
                            self.samples.append({"prompt": item})
                        elif isinstance(item, dict) and self.prompt_key in item:
                            metadata = {k: v for k, v in item.items() if k != self.prompt_key}
                            sample = {"prompt": item[self.prompt_key]}
                            if metadata:
                                sample["metadata"] = metadata
                            self.samples.append(sample)
                        else:
                            logger.warning(f"Skipping invalid item at line {line_num}: {item}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        elif self.file_path.endswith('.txt'):
            with open(self.file_path, 'r') as f:
                self.samples = [{"prompt": line.strip()} for line in f if line.strip()]

        else:
            raise ValueError(f"Unsupported file format: {self.file_path}. Supported formats: .json, .jsonl, .txt")

        if not self.samples:
            raise ValueError(f"No prompts found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class BaseRLDataset(Dataset):
    """
    Base class for RL datasets with pre-computed embeddings.

    The dataset stores:
    - Pre-computed prompt embeddings (from text encoder)
    - Original captions (for reward computation and logging)
    """

    def __init__(
        self,
        json_path: str,
        embedding_dir: Optional[str] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Initialize base RL dataset.

        Args:
            json_path: Path to JSON metadata file
            embedding_dir: Directory containing embedding files (if not in JSON)
            seed: Random seed for shuffling
            shuffle: Whether to shuffle data on load
        """
        self.json_path = json_path
        self.embedding_dir = embedding_dir or os.path.dirname(json_path)
        self.data: List[Dict[str, Any]] = []

        # Load metadata
        self._load_metadata()

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.data)

    def _load_metadata(self) -> None:
        """Load metadata from JSON file."""
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        if not isinstance(self.data, list):
            raise ValueError(f"Expected list in JSON file, got {type(self.data)}")

        logger.info(f"Loaded {len(self.data)} samples from {self.json_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement __getitem__")


class EmbeddingRLDataset(BaseRLDataset):
    """
    Unified dataset for embedding-based RL training.

    Expected JSON format:
    [
        {
            "caption": "A beautiful sunset",
            "prompt_embed_path": "embeddings/0001_prompt.pt",
            "pooled_embed_path": "embeddings/0001_pooled.pt"
        },
        ...
    ]

    Each embedding file contains a torch tensor:
    - prompt_embeds: [seq_len, hidden_dim]
    - pooled_embeds: [hidden_dim]
    - text_ids: [seq_len, 3] (optional, loaded only when enabled)
    """

    def __init__(
        self,
        json_path: str,
        embedding_dir: Optional[str] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        load_text_ids: bool = False,
    ):
        self.load_text_ids = load_text_ids
        super().__init__(json_path=json_path, embedding_dir=embedding_dir, seed=seed, shuffle=shuffle)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load embeddings
        prompt_embed_key = "prompt_embed_path"
        if prompt_embed_key not in item:
            raise KeyError(f"Missing '{prompt_embed_key}' in dataset item")

        pooled_key = "pooled_embed_path"
        if pooled_key not in item:
            # Accept alternate pooled embedding field from existing datasets.
            pooled_key = "pooled_prompt_embeds_path" if "pooled_prompt_embeds_path" in item else pooled_key

        prompt_embed_path = os.path.join(self.embedding_dir, item[prompt_embed_key])
        pooled_embed_path = os.path.join(self.embedding_dir, item[pooled_key])

        prompt_embeds = torch.load(prompt_embed_path, map_location="cpu")
        pooled_prompt_embeds = torch.load(pooled_embed_path, map_location="cpu")

        result = {
            "prompt": item.get("caption", item.get("prompt", "")),
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        metadata = {
            k: v
            for k, v in item.items()
            if k
            not in {
                "caption",
                "prompt",
                "prompt_embed_path",
                "pooled_embed_path",
                "pooled_prompt_embeds_path",
                "text_ids_path",
                "text_ids",
            }
        }
        if metadata:
            result["metadata"] = metadata

        if self.load_text_ids:
            # Load text_ids if available (path or inline)
            if "text_ids_path" in item:
                text_ids_path = os.path.join(self.embedding_dir, item["text_ids_path"])
                result["text_ids"] = torch.load(text_ids_path, map_location="cpu")
            elif "text_ids" in item:
                text_ids_val = item["text_ids"]
                if isinstance(text_ids_val, str):
                    text_ids_path = os.path.join(self.embedding_dir, text_ids_val)
                    result["text_ids"] = torch.load(text_ids_path, map_location="cpu")
                else:
                    result["text_ids"] = text_ids_val

        return result


def create_rl_dataset(
    json_path: str,
    model_path: str,
    **kwargs,
) -> BaseRLDataset:
    """
    Build embedding dataset using model class self-description.

    Args:
        json_path: Path to JSON metadata file
        model_path: Model bundle class dotpath
        **kwargs: Additional arguments passed to dataset

    Returns:
        Dataset instance returned by model_cls.create_embedding_dataset()
    """
    model_cls = load_function(model_path)
    build_dataset = getattr(model_cls, "create_embedding_dataset", None)
    if not callable(build_dataset):
        raise ValueError(
            f"Model class {model_path} must provide classmethod create_embedding_dataset()."
        )
    dataset = build_dataset(json_path=json_path, **kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"{model_path}.create_embedding_dataset() must return torch.utils.data.Dataset, "
            f"got {type(dataset)!r}."
        )
    return dataset


def collate_embeddings(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for embedding datasets.

    Args:
        batch: List of dataset items

    Returns:
        Batched dictionary with stacked tensors
    """
    result = {
        "prompts": [item["prompt"] for item in batch],
    }

    metadata = [item.get("metadata") for item in batch]
    if any(m is not None for m in metadata):
        result["metadata"] = metadata

    # Stack tensors
    if "prompt_embeds" in batch[0]:
        result["prompt_embeds"] = torch.stack([item["prompt_embeds"] for item in batch])

    if "pooled_prompt_embeds" in batch[0]:
        result["pooled_prompt_embeds"] = torch.stack([item["pooled_prompt_embeds"] for item in batch])

    if "text_ids" in batch[0] and batch[0]["text_ids"] is not None:
        result["text_ids"] = torch.stack([item["text_ids"] for item in batch])

    return result
