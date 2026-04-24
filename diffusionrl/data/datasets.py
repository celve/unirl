"""
Dataset implementations for GRPO training.

The default user-facing data input contract is prompt-first:
- Text prompts plus optional metadata (TextPromptDataset)
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_PROMPT_EXAMPLE_EXCLUDED_KEYS = {
    "prompt",
    "caption",
    "metadata",
    "prompt_embed_path",
    "pooled_embed_path",
    "pooled_prompt_embeds_path",
    "text_ids_path",
    "text_ids",
    "prompt_embeds",
    "pooled_prompt_embeds",
    "prompt_id",
}

_LEGACY_EMBEDDING_FIELDS = {
    "prompt_embed_path",
    "pooled_embed_path",
    "pooled_prompt_embeds_path",
    "text_ids_path",
    "text_ids",
    "prompt_embeds",
    "pooled_prompt_embeds",
}


def normalize_prompt_example(
    item: Dict[str, Any],
    *,
    default_prompt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one raw dataset entry into prompt/metadata form."""
    if not isinstance(item, dict):
        raise TypeError(f"Prompt example must be a dict, got {type(item).__name__}.")

    prompt = item.get("prompt", item.get("caption", ""))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt example is missing a non-empty 'prompt' or 'caption' field.")

    metadata = item.get("metadata")
    if metadata is None:
        metadata = {key: value for key, value in item.items() if key not in _PROMPT_EXAMPLE_EXCLUDED_KEYS}
    elif not isinstance(metadata, dict):
        raise TypeError(f"Prompt example metadata must be a dict when provided, got {type(metadata).__name__}.")

    result: Dict[str, Any] = {"prompt": prompt}
    prompt_id = item.get("prompt_id")
    if prompt_id is None and default_prompt_id is not None:
        prompt_id = default_prompt_id
    if prompt_id is not None:
        result["prompt_id"] = str(prompt_id)
    if metadata:
        result["metadata"] = dict(metadata)
    return result


class PromptExampleDataset(Dataset):
    """
    Dataset that can expose prompt/metadata examples without loading training tensors.

    This is the framework-level interface used by evaluation prompt selection.
    """

    def get_prompt_example(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement get_prompt_example().")


class TextPromptDataset(PromptExampleDataset):
    """
    Simple dataset for text prompts.

    Supports:
    - JSON file with list of strings
    - JSON file with list of dicts containing 'prompt' or 'caption'
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
        self._source_prefix = os.path.basename(self.file_path) or "prompt_source"
        self._load_prompts()

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.Random(seed).shuffle(self.samples)
            else:
                random.shuffle(self.samples)

        logger.info(f"Loaded {len(self.samples)} prompts from {file_path}")

    def _load_prompts(self) -> None:
        """Load prompts from file."""

        def _append_item(item: Any, *, context: str) -> None:
            sample_idx = len(self.samples)
            default_prompt_id = f"{self._source_prefix}:{sample_idx}"
            if isinstance(item, str):
                candidate: Any = {"prompt": item}
            else:
                candidate = item
            if not isinstance(item, dict):
                if not isinstance(candidate, dict):
                    logger.warning("Skipping invalid %s: %r", context, item)
                    return
            candidate = dict(candidate)
            if self.prompt_key in candidate and self.prompt_key != "prompt":
                candidate["prompt"] = candidate.pop(self.prompt_key)
            legacy_fields = sorted(field for field in _LEGACY_EMBEDDING_FIELDS if field in candidate)
            if legacy_fields:
                raise ValueError(
                    "Prompt manifests must be prompt-first and may not include legacy embedding fields. "
                    f"Got fields={legacy_fields} in {self.file_path}."
                )

            try:
                normalized = normalize_prompt_example(
                    candidate,
                    default_prompt_id=default_prompt_id,
                )
                self.samples.append(normalized)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping invalid %s: %s", context, exc)

        if self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    _append_item(item, context="item")
            elif isinstance(data, dict):
                # Handle dict format with prompts key
                if "prompts" in data:
                    prompts = data["prompts"]
                    if isinstance(prompts, list):
                        for item in prompts:
                            _append_item(item, context="prompts item")
                    elif isinstance(prompts, str):
                        _append_item(prompts, context="prompts item")
                elif self.prompt_key in data:
                    prompt_val = data[self.prompt_key]
                    if isinstance(prompt_val, list):
                        for item in prompt_val:
                            if isinstance(item, dict):
                                _append_item(item, context="prompt list item")
                            else:
                                _append_item({self.prompt_key: item}, context="prompt list item")
                    elif isinstance(prompt_val, str):
                        _append_item(data, context="top-level prompt item")
                elif "caption" in data:
                    _append_item(data, context="top-level caption item")

        elif self.file_path.endswith(".jsonl"):
            # JSON Lines format: one JSON object per line
            with open(self.file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        _append_item(item, context=f"item at line {line_num}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        elif self.file_path.endswith(".txt"):
            with open(self.file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    prompt = line.strip()
                    if not prompt:
                        continue
                    _append_item(prompt, context=f"txt line {line_num}")

        else:
            raise ValueError(f"Unsupported file format: {self.file_path}. Supported formats: .json, .jsonl, .txt")

        if not self.samples:
            raise ValueError(f"No prompts found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_prompt_example(self, idx: int) -> Dict[str, Any]:
        return normalize_prompt_example(
            self.samples[idx],
            default_prompt_id=f"{self._source_prefix}:{idx}",
        )
