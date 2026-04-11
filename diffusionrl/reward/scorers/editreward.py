"""EditReward scorer for instruction-guided image editing."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image

from diffusionrl.types.reward import RewardRequest, RewardType

from .base_local import BaseLocalRewardScorer


class EditRewardScorer(BaseLocalRewardScorer):
    """EditReward scorer that evaluates edited images against source images.

    This scorer expects per-sample metadata to include a source image reference.
    Accepted metadata keys include ``source_image_path``, ``source_image``,
    ``image_src``, ``input_image_path``, and related aliases.
    """

    canonical_model_name = "editreward"

    _SOURCE_IMAGE_KEYS = (
        "source_image_path",
        "source_image",
        "source_path",
        "image_src",
        "src_image_path",
        "src_image",
        "input_image_path",
        "input_image",
        "original_image_path",
        "original_image",
        "condition_image_path",
        "condition_image",
        "conditioning_image",
        "init_image",
        "source",
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        timeout: float = 60.0,
        *,
        reward_model_ckpt_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        reward_dim: Optional[str] = None,
        rm_head_type: Optional[str] = None,
        source_image_key: Optional[str] = None,
        editreward_python_path: Optional[str] = None,
        **model_kwargs,
    ) -> None:
        self.reward_model_ckpt_path = (
            str(reward_model_ckpt_path or checkpoint_path or os.getenv("EDITREWARD_CHECKPOINT_PATH", "")).strip()
        )
        self.config_path = str(config_path or os.getenv("EDITREWARD_CONFIG_PATH", "")).strip() or None
        self.reward_dim = str(reward_dim or os.getenv("EDITREWARD_REWARD_DIM", "overall_detail")).strip()
        self.rm_head_type = str(rm_head_type or os.getenv("EDITREWARD_RM_HEAD_TYPE", "ranknet_multi_head")).strip()
        self.source_image_key = str(source_image_key or os.getenv("EDITREWARD_SOURCE_IMAGE_KEY", "")).strip() or None
        self.editreward_python_path = str(
            editreward_python_path or os.getenv("EDITREWARD_PYTHON_PATH", "")
        ).strip() or None
        self._editreward_inferencer = None
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            timeout=timeout,
            **model_kwargs,
        )

    @classmethod
    def _candidate_editreward_paths(cls, explicit_path: Optional[str] = None) -> Iterable[Path]:
        if explicit_path:
            yield Path(explicit_path)

        repo_root = Path(__file__).resolve().parents[3]
        workspace_root = repo_root.parents[1] if len(repo_root.parents) >= 2 else repo_root.parent

        yield repo_root / "third_party" / "EditReward"
        yield workspace_root / "My_Code" / "EditReward"

    def _import_editreward(self):
        try:
            module = importlib.import_module("EditReward")
            inferencer_cls = getattr(module, "EditRewardInferencer")
            return inferencer_cls
        except Exception as first_error:
            for candidate in self._candidate_editreward_paths(self.editreward_python_path):
                if not candidate.is_dir():
                    continue
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                try:
                    module = importlib.import_module("EditReward")
                    inferencer_cls = getattr(module, "EditRewardInferencer")
                    return inferencer_cls
                except Exception:
                    continue
            raise ImportError(
                "EditReward is required for the 'editreward' scorer. "
                "Install the package or set EDITREWARD_PYTHON_PATH to a local checkout."
            ) from first_error

    def _load_model(self) -> None:
        if not self.reward_model_ckpt_path:
            raise ValueError(
                "EditReward requires reward_model_ckpt_path (or EDITREWARD_CHECKPOINT_PATH) "
                "to point at a checkpoint directory."
            )

        inferencer_cls = self._import_editreward()
        self._editreward_inferencer = inferencer_cls(
            config_path=self.config_path,
            checkpoint_path=self.reward_model_ckpt_path,
            device=self.device,
            reward_dim=self.reward_dim,
            rm_head_type=self.rm_head_type,
        )
        self.model = self._editreward_inferencer.model
        self.processor = getattr(self._editreward_inferencer, "processor", None)
        self.reward_types = [RewardType.CUSTOM]

    def _normalize_image_input(self, image: Any) -> Any:
        if isinstance(image, (str, os.PathLike)):
            return os.fspath(image)
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            return image
        return Image.fromarray(image).convert("RGB")

    def _extract_source_image(self, sample_metadata: Optional[Dict[str, Any]], index: int) -> Any:
        if not isinstance(sample_metadata, dict):
            raise ValueError(
                "EditReward requires sample metadata with a source image reference for every sample. "
                f"Missing metadata at index={index}."
            )

        candidate_keys = []
        if self.source_image_key:
            candidate_keys.append(self.source_image_key)
        candidate_keys.extend(key for key in self._SOURCE_IMAGE_KEYS if key not in candidate_keys)

        for key in candidate_keys:
            value = sample_metadata.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                for nested_key in ("path", "image", "value", "uri", "url"):
                    nested_value = value.get(nested_key)
                    if nested_value is not None:
                        value = nested_value
                        break
            return self._normalize_image_input(value)

        available_keys = ", ".join(sorted(str(key) for key in sample_metadata.keys()))
        raise ValueError(
            "EditReward could not find a source image in sample metadata. "
            f"Checked keys={candidate_keys}. Available keys=[{available_keys}] at index={index}."
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        if request.images is None:
            raise ValueError("EditReward requires request.images for edited-image scoring.")

        images = [self._normalize_image_input(image) for image in request.images]
        prompts = list(request.prompts)

        if len(images) != len(prompts):
            raise ValueError(
                "EditReward requires images and prompts to have the same length. "
                f"Got images={len(images)} prompts={len(prompts)}."
            )

        metadata = request.metadata
        if not isinstance(metadata, list) or len(metadata) != len(prompts):
            raise ValueError(
                "EditReward requires sample-aligned metadata entries containing source images. "
                f"Got metadata_len={len(metadata) if isinstance(metadata, list) else None}, "
                f"prompts={len(prompts)}."
            )

        source_images = [
            self._extract_source_image(sample_metadata, index=i)
            for i, sample_metadata in enumerate(metadata)
        ]

        all_rewards: List[float] = []
        for i in range(0, len(images), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_source_images = source_images[i : i + self.batch_size]
            batch_images = images[i : i + self.batch_size]

            with torch.no_grad():
                scores = self._editreward_inferencer.reward(
                    prompts=batch_prompts,
                    image_src=batch_source_images,
                    image_paths=batch_images,
                )

            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores)
            scores = scores.detach()
            if scores.ndim == 0:
                scores = scores.unsqueeze(0)
            if scores.ndim >= 2:
                scores = scores[:, 0]
            all_rewards.extend(scores.float().cpu().tolist())

        return [float(score) for score in all_rewards]
