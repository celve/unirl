"""Integration tests for the WAN I2V pipeline.

Tests cover:
- Dataset loading with canonical "media" format → MediaRef conversion
- Relative path resolution against dataset base_dir
- End-to-end data pipeline threading (DataSource → Prompts → Sampler multimodal kwargs)
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

import torch

from diffusionrl.data import MultimodalRLDataSource, TextPromptDataset
from diffusionrl.data.datasets import (
    _resolve_media_uri,
    normalize_prompt_example,
)
from diffusionrl.types.media import MediaRef
from diffusionrl.types.prompts import Prompts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args(data_path, *, seed: int = 42, prompts_per_rollout: int = 2):
    return SimpleNamespace(
        run=SimpleNamespace(
            data_path=str(data_path),
            eval_data_path=None,
            seed=seed,
        ),
        algorithm=SimpleNamespace(prompts_per_rollout=prompts_per_rollout),
    )


def _write_i2v_jsonl(tmp_path, entries: List[Dict[str, Any]], filename: str = "train.jsonl"):
    """Write an I2V-style JSONL dataset file."""
    path = tmp_path / filename
    lines = [json.dumps(entry) for entry in entries]
    path.write_text("\n".join(lines))
    return path


def _media_entry(prompt: str, uri: str) -> Dict[str, Any]:
    """Build a canonical I2V dataset entry with media field."""
    return {"prompt": prompt, "media": [{"modality": "image", "role": "condition", "uri": uri}]}


# ---------------------------------------------------------------------------
# Tests: canonical "media" format → MediaRef conversion
# ---------------------------------------------------------------------------


class TestMediaRefParsing:
    """Tests for canonical media list → MediaRef conversion."""

    def test_relative_uri_resolved_against_base_dir(self, tmp_path):
        """A relative URI in media is resolved to absolute."""
        entry = _media_entry("A cat jumping", "cat.png")
        result = normalize_prompt_example(entry, base_dir=str(tmp_path))

        assert "media_refs" in result
        assert len(result["media_refs"]) == 1
        ref = result["media_refs"][0]
        assert ref.modality == "image"
        assert ref.role == "condition"
        assert ref.uri == os.path.join(str(tmp_path), "cat.png")

    def test_absolute_uri_unchanged(self, tmp_path):
        """An absolute URI is kept as-is."""
        entry = _media_entry("A cat", "/data/images/cat.png")
        result = normalize_prompt_example(entry, base_dir=str(tmp_path))

        assert result["media_refs"][0].uri == "/data/images/cat.png"

    def test_url_uri_unchanged(self, tmp_path):
        """A URL URI is kept as-is."""
        entry = _media_entry("A cat", "https://example.com/cat.png")
        result = normalize_prompt_example(entry, base_dir=str(tmp_path))

        assert result["media_refs"][0].uri == "https://example.com/cat.png"

    def test_no_media_field_produces_no_media_refs(self):
        """Without 'media' field, media_refs is absent."""
        entry = {"prompt": "A cat running"}
        result = normalize_prompt_example(entry)

        assert "media_refs" not in result

    def test_media_field_not_in_metadata(self, tmp_path):
        """The 'media' field should not leak into metadata."""
        entry = {
            "prompt": "A cat",
            "media": [{"modality": "image", "role": "condition", "uri": "cat.png"}],
            "extra_field": "val",
        }
        result = normalize_prompt_example(entry, base_dir=str(tmp_path))

        assert "media" not in result.get("metadata", {})
        assert result["metadata"] == {"extra_field": "val"}


class TestResolveMediaUri:
    """Unit tests for _resolve_media_uri helper."""

    def test_relative_path_with_base_dir(self):
        assert _resolve_media_uri("img.png", base_dir="/data/ds") == "/data/ds/img.png"

    def test_relative_path_without_base_dir(self):
        assert _resolve_media_uri("img.png", base_dir=None) == "img.png"

    def test_absolute_path(self):
        assert _resolve_media_uri("/abs/img.png", base_dir="/data") == "/abs/img.png"

    def test_http_url(self):
        assert _resolve_media_uri("https://cdn.com/img.png", base_dir="/data") == "https://cdn.com/img.png"

    def test_s3_url(self):
        assert _resolve_media_uri("s3://bucket/img.png", base_dir="/data") == "s3://bucket/img.png"


# ---------------------------------------------------------------------------
# Tests: TextPromptDataset with canonical media format
# ---------------------------------------------------------------------------


class TestTextPromptDatasetI2V:
    """Tests for TextPromptDataset loading I2V-style JSONL files."""

    def test_load_canonical_media_format(self, tmp_path):
        """Dataset with 'media' field loads correctly."""
        entries = [
            _media_entry("Transform into watercolor", "v2v_001.png"),
            _media_entry("Add sunset lighting", "v2v_002.png"),
        ]
        path = _write_i2v_jsonl(tmp_path, entries)

        dataset = TextPromptDataset(str(path))

        assert len(dataset) == 2
        assert dataset[0]["prompt"] == "Transform into watercolor"
        assert dataset[0]["media_refs"][0].uri == os.path.join(str(tmp_path), "v2v_001.png")
        assert dataset[0]["media_refs"][0].modality == "image"
        assert dataset[0]["media_refs"][0].role == "condition"

    def test_mixed_with_and_without_media(self, tmp_path):
        """Dataset with some entries having 'media' and some without."""
        entries = [
            _media_entry("Transform into watercolor", "v2v_001.png"),
            {"prompt": "A simple text prompt"},  # T2V entry, no media
        ]
        path = _write_i2v_jsonl(tmp_path, entries)

        dataset = TextPromptDataset(str(path))

        assert len(dataset) == 2
        assert "media_refs" in dataset[0]
        assert "media_refs" not in dataset[1]

    def test_nested_directory_uris(self, tmp_path):
        """URIs with subdirectories resolve correctly."""
        entries = [
            _media_entry("Style transfer", "images/subdir/img.png"),
        ]
        path = _write_i2v_jsonl(tmp_path, entries)

        dataset = TextPromptDataset(str(path))

        expected_uri = os.path.join(str(tmp_path), "images/subdir/img.png")
        assert dataset[0]["media_refs"][0].uri == expected_uri


# ---------------------------------------------------------------------------
# Tests: MultimodalRLDataSource with I2V data
# ---------------------------------------------------------------------------


class TestMultimodalDataSourceI2V:
    """Tests for the full data source pipeline with I2V datasets."""

    def test_data_source_collates_media_refs(self, tmp_path):
        """DataSource collates media refs into batches."""
        entries = [
            _media_entry("Style A", "img_a.png"),
            _media_entry("Style B", "img_b.png"),
        ]
        path = _write_i2v_jsonl(tmp_path, entries)

        data_source = MultimodalRLDataSource(_args(path, prompts_per_rollout=2))
        batch = data_source.get_samples(2)

        assert "media_refs" in batch
        assert len(batch["media_refs"]) == 2
        # URIs should be resolved to absolute paths
        for refs in batch["media_refs"]:
            assert len(refs) == 1
            assert refs[0].modality == "image"
            assert refs[0].role == "condition"
            assert os.path.isabs(refs[0].uri)

    def test_eval_samples_include_media_refs(self, tmp_path):
        """Eval sampling also threads media_refs through."""
        entries = [
            _media_entry("Style A", "img_a.png"),
            _media_entry("Style B", "img_b.png"),
            _media_entry("Style C", "img_c.png"),
        ]
        path = _write_i2v_jsonl(tmp_path, entries)

        data_source = MultimodalRLDataSource(_args(path, prompts_per_rollout=2))
        eval_batch = data_source.get_eval_samples(2)

        assert "media_refs" in eval_batch
        assert len(eval_batch["media_refs"]) == 2


# ---------------------------------------------------------------------------
# Tests: Sampler multimodal encode kwargs (mocked model bundle)
# ---------------------------------------------------------------------------


class TestSamplerMultimodalEncode:
    """Tests for prepare_multimodal_encode_kwargs with I2V media refs."""

    def test_prepare_multimodal_encode_kwargs_with_image_refs(self):
        """Sampler correctly extracts condition image refs and calls model bundle."""
        from diffusionrl.samplers.fsdp.base_sampler import FSDPBaseSampler
        from diffusionrl.types.request import RolloutRequest
        from diffusionrl.types.sampling import SamplingParams

        # Build a request with media refs
        prompts = Prompts.from_unique_prompts(
            ["A cat jumping"],
            media_refs=[
                [MediaRef(modality="image", role="condition", uri="/tmp/cat.png")],
            ],
        )
        sampling_params = MagicMock(spec=SamplingParams)
        request = RolloutRequest(prompts=prompts, sampling_params=sampling_params)

        # Mock model bundle that accepts image input
        mock_bundle = MagicMock()
        mock_bundle.accepts_image_input = True

        sampler = FSDPBaseSampler.__new__(FSDPBaseSampler)
        sampler.model_bundle = mock_bundle

        # Mock _load_image_refs to avoid actual file I/O
        fake_tensor = torch.randn(1, 3, 480, 832)
        with patch.object(sampler, "_load_image_refs", return_value=fake_tensor):
            result = sampler.prepare_multimodal_encode_kwargs(request, height=480, width=832, num_frames=81)

        assert "image" in result
        assert torch.equal(result["image"], fake_tensor)
        assert result["height"] == 480
        assert result["width"] == 832
        assert result["num_frames"] == 81

    def test_prepare_multimodal_encode_kwargs_no_refs_returns_empty(self):
        """Sampler returns empty dict when no image refs are present."""
        from diffusionrl.samplers.fsdp.base_sampler import FSDPBaseSampler
        from diffusionrl.types.request import RolloutRequest
        from diffusionrl.types.sampling import SamplingParams

        # T2V prompts - no media refs
        prompts = Prompts.from_unique_prompts(["A sunset over ocean"])
        sampling_params = MagicMock(spec=SamplingParams)
        request = RolloutRequest(prompts=prompts, sampling_params=sampling_params)

        sampler = FSDPBaseSampler.__new__(FSDPBaseSampler)
        sampler.model_bundle = MagicMock()

        result = sampler.prepare_multimodal_encode_kwargs(request, height=480, width=832, num_frames=81)

        assert result == {}

    def test_mixed_batch_raises_error(self):
        """Sampler raises ValueError on mixed I2V/T2V batch."""
        from diffusionrl.samplers.fsdp.base_sampler import FSDPBaseSampler
        from diffusionrl.types.request import RolloutRequest
        from diffusionrl.types.sampling import SamplingParams

        # Mixed batch: first sample has image, second doesn't
        prompts = Prompts.from_unique_prompts(
            ["With image", "Without image"],
            media_refs=[
                [MediaRef(modality="image", role="condition", uri="/tmp/a.png")],
                [],  # no media ref
            ],
        )
        sampling_params = MagicMock(spec=SamplingParams)
        request = RolloutRequest(prompts=prompts, sampling_params=sampling_params)

        mock_bundle = MagicMock()
        mock_bundle.accepts_image_input = True

        sampler = FSDPBaseSampler.__new__(FSDPBaseSampler)
        sampler.model_bundle = mock_bundle

        with pytest.raises(ValueError, match="mixed"):
            sampler.prepare_multimodal_encode_kwargs(request, height=480, width=832, num_frames=81)
