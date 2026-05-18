from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from diffusionrl.data import MultimodalRLDataSource, TextPromptDataset
from diffusionrl.types.media import MediaRef
from diffusionrl.types.prompts import Prompts


def _write_prompt_file(tmp_path, prompts: list[str]):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"prompt": prompt} for prompt in prompts]))
    return path


def _args(data_path, *, seed: int = 123, prompts_per_rollout: int = 2):
    return SimpleNamespace(
        run=SimpleNamespace(
            data_path=str(data_path),
            eval_data_path=None,
            seed=seed,
        ),
        algorithm=SimpleNamespace(prompts_per_rollout=prompts_per_rollout),
    )


def test_text_prompt_dataset_preserves_file_order_by_default(tmp_path):
    path = _write_prompt_file(tmp_path, ["a", "b", "c"])

    dataset = TextPromptDataset(str(path))

    assert [dataset[idx]["prompt"] for idx in range(len(dataset))] == ["a", "b", "c"]
    assert [dataset[idx]["prompt_id"] for idx in range(len(dataset))] == [
        "prompts.json:0",
        "prompts.json:1",
        "prompts.json:2",
    ]


def test_multimodal_data_source_collates_typed_media_refs(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {
                    "prompt": "a prompt",
                    "media": [{"modality": "image", "role": "condition", "uri": "/shared/a.png"}],
                    "metadata": {"dataset": "toy"},
                },
                {
                    "prompt": "b prompt",
                    "media": [{"modality": "image", "role": "condition", "uri": "/shared/b.png"}],
                },
            ]
        )
    )

    data_source = MultimodalRLDataSource(_args(path, seed=0, prompts_per_rollout=2))
    batch = data_source.get_samples(2)

    assert sorted(refs[0].uri for refs in batch["media_refs"]) == ["/shared/a.png", "/shared/b.png"]
    assert any(metadata == {"dataset": "toy"} for metadata in batch["metadata"])

    eval_batch = data_source.get_eval_samples(2)
    assert [refs[0] for refs in eval_batch["media_refs"]] == [
        MediaRef(modality="image", role="condition", uri="/shared/a.png"),
        MediaRef(modality="image", role="condition", uri="/shared/b.png"),
    ]


def test_prompts_expand_preserves_sample_aligned_media_refs():
    prompts = Prompts.from_unique_prompts(
        ["a", "b"],
        prompt_ids=["a-id", "b-id"],
        media_refs=[
            [MediaRef(modality="image", role="condition", uri="/shared/a.png")],
            [MediaRef(modality="image", role="condition", uri="/shared/b.png")],
        ],
    )

    expanded = prompts.expand(2)

    assert [refs[0].uri for refs in expanded.media_refs] == [
        "/shared/a.png",
        "/shared/a.png",
        "/shared/b.png",
        "/shared/b.png",
    ]
