from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from diffusionrl.data import MultimodalRLDataSource, TextPromptDataset
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.prompts import RolloutInputs


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


def test_multimodal_data_source_returns_rollout_inputs_with_images(tmp_path):
    import PIL.Image

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    PIL.Image.new("RGB", (4, 4), color=(255, 0, 0)).save(a_path)
    PIL.Image.new("RGB", (4, 4), color=(0, 255, 0)).save(b_path)

    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {
                    "prompt": "a prompt",
                    "media": [{"modality": "image", "role": "condition", "uri": str(a_path)}],
                    "metadata": {"dataset": "toy"},
                },
                {
                    "prompt": "b prompt",
                    "media": [{"modality": "image", "role": "condition", "uri": str(b_path)}],
                },
            ]
        )
    )

    data_source = MultimodalRLDataSource(_args(path, seed=0, prompts_per_rollout=2))
    inputs = data_source.get_samples(2)

    assert isinstance(inputs, RolloutInputs)
    assert isinstance(inputs.primitives["text"], Texts)
    assert len(inputs.primitives["text"].texts) == 2
    assert "image" in inputs.primitives
    assert isinstance(inputs.primitives["image"], Images)
    assert inputs.primitives["image"].pixels.shape == (2, 3, 4, 4)
    assert len(inputs.sample_ids) == 2
    assert len(inputs.group_ids) == 2


def test_rollout_inputs_expand_preserves_text_and_ids():
    inputs = RolloutInputs(
        primitives={"text": Texts(texts=["a", "b"])},
        sample_ids=["prompt:a-id:sample:0", "prompt:b-id:sample:0"],
        group_ids=["a-id", "b-id"],
    )

    expanded = inputs.expand(2)

    assert expanded.primitives["text"].texts == ["a", "a", "b", "b"]
    assert expanded.sample_ids == [
        "prompt:a-id:sample:0",
        "prompt:a-id:sample:1",
        "prompt:b-id:sample:0",
        "prompt:b-id:sample:1",
    ]
    assert expanded.group_ids == ["a-id", "a-id", "b-id", "b-id"]
