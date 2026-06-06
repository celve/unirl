"""QwenImageBundle ``meta_init_transformer`` lifecycle (fakes, no checkpoints).

Monkeypatches the diffusers/transformers classes that
``QwenImageBundle.from_config`` imports at call time, per the
add-model-bundle skill's "fakes over real checkpoints" rule.  Asserts the
contract VeOmniBackend relies on:

* transformer built on the meta device (no weight allocation), dtype-cast;
* ``init_weights`` stamped to a no-op (VeOmni calls it unconditionally
  after ``to_empty``);
* ``_transformer_weights_path`` stashed for the post-parallelize load;
* the eager path is byte-for-byte unaffected when the flag is False.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="bundle construction needs torch")
pytest.importorskip("diffusers", reason="bundle imports diffusers at call time")
pytest.importorskip("transformers", reason="bundle imports transformers at call time")

import torch.nn as nn  # noqa: E402


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    @classmethod
    def load_config(cls, path, subfolder=None):
        return {"path": path, "subfolder": subfolder}

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def from_pretrained(cls, path, subfolder=None, torch_dtype=None):
        m = cls()
        return m.to(torch_dtype) if torch_dtype is not None else m


class _FakeAux(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Linear(2, 2)

    @classmethod
    def from_pretrained(cls, path, subfolder=None, torch_dtype=None):
        m = cls()
        return m.to(torch_dtype) if torch_dtype is not None else m


class _FakeFromPretrained:
    @classmethod
    def from_pretrained(cls, path, subfolder=None):
        return cls()


@pytest.fixture()
def patched_components(monkeypatch):
    import diffusers
    import diffusers.schedulers as diffusers_schedulers
    import transformers

    monkeypatch.setattr(diffusers, "QwenImageTransformer2DModel", _FakeTransformer, raising=True)
    monkeypatch.setattr(diffusers, "AutoencoderKLQwenImage", _FakeAux, raising=True)
    monkeypatch.setattr(
        diffusers_schedulers, "FlowMatchEulerDiscreteScheduler", _FakeFromPretrained, raising=True
    )
    monkeypatch.setattr(transformers, "Qwen2_5_VLForConditionalGeneration", _FakeAux, raising=True)
    monkeypatch.setattr(transformers, "Qwen2Tokenizer", _FakeFromPretrained, raising=True)


def _make_config(**overrides):
    from unirl.models.qwen_image.config import QwenImagePipelineConfig

    defaults = dict(pretrained_model_ckpt_path="/ckpt/qwen-image", device="cpu")
    defaults.update(overrides)
    return QwenImagePipelineConfig(**defaults)


def test_meta_init_builds_on_meta_with_noop_init_weights(patched_components) -> None:
    from unirl.models.qwen_image.bundle import QwenImageBundle

    bundle = QwenImageBundle.from_config(_make_config(meta_init_transformer=True))

    assert all(p.is_meta for p in bundle.transformer.parameters())
    assert all(p.dtype == torch.bfloat16 for p in bundle.transformer.parameters())
    assert bundle.transformer.init_weights() is None  # stamped no-op
    assert bundle._transformer_weights_path == "/ckpt/qwen-image/transformer"
    # Aux components stay eager and frozen.
    assert not any(p.is_meta for p in bundle.vae.parameters())
    assert not any(p.requires_grad for p in bundle.vae.parameters())
    assert not any(p.requires_grad for p in bundle.text_encoder.parameters())


def test_eager_path_unchanged_when_flag_off(patched_components) -> None:
    from unirl.models.qwen_image.bundle import QwenImageBundle

    bundle = QwenImageBundle.from_config(_make_config())

    assert not any(p.is_meta for p in bundle.transformer.parameters())
    assert not hasattr(bundle, "_transformer_weights_path")
    assert "init_weights" not in vars(bundle.transformer)  # no stamp on the eager path
