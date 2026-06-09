"""Meta-init rollout: the shared ``finalize_meta_init`` helper + the per-model
``meta_init_transformer`` contract for the no-quirk single-transformer bundles
(wan21, hunyuan_video, hunyuan_video15).

Fakes over real checkpoints (per the add-model-bundle skill): the
diffusers/transformers classes each ``from_config`` imports at call time are
monkeypatched. Asserts the contract both backends rely on — transformer built
on meta, dtype-cast, ``init_weights`` no-op, ``_transformer_weights_path``
stashed at ``<ckpt>/transformer``.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch", reason="bundle construction needs torch")
pytest.importorskip("diffusers", reason="bundles import diffusers at call time")
pytest.importorskip("transformers", reason="bundles import transformers at call time")

import torch.nn as nn  # noqa: E402


def test_finalize_meta_init_casts_noops_and_warns(caplog) -> None:
    from unirl.models.types.meta_init import finalize_meta_init

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.register_buffer("np_buf", torch.zeros(3), persistent=False)

    with torch.device("meta"):
        m = M()

    with caplog.at_level(logging.WARNING, logger="unirl.models.types.meta_init"):
        out = finalize_meta_init(m, dtype=torch.bfloat16)

    assert out is m
    assert all(p.is_meta for p in m.parameters())
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    assert m.init_weights() is None  # stamped no-op
    assert "np_buf" in caplog.text  # non-persistent buffer surfaced


# ----------------------------------------------------------------------
# Shared fakes (meta-friendly: from_config builds under torch.device("meta")).
# ----------------------------------------------------------------------


class _FakeCfg:
    image_dim = 0  # wan21: T2V → no CLIP vision tower
    use_meanflow = False  # hv15: not a meanflow checkpoint


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.config = _FakeCfg()

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


def _assert_meta_contract(bundle, expected_weights_dir: str) -> None:
    assert all(p.is_meta for p in bundle.transformer.parameters())
    assert all(p.dtype == torch.bfloat16 for p in bundle.transformer.parameters())
    assert bundle.transformer.init_weights() is None
    assert bundle._transformer_weights_path == expected_weights_dir
    assert not any(p.is_meta for p in bundle.vae.parameters())  # aux stays eager


def test_wan21_meta_init(monkeypatch) -> None:
    import diffusers
    import transformers

    monkeypatch.setattr(diffusers, "WanTransformer3DModel", _FakeTransformer, raising=False)
    monkeypatch.setattr(diffusers, "AutoencoderKLWan", _FakeAux, raising=False)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeFromPretrained, raising=False)
    monkeypatch.setattr(transformers, "UMT5EncoderModel", _FakeAux, raising=False)

    from unirl.models.wan21.bundle import WAN21Bundle
    from unirl.models.wan21.config import WAN21PipelineConfig

    bundle = WAN21Bundle.from_config(
        WAN21PipelineConfig(pretrained_model_ckpt_path="/ckpt/wan", device="cpu", meta_init_transformer=True)
    )
    _assert_meta_contract(bundle, "/ckpt/wan/transformer")


def test_hunyuan_video_meta_init(monkeypatch) -> None:
    import diffusers
    import diffusers.schedulers as diffusers_schedulers
    import transformers

    monkeypatch.setattr(diffusers, "HunyuanVideoTransformer3DModel", _FakeTransformer, raising=False)
    monkeypatch.setattr(diffusers, "AutoencoderKLHunyuanVideo", _FakeAux, raising=False)
    monkeypatch.setattr(diffusers_schedulers, "FlowMatchEulerDiscreteScheduler", _FakeFromPretrained, raising=False)
    monkeypatch.setattr(transformers, "LlamaModel", _FakeAux, raising=False)
    monkeypatch.setattr(transformers, "LlamaTokenizerFast", _FakeFromPretrained, raising=False)
    monkeypatch.setattr(transformers, "CLIPTextModel", _FakeAux, raising=False)
    monkeypatch.setattr(transformers, "CLIPTokenizer", _FakeFromPretrained, raising=False)

    from unirl.models.hunyuan_video.bundle import HunyuanVideoBundle
    from unirl.models.hunyuan_video.config import HunyuanVideoPipelineConfig

    bundle = HunyuanVideoBundle.from_config(
        HunyuanVideoPipelineConfig(pretrained_model_ckpt_path="/ckpt/hv", device="cpu", meta_init_transformer=True)
    )
    _assert_meta_contract(bundle, "/ckpt/hv/transformer")


def test_hunyuan_video15_meta_init(monkeypatch) -> None:
    import diffusers
    import diffusers.schedulers as diffusers_schedulers
    import transformers

    monkeypatch.setattr(diffusers, "HunyuanVideo15Transformer3DModel", _FakeTransformer, raising=False)
    monkeypatch.setattr(diffusers, "AutoencoderKLHunyuanVideo15", _FakeAux, raising=False)
    monkeypatch.setattr(diffusers_schedulers, "FlowMatchEulerDiscreteScheduler", _FakeFromPretrained, raising=False)
    monkeypatch.setattr(transformers, "Qwen2_5_VLTextModel", _FakeAux, raising=False)
    monkeypatch.setattr(transformers, "Qwen2Tokenizer", _FakeFromPretrained, raising=False)
    monkeypatch.setattr(transformers, "T5EncoderModel", _FakeAux, raising=False)
    monkeypatch.setattr(transformers, "ByT5Tokenizer", _FakeFromPretrained, raising=False)

    from unirl.models.hunyuan_video15.bundle import HunyuanVideo15Bundle
    from unirl.models.hunyuan_video15.config import HunyuanVideo15PipelineConfig

    bundle = HunyuanVideo15Bundle.from_config(
        HunyuanVideo15PipelineConfig(
            pretrained_model_ckpt_path="/ckpt/hv15", device="cpu", meta_init_transformer=True
        )
    )
    _assert_meta_contract(bundle, "/ckpt/hv15/transformer")


class _FakeTokenizer:
    pad_token = None
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, path, subfolder=None):
        return cls()


def test_flux2_klein_meta_init(monkeypatch) -> None:
    import diffusers
    import diffusers.schedulers as diffusers_schedulers
    import transformers

    monkeypatch.setattr(diffusers, "Flux2Transformer2DModel", _FakeTransformer, raising=False)
    monkeypatch.setattr(diffusers, "AutoencoderKLFlux2", _FakeAux, raising=False)
    monkeypatch.setattr(diffusers_schedulers, "FlowMatchEulerDiscreteScheduler", _FakeFromPretrained, raising=False)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeAux, raising=False)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer, raising=False)

    from unirl.models.flux2_klein.bundle import Flux2KleinBundle
    from unirl.models.flux2_klein.config import Flux2KleinPipelineConfig

    bundle = Flux2KleinBundle.from_config(
        Flux2KleinPipelineConfig(pretrained_model_ckpt_path="/ckpt/flux2", device="cpu", meta_init_transformer=True)
    )
    _assert_meta_contract(bundle, "/ckpt/flux2/transformer")


def test_flux2_zero_checkpoint_absent_params(tmp_path) -> None:
    """The guidance-embedder quirk: params absent from the checkpoint are zeroed
    post-load (deferred), not left as to_empty garbage."""
    from safetensors.torch import save_file

    from unirl.models.flux2_klein.bundle import _stamp_zero_checkpoint_absent_params
    from unirl.train.deferred import apply_deferred_ops

    class T(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kept = nn.Linear(2, 2)
            self.guidance = nn.Linear(2, 2)  # absent from the checkpoint

    wdir = tmp_path / "transformer"
    wdir.mkdir()
    real = T()
    save_file(
        {"kept.weight": real.kept.weight.detach().contiguous(), "kept.bias": real.kept.bias.detach().contiguous()},
        str(wdir / "diffusion_pytorch_model.safetensors"),
    )

    with torch.device("meta"):
        meta = T()
    _stamp_zero_checkpoint_absent_params(meta, str(wdir))

    meta.to_empty(device="cpu")  # backend materialize: garbage storage for every param
    with torch.no_grad():  # simulate the load filling everything, incl. absent garbage
        for p in meta.parameters():
            p.fill_(5.0)
    apply_deferred_ops(meta)

    assert torch.all(meta.guidance.weight == 0)
    assert torch.all(meta.guidance.bias == 0)
    assert torch.all(meta.kept.weight == 5.0)  # present in ckpt → not touched by the deferred op
