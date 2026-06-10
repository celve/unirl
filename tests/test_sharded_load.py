"""Shared sharded-weight loader helpers (no dist, no checkpoints).

Unit-tests the pieces FSDPBackend and VeOmniBackend share via
``unirl.train.backend.sharded_load``: the meta-gate predicate, the
diffusers-layout safetensors reader, and the LoRA ``base_layer`` key remap.
The full broadcast-load collective
(``set_model_state_dict(broadcast_from_rank0=True)``) needs a process group
and is exercised by the GPU lifecycle scripts, not here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="sharded_load imports torch")

import torch.nn as nn  # noqa: E402

from unirl.train.backend.sharded_load import (  # noqa: E402
    _module_has_meta_param,
    _read_safetensors_dir,
    _remap_lora_base_keys,
)


def test_module_has_meta_param() -> None:
    with torch.device("meta"):
        meta = nn.Linear(4, 4)
    assert _module_has_meta_param(meta) is True
    assert _module_has_meta_param(nn.Linear(4, 4)) is False


def test_read_safetensors_dir_merges_shards(tmp_path) -> None:
    from safetensors.torch import save_file

    save_file({"a.weight": torch.ones(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b.weight": torch.zeros(3)}, str(tmp_path / "model-00002-of-00002.safetensors"))

    sd = _read_safetensors_dir(str(tmp_path))

    assert set(sd) == {"a.weight", "b.weight"}
    assert torch.equal(sd["a.weight"], torch.ones(2))
    assert torch.equal(sd["b.weight"], torch.zeros(3))


def test_read_safetensors_dir_raises_on_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        _read_safetensors_dir(str(tmp_path / "does-not-exist"))


def test_read_safetensors_dir_raises_on_empty_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        _read_safetensors_dir(str(tmp_path))


def test_remap_lora_base_keys_inserts_base_layer_hop() -> None:
    # A peft-style wrapped Linear exposes its frozen base weight at
    # ``<module>.base_layer.{weight,bias}``; the base checkpoint still uses
    # ``<module>.{weight,bias}``.
    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv_proj = nn.Module()
            self.qkv_proj.base_layer = nn.Linear(4, 4)

    model = Wrapped()
    ckpt = {"qkv_proj.weight": torch.ones(4, 4), "qkv_proj.bias": torch.ones(4)}

    remapped = _remap_lora_base_keys(ckpt, model)

    assert "qkv_proj.base_layer.weight" in remapped
    assert "qkv_proj.base_layer.bias" in remapped
    assert "qkv_proj.weight" not in remapped


def test_remap_lora_base_keys_passthrough_when_no_wrap() -> None:
    model = nn.Linear(4, 4)  # keys: weight, bias
    ckpt = {"weight": torch.ones(4, 4), "bias": torch.ones(4)}

    remapped = _remap_lora_base_keys(ckpt, model)

    assert set(remapped) == {"weight", "bias"}


# ----------------------------------------------------------------------
# load_trainable_weights — the shared post-wrap dispatch both backends call.
# (The full broadcast-load is mocked; only the branch selection is tested.)
# ----------------------------------------------------------------------


def test_load_trainable_weights_uses_stash_path(monkeypatch) -> None:
    import unirl.train.backend.sharded_load as sl

    calls: dict = {}
    monkeypatch.setattr(
        sl,
        "load_sharded",
        lambda model, weights_dir, *, device, strict=False: calls.update(
            weights_dir=weights_dir, strict=strict
        ),
    )

    class FakeBundle:
        _transformer_weights_path = "/ckpt/transformer"

    sl.load_trainable_weights(nn.Linear(2, 2), FakeBundle(), device=torch.device("cpu"), eager_ok=True)

    assert calls == {"weights_dir": "/ckpt/transformer", "strict": False}


def test_load_trainable_weights_calls_materialize(monkeypatch) -> None:
    import unirl.train.backend.sharded_load as sl

    monkeypatch.setattr(
        sl, "load_sharded", lambda *a, **k: pytest.fail("load_sharded must not run for Pattern A")
    )
    seen: dict = {}

    class FakeBundle:
        def materialize(self, *, device, with_aux=()):
            seen.update(device=device, with_aux=with_aux)

    sl.load_trainable_weights(
        nn.Linear(2, 2), FakeBundle(), device=torch.device("cpu"), with_aux=("vae",), eager_ok=False
    )

    assert seen == {"device": torch.device("cpu"), "with_aux": ("vae",)}


def test_load_trainable_weights_eager_ok_is_noop() -> None:
    import unirl.train.backend.sharded_load as sl

    # No weight source + eager_ok (FSDP): weights already present → no-op, no raise.
    sl.load_trainable_weights(nn.Linear(2, 2), object(), device=torch.device("cpu"), eager_ok=True)


def test_load_trainable_weights_eager_rejected_when_not_ok() -> None:
    import unirl.train.backend.sharded_load as sl

    # No weight source + not eager_ok (VeOmni already to_empty'd) → hard error.
    with pytest.raises(ValueError, match="no weight source"):
        sl.load_trainable_weights(nn.Linear(2, 2), object(), device=torch.device("cpu"), eager_ok=False)


# ----------------------------------------------------------------------
# resolve_trainable_module — backends hand over a nested submodule when the
# bundle exposes trainable_module() (composite, e.g. hi3 decoder), else the attr.
# ----------------------------------------------------------------------


def test_resolve_trainable_module_prefers_bundle_method() -> None:
    from unirl.train.backend.base import resolve_trainable_module

    decoder = object()
    wrapper = object()

    class CompositeBundle:  # hi3-shaped: hands over the nested decoder
        transformer = wrapper

        def trainable_module(self):
            return decoder

    class SingleModuleBundle:  # sd3/qwen_image-shaped: no method → use the attr
        transformer = wrapper

    assert resolve_trainable_module(CompositeBundle(), "transformer") is decoder
    assert resolve_trainable_module(SingleModuleBundle(), "transformer") is wrapper
