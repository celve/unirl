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
