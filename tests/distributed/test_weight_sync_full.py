"""Unit tests for the v2 full-weight sync name-rewrite logic.

Covers ``_apply_name_remap`` (the ordered single-``*`` rewrite that replaces the
old param_name_prefix / skip_param_patterns / name_remap trio),
``_iter_full_tensors`` (the state-dict walk that applies it), and the
``_validate_name_remap`` fail-closed config check.

No FSDP / DTensor / CUDA / Ray needed: ``_apply_name_remap`` is a pure module
function, and ``_iter_full_tensors`` imports ``raw_state_dict`` /
``merged_state_dict`` *function-locally* from ``diffusionrl.utils.peft_merge``,
so we monkeypatch them on that module (their internal ``_to_full_tensor``
collective is thereby never reached).
"""

from __future__ import annotations

import types

import pytest
import torch
from omegaconf import OmegaConf

from diffusionrl.distributed.weight_sync.full.base import (
    FullWeightSync,
    _apply_name_remap,
    _validate_name_remap,
)
from diffusionrl.utils import peft_merge

# ---------------------------------------------------------------------------
# _apply_name_remap — pure transform
# ---------------------------------------------------------------------------


def test_apply_name_remap_prepend_all():
    remap = {"*": "transformer.*"}
    assert _apply_name_remap("blocks.0.weight", remap) == "transformer.blocks.0.weight"


def test_apply_name_remap_prefix_replace():
    remap = {"model.language_model.*": "model.*"}
    assert _apply_name_remap("model.language_model.layers.0.q_proj.weight", remap) == "model.layers.0.q_proj.weight"


def test_apply_name_remap_drop():
    remap = {"model.visual.*": None}
    assert _apply_name_remap("model.visual.patch_embed.proj.weight", remap) is None


def test_apply_name_remap_no_match_passthrough():
    remap = {"model.visual.*": None}
    assert _apply_name_remap("lm_head.weight", remap) == "lm_head.weight"


def test_apply_name_remap_first_match_wins():
    remap = {"model.language_model.*": "model.*", "model.*": "X.*"}
    # the more-specific rule is first, so it wins for language_model keys
    assert _apply_name_remap("model.language_model.norm.weight", remap) == "model.norm.weight"
    # a bare model.* key only matches the second rule
    assert _apply_name_remap("model.foo", remap) == "X.foo"


def test_apply_name_remap_empty_passthrough():
    assert _apply_name_remap("anything.weight", {}) == "anything.weight"


def test_apply_name_remap_qwen_vl_golden():
    """Bit-equivalent to the old skip=['visual'] + remap={'model.language_model.':'model.'}
    over the real Qwen2.5-VL state_dict top-levels (model.visual.* / model.language_model.* /
    lm_head.weight)."""
    remap = {"model.visual.*": None, "model.language_model.*": "model.*"}
    keys = [
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.merger.mlp.0.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    assert [_apply_name_remap(k, remap) for k in keys] == [
        None,
        None,
        None,
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def test_apply_name_remap_suffix_strip():
    # leading '*' = suffix match: capture the part before, drop the '.lora' tail
    assert _apply_name_remap("blocks.0.attn.lora", {"*.lora": "*"}) == "blocks.0.attn"


def test_apply_name_remap_suffix_rewrite():
    assert _apply_name_remap("a.b.weight", {"*.weight": "*.w"}) == "a.b.w"


def test_apply_name_remap_infix():
    assert _apply_name_remap("model.layer0.bias", {"model.*.bias": "x.*.bias"}) == "x.layer0.bias"


def test_apply_name_remap_suffix_to_prefix_reorder():
    assert _apply_name_remap("blocks.0.lora", {"*.lora": "lora.*"}) == "lora.blocks.0"


def test_apply_name_remap_overlap_length_guard_no_match():
    # key 'a*ab' -> pre='a', post='ab'; 'ab' can't host both without overlap
    # (len 2 < 1 + 2), so it must NOT match -> passthrough unchanged.
    assert _apply_name_remap("ab", {"a*ab": "x*"}) == "ab"


def test_apply_name_remap_suffix_no_match_passthrough():
    assert _apply_name_remap("blocks.0.weight", {"*.lora": "*"}) == "blocks.0.weight"


# ---------------------------------------------------------------------------
# _iter_full_tensors — state-dict walk (peft_merge monkeypatched)
# ---------------------------------------------------------------------------


def _make_handler(*, lora_merged, name_remap):
    handler = object.__new__(FullWeightSync)
    handler._backend = types.SimpleNamespace(model=object())
    handler._lora_merged = lora_merged
    handler._name_remap = name_remap
    return handler


def test_iter_full_tensors_raw_filters_lora_and_applies_catchall(monkeypatch):
    t0, t1 = torch.ones(1), torch.ones(2)
    monkeypatch.setattr(
        peft_merge,
        "raw_state_dict",
        lambda _model: iter(
            [
                ("blocks.0.weight", t0),
                ("blocks.0.lora_A", torch.ones(1)),
                ("blocks.0.lora_B", torch.ones(1)),
                ("blocks.1.weight", t1),
            ]
        ),
    )
    handler = _make_handler(lora_merged=False, name_remap={"*": "transformer.*"})
    out = list(handler._iter_full_tensors())

    assert [name for name, _ in out] == ["transformer.blocks.0.weight", "transformer.blocks.1.weight"]
    # tensor objects are passed through untouched (no copy/transform on values)
    assert out[0][1] is t0 and out[1][1] is t1


def test_iter_full_tensors_merged_drops_and_remaps(monkeypatch):
    keep = torch.ones(3)
    monkeypatch.setattr(
        peft_merge,
        "merged_state_dict",
        lambda _model: iter(
            [
                ("model.visual.blocks.0.weight", torch.ones(1)),
                ("model.language_model.layers.0.q_proj.weight", keep),
                ("lm_head.weight", torch.ones(2)),
            ]
        ),
    )
    handler = _make_handler(
        lora_merged=True,
        name_remap={"model.visual.*": None, "model.language_model.*": "model.*"},
    )
    out = list(handler._iter_full_tensors())

    assert [name for name, _ in out] == ["model.layers.0.q_proj.weight", "lm_head.weight"]
    assert out[0][1] is keep


def test_iter_full_tensors_merged_does_not_filter_lora_suffix(monkeypatch):
    # merged_state_dict already absorbs LoRA deltas; the merged branch must NOT
    # re-apply the raw-path .lora_A/.lora_B skip (documents the branch difference).
    monkeypatch.setattr(
        peft_merge,
        "merged_state_dict",
        lambda _model: iter([("blocks.0.lora_A", torch.ones(1))]),
    )
    handler = _make_handler(lora_merged=True, name_remap={})
    assert [name for name, _ in handler._iter_full_tensors()] == ["blocks.0.lora_A"]


# ---------------------------------------------------------------------------
# _validate_name_remap — fail-closed config check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"model.": "x.*"},  # key has 0 '*'
        {"a*b*": "x*"},  # key has 2 '*'
        {"a.*": "x."},  # value is a string with 0 '*'
        {"a.*": "x*y*"},  # value has 2 '*'
    ],
)
def test_validate_name_remap_rejects_malformed(bad):
    with pytest.raises(ValueError):
        _validate_name_remap(bad)


def test_validate_name_remap_rejects_catchall_not_last():
    with pytest.raises(ValueError):
        _validate_name_remap({"*": "transformer.*", "model.visual.*": None})


def test_validate_name_remap_accepts_valid_and_preserves_order():
    rules = _validate_name_remap({"model.visual.*": None, "model.language_model.*": "model.*"})
    assert list(rules.items()) == [
        ("model.visual.*", None),
        ("model.language_model.*", "model.*"),
    ]


def test_validate_name_remap_none_is_empty():
    assert _validate_name_remap(None) == {}


def test_validate_name_remap_accepts_shipped_config_literals():
    # the exact name_remap literals migrated into the conf_v2 full-sync blocks
    _validate_name_remap({"*": "transformer.*"})
    _validate_name_remap({"model.visual.*": None, "model.language_model.*": "model.*"})


def test_validate_name_remap_accepts_suffix_and_infix_patterns():
    # L1: a single '*' anywhere (not just trailing) is valid
    _validate_name_remap({"*.lora": "*"})
    _validate_name_remap({"model.*.bias": "x.*.bias"})


def test_name_remap_omegaconf_roundtrip_preserves_order_and_none():
    # Guards the YAML contract: a future OmegaConf bump must not reorder keys or
    # coerce null. Mirrors the parse_hydra_cfg path (OmegaConf.to_container).
    cfg = OmegaConf.create({"name_remap": {"a.*": None, "b.*": "c.*", "*": "d.*"}})
    remap = OmegaConf.to_container(cfg, resolve=True)["name_remap"]

    assert isinstance(remap, dict)
    assert list(remap.keys()) == ["a.*", "b.*", "*"]
    assert remap["a.*"] is None
    assert _validate_name_remap(remap) == remap  # validates + catch-all is last
