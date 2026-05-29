"""Tests for the HunyuanImage3 typed conditions containers.

Covers:

- ``HunyuanImage3FusedMultimodalCondition`` — the subclass of
  ``FusedMultimodalCondition`` that adds Hunyuan's 5 scatter-layout fields.
- ``HunyuanImage3DiffusionConditions`` — composes ``fused`` +
  ``cond_vae`` + ``cond_vit`` + ``cond_timestep`` + ``tokenizer_output``.
- ``HunyuanImage3ARConditions`` — composes ``fused`` + ``cond_vit`` +
  ``tokenizer_output``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from diffusionrl.models.hunyuan_image3.conditions import (
    HunyuanImage3ARConditions,
    HunyuanImage3DiffusionConditions,
    HunyuanImage3FusedMultimodalCondition,
)
from diffusionrl.types.conditions import (
    FusedMultimodalCondition,
    ImageEmbedCondition,
    ImageLatentCondition,
    Modality,
)


def _make_fused(
    n: int = 2,
    seq_len: int = 6,
    head_dim: int = 4,
    *,
    with_gen_scatter: bool = True,
    with_cond_vae_scatter: bool = False,
    with_cond_vit_scatter: bool = False,
) -> HunyuanImage3FusedMultimodalCondition:
    """Build a fake fused condition with the requested scatter fields populated."""
    return HunyuanImage3FusedMultimodalCondition(
        input_ids=torch.zeros(n, seq_len, dtype=torch.long),
        attention_mask=torch.ones(n, 1, seq_len, seq_len, dtype=torch.bool),
        position_ids=torch.arange(seq_len, dtype=torch.long)[None].expand(n, -1),
        rope_cache=(
            torch.zeros(n, seq_len, head_dim),
            torch.zeros(n, seq_len, head_dim),
        ),
        gen_image_mask=(torch.zeros(n, seq_len, dtype=torch.bool) if with_gen_scatter else None),
        gen_timestep_scatter_index=(torch.zeros(n, 1, dtype=torch.long) if with_gen_scatter else None),
        cond_vae_image_mask=(torch.zeros(n, seq_len, dtype=torch.bool) if with_cond_vae_scatter else None),
        cond_vit_image_mask=(torch.zeros(n, seq_len, dtype=torch.bool) if with_cond_vit_scatter else None),
        cond_timestep_scatter_index=(torch.zeros(n, 1, dtype=torch.long) if with_cond_vae_scatter else None),
    )


# ---------------------------------------------------------------------------
# HunyuanImage3FusedMultimodalCondition — subclass mechanics
# ---------------------------------------------------------------------------


def test_fused_subclass_inherits_parent_fields():
    """Subclass exposes the 4 parent fields plus its own 5 scatter fields."""
    field_names = [f.name for f in dataclasses.fields(HunyuanImage3FusedMultimodalCondition)]
    assert field_names == [
        "input_ids",
        "attention_mask",
        "position_ids",
        "rope_cache",
        "gen_image_mask",
        "gen_timestep_scatter_index",
        "cond_vae_image_mask",
        "cond_vit_image_mask",
        "cond_timestep_scatter_index",
    ]


def test_fused_subclass_is_a_fused_multimodal_condition():
    f = _make_fused()
    assert isinstance(f, FusedMultimodalCondition)
    assert f.modality is Modality.MULTIMODAL


def test_fused_subclass_field_metadata_preserved():
    """All tensor fields are CONCAT; rope_cache is SHARED.

    Compares ``FieldKind`` by ``.name`` rather than identity since module
    reloads (e.g. under hydra-using sibling tests) can yield a fresh
    enum class with the same values but distinct identity.
    """
    by_name = {f.name: f for f in dataclasses.fields(HunyuanImage3FusedMultimodalCondition)}
    for name in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "gen_image_mask",
        "gen_timestep_scatter_index",
        "cond_vae_image_mask",
        "cond_vit_image_mask",
        "cond_timestep_scatter_index",
    ):
        kind = by_name[name].metadata.get("kind")
        assert kind is not None and kind.name == "CONCAT"
    rope_kind = by_name["rope_cache"].metadata.get("kind")
    assert rope_kind is not None and rope_kind.name == "SHARED"


def test_fused_subclass_to_dict_only_emits_set_fields():
    f = HunyuanImage3FusedMultimodalCondition(
        input_ids=torch.zeros(2, 4, dtype=torch.long),
    )
    d = f.to_dict()
    assert set(d.keys()) == {"input_ids"}


def test_fused_subclass_roundtrip_full():
    f = _make_fused(with_gen_scatter=True, with_cond_vae_scatter=True, with_cond_vit_scatter=True)
    f_round = HunyuanImage3FusedMultimodalCondition.from_dict(f.to_dict())
    assert f_round.input_ids is f.input_ids
    assert f_round.gen_image_mask is f.gen_image_mask
    assert f_round.cond_vae_image_mask is f.cond_vae_image_mask
    assert f_round.cond_vit_image_mask is f.cond_vit_image_mask


# ---------------------------------------------------------------------------
# HunyuanImage3DiffusionConditions
# ---------------------------------------------------------------------------


def test_diffusion_conditions_t2i_roundtrip():
    """Vanilla t2i: only ``fused`` is populated."""
    fused = _make_fused()
    diff = HunyuanImage3DiffusionConditions(fused=fused)
    d = diff.to_dict()
    assert set(d.keys()) == {"fused"}
    assert d["fused"] is fused

    diff_round = HunyuanImage3DiffusionConditions.from_dict(d)
    assert diff_round.fused is fused
    assert diff_round.cond_vae is None
    assert diff_round.cond_vit is None
    assert diff_round.cond_timestep is None


def test_diffusion_conditions_it2i_roundtrip():
    """it2i path: cond_vae / cond_vit / cond_timestep / tokenizer_output round-trip."""
    fused = _make_fused(with_cond_vae_scatter=True, with_cond_vit_scatter=True)
    cond_vae = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    cond_vit = ImageEmbedCondition(
        embeds=torch.randn(2, 16, 8),
        attn_mask=torch.ones(2, 16, dtype=torch.long),
        spatial_shapes=[(2, 8), (2, 8)],
    )
    cond_t = torch.zeros(2)
    tok_out = object()
    diff = HunyuanImage3DiffusionConditions(
        fused=fused,
        cond_vae=cond_vae,
        cond_vit=cond_vit,
        cond_timestep=cond_t,
        tokenizer_output=tok_out,
    )
    d = diff.to_dict()
    assert set(d.keys()) == {"fused", "cond_vae", "cond_vit", "cond_timestep", "tokenizer_output"}

    diff_round = HunyuanImage3DiffusionConditions.from_dict(d)
    assert diff_round.fused is fused
    assert diff_round.cond_vae is cond_vae
    assert diff_round.cond_vit is cond_vit
    assert diff_round.cond_timestep is cond_t
    assert diff_round.tokenizer_output is tok_out


def test_diffusion_conditions_from_dict_rejects_missing_fused_input_ids():
    with pytest.raises(TypeError, match="input_ids"):
        HunyuanImage3DiffusionConditions.from_dict({})


def test_diffusion_conditions_to_dict_rejects_unset_fused_input_ids():
    diff = HunyuanImage3DiffusionConditions()
    with pytest.raises(ValueError, match="input_ids"):
        diff.to_dict()


def test_diffusion_conditions_from_dict_rejects_wrong_cond_vae_type():
    """``cond_vae`` must be an ImageLatentCondition (or absent)."""
    fused = _make_fused()
    bad = torch.zeros(2, 4, 8, 8)
    with pytest.raises(TypeError, match="cond_vae"):
        HunyuanImage3DiffusionConditions.from_dict({"fused": fused, "cond_vae": bad})


def test_diffusion_conditions_from_dict_rejects_wrong_cond_vit_type():
    """``cond_vit`` must be an ImageEmbedCondition (or absent)."""
    fused = _make_fused()
    bad = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="cond_vit"):
        HunyuanImage3DiffusionConditions.from_dict({"fused": fused, "cond_vit": bad})


# ---------------------------------------------------------------------------
# HunyuanImage3ARConditions
# ---------------------------------------------------------------------------


def test_ar_conditions_t2t_roundtrip():
    """t2t path: only ``fused`` is populated."""
    fused = _make_fused(with_gen_scatter=False)  # AR mode doesn't use gen scatter
    ar = HunyuanImage3ARConditions(fused=fused)
    d = ar.to_dict()
    assert set(d.keys()) == {"fused"}

    ar_round = HunyuanImage3ARConditions.from_dict(d)
    assert ar_round.fused is fused
    assert ar_round.cond_vit is None


def test_ar_conditions_i2t_roundtrip():
    """i2t path: ``fused.cond_vit_image_mask`` + ``cond_vit`` ImageEmbedCondition."""
    fused = _make_fused(with_gen_scatter=False, with_cond_vit_scatter=True)
    cond_vit = ImageEmbedCondition(
        embeds=torch.randn(2, 16, 8),
        attn_mask=torch.ones(2, 16, dtype=torch.long),
        spatial_shapes=[(2, 8), (2, 8)],
    )
    tok_out = object()
    ar = HunyuanImage3ARConditions(fused=fused, cond_vit=cond_vit, tokenizer_output=tok_out)
    d = ar.to_dict()
    assert set(d.keys()) == {"fused", "cond_vit", "tokenizer_output"}

    ar_round = HunyuanImage3ARConditions.from_dict(d)
    assert ar_round.fused is fused
    assert ar_round.cond_vit is cond_vit
    assert ar_round.tokenizer_output is tok_out


def test_ar_conditions_from_dict_rejects_missing_fused_input_ids():
    with pytest.raises(TypeError, match="input_ids"):
        HunyuanImage3ARConditions.from_dict({})


def test_ar_conditions_from_dict_rejects_wrong_cond_vit_type():
    fused = _make_fused()
    bad = ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8))
    with pytest.raises(TypeError, match="cond_vit"):
        HunyuanImage3ARConditions.from_dict({"fused": fused, "cond_vit": bad})


def test_ar_conditions_to_dict_rejects_unset_fused_input_ids():
    ar = HunyuanImage3ARConditions()
    with pytest.raises(ValueError, match="input_ids"):
        ar.to_dict()
