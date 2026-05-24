"""Tests for the pure-math helpers in :mod:`diffusionrl.models.flux2_klein.flux2_klein_utils`.

Covers:

* ``compute_empirical_mu`` — both the high-token piecewise branch
  (``image_seq_len > 4300``) and the low-token linear-blend branch.
* ``patchify_latents`` / ``unpatchify_latents`` — round-trip on a
  hand-rolled tensor, plus the channel-count shape transform
  ``[B, 32, H, W] -> [B, 128, H/2, W/2]``.
* ``pack_latents`` / ``unpack_latents`` — round-trip on patchified
  spatial shape ``[B, 128, H, W] -> [B, H*W, 128] -> [B, 128, H, W]``.
* ``prepare_text_ids`` / ``prepare_latent_ids`` — verify the 4-axis
  ``(T, H, W, L)`` layout matches what ``Flux2Transformer2DModel``
  consumes (``axes_dims_rope=[32, 32, 32, 32]``).
* BN normalize/denormalize — round-trip with a stub VAE.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from diffusionrl.models.flux2_klein.flux2_klein_utils import (
    compute_empirical_mu,
    denormalize_patchified_latents,
    normalize_patchified_latents,
    pack_latents,
    patchify_latents,
    prepare_latent_ids,
    prepare_text_ids,
    unpack_latents,
    unpatchify_latents,
    vae_bn_stats,
)

# ---- compute_empirical_mu ------------------------------------------------


def test_compute_empirical_mu_high_token_branch_uses_only_line2():
    """``image_seq_len > 4300`` short-circuits to ``a2 * L + b2`` and ignores num_steps."""

    L = 8192
    expected = 0.00016927 * L + 0.45666666
    assert compute_empirical_mu(L, 10) == compute_empirical_mu(L, 200)
    assert abs(compute_empirical_mu(L, 10) - expected) < 1e-9


def test_compute_empirical_mu_low_token_branch_blends_10_to_200():
    """For ``image_seq_len <= 4300`` the result blends m_10 at N=10 and m_200 at N=200."""

    L = 1024
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    m_10 = a1 * L + b1
    m_200 = a2 * L + b2

    assert abs(compute_empirical_mu(L, 10) - m_10) < 1e-6
    assert abs(compute_empirical_mu(L, 200) - m_200) < 1e-6

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    for num_steps in (20, 50, 100):
        expected = a * num_steps + b
        assert abs(compute_empirical_mu(L, num_steps) - expected) < 1e-6


# ---- patchify / unpatchify ----------------------------------------------


def test_patchify_unpatchify_roundtrip():
    """``unpatchify ∘ patchify == id`` on the typical Klein shape."""

    latents = torch.arange(2 * 32 * 4 * 4, dtype=torch.float32).reshape(2, 32, 4, 4)
    patched = patchify_latents(latents)
    assert patched.shape == (2, 128, 2, 2)
    restored = unpatchify_latents(patched)
    assert restored.shape == latents.shape
    assert torch.equal(restored, latents)


def test_patchify_shape_invariants():
    """Channel/spatial reshape obeys the official ``C*4, H/2, W/2`` law."""

    latents = torch.randn(1, 32, 8, 6)
    patched = patchify_latents(latents)
    assert patched.shape == (1, 128, 4, 3)


# ---- pack / unpack -------------------------------------------------------


def test_pack_unpack_roundtrip_known_dims():
    """When the patchified spatial size is known, pack/unpack round-trips exactly."""

    latents = torch.randn(2, 128, 3, 5)
    tokens = pack_latents(latents)
    assert tokens.shape == (2, 15, 128)
    restored = unpack_latents(tokens, height=3, width=5)
    assert torch.equal(restored, latents)


# ---- RoPE ids ------------------------------------------------------------


def test_prepare_text_ids_layout():
    """Text RoPE ids encode ``(0, 0, 0, l)`` for ``l in range(L)``."""

    prompt_embeds = torch.zeros(2, 7, 4)
    ids = prepare_text_ids(prompt_embeds)
    assert ids.shape == (2, 7, 4)
    expected = torch.zeros(7, 4, dtype=ids.dtype)
    expected[:, 3] = torch.arange(7)
    assert torch.equal(ids[0], expected)
    assert torch.equal(ids[1], expected)


def test_prepare_latent_ids_layout():
    """Latent RoPE ids encode ``(0, h, w, 0)`` over the patchified grid."""

    latents = torch.zeros(2, 128, 2, 3)
    ids = prepare_latent_ids(latents)
    assert ids.shape == (2, 6, 4)
    expected = torch.zeros(6, 4, dtype=ids.dtype)
    coords = []
    for h in range(2):
        for w in range(3):
            coords.append((0, h, w, 0))
    for i, (t, h, w, s) in enumerate(coords):
        expected[i, 0] = t
        expected[i, 1] = h
        expected[i, 2] = w
        expected[i, 3] = s
    assert torch.equal(ids[0], expected)


# ---- VAE BN normalize / denormalize -------------------------------------


class _StubVAE:
    """Minimal stand-in for ``AutoencoderKLFlux2`` BN head."""

    def __init__(self, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-4) -> None:
        self.bn = SimpleNamespace(running_mean=mean, running_var=var)
        self.config = SimpleNamespace(batch_norm_eps=eps)


def test_vae_bn_stats_returns_none_when_no_bn():
    vae = SimpleNamespace()
    assert vae_bn_stats(vae, device=torch.device("cpu"), dtype=torch.float32) is None


def test_normalize_denormalize_roundtrip():
    """``(latents - mean) / std`` is reversed by ``latents * std + mean``."""

    C = 128
    mean = torch.linspace(-0.5, 0.5, C)
    var = torch.linspace(0.1, 1.0, C)
    vae = _StubVAE(mean=mean, var=var)
    latents = torch.randn(2, C, 4, 5)
    normalized = normalize_patchified_latents(latents, vae)
    restored = denormalize_patchified_latents(normalized, vae)
    assert torch.allclose(restored, latents, atol=1e-5)
