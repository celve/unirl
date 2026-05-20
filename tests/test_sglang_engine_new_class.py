"""Unit tests for ``SGLangRolloutEngine`` parts that don't need SGLang runtime.

Pins behavior of:

- ``_resolve_sde_label`` — strategy → SGLang kernel label mapping (returns
  ``"sde"`` for flow, ``"cps"`` for cps, ``None`` for missing strategy, raises
  on unrecognized strategies).
- One-shot ctor surface — confirms the class is constructible-shaped (no
  ``initialize`` method, no ``_is_initialized`` flag, has ``shutdown``,
  ``sleep``, ``wake_up``, ``health_check``, ``is_offloaded`` property).

Ctor / generate / weight-sync RPC plumbing is covered by the smoke test —
those paths require a live SGLang scheduler client and SGLang server
subprocess. See ``scripts/smoke_sd3_t2i_replay_sglang.py``.
"""

from __future__ import annotations

import pytest

from diffusionrl.rollout.engine.sglang.engine import SGLangRolloutEngine


class _FakeStrategy:
    """Minimal stand-in for an SDE strategy with a ``canonical_name`` class attr."""

    canonical_name = "flow"


class _FakeCpsStrategy:
    canonical_name = "cps"


class _FakeUnknownStrategy:
    canonical_name = "dpm2"


def test_sde_label_flow_maps_to_sde():
    assert SGLangRolloutEngine._resolve_sde_label(_FakeStrategy()) == "sde"


def test_sde_label_cps_maps_to_cps():
    assert SGLangRolloutEngine._resolve_sde_label(_FakeCpsStrategy()) == "cps"


def test_sde_label_none_strategy_returns_none():
    """ODE-mode callers (eval, NFT) pass strategy=None; mapping returns None."""
    assert SGLangRolloutEngine._resolve_sde_label(None) is None


def test_sde_label_unknown_strategy_raises():
    """Unsupported kernels must fail loudly rather than silently picking 'sde'."""
    with pytest.raises(ValueError):
        SGLangRolloutEngine._resolve_sde_label(_FakeUnknownStrategy())


def test_class_has_required_lifecycle_methods():
    """One-shot ctor protocol: no `initialize`, must have shutdown / sleep / wake_up."""
    assert hasattr(SGLangRolloutEngine, "shutdown")
    assert hasattr(SGLangRolloutEngine, "sleep")
    assert hasattr(SGLangRolloutEngine, "wake_up")
    assert hasattr(SGLangRolloutEngine, "health_check")
    assert hasattr(SGLangRolloutEngine, "is_offloaded")
    # The legacy two-step lifecycle is intentionally gone.
    assert not hasattr(SGLangRolloutEngine, "initialize")
    assert not hasattr(SGLangRolloutEngine, "is_initialized")


def test_class_has_all_weight_sync_methods():
    """Every transport on the new ABC has a concrete or NotImplemented override."""
    assert hasattr(SGLangRolloutEngine, "update_weights_from_tensor")
    assert hasattr(SGLangRolloutEngine, "init_weights_update_group")
    assert hasattr(SGLangRolloutEngine, "update_weights_from_distributed")
    assert hasattr(SGLangRolloutEngine, "destroy_weights_update_group")
    assert hasattr(SGLangRolloutEngine, "set_lora_from_tensors")
    assert hasattr(SGLangRolloutEngine, "update_weights_from_ipc")
    assert hasattr(SGLangRolloutEngine, "loaded_param_checksums")


def test_class_drops_legacy_only_methods():
    """Methods that lived on the legacy engine are intentionally not ported."""
    # Trainer-side now packages state dicts and pushes via update_weights_from_tensor.
    assert not hasattr(SGLangRolloutEngine, "update_weights")
    assert not hasattr(SGLangRolloutEngine, "update_weights_from_path")
    # Replaced by loaded_param_checksums.
    assert not hasattr(SGLangRolloutEngine, "get_last_weight_checksum")
    # Not on new ABC.
    assert not hasattr(SGLangRolloutEngine, "encode_prompt")
    assert not hasattr(SGLangRolloutEngine, "decode_latents")
    # Substring-based model_type inference replaced by cfg.model_family.
    assert not hasattr(SGLangRolloutEngine, "_infer_model_type")


def test_update_weights_from_ipc_signals_not_implemented():
    """SGLang has no BucketedWeightReceiver — the override must say so explicitly."""
    # We can't instantiate without SGLang runtime, but the method itself is a
    # class-level callable; bind it to a no-op instance via __new__ to invoke.
    instance = SGLangRolloutEngine.__new__(SGLangRolloutEngine)
    with pytest.raises(NotImplementedError):
        instance.update_weights_from_ipc()
