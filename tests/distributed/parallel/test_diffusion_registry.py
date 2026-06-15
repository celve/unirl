"""Importing the diffusion SP package must self-register every per-model wrapper.

CPU-only: importing the package needs only ``torch`` (veomni/diffusers are lazy and
unused at import time), so this runs without a GPU. It locks the decorator-registry
wiring; the moved code's numerics are covered by the ``sp_*_parity`` torchrun tests.
"""
from unirl.train.backend.veomni.sp import diffusion

EXPECTED_MODELS = {
    "QwenImageTransformer2DModel",
    "SD3Transformer2DModel",
    "WanTransformer3DModel",
    "Flux2Transformer2DModel",
}


def test_all_model_wrappers_register_on_import():
    assert set(diffusion.FORWARD_WRAPPERS) == EXPECTED_MODELS
    assert all(callable(fn) for fn in diffusion.FORWARD_WRAPPERS.values())
