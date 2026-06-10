"""Shim closure audit for ``unirl.train.backend.veomni._compat``.

The single most important de-risk gate for the VeOmni backend: proves the
selective-import shim loads the distributed layer WITHOUT executing either
poisoned ``__init__`` (``veomni`` root: ops monkey-patching; ``veomni.models``:
transformers-5.9 model zoo).  Run on any machine with torch + veomni
installed (pod CPU venv is enough — no GPU, no process group).

If a veomni version bump adds a new cross-package import to the distributed
layer, this test fails loudly at import time — that is its job.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

pytest.importorskip("torch", reason="shim audit needs torch (pod venv)")
if importlib.util.find_spec("veomni") is None:
    pytest.skip("veomni not installed (install the [veomni] extra)", allow_module_level=True)


@pytest.fixture(scope="module")
def api():
    from unirl.train.backend.veomni import _compat

    return _compat.load()


def test_load_returns_full_api(api) -> None:
    assert callable(api.init_parallel_state)
    assert callable(api.get_parallel_state)
    assert callable(api.parallelize_model_fsdp2)
    assert callable(api.clip_grad_norm)
    assert callable(api.offload_model_to_cpu)
    assert callable(api.load_model_to_gpu)
    assert isinstance(api.MixedPrecisionConfig, type)


def test_poisoned_inits_never_ran(api) -> None:
    # Root __init__ would have imported the ops/kernel registry.
    assert "veomni.ops" not in sys.modules, "veomni/__init__.py executed (ops registry imported)"
    # models __init__ would have pulled the transformers-5.9 model zoo.
    # (Loader-support siblings of module_utils — e.g. checkpoint_tensor_loading
    # — import through the stub parent WITHOUT executing the real __init__;
    # the zoo signal is the modeling subpackages + the eager registry.)
    zoo_prefixes = (
        "veomni.models.transformers",
        "veomni.models.diffusers",
        "veomni.models.seed_omni",
        "veomni.models.auto",
        "veomni.models.loader",
    )
    zoo_modules = [m for m in sys.modules if m.startswith(zoo_prefixes)]
    assert not zoo_modules, f"veomni/models/__init__.py executed (zoo imported: {zoo_modules[:5]})"
    # The stubs themselves are package-shaped but init-less — sys.modules
    # still holding our loader-less stubs IS the proof neither real
    # __init__ ever executed (a real import would have replaced them).
    assert sys.modules["veomni"].__spec__.loader is None
    assert sys.modules["veomni.models"].__spec__.loader is None


def test_models_stub_carries_loader_names(api) -> None:
    # torch_parallelize does `from ..models import load_model_weights, ...`;
    # the stub must satisfy that name import.
    models_mod = sys.modules["veomni.models"]
    assert callable(models_mod.load_model_weights)
    assert callable(models_mod.rank0_load_and_broadcast_weights)


def test_mixed_precision_config_bf16_parity(api) -> None:
    mp = api.MixedPrecisionConfig(enable=True, param_dtype="bfloat16", reduce_dtype="float32")
    assert mp.enable and mp.param_dtype == "bfloat16" and mp.reduce_dtype == "float32"


def test_load_is_cached(api) -> None:
    from unirl.train.backend.veomni import _compat

    assert _compat.load() is api
