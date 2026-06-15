"""Structural guards for the shared-base FSDP2 backend refactor (CPU, pytest).

Lock the invariants the refactor relies on:

* the veomni package import stays torch-free + lazy (Hydra ``_target_`` resolution
  on a torch-less machine, and the selective-import shim discipline);
* both backends share :class:`BaseFSDP2Backend` and inherit its ``@distributed``
  methods (so the dispatch ``Handle`` still discovers them);
* the optimizer save/load mechanism stays engine-specific (DCP vs plain);
* ``VeOmniBackend.save`` matches the trainer's ``save(path, step=, mode=)`` call
  (the drift that used to ``TypeError`` the moment a VeOmni recipe checkpointed).
"""

import subprocess
import sys


def test_veomni_package_import_is_torch_free():
    """Importing the veomni package must not import torch (config-compose path)."""
    code = (
        "import sys; import unirl.train.backend.veomni as v; "
        "assert 'torch' not in sys.modules, 'torch leaked on veomni package import'; "
        "assert hasattr(v, 'VeOmniBackend'), 'lazy __getattr__ missing VeOmniBackend'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_veomni_package_is_lazy():
    """The veomni package must not eagerly import its torch-heavy submodules."""
    import importlib

    importlib.import_module("unirl.train.backend.veomni")
    for sub in ("backend", "state", "wrap"):
        assert f"unirl.train.backend.veomni.{sub}" not in sys.modules


def test_backends_share_base():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("ray")  # base_backend -> Remote -> handle imports ray

    from unirl.train.backend.base_backend import BaseFSDP2Backend
    from unirl.train.backend.fsdp import FSDPBackend
    from unirl.train.backend.veomni import VeOmniBackend

    assert issubclass(FSDPBackend, BaseFSDP2Backend)
    assert issubclass(VeOmniBackend, BaseFSDP2Backend)


def test_distributed_methods_inherited():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("ray")

    from unirl.distributed.group.dispatch import DISTRIBUTED_CONFIG_ATTR
    from unirl.train.backend.fsdp import FSDPBackend
    from unirl.train.backend.veomni import VeOmniBackend

    methods = ("save", "load", "onload", "offload", "apply_eval_ema", "restore_from_eval")
    for cls in (FSDPBackend, VeOmniBackend):
        for m in methods:
            cfg = getattr(getattr(cls, m), DISTRIBUTED_CONFIG_ATTR, None)
            assert cfg is not None, f"{cls.__name__}.{m} lost its @distributed dispatch config"


def test_optimizer_hooks_are_engine_specific():
    """FSDP uses DCP optimizer gather; VeOmni uses a plain per-rank state_dict."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("ray")

    from unirl.train.backend.fsdp import FSDPBackend
    from unirl.train.backend.veomni import VeOmniBackend

    assert FSDPBackend._gather_optimizer_state is not VeOmniBackend._gather_optimizer_state
    assert FSDPBackend._load_optimizer_state is not VeOmniBackend._load_optimizer_state


def test_veomni_save_signature_matches_trainer():
    """trainer/base.py calls backend.save(path, step=step, mode=save_mode)."""
    import inspect

    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("ray")

    from unirl.train.backend.veomni import VeOmniBackend

    params = inspect.signature(VeOmniBackend.save).parameters
    assert "step" in params and "mode" in params
