"""E0 gate: the modules the parity gate touches import, and their hot paths resolve.

A missing import in an algorithm's hot path does not surface until a worker executes that
line — minutes into a GPU run, inside a Ray traceback. `TensorWeightSync` in the test-2
matrix died exactly that way on `NameError: parity_measurement_enabled`. These are cheap
and catch it in the suite instead.
"""

from __future__ import annotations

import importlib

import pytest

TOUCHED = [
    "unirl.utils.parity_gate",
    "unirl.utils.driver_clock",
    "unirl.utils.worker_rng",
    "unirl.algorithms.base",
    "unirl.algorithms.flowgrpo",
    "unirl.algorithms.grpo",
    "unirl.trainer.ar",
    "unirl.trainer.diffusion",
    "unirl.trainer.async_rollout",
]

# Every name the parity work added to a module that did not previously use it. A NameError
# here is the failure the matrix hit; hasattr on the module namespace is what catches it.
REQUIRED_BINDINGS = [
    ("unirl.algorithms.flowgrpo", "parity_measurement_enabled"),
    ("unirl.algorithms.flowgrpo", "rollout_replay_logp_absdiff"),
    ("unirl.algorithms.flowgrpo", "gather_sde_field"),
    ("unirl.algorithms.grpo", "rollout_replay_logp_absdiff"),
    ("unirl.trainer.ar", "ParityGate"),
    ("unirl.trainer.ar", "publication_path_name"),
    ("unirl.trainer.diffusion", "ParityGate"),
    ("unirl.trainer.diffusion", "publication_path_name"),
    ("unirl.trainer.async_rollout", "ParityGate"),
    ("unirl.trainer.async_rollout", "publication_path_name"),
    ("unirl.trainer.async_rollout", "DriverClock"),
    ("unirl.trainer.async_rollout", "buffer_accounting_metrics"),
]


@pytest.mark.parametrize("module", TOUCHED)
def test_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module,symbol", REQUIRED_BINDINGS)
def test_symbol_is_bound_in_module_namespace(module, symbol):
    assert hasattr(importlib.import_module(module), symbol), f"{module} never imported {symbol}"


def test_the_parity_witness_is_not_in_the_transported_schema():
    """E4a compares segment schemas and E5-R prices them, so the witness stays off the IR."""
    from unirl.types.segments.latent import LatentSegment

    assert not hasattr(LatentSegment, "rollout_sde_logp")


def test_measurement_and_raising_are_separately_controlled(monkeypatch):
    """A timing run must be able to take the clock without paying for the gauge."""
    from unirl.utils.parity_gate import ParityGate, parity_measurement_enabled

    monkeypatch.delenv("UNIRL_PARITY_TOLERANCE", raising=False)
    monkeypatch.delenv("UNIRL_PARITY_MEASURE", raising=False)
    assert not parity_measurement_enabled()
    assert not ParityGate.from_env().armed

    monkeypatch.setenv("UNIRL_PARITY_MEASURE", "1")
    assert parity_measurement_enabled()
    assert not ParityGate.from_env().armed

    monkeypatch.delenv("UNIRL_PARITY_MEASURE")
    monkeypatch.setenv("UNIRL_PARITY_TOLERANCE", "0.01")
    assert parity_measurement_enabled()
    assert ParityGate.from_env().armed

    monkeypatch.setenv("UNIRL_PARITY_MEASURE", "0")
    assert not parity_measurement_enabled()
