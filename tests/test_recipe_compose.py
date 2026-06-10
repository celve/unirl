"""Recipe compose + lazy-import discipline for the VeOmni backend.

Pure-python where possible (runs on torch-less dev boxes): Hydra-composes
the qwen_image trainside recipes and asserts the VeOmni backend package
imports without dragging in torch or veomni — the discipline that keeps
Hydra ``_target_`` resolution cheap driver-side and keeps veomni's import
side effects out of every process that never constructs the backend.

The sys.modules assertions run in SUBPROCESSES: popping an already-imported
torch in-process and re-importing it re-registers its TORCH_LIBRARY
namespaces and crashes the interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run_in_clean_interpreter(code: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout


def test_veomni_package_imports_without_torch_or_veomni() -> None:
    _run_in_clean_interpreter(
        "import sys; import unirl.train.backend.veomni; "
        "assert 'torch' not in sys.modules, 'package import must stay torch-free'; "
        "assert 'veomni' not in sys.modules, 'veomni must only load inside _compat.load()'; "
        "print('OK')"
    )


@pytest.mark.parametrize(
    "config_name",
    ["diffusion/qwen_image_trainside", "diffusion/qwen_image_trainside_veomni"],
)
def test_trainside_recipes_compose(config_name: str) -> None:
    pytest.importorskip("hydra", reason="hydra-core not installed")
    from hydra import compose, initialize_config_dir

    if not (EXAMPLES_DIR / f"{config_name}.yaml").exists():
        pytest.skip(f"{config_name}.yaml not present")
    with initialize_config_dir(config_dir=str(EXAMPLES_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    assert cfg.backend._target_.endswith("Backend")
    assert cfg.bundle._target_.startswith("unirl.models.qwen_image")


def test_veomni_recipe_targets_resolve_lazily() -> None:
    """`get_class` on the _target_ must succeed and still not import veomni
    (construction, not resolution, is what triggers _compat.load())."""
    pytest.importorskip("hydra", reason="hydra-core not installed")
    pytest.importorskip("torch", reason="class resolution imports the backend module (torch)")
    _run_in_clean_interpreter(
        "import sys; from hydra.utils import get_class; "
        "cls = get_class('unirl.train.backend.veomni.VeOmniBackend'); "
        "assert cls.__name__ == 'VeOmniBackend'; "
        "assert 'veomni' not in sys.modules, 'resolving the class must not import veomni'; "
        "print('OK')"
    )
