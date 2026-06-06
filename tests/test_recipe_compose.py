"""Recipe compose + lazy-import discipline for the VeOmni backend.

Pure-python where possible (runs on torch-less dev boxes): Hydra-composes
the qwen_image trainside recipes and asserts the VeOmni backend package
imports without dragging in torch or veomni — the discipline that keeps
Hydra ``_target_`` resolution cheap driver-side and keeps veomni's import
side effects out of every process that never constructs the backend.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_veomni_package_imports_without_torch_or_veomni() -> None:
    for mod in ("torch", "veomni"):
        sys.modules.pop(mod, None)
    importlib.import_module("unirl.train.backend.veomni")
    assert "torch" not in sys.modules, "package import must stay torch-free (PEP 562 lazy re-export)"
    assert "veomni" not in sys.modules, "veomni must only load inside _compat.load()"


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
    from hydra.utils import get_class

    sys.modules.pop("veomni", None)
    cls = get_class("unirl.train.backend.veomni.VeOmniBackend")
    assert cls.__name__ == "VeOmniBackend"
    assert "veomni" not in sys.modules, "resolving the class must not import veomni"
