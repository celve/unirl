from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run heavy end-to-end tests that require external runtime dependencies.",
    )
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Local model checkpoint path for heavy E2E tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e"):
        return

    skip_e2e = pytest.mark.skip(reason="need --run-e2e to execute heavy E2E tests")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.fixture(scope="session")
def model_path(pytestconfig) -> str:
    path = pytestconfig.getoption("--model-path") or os.environ.get("DIFFUSIONRL_TEST_MODEL_PATH")
    if not path:
        pytest.skip("heavy E2E test requires --model-path or DIFFUSIONRL_TEST_MODEL_PATH")
    return str(path)
