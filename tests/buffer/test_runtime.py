"""Lifecycle tests for ``TransferQueueRuntime``.

Covers the process-singleton plumbing (``current()`` / ``install()`` /
``clear_current()``), the ``is_enabled()`` predicate, and that ``init()``
returns ``None`` (without touching driver-only state) when
``cfg.transfer_queue`` is absent.

These tests do not require Ray, ZMQ, or a real TQ controller.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from diffusionrl.distributed.transfer_queue import TransferQueueRuntime


@pytest.fixture(autouse=True)
def _reset_current():
    """Each test starts and ends with no runtime bound to the process."""
    TransferQueueRuntime.clear_current()
    yield
    TransferQueueRuntime.clear_current()


def test_current_starts_unbound():
    assert TransferQueueRuntime.current() is None


def test_install_binds_self_and_returns_self():
    runtime = TransferQueueRuntime()
    returned = runtime.install()
    assert returned is runtime
    assert TransferQueueRuntime.current() is runtime


def test_clear_current_unbinds():
    TransferQueueRuntime().install()
    TransferQueueRuntime.clear_current()
    assert TransferQueueRuntime.current() is None


def test_install_replacing_existing_warns(caplog):
    first = TransferQueueRuntime().install()
    with caplog.at_level("WARNING", logger="diffusionrl.distributed.transfer_queue.runtime"):
        second = TransferQueueRuntime().install()
    assert TransferQueueRuntime.current() is second
    assert any("replacing existing current runtime" in rec.message for rec in caplog.records)
    assert first is not second


def test_install_idempotent_for_same_instance(caplog):
    runtime = TransferQueueRuntime()
    runtime.install()
    with caplog.at_level("WARNING", logger="diffusionrl.distributed.transfer_queue.runtime"):
        runtime.install()
    # Re-installing the same instance should not warn.
    assert not any("replacing existing current runtime" in rec.message for rec in caplog.records)
    assert TransferQueueRuntime.current() is runtime


def test_is_enabled_tracks_client():
    runtime = TransferQueueRuntime()
    assert runtime.is_enabled() is False
    runtime.client = object()  # any non-None placeholder
    assert runtime.is_enabled() is True


def test_init_returns_none_when_transfer_queue_absent():
    cfg = OmegaConf.create({"run": {"seed": 0}})
    runtime = TransferQueueRuntime()
    assert runtime.init(cfg) is None
    assert runtime.backend is None
    assert runtime.controller is None
    assert runtime.client is None


def test_clear_partition_noop_when_disabled():
    # No client wired up — should silently no-op rather than raise.
    TransferQueueRuntime().clear_partition()
    TransferQueueRuntime().clear_partition("custom_partition")


def test_reset_zero_copy_buffer_free_noop_when_disabled():
    TransferQueueRuntime().reset_zero_copy_buffer_free()
