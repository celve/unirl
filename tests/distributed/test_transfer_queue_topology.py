"""Unit tests for the RDMA HCA discovery helper."""

from __future__ import annotations

import pytest

from diffusionrl.distributed.tensor.backend.transfer_queue import topology
from diffusionrl.distributed.tensor.backend.transfer_queue.topology import list_rdma_bonds


def _populate_sysfs(tmp_path, names):
    """Create empty subdirs under tmp_path mimicking /sys/class/infiniband/<name>."""
    for n in names:
        (tmp_path / n).mkdir()


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """list_rdma_bonds is lru_cache-wrapped; reset between tests."""
    list_rdma_bonds.cache_clear()
    yield
    list_rdma_bonds.cache_clear()


def test_eight_bonds_returns_sorted_list(monkeypatch, tmp_path):
    names = [f"mlx5_bond_{i}" for i in range(1, 9)]
    _populate_sysfs(tmp_path, names)
    monkeypatch.setattr(topology, "_IB_CLASS_DIR", str(tmp_path))

    assert list_rdma_bonds() == sorted(names)


def test_bonds_preferred_over_individual_ports(monkeypatch, tmp_path):
    _populate_sysfs(tmp_path, ["mlx5_0", "mlx5_1", "mlx5_bond_1", "mlx5_bond_3"])
    monkeypatch.setattr(topology, "_IB_CLASS_DIR", str(tmp_path))

    # Only the bonds; individual ports filtered out.
    assert list_rdma_bonds() == ["mlx5_bond_1", "mlx5_bond_3"]


def test_no_bonds_falls_back_to_all_ib_devices(monkeypatch, tmp_path):
    _populate_sysfs(tmp_path, ["mlx5_0", "mlx5_1", "mlx5_2"])
    monkeypatch.setattr(topology, "_IB_CLASS_DIR", str(tmp_path))

    assert list_rdma_bonds() == ["mlx5_0", "mlx5_1", "mlx5_2"]


def test_missing_sysfs_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(topology, "_IB_CLASS_DIR", str(tmp_path / "does_not_exist"))
    assert list_rdma_bonds() == []


def test_empty_sysfs_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(topology, "_IB_CLASS_DIR", str(tmp_path))
    assert list_rdma_bonds() == []
