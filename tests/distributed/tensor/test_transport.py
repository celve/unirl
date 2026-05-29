from __future__ import annotations

import logging

import pytest
import torch

from diffusionrl.distributed.tensor.transport import (
    TensorMeta,
    TensorTransportRuntime,
)

# ---------------------------------------------------------------------------
# TensorMeta — per-handle refs
# ---------------------------------------------------------------------------


def test_tensor_meta_batch_size():
    m = TensorMeta(refs=["h0"], sizes=[5])
    assert m.batch_size == 5


def test_tensor_meta_batch_size_multi_handle():
    m = TensorMeta(refs=["h0", "h1"], sizes=[2, 3])
    assert m.batch_size == 5


def test_tensor_meta_len():
    m = TensorMeta(refs=["h0", "h1"], sizes=[100, 200])
    assert len(m) == 300


def test_tensor_meta_concat():
    a = TensorMeta(refs=["h0"], sizes=[2], shape=(2, 4), dtype=torch.float32, device="cpu")
    b = TensorMeta(refs=["h1"], sizes=[3], shape=(3, 4), dtype=torch.float32, device="cpu")
    merged = TensorMeta.concat([a, b])
    assert merged.refs == ["h0", "h1"]
    assert merged.sizes == [2, 3]
    assert merged.batch_size == 5
    assert merged.shape == (5, 4)


def test_tensor_meta_concat_multi_handle():
    a = TensorMeta(refs=["h0", "h1"], sizes=[2, 3], shape=(5, 4))
    b = TensorMeta(refs=["h2"], sizes=[1], shape=(1, 4))
    merged = TensorMeta.concat([a, b])
    assert merged.refs == ["h0", "h1", "h2"]
    assert merged.sizes == [2, 3, 1]
    assert merged.batch_size == 6
    assert merged.shape == (6, 4)


def test_tensor_meta_select_raises():
    m = TensorMeta(refs=["h0"], sizes=[5])
    with pytest.raises(NotImplementedError, match="hydrate first"):
        m.select(torch.tensor([0, 2]))


def test_tensor_meta_slice_raises():
    m = TensorMeta(refs=["h0"], sizes=[5])
    with pytest.raises(NotImplementedError, match="hydrate first"):
        m.slice(0, 2)


# ---------------------------------------------------------------------------
# TensorMeta compute proxy
# ---------------------------------------------------------------------------


def test_transform_requires_runtime():
    m = TensorMeta(refs=["r0"], sizes=[1])
    with pytest.raises(RuntimeError, match="No TensorTransport backend"):
        m.transform(lambda t: t)


def test_local_requires_runtime():
    m = TensorMeta(refs=["r0"], sizes=[1])
    with pytest.raises(RuntimeError, match="No TensorTransport backend"):
        m.local()


def test_transform_with_mock_backend(in_memory_transport):
    TensorTransportRuntime.install(in_memory_transport)
    t = torch.arange(6.0).reshape(3, 2)
    ref = in_memory_transport.put(t)
    m = TensorMeta(refs=[ref], sizes=[3], shape=(3, 2), dtype=torch.float32, device="cpu")

    result = m.transform(lambda x: x * 2)

    assert result.shape == (3, 2)
    recovered = in_memory_transport.get(result.refs)
    torch.testing.assert_close(recovered, t * 2)


def test_retain_grad_sets_flag():
    m = TensorMeta(refs=["r0"], sizes=[1])
    assert m.retain_grad_flag is False
    ret = m.retain_grad()
    assert ret is m
    assert m.retain_grad_flag is True


# ---------------------------------------------------------------------------
# TensorTransportRuntime
# ---------------------------------------------------------------------------


def test_current_starts_none():
    assert TensorTransportRuntime.current() is None


def test_install_and_current(in_memory_transport):
    TensorTransportRuntime.install(in_memory_transport)
    assert TensorTransportRuntime.current() is in_memory_transport


def test_install_replacing_warns(in_memory_transport, caplog):
    from tests.distributed.tensor.conftest import InMemoryTransport

    TensorTransportRuntime.install(in_memory_transport)
    other = InMemoryTransport()
    with caplog.at_level(logging.WARNING):
        TensorTransportRuntime.install(other)
    assert "replacing" in caplog.text.lower()
    assert TensorTransportRuntime.current() is other


def test_install_same_is_idempotent(in_memory_transport, caplog):
    TensorTransportRuntime.install(in_memory_transport)
    with caplog.at_level(logging.WARNING):
        TensorTransportRuntime.install(in_memory_transport)
    assert "replacing" not in caplog.text.lower()


def test_clear_current(in_memory_transport):
    TensorTransportRuntime.install(in_memory_transport)
    TensorTransportRuntime.clear_current()
    assert TensorTransportRuntime.current() is None


# ---------------------------------------------------------------------------
# TensorTransport base defaults
# ---------------------------------------------------------------------------


def test_put_batch_default_iterates(in_memory_transport):
    tensors = {
        "a": torch.randn(2, 3),
        "b": torch.randn(2, 4),
    }
    metas = in_memory_transport.put_batch(tensors)
    assert set(metas.keys()) == {"a", "b"}
    assert isinstance(metas["a"], TensorMeta)
    assert len(metas["a"].refs) == 1
    assert metas["a"].sizes == [2]
    assert metas["a"].shape == (2, 3)
    assert metas["b"].shape == (2, 4)


def test_get_batch_default_iterates(in_memory_transport):
    tensors = {
        "x": torch.randn(3, 2),
        "y": torch.randn(3, 5),
    }
    metas = in_memory_transport.put_batch(tensors)
    recovered = in_memory_transport.get_batch(metas)
    assert set(recovered.keys()) == {"x", "y"}
    torch.testing.assert_close(recovered["x"], tensors["x"])
    torch.testing.assert_close(recovered["y"], tensors["y"])


def test_transform_default_roundtrips(in_memory_transport):
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ref = in_memory_transport.put(t)
    m = TensorMeta(refs=[ref], sizes=[2], shape=(2, 2), dtype=torch.float32, device="cpu")

    result = in_memory_transport.transform(m, lambda x: x + 10)

    recovered = in_memory_transport.get(result.refs)
    torch.testing.assert_close(recovered, t + 10)
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Concat then hydrate (the motivating feature)
# ---------------------------------------------------------------------------


def test_concat_then_hydrate_produces_catted_tensor(in_memory_transport):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    ref_a = in_memory_transport.put(a)
    ref_b = in_memory_transport.put(b)
    meta_a = TensorMeta(refs=[ref_a], sizes=[2], shape=(2, 2))
    meta_b = TensorMeta(refs=[ref_b], sizes=[3], shape=(3, 2))

    merged = TensorMeta.concat([meta_a, meta_b])
    result = in_memory_transport.get(merged.refs)

    expected = torch.cat([a, b], dim=0)
    torch.testing.assert_close(result, expected)
