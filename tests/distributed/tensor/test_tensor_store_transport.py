from __future__ import annotations

import pytest
import torch

ray = pytest.importorskip("ray")

from diffusionrl.distributed.tensor.backend.tensor_store.store import TensorStore  # noqa: E402
from diffusionrl.distributed.tensor.backend.tensor_store.transport import (  # noqa: E402
    TensorStoreTransport,
)
from diffusionrl.distributed.tensor.transport import (  # noqa: E402
    TensorMeta,
    TensorTransportRuntime,
)

from .conftest import TensorBatch  # noqa: E402

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def store():
    return TensorStore(worker_id="test-worker", device="cuda:0")


@pytest.fixture
def transport(store):
    return TensorStoreTransport(store)


# ---------------------------------------------------------------------------
# put / get
# ---------------------------------------------------------------------------


def test_put_returns_single_ref(transport):
    t = torch.randn(4, 3, device="cuda:0")
    ref = transport.put(t)
    assert ref is not None


def test_put_batch_returns_dict_of_tensor_meta(transport):
    tensors = {
        "a": torch.randn(3, 2, device="cuda:0"),
        "b": torch.randn(3, 5, device="cuda:0"),
    }
    metas = transport.put_batch(tensors)
    assert set(metas.keys()) == {"a", "b"}
    assert isinstance(metas["a"], TensorMeta)
    assert metas["a"].sizes == [3]
    assert metas["a"].batch_size == 3


def test_get_reconstructs_full_tensor(transport):
    t = torch.randn(4, 3, device="cuda:0")
    ref = transport.put(t)
    restored = transport.get([ref])
    torch.testing.assert_close(restored, t)


def test_get_multi_handle_cats(transport):
    a = torch.randn(2, 3, device="cuda:0")
    b = torch.randn(3, 3, device="cuda:0")
    ref_a = transport.put(a)
    ref_b = transport.put(b)
    result = transport.get([ref_a, ref_b])
    expected = torch.cat([a, b], dim=0)
    torch.testing.assert_close(result, expected)


def test_get_empty_refs_raises(transport):
    with pytest.raises(ValueError):
        transport.get([])


# ---------------------------------------------------------------------------
# end-to-end dehydrate / hydrate
# ---------------------------------------------------------------------------


def test_end_to_end_dehydrate_hydrate(transport):
    TensorTransportRuntime.install(transport)

    original = torch.randn(4, 3, device="cuda:0")
    obj = TensorBatch(data=original.clone())

    transport.dehydrate(obj)
    assert isinstance(obj.data, TensorMeta)
    assert obj.data.sizes == [4]

    transport.hydrate(obj)
    assert isinstance(obj.data, torch.Tensor)
    torch.testing.assert_close(obj.data, original)


def test_concat_dehydrated_then_hydrate(transport):
    TensorTransportRuntime.install(transport)

    a_data = torch.randn(2, 3, device="cuda:0")
    b_data = torch.randn(3, 3, device="cuda:0")
    a = TensorBatch(data=a_data.clone())
    b = TensorBatch(data=b_data.clone())

    transport.dehydrate(a)
    transport.dehydrate(b)
    merged = TensorBatch.concat([a, b])

    assert isinstance(merged.data, TensorMeta)
    assert merged.data.batch_size == 5
    assert merged.data.refs == a.data.refs + b.data.refs

    transport.hydrate(merged)
    expected = torch.cat([a_data, b_data], dim=0)
    torch.testing.assert_close(merged.data, expected)
