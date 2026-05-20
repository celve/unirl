"""Round-trip test for the lifted ``BucketedWeightSender`` / ``BucketedWeightReceiver``.

Covers two routes:

1. **CUDA IPC** — sender + receiver in two threads of one process; only
   exercised under ``--run-e2e`` because CUDA IPC needs an actual GPU
   plus the receiver living in a different process for the device-id
   rewrite to actually mean anything. Same-process is enough to exercise
   the metadata + buffer-view code paths.
2. **Shared-memory fallback** — ``use_shm=True``; CPU-only, runs by
   default in CI.

The shared-memory test pushes ~16 named tensors across a 2 MB bucket so
we exercise the multi-bucket flush path and assert names + shapes +
values round-trip exactly.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import List, Tuple

import pytest
import torch

from diffusionrl.rollout.engine.vllm_omni.weight_sync import (
    BucketedWeightReceiver,
    BucketedWeightSender,
)


def _build_named_tensors(num: int = 16, dim: int = 512) -> List[Tuple[str, torch.Tensor]]:
    """Build a state-dict-shaped list of (name, fp32 cpu tensor) pairs.

    16 × 512 × 512 × 4 B = 16 MB total; with 2 MB bucket → 8 buckets.
    """
    return [(f"layer.{i}.weight", torch.randn(dim, dim, dtype=torch.float32)) for i in range(num)]


def _start_sender(handle: str, weights, bucket_size_mb: int, use_shm: bool) -> threading.Thread:
    def run():
        sender = BucketedWeightSender(zmq_handle=handle, bucket_size_mb=bucket_size_mb, use_shm=use_shm)
        # One async wrapper for the sync test entry; sender's API is async.
        asyncio.run(sender.async_send_weights(weights))

    t = threading.Thread(target=run, name="bucketed-sender", daemon=True)
    t.start()
    return t


def test_bucketed_round_trip_shm() -> None:
    """Shared-memory bucketed transfer end-to-end on CPU."""
    handle = f"ipc:///tmp/diffrl-test-bucket-{uuid.uuid4().hex}.sock"
    weights = _build_named_tensors(num=16, dim=512)
    expected = {name: t.clone() for name, t in weights}

    sender_thread = _start_sender(handle, weights, bucket_size_mb=2, use_shm=True)

    received: dict[str, torch.Tensor] = {}

    def on_bucket(bucket: list) -> None:
        # Tensors are views into the shared buffer; clone before next bucket
        # overwrites.
        for name, tensor in bucket:
            received[name] = tensor.clone()

    receiver = BucketedWeightReceiver(
        zmq_handle=handle,
        device=torch.device("cpu"),
        use_shm=True,
    )
    receiver.receive_weights(on_bucket_received=on_bucket)
    sender_thread.join(timeout=30)
    assert not sender_thread.is_alive(), "sender thread did not finish"

    assert set(received.keys()) == set(expected.keys()), "names mismatch"
    for name, ref in expected.items():
        got = received[name]
        assert got.shape == ref.shape, f"{name} shape {got.shape} != {ref.shape}"
        assert got.dtype == ref.dtype, f"{name} dtype mismatch"
        assert torch.equal(got, ref), f"{name} value mismatch"


@pytest.mark.e2e
def test_bucketed_round_trip_cuda_ipc() -> None:
    """CUDA-IPC path; needs --run-e2e and an actual GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for IPC path")

    handle = f"ipc:///tmp/diffrl-test-bucket-{uuid.uuid4().hex}.sock"
    weights = [(f"layer.{i}.weight", torch.randn(256, 256, dtype=torch.float32, device="cuda")) for i in range(8)]
    expected = {name: t.detach().clone() for name, t in weights}

    sender_thread = _start_sender(handle, weights, bucket_size_mb=2, use_shm=False)

    received: dict[str, torch.Tensor] = {}

    def on_bucket(bucket: list) -> None:
        for name, tensor in bucket:
            received[name] = tensor.detach().clone()

    receiver = BucketedWeightReceiver(
        zmq_handle=handle,
        device=torch.device("cuda", torch.cuda.current_device()),
        use_shm=False,
    )
    receiver.receive_weights(on_bucket_received=on_bucket)
    sender_thread.join(timeout=30)
    assert not sender_thread.is_alive()

    assert set(received.keys()) == set(expected.keys())
    for name, ref in expected.items():
        got = received[name]
        assert torch.equal(got.cpu(), ref.cpu()), f"{name} value mismatch"
