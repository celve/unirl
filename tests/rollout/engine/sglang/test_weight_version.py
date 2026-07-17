from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import ray
import torch

from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine


class _ReceiverWeightSync:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def update_weights_from_distributed(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _RemoteCall:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    def remote(self, role: str, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
        del role, method, args
        self.kwargs.append(kwargs)
        return object()


class _RolloutTarget:
    def __init__(self) -> None:
        self.call = _RemoteCall()


def test_sglang_version_advances_once_at_publication_boundary() -> None:
    engine = SGLangRolloutEngine.__new__(SGLangRolloutEngine)
    engine._weight_sync = _ReceiverWeightSync()
    engine._weight_version = 0
    payload = {
        "names": ["weight"],
        "dtypes": ["float32"],
        "shapes": [[1]],
        "group_name": "sync",
    }

    engine.update_weights_from_distributed(**payload, publication_complete=False)
    engine.update_weights_from_distributed(**payload, publication_complete=False)
    assert engine._weight_version == 0

    engine.update_weights_from_distributed(**payload, publication_complete=True)
    assert engine._weight_version == 1

    # An older direct caller sends one complete publication per call and omits
    # the new marker; retain that behavior for compatibility.
    engine.update_weights_from_distributed(**payload)
    assert engine._weight_version == 2


def test_nccl_sender_marks_only_the_final_bucket_complete(monkeypatch) -> None:
    target = _RolloutTarget()
    sync = NCCLWeightSync.__new__(NCCLWeightSync)
    sync.rank_info = SimpleNamespace(rank=0)
    sync._rollout_targets = [target]
    sync._rollout_role = "rollout"
    sync._group_name = "sync"
    sync._model_update_group = object()
    sync._flush_cache = False
    sync._track_prefix = ""
    sync.weight_version = 0
    sync._iter_buckets = lambda: iter(
        [
            ([("first", torch.ones(1))], False),
            ([("last", torch.ones(1))], True),
        ]
    )

    monkeypatch.setattr(ray, "get", lambda refs: refs)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)

    sync.sync()

    assert [call["publication_complete"] for call in target.call.kwargs] == [False, True]
    assert [call["flush_cache"] for call in target.call.kwargs] == [False, False]
    assert sync.weight_version == 1
