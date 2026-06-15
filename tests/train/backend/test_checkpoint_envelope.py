"""Checkpoint envelope round-trip for BaseFSDP2Backend (CPU + gloo, pytest).

Locks the unified save/load envelope and the VeOmni drift fix: ``step``/``mode``
are accepted, the saved step round-trips, optimizer + step-count restore,
``lora_config`` is recorded, and ``mode='adapter'`` persists only the LoRA keys.

A minimal concrete backend over a plain (unsharded) ``nn.Module`` exercises the
shared envelope without FSDP/VeOmni sharding (which needs GPUs); its plain
``state_dict()`` optimizer hooks mirror VeOmni's mechanism. A single-rank gloo
group is required because ``load_model_state_dict`` broadcasts from rank 0.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ray")  # base_backend -> Remote -> handle imports ray

import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from unirl.train.backend.base_backend import BaseFSDP2Backend  # noqa: E402
from unirl.train.backend.sharded_state import move_optimizer_state  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _gloo_pg():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    yield
    if created:
        dist.destroy_process_group()


class _PlainBackend(BaseFSDP2Backend):
    """Concrete backend with a plain module + VeOmni-style plain optimizer hooks."""

    def __init__(self, model, *, lora_meta=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.ema = None
        self._optimizer_step_count = 7
        self._eval_ema_active = False
        self._lora_meta = lora_meta
        self._rollout_adapter_name = "default"
        self._defer_grad_sync = False
        self._grad_sync_enabled = True
        self._rank = 0
        self._device = torch.device("cpu")

    def _clip_grad_norm(self, max_grad_norm):
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

    def _gather_optimizer_state(self):
        return self.optimizer.state_dict()

    def _load_optimizer_state(self, sd):
        self.optimizer.load_state_dict(sd)
        move_optimizer_state(self.optimizer, self._device)

    def _onload_model(self):
        pass

    def _offload_model(self):
        pass


class _Loraish(nn.Module):
    """A module whose param FQNs include lora_A / lora_B for the adapter filter."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Parameter(torch.randn(2, 4))
        self.lora_B = nn.Parameter(torch.randn(4, 2))

    def forward(self, x):
        return self.base(x)


def _take_step(backend):
    backend.model(torch.randn(2, 4)).sum().backward()
    backend.optimizer.step()


def test_full_checkpoint_roundtrip(tmp_path):
    src = _PlainBackend(nn.Linear(4, 4))
    _take_step(src)
    ref = {k: v.clone() for k, v in src.model.state_dict().items()}

    src.save(str(tmp_path), step=5, mode="full")

    dst = _PlainBackend(nn.Linear(4, 4))
    returned = dst.load(str(tmp_path))

    assert returned == 5
    assert dst._optimizer_step_count == 7
    for k, v in dst.model.state_dict().items():
        assert torch.allclose(v, ref[k]), k


def test_checkpoint_records_lora_config(tmp_path):
    meta = {"rank": 8, "alpha": 16, "target_modules": ["q_proj"]}
    src = _PlainBackend(nn.Linear(4, 4), lora_meta=meta)
    src.save(str(tmp_path), step=1, mode="full")
    ck = torch.load(os.path.join(str(tmp_path), "checkpoint.pt"))
    assert ck["lora_config"] == meta
    assert ck["save_mode"] == "full"


def test_adapter_mode_keeps_only_lora_keys(tmp_path):
    src = _PlainBackend(_Loraish(), lora_meta={"rank": 2, "alpha": 4, "target_modules": []})
    src.save(str(tmp_path), step=3, mode="adapter")

    ck = torch.load(os.path.join(str(tmp_path), "checkpoint.pt"))
    assert set(ck["policy_state_dict"].keys()) == {"lora_A", "lora_B"}
    assert ck["save_mode"] == "adapter"

    # Adapter checkpoints load non-strict (the frozen base keys are absent).
    dst = _PlainBackend(_Loraish())
    assert dst.load(str(tmp_path)) == 3


def test_unknown_mode_rejected(tmp_path):
    src = _PlainBackend(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="unknown mode"):
        src.save(str(tmp_path), mode="bogus")


def test_adapter_mode_without_lora_rejected(tmp_path):
    src = _PlainBackend(nn.Linear(4, 4))  # no lora params
    with pytest.raises(RuntimeError, match="no LoRA params"):
        src.save(str(tmp_path), mode="adapter")
