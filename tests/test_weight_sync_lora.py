from __future__ import annotations

import types

import torch

from diffusionrl.distributed.weight_sync import base as base_mod
from diffusionrl.distributed.weight_sync import ipc as ipc_mod
from diffusionrl.distributed.weight_sync import nccl as nccl_mod
from diffusionrl.distributed.weight_sync import tensor as tensor_mod
from diffusionrl.distributed.weight_sync.base import BucketedUpdateWeight
from diffusionrl.ray.mixins.training_weight_sync import TrainingWeightSyncMixin
from diffusionrl.utils import peft_merge


def test_lora_tensors_for_vllm_uses_peft_envelope_and_weight_suffix(monkeypatch):
    monkeypatch.setattr(peft_merge, "_to_full_tensor", lambda tensor: tensor)

    class Model:
        def state_dict(self):
            return {
                "base_model.model.block.attn.lora_A.default.weight": torch.ones(2, 3),
                "base_model.model.block.attn.lora_B.default.weight": torch.ones(4, 2),
                "base_model.model.block.attn.base_layer.weight": torch.ones(4, 3),
            }

    tensors = peft_merge.lora_tensors_for_vllm(
        Model(),
        param_name_prefix="transformer.",
    )

    assert set(tensors) == {
        "base_model.model.transformer.block.attn.lora_A.weight",
        "base_model.model.transformer.block.attn.lora_B.weight",
    }
    assert all(not key.endswith(".alpha") for key in tensors)


def test_bucketed_base_sync_filters_lora_keys(monkeypatch):
    monkeypatch.setattr(
        base_mod,
        "raw_state_dict",
        lambda _model: iter(
            [
                ("block.weight", torch.ones(1)),
                ("block.lora_A", torch.ones(1)),
                ("block.lora_B", torch.ones(1)),
            ]
        ),
    )

    class CaptureBuckets(BucketedUpdateWeight):
        def connect_rollout_engines(self) -> None:
            return None

        def update_bucket_weights(self, named_tensors, weight_version=None, is_last_bucket: bool = False) -> None:
            self.captured = list(named_tensors)

    handler = CaptureBuckets(
        model=object(),
        rollout_runtime=object(),
        placement_cfg=object(),
        bucket_size=256,
        flush_cache=True,
        target_modules=("transformer",),
        param_name_prefix="transformer.",
    )
    handler.update_weights()

    assert [name for name, _ in handler.captured] == ["transformer.block.weight"]


def test_training_weight_sync_lora_uses_base_once_then_lora_every_call():
    class PeftConfig:
        def to_dict(self):
            return {"r": 4, "lora_alpha": 8, "target_modules": {"to_q", "to_v"}}

    class Model:
        peft_config = {"default": PeftConfig()}

    class Handler:
        def __init__(self) -> None:
            self.calls = []

        def update_weights(self, *, peft_config=None, base_sync_done=False) -> None:
            self.calls.append((peft_config, base_sync_done))

    class Host(TrainingWeightSyncMixin):
        def __init__(self) -> None:
            self.model = Model()
            self.rank = 0
            self._use_lora = True
            self._init_weight_sync_state()

    host = Host()
    handler = Handler()
    host._update_weight_handler = handler

    host.sync_weights_to_rollout()
    host.sync_weights_to_rollout()

    assert handler.calls[0] == (None, False)
    assert handler.calls[1][0]["target_modules"] == ["to_q", "to_v"]
    assert handler.calls[1][1] is True
    assert handler.calls[2][0]["target_modules"] == ["to_q", "to_v"]
    assert handler.calls[2][1] is True


def test_ipc_orphan_rank_drains_iterator_without_sender(monkeypatch):
    monkeypatch.setattr(ipc_mod.dist, "is_initialized", lambda: False)

    handler = object.__new__(ipc_mod.UpdateWeightFromIPC)
    handler._rollout_actors = [object()]
    handler._this_rank = 2
    handler._placement_cfg = types.SimpleNamespace(num_rollout_gpus_per_actor=2)
    handler._actor_num_stages = [2]
    handler._stage_ids = (0,)
    handler.weight_version = 0

    calls = []

    def iter_named_params(self, *, peft_config=None, base_sync_done=False):
        calls.append((peft_config, base_sync_done))
        yield "name", torch.ones(1)

    handler._iter_named_params = types.MethodType(iter_named_params, handler)

    handler.update_weights(peft_config={"r": 4}, base_sync_done=True)

    assert calls == [({"r": 4}, True)]
    assert handler.weight_version == 1


def test_nccl_lora_phase_materializes_on_non_source_rank(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nccl_mod,
        "lora_tensors_for_vllm",
        lambda model, *, param_name_prefix: calls.append((model, param_name_prefix)) or {},
    )

    handler = object.__new__(nccl_mod.UpdateWeightFromDistributed)
    handler.model = object()
    handler._param_name_prefix = "transformer."
    handler._is_src_rank = False

    handler.update_weights(peft_config={"r": 4}, base_sync_done=True)

    assert calls == [(handler.model, "transformer.")]


def test_tensor_lora_phase_materializes_on_non_sender_rank(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tensor_mod,
        "lora_tensors_for_vllm",
        lambda model, *, param_name_prefix: calls.append((model, param_name_prefix)) or {},
    )

    handler = object.__new__(tensor_mod.UpdateWeightFromTensor)
    handler.model = object()
    handler._param_name_prefix = "transformer."
    handler._ipc_engine = None
    handler._ipc_gather_src = None

    handler.update_weights(peft_config={"r": 4}, base_sync_done=True)

    assert calls == [(handler.model, "transformer.")]
