"""Rollout engine base class for the ``RolloutReq``/``RolloutResp`` path.

Subclasses (``VLLMOmniRolloutEngine``, ``SGLangRolloutEngine``,
``TrainsideRolloutEngine``) take all runtime deps as ``__init__`` kwargs
and complete construction in one shot — no separate ``initialize(device)``
step. After ``__init__`` returns the engine is fully usable: model loaded,
worker subprocesses spawned, dist groups brought up. This matches the
actor flow where ``_setup_distributed_env`` runs before the engine is
built.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp


class BaseRolloutEngine(ABC):
    """New rollout engine ABC. One-shot construction; new types only.

    ``stage_ids`` on the weight-sync methods is the multi-stage
    discriminator for engines like vllm-omni HI3 (AR=0, DiT=1). Single-
    stage engines may ignore the kwarg.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def shutdown(self) -> None:
        """Release worker subprocesses and any other engine-owned resources."""

    def sleep(self) -> None:
        """Best-effort runtime offload. Default no-op."""

    def wake_up(self) -> None:
        """Restore runtime resources after ``sleep``. Default no-op."""

    @property
    def is_offloaded(self) -> bool:
        """Whether the engine has released its runtime resources."""
        return False

    def health_check(self) -> bool:
        """Return True iff the engine is ready to serve a generate call."""
        return True

    def get_memory_info(self) -> Dict[str, float]:
        """Per-engine GPU memory snapshot. Default reads CUDA totals."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run one rollout against the engine and return its typed response."""

    # ------------------------------------------------------------------
    # Weight sync — bucketed CUDA-IPC (verl-omni pattern)
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Receive a state dict over a per-rank ZMQ + CUDA-IPC channel.

        The trainer-side counterpart (``UpdateWeightFromIPC`` in
        ``diffusionrl.distributed.weight_sync.ipc``) sets up a
        ``BucketedWeightSender`` on the matching socket and pumps tensors
        bucket by bucket; the engine wires a ``BucketedWeightReceiver`` and
        forwards each bucket into the relevant worker's ``load_weights``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — NCCL broadcast
    # ------------------------------------------------------------------

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Bring up a trainer↔rollout NCCL process group on the engine side."""
        raise NotImplementedError

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Receive a bucket of weights via the established NCCL group."""
        raise NotImplementedError

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Tear down a previously-initialized NCCL update group."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — LoRA tensor bag
    # ------------------------------------------------------------------

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Load a LoRA adapter directly from in-memory tensors."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — SGLang-shape one-bag tensor payload
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Receive a state-dict packed as a single SGLang-shape payload per TP rank.

        The trainer-side counterpart is
        :class:`diffusionrl.distributed.weight_sync.tensor.UpdateWeightFromTensor`,
        which builds one ``{flattened_tensor, metadata}`` payload per dtype
        bucket via ``sglang.srt.weight_sync.tensor_bucket.FlattenedTensorBucket``,
        wraps in ``sglang.srt.utils.MultiprocessingSerializer.serialize(...,
        output_str=True)`` (base64-pickle), and replicates per TP rank.

        ``serialized_named_tensors[i]`` is consumed by the worker at TP rank ``i``.
        The IPC-bucketed transport (``update_weights_from_ipc``) is preferred for
        large models — this entry exists for protocol compat with the legacy
        SGLang-bound trainer handler.
        """
        raise NotImplementedError


__all__ = ["BaseRolloutEngine"]
