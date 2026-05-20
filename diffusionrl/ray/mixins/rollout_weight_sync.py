"""Rollout-side weight synchronization mixin (weight receiver)."""

import logging
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.distributed.weight_sync_checkpoint import wait_for_published_checkpoint
from diffusionrl.rollout.engine.base import BaseRolloutEngine

logger = logging.getLogger(__name__)


class RolloutWeightSyncMixin:
    """Mixin that provides weight-sync methods for rollout actors.

    Host class must set these instance attributes before calling any mixin method:
        engine: Optional[BaseRolloutEngine]
        rank: int
    """

    engine: Optional[BaseRolloutEngine]
    rank: int

    def _prepare_engine_for_weight_update(self) -> None:
        """Ensure engine is active before updating weights."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self.engine.wake_up()

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Update model weights from a shared checkpoint path."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return {"rank": int(self.rank), "checksum": None}

        wait_for_published_checkpoint(checkpoint_path)
        self._prepare_engine_for_weight_update()
        if hasattr(self.engine, "update_weights_from_path"):
            self.engine.update_weights_from_path(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.engine.update_weights(state_dict)
        logger.info("Rank %s: Weights updated from path %s", self.rank, checkpoint_path)
        checksum = None
        get_checksum_fn = getattr(self.engine, "get_last_weight_checksum", None)
        if callable(get_checksum_fn):
            try:
                raw_checksum = get_checksum_fn()
                if isinstance(raw_checksum, dict) and raw_checksum:
                    checksum = {str(k): str(v) for k, v in raw_checksum.items()}
            except Exception as exc:
                logger.warning(
                    "Rank %s: failed to query engine checksum after update: %s",
                    self.rank,
                    exc,
                )
        return {"rank": int(self.rank), "checksum": checksum}

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Update weights using serialized tensor payload."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self._prepare_engine_for_weight_update()
        self.engine.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: dict,
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Set LoRA adapter weights on rollout engines from in-memory tensors.

        ``peft_config`` is optional and only consumed by engines that
        support tensor-bag LoRA loading (vllm-omni via the ``VLLMOmniHijack``
        + ``OmniTensorLoRARequest`` path). Engines that ignore it (SGLang)
        accept the call as before via ``**kwargs``-tolerant dispatch.
        """
        if self.engine is None:
            logger.warning("No engine to set LoRA tensors")
            return
        try:
            self.engine.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)
        except TypeError:
            # Legacy engine signature without peft_config — fall back.
            self.engine.set_lora_from_tensors(adapter_name, lora_tensors)

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
        """Initialize custom distributed weight-update group in engine workers.

        ``stage_ids`` (optional) scopes the call to specific stages of a
        multi-stage engine (vllm-omni HI3). Engines that ignore it (SGLang)
        accept the call as before via the ``TypeError`` fall-back.
        """
        if self.engine is None:
            logger.warning("No engine to initialize weight update group")
            return
        self._prepare_engine_for_weight_update()
        try:
            self.engine.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
                stage_ids=stage_ids,
            )
        except TypeError:
            self.engine.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Destroy custom distributed weight-update group in engine workers."""
        if self.engine is None:
            logger.warning("No engine to destroy weight update group")
            return
        try:
            self.engine.destroy_weights_update_group(group_name=group_name, stage_ids=stage_ids)
        except TypeError:
            self.engine.destroy_weights_update_group(group_name=group_name)

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
        """Receive weights from custom distributed broadcast group."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self._prepare_engine_for_weight_update()
        try:
            self.engine.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                target_modules=target_modules,
                flush_cache=flush_cache,
                stage_ids=stage_ids,
            )
        except TypeError:
            self.engine.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                target_modules=target_modules,
                flush_cache=flush_cache,
            )

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Spawn bucketed-IPC receivers on the rollout engine workers.

        Trainer-side counterpart in ``diffusionrl.distributed.weight_sync.ipc``
        opens matching ``BucketedWeightSender`` instances on the same per-rank
        ZMQ sockets and pumps the state dict bucket-by-bucket. This method
        returns immediately after the receivers are listening; the actual
        bucket pump runs synchronously inside the engine's ``collective_rpc``
        fan-out (one call per stage in ``stage_ids``).
        """
        if self.engine is None:
            logger.warning("No engine to update weights via IPC")
            return
        self._prepare_engine_for_weight_update()
        update_fn = getattr(self.engine, "update_weights_from_ipc", None)
        if update_fn is None:
            raise NotImplementedError(f"engine {type(self.engine).__name__} does not implement update_weights_from_ipc")
        update_fn(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            stage_ids=stage_ids,
        )
