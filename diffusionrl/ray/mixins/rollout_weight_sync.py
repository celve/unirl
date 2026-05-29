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

    def _prepare_engine_for_weight_update(self, *, track_prefix: str = "") -> None:
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self.engine.onload_weights(track_prefix=track_prefix)

    @staticmethod
    def _infer_single_track_prefix(tensors: dict) -> str:
        prefixes = set()
        for name in tensors:
            head, sep, _ = str(name).partition(".")
            if not sep:
                return ""
            prefixes.add(head)
            if len(prefixes) > 1:
                return ""
        return next(iter(prefixes), "")

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
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
        track_prefix: str = "",
    ) -> None:
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self._prepare_engine_for_weight_update(track_prefix=track_prefix)
        self.engine.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
            track_prefix=track_prefix,
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: dict,
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        if self.engine is None:
            logger.warning("No engine to set LoRA tensors")
            return
        self._prepare_engine_for_weight_update(track_prefix=self._infer_single_track_prefix(lora_tensors))
        self.engine.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        if self.engine is None:
            logger.warning("No engine to initialize weight update group")
            return
        self._prepare_engine_for_weight_update()
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
    ) -> None:
        if self.engine is None:
            logger.warning("No engine to destroy weight update group")
            return
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
    ) -> None:
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self._prepare_engine_for_weight_update()
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
    ) -> None:
        if self.engine is None:
            logger.warning("No engine to update weights via IPC")
            return
        self._prepare_engine_for_weight_update()
        self.engine.update_weights_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
        )
