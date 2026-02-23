"""Checkpoint and weight-sync helpers for TrainingActor."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from diffusionrl.utils.weight_sync_checkpoint import (
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
)

logger = logging.getLogger(__name__)


class TrainingActorStateIOService:
    """Owns state_dict/checkpoint boundaries for training actors."""

    def get_weights(self, actor: Any) -> Dict[str, torch.Tensor]:
        """Get current model weights for syncing to inference actors."""
        was_offloaded = actor._is_offloaded
        if was_offloaded:
            actor.onload()

        try:
            if actor._use_lora:
                try:
                    if actor._use_fsdp:
                        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(actor.model, StateDictType.FULL_STATE_DICT, save_policy):
                            full_state_dict = actor.model.state_dict()

                        if actor.rank != 0:
                            return {}

                        lora_state = {
                            k: (v.cpu() if v.is_cuda else v)
                            for k, v in full_state_dict.items()
                            if "lora" in k.lower()
                        }
                        if lora_state:
                            return lora_state
                        logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")

                        full_state_dict = {k: v.cpu() if v.is_cuda else v for k, v in full_state_dict.items()}
                        return full_state_dict

                    try:
                        from peft.utils import get_peft_model_state_dict

                        base_model = actor.model
                        if hasattr(base_model, "module"):
                            base_model = base_model.module

                        adapter_names = []
                        if hasattr(base_model, "peft_config"):
                            adapter_names = list(base_model.peft_config.keys())
                        if not adapter_names:
                            adapter_names = [getattr(base_model, "active_adapter", "default")]

                        lora_state = {}
                        for adapter_name in adapter_names:
                            lora_state.update(
                                get_peft_model_state_dict(base_model, adapter_name=adapter_name)
                            )

                        if lora_state:
                            return {k: v.cpu() if v.is_cuda else v for k, v in lora_state.items()}
                    except Exception as e:
                        logger.warning("PEFT LoRA-only sync failed; falling back to key filter: %s", e)

                    local_state = actor.model.state_dict()
                    lora_state = {
                        k: (v.cpu() if v.is_cuda else v)
                        for k, v in local_state.items()
                        if "lora" in k.lower()
                    }
                    if lora_state:
                        return lora_state
                    logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")

                except Exception as e:
                    logger.warning("LoRA-only sync failed; falling back to full sync: %s", e)

            if actor._use_fsdp:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(actor.model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = actor.model.state_dict()

                if actor.rank != 0:
                    return {}

                state_dict = {k: v.cpu() if v.is_cuda else v for k, v in state_dict.items()}
            else:
                state_dict = {k: v.cpu() for k, v in actor.model.state_dict().items()}

            return state_dict
        finally:
            if was_offloaded:
                actor.offload()

    def export_weights_to_path(
        self,
        actor: Any,
        checkpoint_path: str,
        export_format: str = "state_dict",
    ) -> Optional[str]:
        """Export synchronized weights to a shared path."""
        state_dict = self.get_weights(actor)
        if actor.rank != 0:
            return None
        if export_format == "state_dict":
            return publish_checkpoint_atomic(state_dict, checkpoint_path)
        if export_format == "sglang_transformer_safetensors":
            return publish_sglang_transformer_checkpoint_atomic(
                state_dict,
                checkpoint_path,
                module_name="transformer",
            )
        raise ValueError(
            f"Unsupported export_format={export_format}. "
            "Expected one of: state_dict, sglang_transformer_safetensors"
        )

    def update_weights(self, actor: Any) -> None:
        """Broadcast weights from rank 0 to all other ranks."""
        if actor._use_fsdp:
            return

        for param in actor.model.parameters():
            dist.broadcast(param.data, src=0)

    def save_model(self, actor: Any, path: str) -> None:
        """Save model checkpoint (FSDP-safe collective)."""
        was_offloaded = actor._is_offloaded
        if was_offloaded:
            actor.onload()

        try:
            if actor._use_fsdp:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(actor.model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = actor.model.state_dict()

                if actor.rank != 0:
                    return
            else:
                if actor.rank != 0:
                    return
                state_dict = actor.model.state_dict()

            os.makedirs(path, exist_ok=True)

            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": actor.optimizer.state_dict(),
            }
            if actor.lr_scheduler is not None:
                checkpoint["scheduler_state_dict"] = actor.lr_scheduler.state_dict()

            torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
            logger.info("Checkpoint saved to %s", path)
        finally:
            if was_offloaded:
                actor.offload()

    def load_checkpoint(self, actor: Any, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=actor._device)

        if actor._use_fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(actor.model, StateDictType.FULL_STATE_DICT, load_policy):
                actor.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            actor.model.load_state_dict(checkpoint["model_state_dict"])

        actor.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if actor.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            actor.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info("Checkpoint loaded from %s", path)
