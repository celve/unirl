"""Checkpoint and weight-sync helpers for TrainingActor."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from diffusionrl.utils.weight_sync_checkpoint import (
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
)

logger = logging.getLogger(__name__)


class TrainingActorStateIOService:
    """Owns state_dict/checkpoint boundaries for training actors."""

    @staticmethod
    def _get_backend(actor: Any) -> Any:
        backend = getattr(actor, "_train_backend", None)
        if backend is None:
            raise RuntimeError(
                "Training backend is not initialized. "
                "Call TrainingActor.init() before state IO operations."
            )
        return backend

    def get_weights(self, actor: Any) -> Dict[str, torch.Tensor]:
        """Get current model weights for syncing to rollout actors."""
        was_offloaded = actor._is_offloaded
        if was_offloaded:
            actor.onload()

        try:
            backend = self._get_backend(actor)
            return backend.get_state_dict(
                actor,
                lora_only=bool(getattr(actor, "_use_lora", False)),
                rank0_only=True,
            )
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
        backend = self._get_backend(actor)
        backend.broadcast_parameters(actor)

    def save_model(self, actor: Any, path: str) -> None:
        """Save model checkpoint (backend-safe collective)."""
        was_offloaded = actor._is_offloaded
        if was_offloaded:
            actor.onload()

        try:
            backend = self._get_backend(actor)
            state_dict = backend.get_state_dict(actor, lora_only=False, rank0_only=True)
            if actor.rank != 0:
                return

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
        backend = self._get_backend(actor)
        backend.load_state_dict(actor, checkpoint["model_state_dict"])
        actor.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if actor.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            actor.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info("Checkpoint loaded from %s", path)
