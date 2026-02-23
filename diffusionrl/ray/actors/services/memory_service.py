"""Memory/device movement helpers for TrainingActor."""

from __future__ import annotations

import logging
from typing import Any, Union

import torch
import torch.nn as nn

from diffusionrl.utils import clear_memory

logger = logging.getLogger(__name__)


class TrainingActorMemoryService:
    """Owns offload/onload transitions for training actors."""

    def _safe_to_device(self, component: nn.Module, device: Union[str, torch.device], name: str) -> None:
        if component is None or not hasattr(component, "to"):
            return
        try:
            component.to(device)
        except Exception as e:
            logger.warning("Could not move %s to %s: %s", name, device, e)

    def move_aux_components(self, actor: Any, device: Union[str, torch.device], include_transformer: bool) -> None:
        # Direct refs
        if include_transformer and actor.model is not None:
            self._safe_to_device(actor.model, device, "model")
        if actor.text_encoder is not None:
            self._safe_to_device(actor.text_encoder, device, "text_encoder")
        if actor.vae is not None:
            self._safe_to_device(actor.vae, device, "vae")

        # Model bundle refs
        if actor.model_bundle is not None:
            for attr_name, component in actor.model_bundle.iter_offloadable_modules(
                include_transformer=include_transformer
            ):
                if not include_transformer and component is actor.model:
                    continue
                self._safe_to_device(component, device, f"model_bundle.{attr_name}")

        # Sampler refs (may hold encoders)
        if actor._sampler is not None:
            for attr_name, component in actor._actor_sampling_executor.iter_reflection_modules(
                actor._sampler,
                include_transformer=include_transformer,
            ):
                if not include_transformer and component is actor.model:
                    continue
                self._safe_to_device(component, device, f"sampler.{attr_name}")

    def offload(self, actor: Any) -> None:
        """Offload model and optimizer to CPU."""
        if getattr(actor, "_fsdp_cpu_offload", False):
            self.move_aux_components(actor, "cpu", include_transformer=False)
            actor._is_offloaded = True
            clear_memory()
            logger.info("Rank %s: FSDP CPU offload mode - just clearing cache", actor.rank)
            return

        actor._is_offloaded = True
        self.move_aux_components(actor, "cpu", include_transformer=False)

        if actor.model is not None:
            actor.model.to("cpu")

        if actor.optimizer is not None:
            for state in actor.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        clear_memory()
        logger.info("Rank %s: Model and optimizer offloaded to CPU", actor.rank)
        actor._log_gpu_state("training_offload")

    def onload(self, actor: Any) -> None:
        """Load model and optimizer back to GPU."""
        if getattr(actor, "_fsdp_cpu_offload", False):
            if actor._device is not None:
                self.move_aux_components(actor, actor._device, include_transformer=False)
            actor._is_offloaded = False
            logger.info("Rank %s: FSDP CPU offload mode - skipping manual onload", actor.rank)
            return

        if actor.model is not None:
            actor.model.to(actor._device)

        if actor._device is not None:
            self.move_aux_components(actor, actor._device, include_transformer=False)

        if actor.optimizer is not None:
            for state in actor.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(actor._device)

        actor._is_offloaded = False
        logger.info("Rank %s: Model and optimizer loaded to GPU", actor.rank)
        actor._log_gpu_state("training_onload")

    def clear_memory(self, actor: Any) -> None:
        """Clear GPU cache without full offload."""
        torch.cuda.empty_cache()
        logger.debug("Rank %s: GPU cache cleared", actor.rank)

    def onload_post_update(self, actor: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
