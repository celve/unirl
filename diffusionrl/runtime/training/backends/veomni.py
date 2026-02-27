"""Built-in VeOmni backend.

This backend now uses the native VeOmni integration path instead of delegating
to diffusionRL's FSDP backend.
"""

from __future__ import annotations

from .base import TrainBackendCapabilities
from .veomni_native import VeOmniNativeTrainBackend


class VeOmniTrainBackend(VeOmniNativeTrainBackend):
    """VeOmni-native backend wired to built-in ``train_backend=veomni``."""

    BACKEND_NAME = "veomni"

    @classmethod
    def declared_capabilities(cls) -> TrainBackendCapabilities:
        caps = super().declared_capabilities()
        return TrainBackendCapabilities(
            name=cls.BACKEND_NAME,
            distributed_backend=caps.distributed_backend,
            supports_training_actor_sampling=caps.supports_training_actor_sampling,
            buffer_partition_mode=caps.buffer_partition_mode,
            supports_state_dict_export=caps.supports_state_dict_export,
            supports_custom_actor_class=caps.supports_custom_actor_class,
            supports_custom_optimizer=caps.supports_custom_optimizer,
            supports_custom_scheduler=caps.supports_custom_scheduler,
            supports_custom_train_step=caps.supports_custom_train_step,
            supports_backend_managed_offload=caps.supports_backend_managed_offload,
            preferred_weight_sync_mode=caps.preferred_weight_sync_mode,
            preferred_weight_export_format=caps.preferred_weight_export_format,
            supported_weight_export_formats=caps.supported_weight_export_formats,
            notes=(
                "Built-in VeOmni backend. Uses VeOmni native APIs for model parallelization, "
                "optimizer/lr scheduler construction, and EP-aware grad clipping."
            ),
        )


__all__ = ["VeOmniTrainBackend"]
