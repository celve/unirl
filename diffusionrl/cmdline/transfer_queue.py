"""Built-in transfer-queue cmdline adaptation helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from diffusionrl.config.spec import (
    MooncakeBackendConfig,
    MooncakeZeroCopyConfig,
    SimpleStorageBackendConfig,
    TransferQueueGlobalConfig,
)


def build_transfer_queue_config_from_args(args: Any):
    transfer_queue_args = args.transfer_queue
    global_config = TransferQueueGlobalConfig(
        enable=transfer_queue_args.enable,
        storage_backend=transfer_queue_args.storage_backend,
    )

    if transfer_queue_args.storage_backend == "AsyncSimpleStorageManager":
        assert hasattr(transfer_queue_args, "simple_storage_backend_config")
        backend_args = transfer_queue_args.simple_storage_backend_config
        backend_config = SimpleStorageBackendConfig(
            train_micro_batch_size=backend_args.train_micro_batch_size,
            num_global_batch=backend_args.num_global_batch,
            item_size_one_batch=backend_args.item_size_one_batch,
            num_data_storage_units_per_node=backend_args.num_data_storage_units_per_node,
            storage_units_nnodes=backend_args.storage_units_nnodes,
            rollout_n=backend_args.rollout_n,
        )
    elif transfer_queue_args.storage_backend == "MooncakeStorageManager":
        assert hasattr(transfer_queue_args, "mooncake_backend_config")
        backend_args = transfer_queue_args.mooncake_backend_config
        zero_copy_config = MooncakeZeroCopyConfig(
            enable=backend_args.zero_copy_enable,
            tensor_buffer_size_gb=backend_args.zero_copy_tensor_buffer_size_gb,
            bytes_buffer_size_gb=backend_args.zero_copy_bytes_buffer_size_gb,
            manager_merge_to_tensordict=backend_args.zero_copy_manager_merge_to_tensordict,
        )
        backend_config = MooncakeBackendConfig(
            metadata_server=backend_args.metadata_server,
            master_server_address=backend_args.master_server_address,
            global_segment_size_gb=backend_args.global_segment_size_gb,
            local_buffer_size_gb=backend_args.local_buffer_size_gb,
            client_name=backend_args.client_name,
            protocol=backend_args.protocol,
            device_name=backend_args.device_name,
            zero_copy=zero_copy_config,
        )
    else:
        raise NotImplementedError(f"tq backend: {transfer_queue_args.storage_backend} is not implement!")

    # NOTE transfer queue backend requires dictionary type config parameters
    return asdict(global_config), asdict(backend_config)


__all__ = [
    "build_transfer_queue_config_from_args",
]
