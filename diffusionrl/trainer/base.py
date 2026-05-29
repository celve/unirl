from diffusionrl.distributed.group.device_pool import DevicePool


class BaseTrainer:
    """Owns a DevicePool. Subclasses use ``placement(self.pool, ...)`` to
    instantiate their ``Remote`` roles inside ``__init__`` / ``setup``.
    """

    def __init__(
        self,
        *,
        num_devices: int,
        devices_per_node: int = 8,
        workers_per_device: int = 2,
    ) -> None:
        self.pool = DevicePool(
            num_devices=num_devices,
            devices_per_node=devices_per_node,
            workers_per_device=workers_per_device,
        )
        self.pool.setup()
