"""FSDP2 model wrapping: per-block ``fully_shard``, HSDP mesh, optional
activation checkpointing / ``torch.compile``.

Runs in the backend constructor after structural injection
(``unirl.train.lora`` / ``unirl.train.ema``) and before materialize.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


def fsdp_wrap(
    model: nn.Module,
    *,
    block_class_names: Tuple[str, ...],
    param_dtype: str = "bf16",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    fsdp_mode: str = "full",
    reshard_after_forward: bool = True,
    activation_checkpointing: bool = False,
    use_torch_compile: bool = False,
) -> None:
    """Apply FSDP2 wrapping to the model.  No handle returned — DTensors
    ARE the handle.  Ported from FSDPPolicy._wrap_model.

    ``block_class_names`` lists the model block classes to ``fully_shard``
    (the backend forwards it from config).
    """
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )

    target_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")

    fsdp_kwargs: Dict[str, object] = {
        "reshard_after_forward": bool(reshard_after_forward),
    }
    if mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=target_dtype,
            reduce_dtype=torch.float32,
        )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    mesh = _create_device_mesh(fsdp_mode)
    if mesh is not None:
        fsdp_kwargs["mesh"] = mesh

    block_instances = _enumerate_block_instances(model, block_class_names)

    casts = 0
    for layer in block_instances:
        for p in layer.parameters(recurse=True):
            if p.dtype.is_floating_point and p.dtype != target_dtype:
                p.data = p.data.to(target_dtype)
                casts += 1

    for layer in block_instances:
        fully_shard(layer, **fsdp_kwargs)

    if activation_checkpointing:
        from torch.utils import checkpoint as _ckpt

        def _make_ckpt_forward(orig_fwd: object) -> object:
            def wrapped(*args: object, **kwargs: object) -> object:
                def fn(*a: object) -> object:
                    return orig_fwd(*a, **kwargs)

                return _ckpt.checkpoint(fn, *args, use_reentrant=False)

            return wrapped

        for layer in block_instances:
            layer.forward = _make_ckpt_forward(layer.forward)

    if use_torch_compile:
        for layer in block_instances:
            layer.forward = torch.compile(layer.forward)

    if _current_rank() == 0:
        logger.info(
            "fsdp_wrap: wrapped %d block(s) of class %r "
            "(%s, cpu_offload=%s, mixed_precision=%s, reshard=%s, "
            "ac=%s, compile=%s, dtype_casts=%d)",
            len(block_instances),
            tuple(block_class_names),
            "HSDP" if mesh is not None else "FSDP2",
            cpu_offload,
            mixed_precision,
            reshard_after_forward,
            activation_checkpointing,
            use_torch_compile,
            casts,
        )


# ------------------------------------------------------------------
# Block-instance enumeration
# ------------------------------------------------------------------


def _enumerate_block_instances(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[nn.Module, ...]:
    if not class_names:
        return ()
    names = set(class_names)
    return tuple(m for _, m in model.named_modules() if type(m).__name__ in names)


# ------------------------------------------------------------------
# HSDP mesh (ported from FSDPPolicy)
# ------------------------------------------------------------------


def _create_device_mesh(fsdp_mode: str) -> Optional[object]:
    if str(fsdp_mode).strip().lower() != "hybrid":
        return None

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None

    world_size = dist.get_world_size()
    shard_size = 8
    if world_size <= shard_size or world_size % shard_size != 0:
        return None

    from torch.distributed.device_mesh import init_device_mesh

    replicate_size = world_size // shard_size
    mesh = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    logger.info("fsdp_wrap: HSDP mesh dp_replicate=%d x dp_shard=%d", replicate_size, shard_size)
    return mesh


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["fsdp_wrap"]
