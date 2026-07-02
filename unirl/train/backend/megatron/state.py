"""mcore param/grad-buffer + optimizer-state onload/offload (M0).

Colocate (SGLang time-shares the GPU via sleep/wake) needs the trainer to vacate
GPU memory between train and rollout. On mcore this means moving the DDP chunks'
param + ``main_grad`` buffers and the (distributed) optimizer state to CPU and back.

VERIFY: mcore's DistributedDataParallel keeps flat param/grad buffers
(``buffers`` / ``param_and_grad_buffer``); the exact attribute names are
version-sensitive. slime's robust path additionally uses ``torch_memory_saver`` +
reloadable process groups + the ``disable_{grad,param}_buffers_cpu_backup`` patch
hooks — that fuller offload is an M1 hardening; M0 does the straightforward move.
"""

from __future__ import annotations

from typing import Any, List

import torch


def offload_model_state(model_chunks: List[Any], optimizer: Any) -> None:
    """Move mcore params + grad buffers + optimizer state to CPU; free the cache."""
    for chunk in model_chunks:
        chunk.to("cpu")  # VERIFY: moves params AND the flat grad buffers on this DDP chunk
    _move_optimizer_state(optimizer, torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def onload_model_state(model_chunks: List[Any], optimizer: Any, device: torch.device) -> None:
    """Move mcore params + grad buffers + optimizer state back to ``device``."""
    for chunk in model_chunks:
        chunk.to(device)
    _move_optimizer_state(optimizer, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _move_optimizer_state(optimizer: Any, device: torch.device) -> None:
    """Move optimizer state tensors to ``device`` in place.

    VERIFY: the mcore distributed optimizer stores its master params / exp-avg
    shards off ``optimizer.state`` differently from a vanilla torch optimizer;
    confirm this reaches the ZeRO shards (or use the mcore-provided offload hook).
    """
    state = getattr(optimizer, "state", None)
    if not state:
        return
    for buf in state.values():
        if isinstance(buf, dict):
            for k, v in buf.items():
                if isinstance(v, torch.Tensor):
                    buf[k] = v.to(device, non_blocking=True)
