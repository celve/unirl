"""v2 full-weight tensor-payload sync (COLOCATE).

Pushes the trained FSDP full base weights into a co-located vLLM-Omni rollout
engine by serializing each bucket (SGLang ``FlattenedTensorBucket`` +
``MultiprocessingSerializer``) and handing it to the local engine sibling's
``update_weights_from_tensor`` — the engine owns the Worker→Omni-subprocess
transfer (serialize already done; ``collective_rpc`` fans to the stage workers).

Full-weight analogue of ``weight_sync/lora/local.py:LocalLoraWeightSync`` and the v2
transport-mate of v1 ``distributed/weight_sync/tensor.py``. Colocate only:
``backend`` and ``rollout`` arrive as LOCAL siblings (same Worker process). At TP=1
each train rank ships to its own co-located engine, which picks
``serialized_named_tensors[0]``. At TP>1 (grouped-TP rollout) sglang's tp_worker
picks ``bag[tp_rank]`` and each bag is a CUDA-IPC handle that must live on that
rank's own GPU, so ``sync`` relays the per-rank handle strings within each TP group
to the group head (a scoped revival of the v1 ``gather_object``), which issues the
single update with the full per-rank window; each rank then reconstructs the full
tensors and ``model.load_weights`` slices its own shard.

Scope: single-node, TP>=1; a single-model engine, or one child of a
``ComposedRolloutEngine`` (via ``track_prefix``). All model / sglang imports are
deferred so the driver can import this module for ``remote(...)``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync


class TensorWeightSync(FullWeightSync):
    """Colocate full-weight sync via serialized tensor payloads."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout  # local engine sibling (single-model, or a ComposedRolloutEngine)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Serialize each bucket and load it into the local engine.

        Runs on every train rank (``BROADCAST``); the ``raw_state_dict`` walk
        all-gathers each shard on every rank in lockstep. Each rank talks to
        its own co-located engine, so no cross-rank gather is needed.
        """
        import torch
        import torch.distributed as dist

        # Use sglang's NATIVE serializer (not unirl's vendored sgl_compat copy):
        # sglang 0.5.12's SafeUnpickler (CVE-2025-10164 guard) runs in the SRT
        # scheduler subprocess and allowlists only its own classes, so a payload
        # referencing unirl.* (vendored FlattenedTensorBucket / rebuild_cuda_tensor)
        # is rejected at update_weights_from_tensor. sglang's native classes carry
        # the same device-UUID IPC mapping and are on its allowlist. Mirrors what
        # the sglang_diffusion engine already does (LIN-365 _patches).
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        monkey_patch_torch_reductions()

        for bucket, is_last in self._iter_buckets():
            # Group by dtype, one FlattenedTensorBucket per dtype (matches the
            # receiver's flattened_bucket load_format).
            by_dtype: dict = {}
            for name, tensor in bucket:
                # Tensors arrive already at the wire dtype: ``wire_dtype`` (sync
                # config) is applied once in the base-class walk, shard-side.
                by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            serialized = []
            for grouped in by_dtype.values():
                flat = FlattenedTensorBucket(named_tensors=grouped)
                payload = {
                    "flattened_tensor": flat.get_flattened_tensor(),
                    "metadata": flat.get_metadata(),
                }
                serialized.append(MultiprocessingSerializer.serialize(payload, output_str=True))

            n_dtypes = len(serialized)
            ri = self._rollout.rank_info
            tp = int(getattr(ri, "tp_size", 1) or 1)
            if tp <= 1:
                # TP=1 (the common path): each train rank ships to its own co-located
                # engine, which picks serialized_named_tensors[0]. flush on the last payload.
                for i, payload in enumerate(serialized):
                    self._rollout.update_weights_from_tensor(
                        serialized_named_tensors=[payload],
                        load_format="flattened_bucket",
                        flush_cache=(self._flush_cache and is_last and i == n_dtypes - 1),
                        track_prefix=self._track_prefix,
                    )
            else:
                # Colocate grouped-TP: sglang's tp_worker deserializes bag[tp_rank], and each
                # bag is a CUDA-IPC handle that MUST live on that rank's own GPU (a GPU-0 handle
                # is unopenable by node_rank>0 → 'Invalid device_uuid'). Every train rank already
                # holds the full weights on its own GPU, so relay the small handle strings within
                # each TP group to the group HEAD, which issues the one update per dtype with the
                # per-rank window [bag_gpu(base+0), bag_gpu(base+1), ...]. Each rank then
                # reconstructs the full tensors and model.load_weights slices its shard.
                if ri.rank != dist.get_rank():
                    raise RuntimeError(
                        "grouped-TP weight sync assumes co-located train rank == rollout worker "
                        f"rank (contiguous 1-GPU placement), but rollout rank_info.rank={ri.rank} "
                        f"!= train rank {dist.get_rank()}"
                    )
                all_ser: list = [None] * dist.get_world_size()
                dist.all_gather_object(all_ser, serialized)  # tiny per-dtype handle strings
                if ri.tp_rank == 0:  # group head drives the single update per dtype
                    base = ri.dp_rank * tp  # first train rank / GPU of this TP group
                    for i in range(n_dtypes):
                        window = [all_ser[base + t][i] for t in range(tp)]  # index t == tp_rank
                        self._rollout.update_weights_from_tensor(
                            serialized_named_tensors=window,
                            load_format="flattened_bucket",
                            flush_cache=(self._flush_cache and is_last and i == n_dtypes - 1),
                            track_prefix=self._track_prefix,
                        )
                # Lifetime barrier: a non-head must not free its GPU weights until the head's
                # update (which opened that rank's handle on node_rank>0) has returned.
                dist.barrier()
            # Release the all-gathered full tensors + IPC payloads for this bucket
            # before gathering the next — else the full model (~13GB) accumulates
            # in the caching allocator and OOMs the colocated SRT server.
            del serialized, by_dtype, bucket
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.weight_version += 1


__all__ = ["TensorWeightSync"]
