"""Base class for v2 full-(base-)weight sync handlers.

Counterpart to ``weight_sync/lora.py`` (LoRA-only). A ``FullWeightSync``
lives on the TRAIN slab as a sibling ``Remote`` of the FSDP backend (and, in
shared-process colocate, of the rollout engine) and pushes the freshly-trained
*full* weights into the rollout engine(s).

Subclasses pick a transport:
  - ``nccl.NCCLWeightSync``   — separate slabs (cross-node capable).
  - ``tensor.TensorWeightSync`` — colocate (serialized-tensor handoff).
  - ``ipc.IPCWeightSync``     — colocate (bucketed CUDA-IPC over ZMQ).

The base provides the transport-agnostic weight walk: full-tensor
materialization (FSDP shard → replicated, a train-mesh collective) and
size-bounded bucketing. All model / torch-heavy imports are deferred into the
methods so the driver can import this module to reference the class for
``remote(...)`` without eagerly pulling torch.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

from diffusionrl.distributed.group.remote import Remote


class FullWeightSync(Remote):
    """Base for full-weight sync Remotes.

    Subclasses implement ``sync()`` (push current weights) plus whatever
    one-time connection setup their transport needs. ``backend`` is the FSDP
    backend sibling whose ``.model`` is the weight source.

    ``use_merged`` selects what gets pushed: ``False`` (default) syncs the raw
    base weights (meaningful for full fine-tuning); ``True`` folds the trained
    LoRA deltas into the base weights and pushes the merged full model
    (meaningful for a LoRA run served without a separate adapter).
    """

    def __init__(
        self,
        *,
        backend,
        bucket_size_mb: int = 512,
        param_name_prefix: str = "",
        target_modules: Tuple[str, ...] = ("transformer",),
        flush_cache: bool = True,
        use_merged: bool = False,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._bucket_bytes = int(bucket_size_mb) * 1024 * 1024
        self._param_name_prefix = str(param_name_prefix or "")
        self._target_modules = list(target_modules)
        self._flush_cache = bool(flush_cache)
        self._use_merged = bool(use_merged)
        self.weight_version = 0

    # ------------------------------------------------------------------
    # Transport-agnostic weight walk
    # ------------------------------------------------------------------

    def _iter_full_tensors(self) -> Iterator[Tuple[str, "object"]]:
        """Yield ``(name, full_tensor)`` one at a time (lazy → bounded memory).

        Both walks apply ``_to_full_tensor`` (redistribute each FSDP ``DTensor``
        shard to Replicate → a collective over the train mesh), so this MUST run
        on every train rank in lockstep; each yields a full (unsharded) CUDA
        tensor per param, in a deterministic order.

        ``use_merged`` selects the walk (mirrors v1 ``BucketedUpdateWeight``):
          - ``True``  → ``merged_state_dict`` folds LoRA deltas into the base
            weights and yields the trained module's own keys (LoRA already
            absorbed, ``.base_layer.`` flattened away).
          - ``False`` → ``raw_state_dict`` base weights, skipping
            ``.lora_A``/``.lora_B``.

        Both walks then get ``param_name_prefix`` applied. ``backend.model`` is
        the trained submodule (e.g. the SD3 transformer), so its state-dict keys
        are bare (``transformer_blocks.0...``). The rollout engine loads them
        into the *full pipeline*, where that submodule lives under
        ``transformer.`` — so the prefix (``"transformer."``) is what makes the
        pushed names resolve on the receiver. It is orthogonal to LoRA-merging.
        """
        from diffusionrl.utils.peft_merge import merged_state_dict, raw_state_dict

        prefix = self._param_name_prefix

        if self._use_merged:
            for name, tensor in merged_state_dict(self._backend.model):
                yield (prefix + name if prefix else name), tensor
            return

        for name, tensor in raw_state_dict(self._backend.model):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            yield (prefix + name if prefix else name), tensor

    def _iter_buckets(self) -> Iterator[Tuple[List[Tuple[str, "object"]], bool]]:
        """Yield ``(bucket, is_last)`` where ``bucket`` is a list of
        ``(name, tensor)`` up to ``bucket_size_mb``.

        Mirrors the accumulation in ``BucketedUpdateWeight.update_weights``
        (``weight_sync/base.py``). ``is_last`` is True only for the final
        bucket — used to drive ``flush_cache`` on the receiver.
        """
        bucket: List[Tuple[str, object]] = []
        nbytes = 0
        for name, tensor in self._iter_full_tensors():
            size = tensor.numel() * tensor.element_size()
            if bucket and nbytes + size >= self._bucket_bytes:
                yield bucket, False
                bucket, nbytes = [], 0
            bucket.append((name, tensor))
            nbytes += size
        if bucket:
            yield bucket, True

    @property
    def _my_rank(self) -> int:
        ri = self.rank_info
        return ri.rank if ri is not None else 0


__all__ = ["FullWeightSync"]
