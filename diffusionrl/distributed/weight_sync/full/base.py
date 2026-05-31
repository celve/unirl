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

from typing import Dict, Iterator, List, Optional, Tuple

from diffusionrl.config.require import require
from diffusionrl.distributed.group.remote import Remote


def _one_star(s: object) -> bool:
    """True if ``s`` is a string containing exactly one ``"*"``."""
    return isinstance(s, str) and s.count("*") == 1


def _validate_name_remap(
    name_remap: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """Validate + return the ordered ``name_remap`` rewrite rules (fail-closed)."""
    rules = dict(name_remap or {})
    keys = list(rules.keys())
    for i, (key, value) in enumerate(rules.items()):
        require(_one_star(key), f"name_remap key {key!r} needs one '*'.")
        require(value is None or _one_star(value), f"name_remap[{key!r}] value: null or one '*'; got {value!r}.")
        require(key != "*" or i == len(keys) - 1, f"name_remap '*' must be last; shadows {keys[i + 1 :]!r}.")
    return rules


def _apply_name_remap(name: str, name_remap: Dict[str, Optional[str]]) -> Optional[str]:
    """Apply the ordered single-``*`` rewrite; return the new name or None to drop."""
    for key, value in name_remap.items():
        pre, _, post = key.partition("*")
        # length guard: pre and post must not overlap inside name
        if name.startswith(pre) and name.endswith(post) and len(name) >= len(pre) + len(post):
            if value is None:
                return None
            vpre, _, vpost = value.partition("*")
            return vpre + name[len(pre) : len(name) - len(post)] + vpost
    return name


class FullWeightSync(Remote):
    """Base for full-weight sync Remotes.

    Subclasses implement ``sync()`` (push current weights) plus whatever
    one-time connection setup their transport needs. ``backend`` is the FSDP
    backend sibling whose ``.model`` is the weight source.

    ``lora_merged`` selects what gets pushed: ``False`` (default) syncs the raw
    base weights (meaningful for full fine-tuning); ``True`` folds the trained
    LoRA deltas into the base weights and pushes the merged full model
    (meaningful for a LoRA run served without a separate adapter).
    """

    def __init__(
        self,
        *,
        backend,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._bucket_bytes = int(bucket_size_mb) * 1024 * 1024
        self._flush_cache = bool(flush_cache)
        self._lora_merged = bool(lora_merged)
        # Ordered, first-match-wins name rewrites (glob -> replacement, or None to
        # drop); see _validate_name_remap / _apply_name_remap for the contract.
        self._name_remap = _validate_name_remap(name_remap)
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

        ``lora_merged`` selects the walk:
          - ``True``  → ``merged_state_dict`` folds LoRA deltas into the base
            weights and yields the trained module's own keys (LoRA already
            absorbed, ``.base_layer.`` flattened away).
          - ``False`` → ``raw_state_dict`` base weights, skipping
            ``.lora_A``/``.lora_B``.

        Each emitted name is then rewritten by ``name_remap`` (see
        ``_apply_name_remap``): a ``None`` rule drops the param (e.g. a frozen
        tower not in the receiver), and the ``"*"`` catch-all nests the trained
        submodule's bare keys under the receiver's namespace (e.g.
        ``transformer.``). It is orthogonal to LoRA-merging.
        """
        from diffusionrl.utils.peft_merge import merged_state_dict, raw_state_dict

        remap = self._name_remap
        if self._lora_merged:
            for name, tensor in merged_state_dict(self._backend.model):
                out = _apply_name_remap(name, remap)
                if out is not None:
                    yield out, tensor
            return

        for name, tensor in raw_state_dict(self._backend.model):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            out = _apply_name_remap(name, remap)
            if out is not None:
                yield out, tensor

    def _iter_buckets(self) -> Iterator[Tuple[List[Tuple[str, "object"]], bool]]:
        """Yield ``(bucket, is_last)`` where ``bucket`` is a list of
        ``(name, tensor)`` up to ``bucket_size_mb``.

        ``is_last`` is True only for the final bucket — used to drive
        ``flush_cache`` on the receiver.
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
