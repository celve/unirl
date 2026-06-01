"""v2 LoRA weight-sync handler: push the trained FSDP LoRA adapter into a
co-located vLLM-Omni rollout engine.

Used only by the v2 trainer
(:class:`diffusionrl.trainer.diffusion.DiffusionTrainer`) in shared-process
colocate, where the :class:`~diffusionrl.train.backend.fsdp.FSDPBackend` and the
:class:`~diffusionrl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine` are
sibling ``Remote`` instances on the same Worker. It extracts the LoRA
tensors from the local FSDP shard and hands them to the local engine's
``set_lora_from_tensors`` — the engine owns the Worker→Omni-subprocess transfer
(serialize + ``collective_rpc``), so there is no separate ZMQ pump and no
sender/receiver overlap to orchestrate.

This deliberately does NOT reuse the full-weight handler family
(``full/ipc.py`` / ``full/tensor.py`` / ``full/nccl.py``): for LoRA the engine's
in-process ``set_lora_from_tensors`` already owns the transfer, so the
sibling handoff is the v2-native equivalent for the shared-process colocate case.

Two senders live here, split by transport topology (NOT by LoRA semantics):
:class:`LoraWeightSync` pushes to a same-Worker sibling engine (colocate);
:class:`LoraDriverExtractSync` extracts on the train workers and returns the
adapter to the driver for a cross-process fan-out (HI3's disjoint-Worker AR / DiT
engines). Full-weight sync is a separate path (``full/``).

All model / vLLM-touching imports are deferred into the methods so the driver
can import this module (to reference the class for ``remote(...)``) without
eagerly pulling torch-heavy or vLLM-only dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


def _extract_canonical_lora(backend: Any, *, param_prefix: str, adapter_name: str):
    """Extract canonical-format LoRA tensors + the PEFT config from the backend.

    ``extract_lora_tensors`` redistributes each FSDP ``DTensor`` shard to a full
    tensor — a collective across the train process group — so the caller MUST run
    this on every train rank in lockstep (``ONE_TO_ALL``). Shared by the sibling
    (:class:`LoraWeightSync`) and driver-extract (:class:`LoraDriverExtractSync`)
    senders; the only difference between them is how the adapter then reaches the
    engine, not how it is read off the model.
    """
    from diffusionrl.distributed.weight_sync.payload import _peft_config_dict
    from diffusionrl.utils.peft_merge import extract_lora_tensors

    model = backend.model
    lora_tensors = extract_lora_tensors(model, param_prefix=param_prefix)
    peft_config = _peft_config_dict(model, adapter_name)
    return lora_tensors, peft_config


class LoraWeightSync(Remote):
    """Push one track's trained FSDP LoRA adapter into a co-located rollout engine.

    Constructed inside the trainer's ``placement(...)`` block with the
    ``backend`` and ``rollout`` siblings; they arrive here as the LOCAL
    ``Remote`` instances (``HandleRef`` resolved by ``Worker.add_remote``), so
    method calls on them run in-process on this Worker.

    ``param_prefix`` is the pipeline prefix prepended to the canonical
    keys (e.g. ``"transformer."``; stripped engine-side by
    ``adapt_lora_for_sglang``). ``track_prefix`` (e.g. ``"ar"`` /
    ``"diffusion"``) further prefixes the keys so a
    :class:`~diffusionrl.rollout.engine.composed.engine.ComposedRolloutEngine`
    can demux the update to one child; empty for a single-model trainer.
    ``verify`` is vLLM-Omni-only and ignored for SGLang.

    ``rollout`` is the same-Worker sibling engine for the ``sync()`` push and is
    REQUIRED. (HI3's cross-process driver push has no sibling engine and lives in
    :class:`LoraDriverExtractSync`.)
    """

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        param_prefix: str = "",
        adapter_name: str = "default",
        verify: bool = False,
        track_prefix: str = "",
    ) -> None:
        super().__init__()
        self._backend = backend
        self._rollout = rollout
        self._param_prefix = str(param_prefix or "")
        self._adapter_name = str(adapter_name or "default")
        self._verify = bool(verify)
        self._track_prefix = str(track_prefix or "")

    @distributed(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync(self) -> None:
        """Extract LoRA from the local FSDP model and load it into the engine.

        Runs on every Worker (``ONE_TO_ALL``). ``extract_lora_tensors`` walks
        ``model.state_dict()`` and redistributes each DTensor shard to a full
        tensor — a collective across the train process group, which lines up
        because every rank runs this method together. The engine must be awake
        (the caller wakes it before ``sync``); ``set_lora_from_tensors`` drops
        any existing adapter and loads the new one on every stage's workers.
        """
        self._sync_lora()

    def _sync_lora(self) -> None:
        """LoRA-adapter path: ``set_lora_from_tensors`` into the engine's pool."""
        lora_tensors, peft_config = _extract_canonical_lora(
            self._backend, param_prefix=self._param_prefix, adapter_name=self._adapter_name
        )
        # Prefix keys so a ComposedRolloutEngine can demux to one child.
        if self._track_prefix:
            lora_tensors = {f"{self._track_prefix}.{k}": v for k, v in lora_tensors.items()}
        self._rollout.set_lora_from_tensors(
            self._adapter_name,
            lora_tensors,
            peft_config=peft_config,
        )
        rank = self.rank_info.rank if self.rank_info is not None else 0
        logger.info(
            "[LoRA-SYNC] rank %s: pushed %d LoRA tensors to rollout (adapter=%s, track=%s)",
            rank,
            len(lora_tensors),
            self._adapter_name,
            self._track_prefix or "<single>",
        )
        if self._verify:
            self._verify_loaded(lora_tensors, peft_config)

    def _verify_loaded(self, lora_tensors: Dict[str, Any], peft_config: Dict) -> None:
        """Assert the engine's loaded LoRA matches what we just pushed.

        Compares trainer-side expected hashes (with ``lora_B`` scaled by
        ``alpha/r`` to match the worker's post-``optimize`` read-back) against
        the engine's per-(stage, rank) ``loaded_lora_checksums``. The engine
        keys by vLLM-internal layer name + field (``lora_a`` / ``lora_b``), so a
        direct dict compare is impossible; instead we compare the *multiset* of
        ``lora_A`` hashes and the multiset of ``lora_B`` hashes. With distinct
        per-layer weights (always true after a step of training) multiset
        equality is a strong bit-equality proof, and it also catches a wrong
        ``param_prefix`` (which yields wrong / zero loaded layers).
        """
        from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
            compute_lora_checksums_post_optimize,
        )
        from diffusionrl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
        )

        expected = compute_lora_checksums_post_optimize(lora_tensors, peft_config)
        exp_a = sorted(h for k, h in expected.items() if ".lora_A." in k)
        exp_b = sorted(h for k, h in expected.items() if ".lora_B." in k)

        loaded = self._rollout.loaded_lora_checksums(adapter_id=int(DIFFRL_LORA_INT_ID))
        rank = self.rank_info.rank if self.rank_info is not None else 0
        for stage_id, per_rank in loaded.items():
            for rank_idx, layer_map in enumerate(per_rank):
                act_a = sorted(f["lora_a"] for f in layer_map.values() if "lora_a" in f)
                act_b = sorted(f["lora_b"] for f in layer_map.values() if "lora_b" in f)
                if act_a != exp_a or act_b != exp_b:
                    raise RuntimeError(
                        f"[LoRA-SYNC] verify FAILED on train-rank {rank}, rollout "
                        f"stage {stage_id} rank {rank_idx}: expected {len(exp_a)} "
                        f"lora_A / {len(exp_b)} lora_B hashes, engine loaded "
                        f"{len(act_a)} / {len(act_b)} (A_match={act_a == exp_a}, "
                        f"B_match={act_b == exp_b}). Likely a transport bug or a "
                        f"param_prefix mismatch ({self._param_prefix!r})."
                    )
        logger.info(
            "[LoRA-SYNC] rank %s: verify OK (%d lora_A / %d lora_B layers match)",
            rank,
            len(exp_a),
            len(exp_b),
        )


class LoraDriverExtractSync(Remote):
    """Extract the trained LoRA adapter and RETURN it to the driver, for a
    cross-process push to engines that are NOT same-Worker siblings.

    Counterpart to :class:`LoraWeightSync` for HI3's two-engine trainer: the
    shared backbone trains across all cards while the AR / DiT engines are
    anchored on separate Workers (disjoint GPU partition), so the backend is not
    a sibling of either engine and ``sync()``'s in-process push can't reach them.
    The driver calls :meth:`extract` (a collective; rank 0 returns the full
    adapter), then pushes the returned adapter into each engine via its Handle
    (``engine.set_lora_from_tensors_copy``). ``verify`` (engine checksum
    read-back) is not available here — there is no sibling ``rollout`` to query.

    Transport only — the adapter is read off the model exactly as in
    :class:`LoraWeightSync` (shared :func:`_extract_canonical_lora`).

    # DELETE-WHEN: HI3's AR / DiT engines can be co-located as same-Worker
    #   siblings of the FSDP backend (e.g. vLLM-Omni grows a colocate placement
    #   that co-tenants the trainer shard and a TP>1 engine on one card). Then
    #   HI3 uses ``LoraWeightSync.sync()`` like SD3 / PE and this class is dead:
    #     1. delete this class + restore conf/hi3_vllmomni.yaml ``sync._target_``
    #        to ``...lora.LoraWeightSync`` (passing ``rollout``).
    #     2. delete the byte-copy receive mates (own DELETE-WHEN tags):
    #        VLLMOmniRolloutEngine.set_lora_from_tensors_copy and
    #        ipc_receive_mixin.BucketedIPCReceiveMixin.set_lora_from_tensor_dict_copy.
    """

    def __init__(
        self,
        *,
        backend: Any,
        param_prefix: str = "",
        adapter_name: str = "default",
    ) -> None:
        super().__init__()
        self._backend = backend
        self._param_prefix = str(param_prefix or "")
        self._adapter_name = str(adapter_name or "default")

    @distributed(dispatch_mode=Dispatch.ONE_TO_ALL)
    def extract(self):
        """Extract the LoRA adapter and RETURN it to the driver (rank 0 only).

        Runs on EVERY Worker (``ONE_TO_ALL``) because ``extract_lora_tensors``'
        DTensor redistribute is a collective across the train process group; only
        rank 0 returns the (full, identical) tensors so the driver gets a single
        copy (the trainer takes the rank-0 entry of the ``ONE_TO_ALL`` list).

        Ships the FUSED qkv adapter (no trainer-side ``packed_modules`` split);
        the engine's vLLM-native packed-modules mapping unpacks q / k / v on
        load, matching the single-engine HI3 path post-#181.
        """
        lora_tensors, peft_config = _extract_canonical_lora(
            self._backend, param_prefix=self._param_prefix, adapter_name=self._adapter_name
        )
        rank = self.rank_info.rank if self.rank_info is not None else 0
        if rank != 0:
            return None
        logger.info("[LoRA-SYNC] rank 0: extracted %d LoRA tensors for driver-side push", len(lora_tensors))
        return {"lora_tensors": lora_tensors, "peft_config": peft_config, "adapter_name": self._adapter_name}


__all__ = ["LoraWeightSync", "LoraDriverExtractSync"]
