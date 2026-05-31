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

Scope: LoRA only. Full-weight / multi-stage (HI3) sync is a different path.

All model / vLLM-touching imports are deferred into the methods so the driver
can import this module (to reference the class for ``remote(...)``) without
eagerly pulling torch-heavy or vLLM-only dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from diffusionrl.config.require import require
from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


class LoraWeightSync(Remote):
    """Push one track's trained FSDP weights into a co-located rollout engine.

    Constructed inside the trainer's ``placement(...)`` block with the
    ``backend`` and ``rollout`` siblings; they arrive here as the LOCAL
    ``Remote`` instances (``HandleRef`` resolved by ``Worker.add_remote``), so
    method calls on them run in-process on this Worker.

    ``mode="lora"`` (default) ships the LoRA adapter via
    ``set_lora_from_tensors`` (vLLM-Omni, SGLang diffusion, SGLang LLM pool).
    ``mode="merged"`` folds ``base + α·B·A`` and ships the full tensors via
    ``update_weights_from_tensor`` (full-parameter transfer; bypasses the LoRA
    pool). ``track_prefix`` (e.g. ``"ar"`` / ``"diffusion"``) routes the update
    to one child of a :class:`ComposedRolloutEngine`; empty for a single-model
    trainer. ``verify`` is vLLM-Omni-only and ignored for SGLang.
    """

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        param_name_prefix: str = "",
        packed_modules: Optional[Dict] = None,
        adapter_name: str = "default",
        verify: bool = False,
        track_prefix: str = "",
        mode: str = "lora",
        flush_cache: bool = True,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._rollout = rollout
        self._param_name_prefix = str(param_name_prefix or "")
        self._packed_modules = dict(packed_modules or {})
        self._adapter_name = str(adapter_name or "default")
        self._verify = bool(verify)
        self._track_prefix = str(track_prefix or "")
        self._mode = str(mode or "lora")
        self._flush_cache = bool(flush_cache)
        require(
            self._mode in ("lora", "merged"),
            f"LoraWeightSync.mode must be 'lora' or 'merged'; got {self._mode!r}",
        )

    @distributed(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync(self) -> None:
        """Extract the local FSDP weights and push them into the engine.

        Runs on every Worker (``ONE_TO_ALL``). Extraction
        (``extract_lora_tensors`` / ``merged_state_dict``) redistributes each
        DTensor shard to a full tensor — a collective across the train process
        group, which lines up because every rank runs this together. The engine
        must be awake (the caller wakes it before ``sync``). ``mode`` selects
        the LoRA-adapter vs merged-full-weight path; ``track_prefix`` routes to
        one child of a composed engine.
        """
        if self._mode == "merged":
            self._sync_merged()
        else:
            self._sync_lora()

    def _sync_lora(self) -> None:
        """LoRA-adapter path: ``set_lora_from_tensors`` into the engine's pool."""
        from diffusionrl.distributed.weight_sync.payload import _peft_config_dict
        from diffusionrl.utils.peft_merge import extract_lora_tensors

        model = self._backend.model
        lora_tensors = extract_lora_tensors(
            model,
            param_name_prefix=self._param_name_prefix,
            packed_modules=self._packed_modules,
        )
        peft_config = _peft_config_dict(model, self._adapter_name)
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
        # verify is vLLM-Omni-only (checksum read-back); skip for SGLang.
        if self._verify:
            self._verify_loaded(lora_tensors, peft_config)

    def _sync_merged(self) -> None:
        """Merged path: fold ``base + α·B·A`` and ship full weights.

        Full-parameter transfer via ``update_weights_from_tensor`` for engines
        without a usable LoRA pool. Sends the whole (merged) state dict in one
        pass, grouped by dtype. NOTE: no size bucketing yet — fine for small
        models (Qwen3-0.6B); add bucketing before promoting to large LLMs.
        """
        from diffusionrl.distributed.weight_sync.payload import serialize_named_tensors
        from diffusionrl.utils.peft_merge import merged_state_dict

        model = self._backend.model
        named = list(merged_state_dict(model, self._adapter_name))
        payloads = serialize_named_tensors(named)
        for i, payload in enumerate(payloads):
            is_last = i == len(payloads) - 1
            self._rollout.update_weights_from_tensor(
                serialized_named_tensors=[payload],
                load_format="flattened_bucket",
                flush_cache=(self._flush_cache and is_last),
                track_prefix=self._track_prefix,
            )
        rank = self.rank_info.rank if self.rank_info is not None else 0
        logger.info(
            "[MERGED-SYNC] rank %s: pushed %d params (%d dtype-bucket(s)) to rollout (track=%s)",
            rank,
            len(named),
            len(payloads),
            self._track_prefix or "<single>",
        )

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
        ``param_name_prefix`` (which yields wrong / zero loaded layers).
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
                        f"param_name_prefix mismatch ({self._param_name_prefix!r})."
                    )
        logger.info(
            "[LoRA-SYNC] rank %s: verify OK (%d lora_A / %d lora_B layers match)",
            rank,
            len(exp_a),
            len(exp_b),
        )


__all__ = ["LoraWeightSync"]
