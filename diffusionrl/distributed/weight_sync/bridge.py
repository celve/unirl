"""v2 LoRA weight-sync bridge: push the trained FSDP LoRA adapter into a
co-located vLLM-Omni rollout engine.

Used only by the v2 trainer
(:class:`diffusionrl.trainer.diffusion.DiffusionTrainer`) in shared-process
colocate, where the :class:`~diffusionrl.train.backend.fsdp.FSDPBackend` and the
:class:`~diffusionrl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine` are
sibling ``Remote`` instances on the same Worker. The bridge extracts the LoRA
tensors from the local FSDP shard and hands them to the local engine's
``set_lora_from_tensors`` — the engine owns the Worker→Omni-subprocess transfer
(serialize + ``collective_rpc``), so there is no separate ZMQ pump and no
sender/receiver overlap to orchestrate.

This deliberately does NOT reuse the ``UpdateWeight`` handler family
(``ipc.py`` / ``tensor.py`` / ``nccl.py``): those assume v1's
``rollout_runtime.get_rollout_actors()`` contract (direct Ray actor methods),
which the v2 ``Handle`` / ``Worker.call`` dispatch model does not provide. The
sibling-bridge is the v2-native equivalent for the shared-process colocate case.

Scope: LoRA only. Full-weight / multi-stage (HI3) sync is a different path.

All model / vLLM-touching imports are deferred into the methods so the driver
can import this module (to reference the class for ``remote(...)``) without
eagerly pulling torch-heavy or vLLM-only dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


class LoraSyncBridge(Remote):
    """Push the local FSDP LoRA adapter into the co-located vLLM-Omni engine.

    Constructed inside the trainer's ``placement(...)`` block with the
    ``backend`` and ``rollout`` siblings; they arrive here as the LOCAL
    ``Remote`` instances (``HandleRef`` resolved by ``Worker.add_remote``), so
    method calls on them run in-process on this Worker.
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
    ) -> None:
        super().__init__()
        self._backend = backend
        self._rollout = rollout
        self._param_name_prefix = str(param_name_prefix or "")
        self._packed_modules = dict(packed_modules or {})
        self._adapter_name = str(adapter_name or "default")
        self._verify = bool(verify)

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
        from diffusionrl.distributed.weight_sync.weight_sync import _peft_config_dict
        from diffusionrl.utils.peft_merge import extract_lora_tensors

        model = self._backend.model
        lora_tensors = extract_lora_tensors(
            model,
            param_name_prefix=self._param_name_prefix,
            packed_modules=self._packed_modules,
        )
        peft_config = _peft_config_dict(model, self._adapter_name)
        self._rollout.set_lora_from_tensors(
            self._adapter_name,
            lora_tensors,
            peft_config=peft_config,
        )
        rank = self.rank_info.rank if self.rank_info is not None else 0
        logger.info(
            "[LoRA-SYNC] rank %s: pushed %d LoRA tensors to rollout (adapter=%s)",
            rank,
            len(lora_tensors),
            self._adapter_name,
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


__all__ = ["LoraSyncBridge"]
