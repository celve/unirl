"""Training-side weight synchronization mixin (weight sender/publisher)."""

import logging
from typing import Any, Dict, List, Optional

import ray
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingWeightSyncMixin:
    """Mixin providing weight-sync sender methods for training actors.

    Host class must provide these instance attributes:
        model: nn.Module          -- the training model (read-only by mixin)
        rank: int                 -- this actor's rank (read-only by mixin)
        _use_lora: bool           -- whether LoRA mode is active (read-only by mixin)
    """

    # -- Declared expectations from host --------------------------------
    model: nn.Module
    rank: int
    _use_lora: bool

    # -- Mixin-owned state (initialized via _init_weight_sync_state) ----
    _update_weight_handler: Any
    _rollout_actors: List[Any]
    _lora_initialized_on_rollout: bool

    def _init_weight_sync_state(self) -> None:
        """Initialize mixin-owned attributes. Call from host __init__."""
        self._update_weight_handler = None
        self._rollout_actors = []
        self._lora_initialized_on_rollout = False

    # -- Handler-based weight sync --------------------------------------

    def setup_weight_sync(self, config: dict) -> None:
        """Configure a handler-based rollout weight-sync path."""
        from argparse import Namespace

        from diffusionrl.utils.fsdp_update_weights_utils import (
            UpdateWeightFromCheckpoint,
            UpdateWeightFromDistributed,
            UpdateWeightFromTensor,
        )

        mode = str(config.get("mode", ""))
        rollout_actors = config.get("rollout_actors", [])
        self._rollout_actors = list(rollout_actors) if rollout_actors else []
        self._lora_initialized_on_rollout = False

        handler_args = Namespace(
            target_modules=config.get("target_modules"),
            flush_cache=config.get("flush_cache", True),
            update_weight_buffer_size=config.get("bucket_size_mb", 256) * 1024 * 1024,
            rollout_num_gpus_per_engine=config.get("rollout_num_gpus_per_engine", 1),
            rollout_num_gpus=config.get("rollout_num_gpus", len(self._rollout_actors)),
        )

        if mode == "tensor_payload":
            self._update_weight_handler = UpdateWeightFromTensor(handler_args, self.model)
        elif mode == "nccl_broadcast":
            self._update_weight_handler = UpdateWeightFromDistributed(handler_args, self.model)
        elif mode == "checkpoint_path":
            handler_args.weight_sync_dir = config.get("weight_sync_dir", "/tmp/diffusionrl_wsync")
            handler_args.export_format = config.get("export_format", "state_dict")
            handler_args.rollout_runtime = config.get("rollout_runtime")
            handler_args.rollout_target = config.get("rollout_target")
            self._update_weight_handler = UpdateWeightFromCheckpoint(handler_args, self.model)
        else:
            raise ValueError(f"Unknown weight sync mode: {mode!r}")

        self._update_weight_handler.connect_rollout_engines(self._rollout_actors, None)
        logger.info(
            "Rank %s: configured weight sync handler mode=%s rollout_actors=%d",
            self.rank,
            mode,
            len(self._rollout_actors),
        )

    def _extract_lora_tensors_with_alpha(self) -> dict:
        """Extract LoRA A/B tensors plus per-layer alpha scalars."""
        from diffusionrl.utils.peft_merge import _strip_peft_prefix, _to_full_tensor

        result = {}
        adapter_name = "default"
        for raw_name, param in self.model.state_dict().items():
            name = _strip_peft_prefix(raw_name)
            if ".lora_A." in name:
                prefix, adapter_suffix = name.split(".lora_A.", 1)
                adapter, *_rest = adapter_suffix.split(".", 1)
                if adapter == adapter_name:
                    result[prefix + ".lora_A"] = _to_full_tensor(param).cpu()
            elif ".lora_B." in name:
                prefix, adapter_suffix = name.split(".lora_B.", 1)
                adapter, *_rest = adapter_suffix.split(".", 1)
                if adapter == adapter_name:
                    result[prefix + ".lora_B"] = _to_full_tensor(param).cpu()

        peft_cfg = getattr(self.model, "peft_config", {}).get(adapter_name)
        if peft_cfg is not None:
            alpha_val = torch.tensor(float(peft_cfg.lora_alpha))
            for key in list(result.keys()):
                if key.endswith(".lora_A"):
                    prefix = key[: -len(".lora_A")]
                    result[prefix + ".alpha"] = alpha_val
        return result

    def sync_weights_to_rollout(self) -> None:
        """Synchronize weights through the configured rollout weight-sync handler."""
        if self._update_weight_handler is None:
            raise RuntimeError(
                "Weight sync handler not configured. "
                "Call setup_weight_sync() before sync_weights_to_rollout()."
            )

        if self._use_lora and not self._lora_initialized_on_rollout:
            lora_tensors = self._extract_lora_tensors_with_alpha()
            if self.rank == 0:
                # Fan out LoRA init in parallel — not serial. The previous
                # ``for actor: ray.get(remote())`` pattern forced each
                # set_lora_from_tensors to complete before the next one
                # started, leaving rank 0 blocking inside Ray while the
                # other training ranks had already reached the subsequent
                # ``update_weights()`` collective (raw_state_dict ->
                # _to_full_tensor all-gather). With N rollout actors, rank 0
                # spent ~N× the per-actor init time holding up the
                # collective. Firing all remote calls first and then a
                # single ray.get() waits for the slowest init, not the
                # sum — and lets all actors work concurrently.
                refs = [
                    actor.set_lora_from_tensors.remote("default", lora_tensors)
                    for actor in self._rollout_actors
                ]
                ray.get(refs)
            # All training ranks MUST wait for rank 0's set_lora to finish
            # on every rollout actor before any of them calls update_weights.
            # Without this barrier, ranks 1..N-1 race ahead and push
            # raw_state_dict (base + .lora_A + .lora_B) to their paired
            # sglang engines — but those engines don't have LoRA layers
            # yet, so the .lora_A / .lora_B keys silently no-op and the
            # subsequent set_lora_from_tensors arrival just sees an empty
            # adapter. Observed on 4×8 pod: only rollout_actors[0] ever
            # emitted ``SGLang LoRA initialized`` because the scheduler on
            # each other engine queued update_weights ahead of set_lora
            # and one of them hung.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self._lora_initialized_on_rollout = True

        self._update_weight_handler.update_weights()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def teardown_weight_sync(self) -> None:
        """Release handler state after rollout weight-sync finishes."""
        self._update_weight_handler = None
        self._rollout_actors = []
        self._lora_initialized_on_rollout = False
        logger.info("Rank %s: weight sync handler torn down", self.rank)
