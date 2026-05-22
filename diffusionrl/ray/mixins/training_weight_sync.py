"""Training-side weight synchronization mixin (weight sender/publisher)."""

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from diffusionrl.ray.placement import PlacementConfig

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
    _base_sync_done: bool

    def _init_weight_sync_state(self) -> None:
        """Initialize mixin-owned attributes. Call from host __init__."""
        self._update_weight_handler = None
        # Skip base sync — rollout already loaded base weights from disk.
        # The base sync path has an unverified prefix/name transform bug
        # (v44b ValueError). LoRA-only sync is the critical path.
        self._base_sync_done = True

    # -- Handler-based weight sync --------------------------------------

    def setup_weight_sync(
        self,
        *,
        sync_cfg: "DictConfig",
        placement_cfg: "PlacementConfig",
        rollout_runtime: Any,
        param_name_prefix: str = "",
        packed_modules: dict | None = None,
    ) -> None:
        """Configure a handler-based rollout weight-sync path."""
        from diffusionrl.config.instantiate import build

        # Keep base sync skipped — rollout already loaded base weights from
        # disk. setup() used to reset this to False (base sync would run once
        # after setup), but base sync has an unverified prefix bug.
        self._base_sync_done = True

        self._update_weight_handler = build(
            sync_cfg,
            model=self.model,
            rollout_runtime=rollout_runtime,
            placement_cfg=placement_cfg,
            param_name_prefix=str(param_name_prefix or ""),
        )
        # Store packed_modules mapping for fused→split LoRA tensor conversion
        self._update_weight_handler._packed_modules = dict(packed_modules or {})
        self._update_weight_handler.connect_rollout_engines()
        logger.info(
            "Rank %s: configured weight sync handler target=%s param_name_prefix=%r",
            self.rank,
            sync_cfg.get("_target_"),
            param_name_prefix,
        )

    def _resolve_peft_config_obj(self, adapter_name: str = "default") -> Any:
        """Walk the model wrap layers and return the per-adapter PEFT config.

        Mirrors ``_detect_lora_on_model`` in ``ray/train_actor.py``: PEFT
        installs ``peft_config`` directly on the model, but FSDP / PEFT
        wrappers sometimes hide it under ``.module`` / ``.base_model``.
        Returns ``None`` if not found (caller decides whether that's fatal).
        """
        cur = self.model
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            pc = getattr(cur, "peft_config", None)
            if isinstance(pc, dict) and adapter_name in pc:
                return pc[adapter_name]
            cur = getattr(cur, "module", None) or getattr(cur, "base_model", None)
        return None

    def _extract_lora_state(self, adapter_name: str = "default") -> tuple[dict, dict]:
        """Return ``(lora_tensors, peft_config_dict)`` for one PEFT adapter.

        ``lora_tensors`` uses the vllm-omni format (PEFT envelope):
        ``base_model.model.<prefix><module>.lora_A.weight`` /
        ``...lora_B.weight``. Alpha is carried by ``peft_config_dict``.
        """
        from diffusionrl.utils.peft_merge import adapt_lora_for_vllm, extract_lora_tensors

        prefix = getattr(self._update_weight_handler, "_param_name_prefix", "")
        tensors = adapt_lora_for_vllm(
            extract_lora_tensors(
                self.model,
                param_name_prefix=str(prefix or ""),
                adapter_name=adapter_name,
            )
        )
        peft_dict = self._peft_config_dict(adapter_name)
        n_pairs = sum(1 for key in tensors if key.endswith(".lora_A.weight"))
        if n_pairs == 0:
            raise RuntimeError(
                f"_extract_lora_state: extracted 0 LoRA layers from "
                f"self.model.state_dict() (looking for adapter "
                f"{adapter_name!r}). Either PEFT didn't inject any "
                f"adapters or the parameter naming has drifted."
            )
        return tensors, peft_dict

    def _peft_config_dict(self, adapter_name: str = "default") -> dict:
        """Return a JSON/Ray-safe PEFT config dict for one adapter."""
        peft_cfg_obj = self._resolve_peft_config_obj(adapter_name)
        if peft_cfg_obj is None:
            raise RuntimeError(
                f"TrainingWeightSyncMixin._peft_config_dict: model has no "
                f"peft_config[{adapter_name!r}] entry. LoRA detection said "
                f"the adapter exists but the per-adapter config is missing "
                f"— probably a wrap mismatch (FSDP unwrap, PEFT base_model)."
            )

        if hasattr(peft_cfg_obj, "to_dict"):
            peft_dict = peft_cfg_obj.to_dict()
        else:
            peft_dict = dict(peft_cfg_obj)

        # Normalize ``target_modules`` — PEFT accepts str (regex) | list |
        # set | tuple, but vllm-omni's worker-side PEFTHelper.from_dict
        # is JSON-shipped over msgspec which doesn't know about set/tuple,
        # and at the receive side the helper expects a list-or-str.
        # CRITICAL: do NOT do ``list(map(str, target_modules))`` blindly —
        # a regex string would be exploded into its individual characters.
        tm = peft_dict.get("target_modules")
        if isinstance(tm, str):
            # regex pattern; leave as-is (PEFTHelper accepts str).
            pass
        elif isinstance(tm, (list, tuple, set, frozenset)):
            peft_dict["target_modules"] = sorted(tm) if isinstance(tm, (set, frozenset)) else list(tm)
        elif tm is None:
            pass  # caught by the required-keys check below
        else:
            raise RuntimeError(
                f"_peft_config_dict: peft_config['target_modules'] has "
                f"unsupported type {type(tm).__name__}; expected str / list / "
                f"set / tuple."
            )

        # vllm-omni's PEFTHelper.from_dict requires these — fail fast here
        # rather than at receive time on the worker subprocess.
        for required in ("r", "lora_alpha", "target_modules"):
            if peft_dict.get(required) in (None, "", [], ()):
                raise RuntimeError(
                    f"_peft_config_dict: peft_config[{required!r}] is "
                    f"missing or empty (got {peft_dict.get(required)!r}); "
                    f"vllm-omni LoRA receive will reject this."
                )

        return peft_dict

    # Back-compat shim: legacy callers still expect the dict-only return.
    # Internal callers should switch to ``_extract_lora_state``.
    def _extract_lora_tensors_with_alpha(self) -> dict:
        tensors, _ = self._extract_lora_state()
        return tensors

    def sync_weights_to_rollout(self) -> None:
        """Synchronize weights through the configured rollout weight-sync handler.

        LoRA mode: sync frozen base weights once, then sync LoRA tensors on
        every call through the configured transport handler. The handler owns
        tensor extraction so it can apply the rollout-side parameter prefix
        consistently for IPC, NCCL and tensor-payload transports.

        Full-FT mode: drop straight to the NCCL/IPC handler that ships the
        whole state dict.
        """
        if self._update_weight_handler is None:
            raise RuntimeError(
                "Weight sync handler not configured. Call setup_weight_sync() before sync_weights_to_rollout()."
            )

        if self._use_lora:
            peft_cfg_dict = self._peft_config_dict()
            if not self._base_sync_done:
                self._update_weight_handler.update_weights(
                    peft_config=None,
                    base_sync_done=False,
                )
                self._base_sync_done = True

            self._update_weight_handler.update_weights(
                peft_config=peft_cfg_dict,
                base_sync_done=True,
            )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            self._update_weight_handler.update_weights()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def teardown_weight_sync(self) -> None:
        """Release handler state after rollout weight-sync finishes."""
        self._update_weight_handler = None
        self._base_sync_done = False
        logger.info("Rank %s: weight sync handler torn down", self.rank)
