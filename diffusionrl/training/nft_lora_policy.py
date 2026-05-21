"""NFTLoRAPolicy — peft adapter injection + dual-adapter EMA as a stackable :class:`Policy`.

NFT needs two LoRA adapters living on the same wrapped ``nn.Module``:

* ``"default"`` — the trainable policy. The optimizer sees only these
  parameters; peft has frozen the base weights and the second adapter.
* ``"old"`` — a frozen reference whose weights track ``"default"``
  via EMA. Used by the NFT loss to compute the un-updated prediction
  without an extra model copy.

EMA update cadence is configurable: ``optimizer_step`` advances the
reference every train micro-step; ``rollout_end`` advances it once per
rollout cycle. The corresponding hooks (``.step`` / ``.on_rollout_end``)
are dispatched by the chain walkers in ``training/stack.py``.

The algorithm consumer (``DiffusionNFT``) never touches peft strings;
it switches adapters through the :meth:`with_old_adapter` context
manager.

Compose order mirrors :class:`LoRAPolicy`::

    pipe = SD3Pipeline.from_meta_config(cfg)
    policy = compose_policy(pipe.diffusion, [nft_lora_cfg, fsdp_cfg])
    pipe.bundle.materialize(device=cuda, with_aux=("vae",))
    policy.post_materialize_init()
    # → resets both adapters, then hard-copies "default" → "old"
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter

from diffusionrl.config.registration import register_config
from diffusionrl.training.fsdp_policy import (
    _extract_peft_lora_state,
    _filter_lora_state,
)
from diffusionrl.training.policy import Policy, PolicyBase

logger = logging.getLogger(__name__)


@register_config(
    group="training/policy",
    name="nft_lora",
    target="diffusionrl.training.nft_lora_policy.NFTLoRAPolicy",
)
@dataclass
class NFTLoRAPolicyConfig:
    """Construction args for :class:`NFTLoRAPolicy`.

    The peft fields (``rank``, ``alpha``, ``target_modules``, ...) match
    :class:`LoRAPolicyConfig`. The remaining fields parameterize the
    dual-adapter EMA: which name to use for the reference adapter, the
    decay schedule, and when the schedule advances.
    """

    name: ClassVar[str] = "nft_lora"

    # --- peft injection (mirror LoRAPolicyConfig) ---
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = field(default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj"))
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"

    # --- NFT dual-adapter + EMA ---
    old_adapter_name: str = "old"
    # ``constant`` → decay always equals ``ema_decay``.
    # ``linear`` → ``min(step * ema_uprate, ema_uphold)`` (climbing schedule).
    # ``warmup`` → 0 for the first ``ema_flat_steps`` updates, then
    #             ``min((step - ema_flat_steps) * ema_uprate, ema_uphold)``.
    ema_decay: float = 0.001
    ema_decay_type: str = "constant"
    ema_flat_steps: int = 0
    ema_uprate: float = 0.001
    ema_uphold: float = 0.5
    # ``optimizer_step`` (high frequency) or ``rollout_end`` (low frequency).
    ema_update_timing: str = "rollout_end"


class NFTLoRAPolicy(PolicyBase):
    """peft dual-adapter (``default`` + ``old``) with EMA-tracked reference.

    Inherits :class:`PolicyBase` defaults for ``trainable_module``,
    ``replay``, ``state_dict``, ``load_state_dict``, ``train``, ``eval``.
    Overrides:

    - ``parameters`` / ``named_parameters`` — filter to
      ``requires_grad=True`` (only ``default`` adapter trains).
    - ``post_materialize_init`` — chain inward, then reset both adapters
      via peft's standard init, then hard-copy ``default → old`` so the
      reference starts at a known baseline.

    Adds (off-Protocol):

    - ``with_old_adapter()`` — context manager that swaps the active peft
      adapter to ``"old"`` for the duration of a forward.
    - ``step(optimization_step)`` — called by :func:`_step_ema_in_chain`
      after every optimizer step. If ``ema_update_timing == "optimizer_step"``
      runs ``old ← decay·old + (1-decay)·default``; otherwise no-op.
    - ``on_rollout_end(step)`` — called by :func:`_on_rollout_end_in_chain`
      at each rollout boundary. If ``ema_update_timing == "rollout_end"``
      runs the EMA update.
    - ``lora_state_dict`` / ``nft_state_dict`` — separate checkpoint
      surfaces for ``default`` (model weights) and ``old`` (derived EMA
      reference); the ``old`` adapter is loaded back independently so it
      doesn't blend with regular state-dict loading.
    - ``disable_adapter`` — peft passthrough (reserved for future KL
      against the un-adapted base model).
    """

    _VALID_DECAY_TYPES: ClassVar[Tuple[str, ...]] = ("constant", "linear", "warmup")
    _VALID_UPDATE_TIMINGS: ClassVar[Tuple[str, ...]] = ("optimizer_step", "rollout_end")

    def __init__(
        self,
        config: NFTLoRAPolicyConfig,
        source: Union[Any, Policy],
    ) -> None:
        if not hasattr(source, "trainable_module"):
            raise TypeError(
                f"NFTLoRAPolicy: source {type(source).__name__} has no "
                "trainable_module() method. Source must be a Stage or "
                "another Policy that exposes the wrap target."
            )
        if config.ema_decay_type not in self._VALID_DECAY_TYPES:
            raise ValueError(
                f"NFTLoRAPolicy: ema_decay_type must be one of "
                f"{self._VALID_DECAY_TYPES}; got {config.ema_decay_type!r}."
            )
        if config.ema_update_timing not in self._VALID_UPDATE_TIMINGS:
            raise ValueError(
                f"NFTLoRAPolicy: ema_update_timing must be one of "
                f"{self._VALID_UPDATE_TIMINGS}; got {config.ema_update_timing!r}."
            )
        if config.old_adapter_name == "default":
            raise ValueError("NFTLoRAPolicy: old_adapter_name must differ from 'default' (the trainable adapter).")

        self.config = config
        self.source = source
        self.model: nn.Module = source.trainable_module()
        self._new_adapter = "default"
        self._old_adapter = str(config.old_adapter_name)
        self._update_counter = 0
        self._inject_lora()

    # ------------------------------------------------------------------
    # Injection: install both adapters
    # ------------------------------------------------------------------

    def _inject_lora(self) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        peft_config = LoraConfig(
            r=int(self.config.rank),
            lora_alpha=int(self.config.alpha),
            lora_dropout=float(self.config.dropout),
            target_modules=list(self.config.target_modules),
            bias=str(self.config.bias),
            task_type=str(self.config.task_type),
        )
        inject_adapter_in_model(peft_config, self.model, adapter_name=self._new_adapter)
        inject_adapter_in_model(peft_config, self.model, adapter_name=self._old_adapter)
        # Activate the trainable adapter (peft's default on second inject
        # might point at "old"; force it to "default" so forward passes
        # outside ``with_old_adapter`` use the trained policy).
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(self._new_adapter)
        # Freeze the ``old`` adapter so optimizer never updates it. peft
        # marks all adapter params trainable on inject; we flip ``old`` to
        # requires_grad=False here, leaving ``default`` as the only set
        # of trainable params.
        self._freeze_adapter(self._old_adapter)
        if _current_rank() == 0:
            n_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
            logger.info(
                "NFTLoRAPolicy: injected adapters %r + %r (rank=%d, alpha=%d, target_modules=%s) — %d trainable params",
                self._new_adapter,
                self._old_adapter,
                self.config.rank,
                self.config.alpha,
                tuple(self.config.target_modules),
                n_trainable,
            )

    def _freeze_adapter(self, adapter_name: str) -> None:
        """Set ``requires_grad=False`` on every LoRA A/B tensor under
        ``adapter_name`` (the ``"old"`` reference adapter must never
        receive optimizer updates — its motion is governed by the EMA
        formula instead).
        """
        from peft.tuners.lora import LoraLayer

        for module in self.model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for matrix in ("lora_A", "lora_B"):
                bank = getattr(module, matrix, None)
                if bank is None or adapter_name not in bank:
                    continue
                bank[adapter_name].weight.requires_grad = False

    # ------------------------------------------------------------------
    # Trainable-only iteration (only ``default`` adapter trains)
    # ------------------------------------------------------------------

    def parameters(self) -> Iterator[Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad)

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, Parameter]]:
        return ((n, p) for n, p in self.model.named_parameters(*args, **kwargs) if p.requires_grad)

    # ------------------------------------------------------------------
    # Post-materialize: reset both adapters + sync default → old
    # ------------------------------------------------------------------

    def post_materialize_init(self) -> None:
        super().post_materialize_init()
        self._reset_lora_parameters(self._new_adapter)
        self._reset_lora_parameters(self._old_adapter)
        self._sync_default_to_old()

    def _reset_lora_parameters(self, adapter_name: str) -> None:
        """peft's standard init: kaiming_uniform_ on ``lora_A``, zeros_ on
        ``lora_B`` for the named adapter. Required after
        ``bundle.materialize`` because ``to_empty`` wipes peft's meta-time
        init.
        """
        from peft.tuners.lora import LoraLayer

        n_reset = 0
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters(adapter_name, init_lora_weights=True)
                n_reset += 1
        if _current_rank() == 0:
            logger.info("NFTLoRAPolicy: reset_lora_parameters(%r) on %d LoraLayer(s)", adapter_name, n_reset)

    # ------------------------------------------------------------------
    # Adapter switching (algorithm consumer's interface)
    # ------------------------------------------------------------------

    @contextmanager
    def with_old_adapter(self):
        """Context manager — temporarily activate the ``"old"`` adapter.

        Algorithm-side usage::

            with torch.no_grad(), nft_lora.with_old_adapter():
                old_pred = stage.predict_noise_at_step(...)

        FSDP-safe: peft's ``set_adapter`` operates on the same in-place
        mutated ``nn.Module`` that ``fully_shard`` wraps.
        """
        if not hasattr(self.model, "set_adapter"):
            raise AttributeError(
                "NFTLoRAPolicy.with_old_adapter: wrapped model has no set_adapter (peft injection didn't take)."
            )
        original = getattr(self.model, "active_adapter", self._new_adapter)
        if isinstance(original, list):
            # peft >=0.7 may store the active adapter list; normalize.
            original = original[0] if original else self._new_adapter
        try:
            self.model.set_adapter(self._old_adapter)
            yield
        finally:
            self.model.set_adapter(original)

    def disable_adapter(self):
        """peft passthrough — context manager that disables LoRA entirely
        (returns to the un-adapted base model). Reserved for future KL
        penalty against the base policy.
        """
        fn = getattr(self.model, "disable_adapter", None)
        if not callable(fn):
            raise AttributeError(
                "NFTLoRAPolicy.disable_adapter: wrapped model has no disable_adapter (peft injection didn't add one)."
            )
        return fn()

    # ------------------------------------------------------------------
    # EMA update hooks (called by Policy-chain walk in stack.py)
    # ------------------------------------------------------------------

    def step(self, optimization_step: Optional[int] = None) -> None:
        """Per-optimizer-step hook (called via ``_step_ema_in_chain``)."""
        if self.config.ema_update_timing != "optimizer_step":
            return
        self._update_old_adapter(optimization_step)

    def on_rollout_end(self, step: Optional[int] = None) -> None:
        """Per-rollout-boundary hook (called via ``_on_rollout_end_in_chain``)."""
        if self.config.ema_update_timing != "rollout_end":
            return
        self._update_old_adapter(step)

    # ------------------------------------------------------------------
    # Dual-adapter EMA: in-place ``old = decay * old + (1 - decay) * default``
    # ------------------------------------------------------------------

    def _current_decay(self, step: Optional[int]) -> float:
        """Decay value for the current update step under the configured schedule.

        For ``constant``: returns ``ema_decay`` always.
        For ``linear``: climbs from 0 toward ``ema_uphold`` at rate
            ``ema_uprate`` per update.
        For ``warmup``: 0 during the first ``ema_flat_steps`` updates,
            then climbs from 0 toward ``ema_uphold`` at ``ema_uprate``.
        """
        s = self._update_counter if step is None else int(step)
        if self.config.ema_decay_type == "linear":
            return float(min(s * self.config.ema_uprate, self.config.ema_uphold))
        if self.config.ema_decay_type == "warmup":
            if s < int(self.config.ema_flat_steps):
                return 0.0
            return float(min((s - int(self.config.ema_flat_steps)) * self.config.ema_uprate, self.config.ema_uphold))
        return float(self.config.ema_decay)

    @torch.no_grad()
    def _update_old_adapter(self, step: Optional[int]) -> None:
        """Run one EMA update step on the ``"old"`` adapter.

        ``decay == 0`` short-circuits to a hard copy (semantically
        equivalent to ``old ← default`` — useful during the warmup-flat
        phase where the reference should track the policy 1:1).
        """
        decay = self._current_decay(step)
        if decay <= 0.0:
            self._sync_default_to_old()
            self._update_counter += 1
            return

        from peft.tuners.lora import LoraLayer

        n_updated = 0
        for module in self.model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for matrix in ("lora_A", "lora_B"):
                bank = getattr(module, matrix, None)
                if bank is None:
                    continue
                src = bank.get(self._new_adapter)
                dst = bank.get(self._old_adapter)
                if src is None or dst is None:
                    raise RuntimeError(
                        f"NFTLoRAPolicy._update_old_adapter: adapter pair missing on "
                        f"{type(module).__name__} ({matrix}); expected {self._new_adapter!r} + {self._old_adapter!r}."
                    )
                dst.weight.data.mul_(decay).add_(src.weight.data, alpha=(1.0 - decay))
                n_updated += 1

        if n_updated == 0:
            raise RuntimeError(
                "NFTLoRAPolicy._update_old_adapter: no LoraLayer adapter pairs found; "
                "peft injection may have failed silently."
            )
        self._update_counter += 1
        if _current_rank() == 0:
            logger.debug(
                "NFTLoRAPolicy: EMA update (step=%s, decay=%.6f, updated=%d layers)",
                step if step is not None else self._update_counter,
                decay,
                n_updated,
            )

    @torch.no_grad()
    def _sync_default_to_old(self) -> None:
        """Hard copy ``default`` adapter weights onto ``old``."""
        from peft.tuners.lora import LoraLayer

        n_copied = 0
        for module in self.model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for matrix in ("lora_A", "lora_B"):
                bank = getattr(module, matrix, None)
                if bank is None:
                    continue
                src = bank.get(self._new_adapter)
                dst = bank.get(self._old_adapter)
                if src is None or dst is None:
                    raise RuntimeError(
                        f"NFTLoRAPolicy._sync_default_to_old: adapter pair missing on "
                        f"{type(module).__name__} ({matrix})."
                    )
                dst.weight.data.copy_(src.weight.data)
                n_copied += 1
        if n_copied == 0:
            raise RuntimeError("NFTLoRAPolicy._sync_default_to_old: no LoraLayer adapter pairs found.")

    # ------------------------------------------------------------------
    # Checkpoint surfaces
    # ------------------------------------------------------------------

    def lora_state_dict(self) -> Dict[str, Any]:
        """Return only the ``default`` adapter's parameters on rank 0.

        Same surface as ``LoRAPolicy.lora_state_dict`` — the ``old``
        adapter is derived state (an EMA tracker) and is persisted
        separately via :meth:`nft_state_dict`.
        """
        full = self.state_dict()
        if _current_rank() != 0:
            return {}
        peft_state = _extract_peft_lora_state(self.model)
        if peft_state:
            return _to_cpu_state_dict(_filter_to_adapter(peft_state, self._new_adapter))
        filtered = _filter_lora_state(full)
        if filtered:
            return _filter_to_adapter(filtered, self._new_adapter)
        raise ValueError(
            "NFTLoRAPolicy.lora_state_dict: no LoRA parameters found on the "
            "wrapped model (neither peft-registered nor key-prefixed)."
        )

    def nft_state_dict(self) -> Dict[str, Any]:
        """Return the ``old`` adapter's parameters + EMA step counter.

        Separate from :meth:`lora_state_dict` so checkpoint loading paths
        can decide whether to resume the EMA reference verbatim (warm
        restart) or re-sync from ``default`` (cold restart).
        """
        full = self.state_dict()
        if _current_rank() != 0:
            return {}
        peft_state = _extract_peft_lora_state(self.model)
        if peft_state:
            adapter_state = _to_cpu_state_dict(_filter_to_adapter(peft_state, self._old_adapter))
        else:
            adapter_state = _filter_to_adapter(_filter_lora_state(full), self._old_adapter)
        return {
            "old_adapter": adapter_state,
            "update_counter": int(self._update_counter),
            "old_adapter_name": str(self._old_adapter),
        }

    def load_nft_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the ``old`` adapter weights + EMA step counter.

        Fail-fast if ``state_dict["old_adapter"]`` is non-empty but no
        target tensors matched the expected key shape — that signals a
        schema mismatch (wrong adapter name, wrong matrix names, peft
        version drift). Letting it pass silently would degrade a warm
        resume into a cold start without anyone noticing.
        """
        adapter_state = state_dict.get("old_adapter") or {}
        # peft-aware load: walk LoraLayer modules and copy in matching keys.
        from peft.tuners.lora import LoraLayer

        copied = 0
        for module in self.model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for matrix in ("lora_A", "lora_B"):
                bank = getattr(module, matrix, None)
                if bank is None or self._old_adapter not in bank:
                    continue
                key_suffix = f".{matrix}.{self._old_adapter}.weight"
                for k, v in adapter_state.items():
                    if k.endswith(key_suffix):
                        bank[self._old_adapter].weight.data.copy_(v.to(bank[self._old_adapter].weight.device))
                        copied += 1
        if adapter_state and copied == 0:
            raise RuntimeError(
                f"NFTLoRAPolicy.load_nft_state_dict: state_dict carried "
                f"{len(adapter_state)} adapter tensors but none matched the "
                f"expected key suffix '.<lora_A|lora_B>.{self._old_adapter}.weight'. "
                f"Schema mismatch — refusing to silently degrade to cold "
                f"start. Sample keys: {list(adapter_state.keys())[:3]}"
            )
        self._update_counter = int(state_dict.get("update_counter", self._update_counter))
        if _current_rank() == 0:
            logger.info("NFTLoRAPolicy.load_nft_state_dict: copied %d tensors into 'old' adapter", copied)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _to_cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
            try:
                value = value.full_tensor()
            except Exception:
                pass
        if isinstance(value, torch.Tensor):
            converted[key] = value.detach().cpu()
        else:
            converted[key] = value
    return converted


def _filter_to_adapter(state_dict: Dict[str, Any], adapter_name: str) -> Dict[str, Any]:
    """Filter a peft-style state dict to entries whose key references the
    named adapter. peft stores LoRA params under keys like
    ``transformer.blocks.0.attn.q_proj.lora_A.<adapter>.weight``.
    """
    token = f".{adapter_name}."
    return {k: v for k, v in state_dict.items() if token in k}


__all__ = ["NFTLoRAPolicyConfig", "NFTLoRAPolicy"]
