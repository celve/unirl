"""LoRAPolicy — peft adapter injection as a stackable :class:`Policy`.

Composes with a Stage or another Policy. peft injection happens
**in place** via :func:`peft.utils.inject_adapter_in_model`, so the
underlying ``nn.Module`` reference stays stable; downstream policies
(notably :class:`FSDPPolicy`) wrap the same object now containing
``LoraLinear`` substitutions.

Standard order::

    pipe = HunyuanImage3Pipeline.from_meta_config(cfg)
    lora = LoRAPolicy(lora_cfg, pipe.diffusion)         # peft inject on meta
    fsdp = FSDPPolicy(fsdp_cfg, lora)                    # FSDP wraps base + adapters
    pipe.bundle.materialize(device=cuda, with_aux=("vae",))
    fsdp.post_materialize_init()                          # walks inward → LoRAPolicy resets adapters

The post-materialize re-init is necessary because ``to_empty`` (called
by ``bundle.materialize``) wipes peft's meta-time init, and the HF
state_dict that DCP loads has no LoRA keys (the bundle's
``_collect_filtered_state_dict`` whitelist excludes them).
``post_materialize_init`` walks the source chain inward first (so the
base is fully materialized + loaded), then runs peft's standard
adapter init (kaiming_uniform_ on ``lora_A``, zeros_ on ``lora_B``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, Tuple, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter

from diffusionrl.config.registration import register_config
from diffusionrl.training_new.fsdp_policy import (
    _extract_peft_lora_state,
    _filter_lora_state,
)
from diffusionrl.training_new.policy import Policy, PolicyBase

logger = logging.getLogger(__name__)


@register_config(
    group="training_new/policy",
    name="lora",
    target="diffusionrl.training_new.lora_policy.LoRAPolicy",
)
@dataclass
class LoRAPolicyConfig:
    """Construction args for :class:`LoRAPolicy`.

    Mirrors the subset of :class:`peft.LoraConfig` fields we use today
    (rank, alpha, target_modules, dropout). Add fields here as new peft
    knobs are needed; the LoRAPolicy constructor maps them onto
    ``peft.LoraConfig`` at injection time.
    """

    name: ClassVar[str] = "lora"

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    # Module names (suffix-matched against the ``nn.Linear`` instances
    # peft walks). For HI3: ``("q_proj", "k_proj", "v_proj", "o_proj")``.
    # For SD3: depends on the JointTransformerBlock internals.
    target_modules: Tuple[str, ...] = field(default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj"))
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"


class LoRAPolicy(PolicyBase):
    """peft adapter injection as a stackable :class:`Policy`.

    Inherits :class:`PolicyBase` for ``trainable_module``, ``replay``,
    ``state_dict``, ``load_state_dict``, ``train``, ``eval``. Overrides:

    - ``parameters`` / ``named_parameters`` — filter to
      ``requires_grad=True`` (peft sets base ``requires_grad=False``).
    - ``post_materialize_init`` — chain inward, then reset LoRA
      parameters via peft's standard init.

    Adds:

    - ``lora_state_dict`` — peft-aware adapter-only state extractor.
    - ``disable_adapter`` — peft passthrough context manager.
    """

    def __init__(
        self,
        config: LoRAPolicyConfig,
        source: Union[Any, Policy],
    ) -> None:
        if not hasattr(source, "trainable_module"):
            raise TypeError(
                f"LoRAPolicy: source {type(source).__name__} has no "
                "trainable_module() method. Source must be a Stage or "
                "another Policy that exposes the wrap target."
            )
        self.config = config
        self.source = source
        # ``self.model`` is the same nn.Module object as
        # ``source.trainable_module()`` — peft's ``inject_adapter_in_model``
        # mutates it in place (replaces nn.Linear with peft's LoraLinear
        # for matched names; sets requires_grad=False on base params).
        self.model: nn.Module = source.trainable_module()
        self._adapter_name = "default"
        self._inject_lora()

    # ------------------------------------------------------------------
    # Injection (in-place)
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
        inject_adapter_in_model(peft_config, self.model, adapter_name=self._adapter_name)
        if _current_rank() == 0:
            n_lora_params = sum(1 for p in self.model.parameters() if p.requires_grad)
            logger.info(
                "LoRAPolicy: injected adapter %r (rank=%d, alpha=%d, target_modules=%s) — %d trainable params",
                self._adapter_name,
                self.config.rank,
                self.config.alpha,
                tuple(self.config.target_modules),
                n_lora_params,
            )

    # ------------------------------------------------------------------
    # Trainable-only iteration (peft sets base requires_grad=False)
    # ------------------------------------------------------------------

    def parameters(self) -> Iterator[Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad)

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, Parameter]]:
        return ((n, p) for n, p in self.model.named_parameters(*args, **kwargs) if p.requires_grad)

    # ------------------------------------------------------------------
    # Post-materialize: reset LoRA adapters (kaiming_uniform_(A), zeros_(B))
    # ------------------------------------------------------------------

    def post_materialize_init(self) -> None:
        # Inner first — base must be materialized + loaded before adapter
        # re-init runs, so that any tensor ops that touch base data work.
        super().post_materialize_init()
        self._reset_lora_parameters()

    def _reset_lora_parameters(self) -> None:
        """Walk LoraLinear modules and re-initialize adapter weights.

        Required after ``bundle.materialize``: ``to_empty`` allocated
        cuda storage but wiped peft's meta-time init, and DCP's HF
        state_dict has no LoRA keys (filtered out by bundle), so adapter
        params currently hold whatever ``to_empty`` left there.
        """
        from peft.tuners.lora import LoraLayer

        n_reset = 0
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters(self._adapter_name, init_lora_weights=True)
                n_reset += 1
        if _current_rank() == 0:
            logger.info(
                "LoRAPolicy: reset_lora_parameters on %d LoraLayer(s)",
                n_reset,
            )

    # ------------------------------------------------------------------
    # LoRA-specific surface (off-Protocol)
    # ------------------------------------------------------------------

    def lora_state_dict(self) -> Dict[str, Any]:
        """Return only LoRA / peft adapter parameters on rank 0.

        Prefers the peft-aware path; falls back to substring-matching
        ``"lora"`` in parameter keys. Goes through the source chain's
        ``state_dict`` so the FSDP-aware DCP gather happens first; this
        method only filters the gathered dict.
        """
        full = self.state_dict()  # delegates inward to FSDPPolicy DCP gather
        if _current_rank() != 0:
            return {}

        peft_state = _extract_peft_lora_state(self.model)
        if peft_state:
            return _to_cpu_state_dict(peft_state)
        filtered = _filter_lora_state(full)
        if filtered:
            return filtered
        raise ValueError(
            "LoRAPolicy.lora_state_dict: no LoRA parameters found on the "
            "wrapped model (neither peft-registered nor key-prefixed)."
        )

    def disable_adapter(self):
        """peft passthrough — context manager that disables LoRA/IA3 etc.
        for the duration of a forward, used by KL-penalty paths comparing
        against the un-adapted policy.
        """
        fn = getattr(self.model, "disable_adapter", None)
        if not callable(fn):
            raise AttributeError(
                "LoRAPolicy.disable_adapter: wrapped model has no disable_adapter() (peft injection didn't add one)."
            )
        return fn()


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


__all__ = ["LoRAPolicyConfig", "LoRAPolicy"]
