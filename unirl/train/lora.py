"""LoRA adapter injection — plain and dual-adapter (DiffusionNFT) variants.

Build-time structural mutation only.  :func:`inject_lora` installs a single
peft adapter; :func:`inject_nft` installs a default + shadow adapter pair and
returns a :class:`~unirl.train.shadow.Shadow` handle for the DiffusionNFT EMA.
Both stamp their post-materialize resets via ``unirl.train.deferred``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Sequence, Tuple

import torch
from torch import nn

from unirl.train.deferred import _stamp
from unirl.train.shadow import Shadow

logger = logging.getLogger(__name__)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = "default",
) -> None:
    """Inject a single LoRA adapter.  No Shadow, no EMA."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_lora: adapter %r (rank=%d, alpha=%d, target_modules=%s) — %d trainable params",
            adapter_name,
            rank,
            alpha,
            tuple(target_modules),
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=adapter_name))


def inject_nft(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    default: str = "default",
    shadow: str = "old",
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> Shadow:
    """Inject dual LoRA adapters for DiffusionNFT-style EMA.  Returns Shadow."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=default)
    inject_adapter_in_model(peft_cfg, model, adapter_name=shadow)

    # peft's inject_adapter_in_model installs the LoRA layers but does not flip
    # diffusers' PeftAdapterMixin `_hf_peft_config_loaded` flag, so the model-level
    # `set_adapter` raises "No adapter loaded". Activate `default` the same
    # per-LoraLayer way swap_out does (works for diffusers + plain modules), and
    # mark the flag so downstream diffusers adapter ops stay consistent.
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = True
    _activate(model, default)
    _freeze_adapter(model, shadow)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_nft: adapters %r + %r (rank=%d, alpha=%d) — %d trainable params",
            default,
            shadow,
            rank,
            alpha,
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=default))
    _stamp(model, partial(_reset_adapter, name=shadow))
    _stamp(model, partial(_copy_adapter, src=default, dst=shadow))

    return Shadow(
        iter_pairs=lambda: _adapter_pairs(model, default, shadow),
        swap_in=lambda: _activate(model, shadow),
        swap_out=lambda: _activate(model, default),
    )


# ------------------------------------------------------------------
# Peft helpers
# ------------------------------------------------------------------


def _freeze_adapter(model: nn.Module, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if name in bank:
                bank[name].weight.requires_grad = False


def _reset_adapter(model: nn.Module, *, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_reset = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters(name, init_lora_weights=True)
            n_reset += 1
    if _current_rank() == 0:
        logger.info("_reset_adapter(%r): %d LoraLayer(s)", name, n_reset)


def _copy_adapter(model: nn.Module, *, src: str, dst: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_copied = 0
    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if src in bank and dst in bank:
                for sp, dp in zip(bank[src].parameters(), bank[dst].parameters()):
                    dp.data.copy_(sp.data)
                n_copied += 1
    if n_copied == 0:
        raise RuntimeError(f"_copy_adapter: no adapter pairs found for {src!r} -> {dst!r}")


def _adapter_pairs(
    model: nn.Module,
    default: str,
    shadow: str,
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    from peft.tuners.lora import LoraLayer

    pairs: list[Tuple[torch.Tensor, torch.Tensor]] = []
    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if default in bank and shadow in bank:
                for sp, dp in zip(bank[default].parameters(), bank[shadow].parameters()):
                    pairs.append((sp, dp))
    return pairs


def _activate(model: nn.Module, adapter_name: str) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.set_adapter(adapter_name)


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["inject_lora", "inject_nft"]
