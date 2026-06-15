"""Ulysses sequence-parallel (SP) patches for the VeOmni backend.

``apply_sequence_parallelism(model, sp_size)`` is called by
:class:`~unirl.train.backend.veomni.backend.VeOmniBackend` immediately after
``veomni_parallelize`` (FSDP2 wrap), and dispatches by trainable-module class to
a per-architecture patcher:

* **AR** (HF causal-LMs, e.g. Qwen3) — :mod:`.ar`: route attention through
  VeOmni's registered ``veomni_flash_attention_2_with_sp`` and wrap the decoder
  forward to slice the sequence in / gather the hidden states out.
* **Diffusion** (diffusers transformers) — :mod:`.diffusion`.
* **HunyuanImage3** (composite custom-SDPA decoder) — :mod:`.hunyuan_image3`:
  monkey-patch ``HunyuanImage3SDPAAttention`` with the Ulysses all-to-all and
  wrap the decoder forward to slice the sequence in / gather hidden out (the
  dense 4D mask stays full).

A true no-op at ``sp_size == 1`` (returns before touching the model). SP itself
is gated at run time on ``get_parallel_state().ulysses_enabled`` inside the
wrappers, so even when installed it is inert unless the folded mesh has
``ulysses_size > 1``.
"""

from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


def apply_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Install the Ulysses SP patch on ``model`` in place (no-op if sp_size<=1)."""
    if sp_size <= 1:
        return

    from unirl.train.backend.veomni.sp import ar, diffusion, hunyuan_image3

    if hunyuan_image3.is_hunyuan_image3_model(model):
        hunyuan_image3.apply_hi3_sequence_parallelism(model, sp_size)
        return

    if ar.is_ar_causal_lm(model):
        ar.apply_ar_sequence_parallelism(model, sp_size)
        return

    if diffusion.is_diffusers_transformer(model):
        diffusion.apply_diffusion_sequence_parallelism(model, sp_size)
        return

    raise NotImplementedError(
        f"apply_sequence_parallelism: no SP patcher for {type(model).__name__} "
        "(neither an HF causal-LM nor a diffusers transformer)."
    )


__all__ = ["apply_sequence_parallelism"]
