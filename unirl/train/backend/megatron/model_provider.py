"""mcore model construction via AutoBridge (M0).

M0 uses AutoBridge (``mbridge`` / ``megatron.bridge``) as the single source of the
mcore ``GPTModel`` AND of HF<->mcore weight conversion: ``from_hf_pretrained``
derives the model + parallelism wiring from the HF config, ``load_hf_weights``
seeds it, and ``export_hf_weights`` (used by the weight-sync walk) reads it back
out — so M0 needs no hand-written per-arch converter. Ported from slime's bridge
path (``slime/backends/megatron_utils/model_provider.py:87-123``).

The raw ``GPTModel`` + hand-written ``megatron_to_hf`` converter path is the M1
alternative for archs the bridge does not cover.

VERIFY: every mcore/bridge symbol below is version-sensitive — pin one
mcore + TransformerEngine + bridge combo and confirm the exact import paths and
signatures (``to_megatron_provider``, ``get_model``, ``ModelType``) against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from unirl.train.configs import MegatronConfig


def build_mcore_model(cfg: "MegatronConfig") -> Tuple[List[Any], Any]:
    """Build the mcore model (list of DDP-wrapped chunks) + the AutoBridge.

    Returns ``(model_chunks, bridge)``. At M0 ``model_chunks`` has exactly one
    entry (pp=vpp=1); the backend exposes ``model_chunks[0]`` as ``self.model``
    for the single-chunk contract (``trainable_module``, weight walk) and keeps
    the list for ``get_forward_backward_func``. Initial weights are loaded by the
    caller via ``bridge.load_hf_weights(model_chunks)``.
    """
    # VERIFY: bridge import path — mbridge exposes `from mbridge import AutoBridge`;
    # NVIDIA's is `from megatron.bridge import AutoBridge`. Pick per the pin.
    from megatron.bridge import AutoBridge
    from megatron.core.enums import ModelType
    from megatron.training import get_model

    bridge = AutoBridge.from_hf_pretrained(cfg.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    # Stamp the parallelism degrees onto the provider (M0: all 1). sequence_parallel
    # is meaningless at tp=1 (slime/verl force it off there). finalize() wires the
    # derived TransformerConfig.
    provider.tensor_model_parallel_size = cfg.tp_size
    provider.pipeline_model_parallel_size = cfg.pp_size
    provider.expert_model_parallel_size = cfg.ep_size
    provider.expert_tensor_parallel_size = cfg.etp_size
    provider.context_parallel_size = cfg.cp_size
    provider.sequence_parallel = cfg.tp_size > 1
    provider.finalize()

    # get_model wraps each pp/vpp stage's GPTModel in mcore DistributedDataParallel
    # and returns the chunk list. VERIFY the get_model signature (wrap_with_ddp kw,
    # ModelType enum) against the pinned mcore.
    model_chunks = get_model(provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
    return model_chunks, bridge
