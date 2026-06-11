"""VeOmni FSDP2 model wrapping for the VeOmni backend.

Calls VeOmni's *inner* ``parallelize_model_fsdp2`` directly — the outer
``build_parallelize_model`` would first upcast the model to fp32 master
weights (``model.float()``), apply HF-API gradient checkpointing, and a
vestigial TP path; bypassing it keeps bf16 master weights and the same
memory/numerics regime as ``unirl.train.backend.fsdp.wrap.fsdp_wrap``
(which force-casts to bf16), so the two backends are A/B-comparable.

Differences vs the fsdp wrap (by VeOmni design, accepted for v1):
* the model root IS ``fully_shard``-ed (root auto-no-reshard) — fine for
  single-module trainables; composites (WAN22/HI3) are out of scope.
* requires the model on the meta device (`init_device="meta"` is asserted
  by VeOmni); materialization happens inside the call (``to_empty`` + the
  model's ``init_weights``, which the bundle stamps to a no-op — real
  weights load afterwards in ``backend.py``).

Runs in the backend constructor after structural injection
(``unirl.train.lora`` / ``unirl.train.ema``) and before the weight load.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def veomni_parallelize(
    model: nn.Module,
    *,
    block_class_names: Tuple[str, ...],
    param_dtype: str = "bf16",
    reshard_after_forward: bool = True,
    activation_checkpointing: bool = False,
    use_torch_compile: bool = False,
    tie_word_embeddings: bool = False,
) -> None:
    """Parallelize ``model`` (on meta) in place via VeOmni FSDP2.

    ``block_class_names`` feeds VeOmni's ``basic_modules`` (its per-module
    ``fully_shard`` targets, unioned with the model's ``_no_split_modules``).
    Word-embedding leaves (``wte`` / ``embed_tokens``) are unioned in too so
    each gets its OWN ``fully_shard`` group (see :func:`_embedding_block_classes`);
    ``tie_word_embeddings`` suppresses that for tied embed/lm_head models.
    """
    from unirl.train.backend.veomni import _compat

    api = _compat.load()

    target_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    dtype_name = _DTYPE_NAMES.get(target_dtype)
    if dtype_name is None:
        raise ValueError(f"veomni_parallelize: unsupported param_dtype {param_dtype!r}")

    # bf16 master weights: cast on meta (dtype-only, no data) so to_empty
    # materializes storage in the target dtype. Mirrors fsdp_wrap's cast.
    model.to(target_dtype)

    # Give word-embedding leaves their OWN fully_shard group by unioning their
    # class names into VeOmni's basic_modules (VeOmni wraps each target class
    # BEFORE the root, torch_parallelize.py). HI3 looks up
    # ``transformer.model.wte(input_ids)`` from the OUTER wrapper's forward,
    # before the decoder's FSDP-managed forward — so without its own pre-forward
    # all-gather hook the root-sharded embedding weight is still a DTensor at the
    # lookup and ``aten.embedding`` raises "mixed torch.Tensor and DTensor".
    # Gated on leaf-name, so diffusion DiTs are untouched (strict no-op).
    embedding_classes = _embedding_block_classes(model, tie_word_embeddings)
    basic_modules = list(block_class_names) + sorted(embedding_classes - set(block_class_names))

    mixed_precision = api.MixedPrecisionConfig(
        enable=True,
        param_dtype=dtype_name,
        reduce_dtype="float32",
    )
    api.parallelize_model_fsdp2(
        model,
        weights_path=None,
        enable_reshard_after_forward=bool(reshard_after_forward),
        mixed_precision=mixed_precision,
        basic_modules=basic_modules,
        init_device="meta",
        enable_fsdp_offload=False,
    )

    # FSDP2 root pre-init. HI3 builds inputs_embeds by calling
    # ``transformer.model.wte(input_ids)`` from the OUTER wrapper's forward,
    # BEFORE the decoder root's own forward runs. FSDP2 marks "the 1st state to
    # run forward" as the root (torch ..._fsdp_state._lazy_init), so wte-first
    # latches ``_is_root=True`` on the embedding and the real root's later init
    # raises "FSDP state has already been lazily initialized for wte ... requires
    # running forward through the root module first". Pre-initializing the root
    # here stamps every nested module a child (``_is_root=False``) up front, so a
    # direct wte() call no-ops its lazy-init and just all-gathers via its own
    # group's hook. Mirrors verl's post-wrap ``_lazy_init(model, model)``. Only
    # needed when an embedding got its own group (a leaf invoked outside the root
    # forward); skipped otherwise so the diffusion recipes stay byte-identical.
    if embedding_classes:
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )

        root_state = _get_module_fsdp_state(model)
        if root_state is None:
            logger.warning(
                "veomni_parallelize: no FSDP state on root after parallelize; "
                "embedding lazy-init guard inactive (wte-before-root may crash)."
            )
        else:
            root_state._lazy_init()

    block_instances = _enumerate_block_instances(model, block_class_names)

    if activation_checkpointing:
        from torch.utils import checkpoint as _ckpt

        def _make_ckpt_forward(orig_fwd: object) -> object:
            def wrapped(*args: object, **kwargs: object) -> object:
                def fn(*a: object) -> object:
                    return orig_fwd(*a, **kwargs)

                return _ckpt.checkpoint(fn, *args, use_reentrant=False)

            return wrapped

        for layer in block_instances:
            layer.forward = _make_ckpt_forward(layer.forward)

    if use_torch_compile:
        for layer in block_instances:
            layer.forward = torch.compile(layer.forward)

    if _current_rank() == 0:
        logger.info(
            "veomni_parallelize: wrapped %d block(s) of class %r + %d embedding "
            "group(s) %r + root (dtype=%s, reshard=%s, ac=%s, compile=%s, tie=%s)",
            len(block_instances),
            tuple(block_class_names),
            len(embedding_classes),
            tuple(sorted(embedding_classes)),
            dtype_name,
            reshard_after_forward,
            activation_checkpointing,
            use_torch_compile,
            tie_word_embeddings,
        )


def _embedding_block_classes(
    model: nn.Module,
    tie_word_embeddings: bool,
) -> frozenset:
    """Class names of word-embedding leaves that need their OWN fully_shard group.

    HI3 builds ``inputs_embeds`` by calling ``transformer.model.wte(input_ids)``
    from the OUTER wrapper's forward (``ar.py`` gen_text), BEFORE the wrapped
    decoder's own FSDP-managed forward runs. If the embedding is only part of the
    root group, its all-gather pre-forward hook has not fired at the lookup, so
    the root-sharded weight is still a DTensor and ``aten.embedding`` raises "got
    mixed torch.Tensor and DTensor". Wrapping the embedding as its own group
    gives it a pre-forward hook that all-gathers it to a plain tensor on that
    direct call (mirrors verl's ``_select_fsdp2_wrap_targets``).

    Gated on leaf-name ``{"wte", "embed_tokens"}`` ONLY (not ``isinstance`` on
    ``nn.Embedding``): diffusion DiTs have no such leaf — QwenImage's
    ``addition_t_embedding`` is a different name — so this is a strict no-op for
    them and their checksum parity is preserved. Returns empty when tied, since a
    shared lm_head/embedding weight must not be split into a separate group.
    """
    if tie_word_embeddings:
        return frozenset()
    classes = set()
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf in {"wte", "embed_tokens"} and hasattr(mod, "weight"):
            classes.add(type(mod).__name__)
    return frozenset(classes)


def _enumerate_block_instances(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[nn.Module, ...]:
    if not class_names:
        return ()
    names = set(class_names)
    return tuple(m for _, m in model.named_modules() if type(m).__name__ in names)


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["veomni_parallelize"]
