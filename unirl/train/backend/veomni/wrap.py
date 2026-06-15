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
from typing import Optional, Tuple

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
    master_dtype: Optional[str] = None,
    reshard_after_forward: bool = True,
    activation_checkpointing: bool = False,
    use_torch_compile: bool = False,
    tie_word_embeddings: bool = False,
    wrap_root_leaves: bool = False,
) -> None:
    """Parallelize ``model`` (on meta) in place via VeOmni FSDP2.

    ``block_class_names`` feeds VeOmni's ``basic_modules`` (its per-module
    ``fully_shard`` targets, unioned with the model's ``_no_split_modules``).

    ``master_dtype`` (e.g. ``"fp32"``) keeps the sharded master weights + optimizer
    states at that dtype while ``MixedPrecisionPolicy(param_dtype)`` still casts the
    all-gathered compute copy to ``param_dtype`` (bf16) — the standard "fp32 master +
    bf16 compute" recipe. Essential for full-finetune RL with tiny gradients (e.g.
    DRPO/GRPO grad-norm ~1e-2): a bf16 master rounds those updates to zero. ``None``
    (default) follows ``param_dtype`` for the master (the prior all-bf16 behavior;
    fine for LoRA, where the trainable adapter update scale is large).

    ``wrap_root_leaves`` (HI3 / outer-wrapper composites): give each root-leaf
    module that carries its own params (``wte``, ``ln_f``, …) its OWN
    ``fully_shard`` group BEFORE delegating to VeOmni, so VeOmni's mandatory root
    group owns no params and each leaf is independently all-gathered when the
    outer wrapper calls it outside the decoder's managed forward (see
    :func:`_root_leaf_modules`). ``tie_word_embeddings`` excludes the embedding
    (a shared embed/lm_head weight must stay in one group). Both off by default
    so single-module (diffusion) trainables are byte-unchanged.
    """
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2

    compute_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    dtype_name = _DTYPE_NAMES.get(compute_dtype)
    if dtype_name is None:
        raise ValueError(f"veomni_parallelize: unsupported param_dtype {param_dtype!r}")

    # Master-weight dtype: cast on meta (dtype-only, no data) so to_empty
    # materializes storage in this dtype; MixedPrecisionPolicy(param_dtype) then
    # casts the compute copy to bf16. master_dtype=None -> master follows
    # param_dtype (all-bf16). Mirrors fsdp_wrap's master/compute split.
    master_t = parse_torch_dtype(master_dtype, field_name="training.fsdp.master_dtype") if master_dtype else compute_dtype
    model.to(master_t)

    # Root-leaf own-group wrap (HI3 / outer-wrapper composites). HI3's OUTER
    # wrapper invokes root submodules of the wrapped decoder DIRECTLY, outside the
    # decoder's managed forward: it builds inputs_embeds via
    # ``transformer.model.wte(input_ids)`` (before the body) and applies
    # ``transformer.model.ln_f`` to the output (after it). Under VeOmni's
    # mandatory root ``fully_shard`` those leaves are sharded DTensors at the
    # direct call site, so ``aten.embedding`` / ``aten.mul`` hit "mixed
    # torch.Tensor and DTensor". Give EACH root-leaf-with-params its OWN group
    # here (instance-level — class-name matching would over-match the per-layer
    # norms), so its own pre-forward hook all-gathers it to a plain tensor; the
    # later root ``fully_shard`` then owns no params. Opt-in, so diffusion
    # trainables are byte-unchanged.
    wrapped_leaves: list[str] = []
    if wrap_root_leaves:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
        from veomni.distributed.parallel_state import get_parallel_state

        leaf_mesh = get_parallel_state().fsdp_mesh
        leaf_mp = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
        for leaf_name, leaf in _root_leaf_modules(model, tie_word_embeddings):
            fully_shard(
                leaf,
                mesh=leaf_mesh,
                mp_policy=leaf_mp,
                reshard_after_forward=bool(reshard_after_forward),
            )
            wrapped_leaves.append(leaf_name)

    mixed_precision = MixedPrecisionConfig(
        enable=True,
        param_dtype=dtype_name,
        reduce_dtype="float32",
    )
    parallelize_model_fsdp2(
        model,
        weights_path=None,
        enable_reshard_after_forward=bool(reshard_after_forward),
        mixed_precision=mixed_precision,
        basic_modules=list(block_class_names),
        init_device="meta",
        enable_fsdp_offload=False,
    )

    # FSDP2 root pre-init (root-leaf composites). With root leaves in their own
    # groups, the OUTER wrapper calls one (e.g. wte) BEFORE the decoder root's
    # forward; FSDP2 marks "the 1st state to run forward" as the root, so
    # leaf-first would latch ``_is_root=True`` on the leaf and the real root's
    # later init raises "already lazily initialized". Pre-initializing the root
    # here stamps every nested module a child up front, so a direct leaf call
    # just all-gathers via its own group's hook. Mirrors verl's post-wrap
    # ``_lazy_init``.
    if wrapped_leaves:
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

        root_state = _get_module_fsdp_state(model)
        if root_state is None:
            logger.warning(
                "veomni_parallelize: no FSDP state on root after parallelize; "
                "root-leaf lazy-init guard inactive (leaf-before-root may crash)."
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
            "veomni_parallelize: wrapped %d block(s) of class %r + %d root-leaf "
            "group(s) %r + root (dtype=%s, reshard=%s, ac=%s, compile=%s, tie=%s)",
            len(block_instances),
            tuple(block_class_names),
            len(wrapped_leaves),
            tuple(wrapped_leaves),
            dtype_name,
            reshard_after_forward,
            activation_checkpointing,
            use_torch_compile,
            tie_word_embeddings,
        )


def _root_leaf_modules(
    model: nn.Module,
    tie_word_embeddings: bool,
) -> "list[Tuple[str, nn.Module]]":
    """Direct-child leaf modules of the wrapped root that carry their OWN params.

    These are the root submodules HI3's OUTER wrapper invokes directly, outside
    the decoder's managed forward — the word embedding ``wte`` (to build
    inputs_embeds before the body) and the final norm ``ln_f`` (applied to the
    output after it). Each needs its own ``fully_shard`` group so a direct call
    all-gathers it to a plain tensor instead of meeting a root-sharded DTensor.

    Selected at the INSTANCE level (the caller ``fully_shard``s these objects)
    rather than by class name, because the final norm shares its class with the
    per-layer norms inside the decoder blocks — a class-name match would
    over-wrap those. Container children (e.g. the decoder ``ModuleList``) carry
    no params of their own and are skipped. The embedding (and any ``lm_head``)
    is skipped when tied, since a shared weight must not be split into a separate
    group.
    """
    skip_when_tied = {"wte", "embed_tokens", "lm_head"}
    out: "list[Tuple[str, nn.Module]]" = []
    for name, child in model.named_children():
        if not any(True for _ in child.parameters(recurse=False)):
            continue
        if tie_word_embeddings and (isinstance(child, nn.Embedding) or name in skip_when_tied):
            continue
        out.append((name, child))
    return out


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
