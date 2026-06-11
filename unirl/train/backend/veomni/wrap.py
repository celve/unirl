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
    wrap_root_leaves: bool = False,
) -> None:
    """Parallelize ``model`` (on meta) in place via VeOmni FSDP2.

    ``block_class_names`` feeds VeOmni's ``basic_modules`` (its per-module
    ``fully_shard`` targets, unioned with the model's ``_no_split_modules``).

    ``wrap_root_leaves`` (HI3 / outer-wrapper models): give each root-leaf module
    that carries its own params (``wte``, ``ln_f``, …) its OWN ``fully_shard``
    group BEFORE delegating to VeOmni, so VeOmni's mandatory root group owns no
    params and each leaf is independently all-gathered when the outer wrapper
    calls it outside the decoder's managed forward (see
    :func:`_root_leaf_modules`). ``tie_word_embeddings`` excludes the embedding
    from that (shared embed/lm_head weight must stay in one group). Off by
    default so single-module (diffusion) trainables are byte-unchanged.
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

    # Root-leaf own-group wrap. HI3's OUTER wrapper invokes root submodules of
    # the wrapped decoder DIRECTLY, outside the decoder's managed forward: it
    # builds inputs_embeds via ``transformer.model.wte(input_ids)`` (before the
    # body) and applies ``transformer.model.ln_f`` to the output (after it).
    # Under VeOmni's mandatory root ``fully_shard`` those leaves are sharded
    # DTensors at the direct call site, so ``aten.embedding`` / ``aten.mul`` hit
    # "mixed torch.Tensor and DTensor". Give EACH root-leaf-with-params its OWN
    # group here (instance-level — class-name matching would over-match the
    # per-layer norms), so its own pre-forward hook all-gathers it to a plain
    # tensor on the direct call; VeOmni's later root ``fully_shard`` then owns no
    # params (an empty container). Opt-in, so diffusion trainables are unchanged.
    wrapped_leaves: list[str] = []
    if wrap_root_leaves:
        from torch.distributed._composable.fsdp import (
            MixedPrecisionPolicy,
            fully_shard,
        )

        leaf_mesh = api.get_parallel_state().fsdp_mesh
        leaf_mp = MixedPrecisionPolicy(param_dtype=target_dtype, reduce_dtype=torch.float32)
        for leaf_name, leaf in _root_leaf_modules(model, tie_word_embeddings):
            fully_shard(
                leaf,
                mesh=leaf_mesh,
                mp_policy=leaf_mp,
                reshard_after_forward=bool(reshard_after_forward),
            )
            wrapped_leaves.append(leaf_name)

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
        basic_modules=list(block_class_names),
        init_device="meta",
        enable_fsdp_offload=False,
    )

    # FSDP2 root pre-init. With root leaves in their own groups, the OUTER
    # wrapper calls one (e.g. wte) BEFORE the decoder root's forward; FSDP2 marks
    # "the 1st state to run forward" as the root (torch ..._fsdp_state._lazy_init),
    # so leaf-first latches ``_is_root=True`` on the leaf and the real root's
    # later init raises "FSDP state has already been lazily initialized ...
    # requires running forward through the root module first". Pre-initializing
    # the root here stamps every nested module a child (``_is_root=False``) up
    # front, so a direct leaf call no-ops its lazy-init and just all-gathers via
    # its own group's hook. Mirrors verl's post-wrap ``_lazy_init(model, model)``.
    if wrapped_leaves:
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )

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
    """Param-carrying leaf modules reachable OUTSIDE the managed decoder-block forward.

    These are the submodules an outer/inner forward invokes directly, outside the
    decoder blocks' managed forward — the word embedding (``wte`` / ``embed_tokens``,
    to build inputs_embeds before the body) and the final norm (``ln_f`` / ``norm``,
    after it). Each needs its own ``fully_shard`` group so a direct call all-gathers
    it to a plain tensor instead of meeting a root-sharded DTensor (``aten.embedding``
    / ``aten.mul`` "mixed torch.Tensor and DTensor").

    Two topologies are covered:
      * HI3 — the wrapped root's DIRECT children carry the params (``wte``, ``ln_f``).
      * Qwen3 (``…ForCausalLM``) — the wrapped root's direct child ``model`` is a
        param-less container; the replay calls ``transformer.model(...)`` directly,
        so ITS leaves (``embed_tokens``, ``norm``) must be reached one level down.

    Selected at the INSTANCE level (the caller ``fully_shard``s these objects)
    rather than by class name, because the final norm shares its class with the
    per-layer norms inside the decoder blocks — a class-name match would over-wrap
    those. ``ModuleList`` / ``ModuleDict`` containers (the decoder blocks, whose
    params live in the groups wrapped via ``block_class_names``) are skipped. The
    embedding (and any ``lm_head``) is skipped when tied, since a shared weight must
    not be split into a separate group.
    """
    skip_when_tied = {"wte", "embed_tokens", "lm_head"}
    out: "list[Tuple[str, nn.Module]]" = []

    def _consider(name: str, mod: nn.Module) -> None:
        if not any(True for _ in mod.parameters(recurse=False)):
            return
        leaf = name.rsplit(".", 1)[-1]
        if tie_word_embeddings and (isinstance(mod, nn.Embedding) or leaf in skip_when_tied):
            return
        out.append((name, mod))

    for name, child in model.named_children():
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
            continue
        if any(True for _ in child.parameters(recurse=False)):
            _consider(name, child)  # HI3 shape: direct param-leaf (wte / ln_f)
        else:
            # Param-less container (e.g. ``…ForCausalLM.model``): descend ONE level
            # to its own param-leaves (embed_tokens, final norm), skipping the
            # decoder ModuleList whose params live in the block groups.
            for sub_name, sub in child.named_children():
                if isinstance(sub, (nn.ModuleList, nn.ModuleDict)):
                    continue
                _consider(f"{name}.{sub_name}", sub)
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
