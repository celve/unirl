"""VeOmniBackend — single-track training-state Remote on VeOmni FSDP2.

Drop-in sibling of :class:`unirl.train.backend.fsdp.FSDPBackend`: both subclass
:class:`~unirl.train.backend.base_backend.BaseFSDP2Backend`, which owns the
training step, EMA swap, checkpoint envelope, and memory lifecycle. This leaf
supplies only the constructor lifecycle and the five engine hooks, whose wrap /
grad-clip / offload internals come from VeOmni's distributed layer via the
:mod:`._compat` selective-import shim. Recipes select it purely by ``_target_``.

Lifecycle differences vs FSDPBackend (all internal to construction):

* The default process group is brought up *explicitly* (VeOmni builds its device
  meshes before any ``fully_shard`` call, so torch's lazy auto-init never fires),
  and ``init_parallel_state`` is invoked — one VeOmni-wrapped model per process.
* The trainable module must arrive on the **meta** device (the bundle's
  ``meta_init_transformer`` flag): VeOmni's parallelize materializes it via
  ``to_empty`` and calls its (no-op-stamped) ``init_weights``; the real weights
  load *after* sharding (``eager_ok=False``).
* LoRA/NFT/mirror injection runs on the meta module — exactly the contract
  ``unirl.train.deferred`` documents — and ``apply_deferred_ops`` drains the
  post-materialize resets *after* the weight load.

Checkpointing: ``save``/``load`` are inherited from the base; the optimizer-state
mechanism is supplied by the hooks below as a plain per-rank ``state_dict()``
(NOT DCP) — VeOmni shards over a folded ``dp_shard x ulysses`` mesh where the DCP
optimizer gather is unverified.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from unirl.models.types.bundle import Bundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig, resolve_trainable_module
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.backend.sharded_load import load_trainable_weights
from unirl.train.backend.sharded_state import StateDict, move_optimizer_state
from unirl.train.backend.veomni.state import clip_grad_norm, veomni_offload, veomni_onload
from unirl.train.backend.veomni.wrap import veomni_parallelize
from unirl.train.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)
from unirl.train.deferred import apply_deferred_ops


class VeOmniBackend(BaseFSDP2Backend):
    """Single-track VeOmni-FSDP2 training backend.

    One-shot construction: after ``__init__`` returns the backend is fully usable
    (model wrapped, weights loaded, optimizer/scheduler/EMA built). ``device`` /
    ``rank`` kwargs are accepted for signature parity with :class:`FSDPBackend`
    but resolved from the actor env + process group — backends are constructed
    before ``Remote.setup()`` delivers rank info.
    """

    def __init__(
        self,
        *,
        bundle: Bundle,
        block_class_names: Tuple[str, ...],
        fsdp_cfg: FSDPConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        device: Optional[torch.device] = None,
        rank: int = 0,
        trainable_attr: str = "transformer",
        wrap_root_leaves: bool = False,
        lora_cfg: Optional[LoraConfig] = None,
        ema_lora_cfg: Optional[EmaLoraConfig] = None,
        ema_cfg: Optional[EmaFullConfig] = None,
        with_aux: Tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._check_lora_exclusivity(lora_cfg, ema_lora_cfg)
        _validate_fsdp_cfg(fsdp_cfg)

        from unirl.train.backend.veomni import _compat

        # 1-3. Distributed bring-up: device binding, default PG, VeOmni parallel
        # state (re-init warns + no-ops, enforcing one VeOmni-wrapped model/process).
        _, _, local_rank = _compat.rank_world_local()
        _compat.ensure_dist_initialized(local_rank)
        import torch.distributed as dist

        self._rank = dist.get_rank() if dist.is_initialized() else int(rank)
        world = dist.get_world_size() if dist.is_initialized() else 1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _compat.ensure_installed()
        from veomni.distributed.parallel_state import init_parallel_state

        # Ulysses sequence parallelism, folded into the FSDP shard mesh
        # (include_sp_in_fsdp=True default): the shard mesh becomes
        # dp_shard(world//sp) x ulysses(sp), so params shard across the whole
        # world and grads reduce-scatter across SP ranks automatically (verified:
        # docs/usp-derisk/sp_fsdp.py, no manual sp_size compensation). sp_size=1 is
        # a true no-op: dp_size=world, ulysses_size=1 -> the prior 1D dp_shard mesh.
        self._sp_size = int(getattr(fsdp_cfg, "sp_size", 1) or 1)
        if world % self._sp_size != 0:
            raise ValueError(
                f"VeOmniBackend: world_size {world} not divisible by sp_size {self._sp_size}"
            )
        init_parallel_state(
            dp_size=world // self._sp_size,
            ulysses_size=self._sp_size,
            dp_mode="fsdp2",
            device_type=self._device.type,
        )

        self._bundle = bundle
        model = resolve_trainable_module(bundle, trainable_attr)

        # Structural injection on the meta module (the documented
        # unirl.train.deferred contract: mutate on meta, stamp resets).
        shadow = self._inject_structural(model, lora_cfg, ema_lora_cfg, ema_cfg)

        # HI3 grouped-MoE: when SP is on, restructure each HunyuanMoE's per-expert
        # ModuleList into stacked frozen params (moe_gate_up/moe_down) on meta BEFORE
        # fully_shard, so FSDP shards the stacked params and the SP adapter runs the
        # experts through veomni's M-invariant bit-exact grouped-GEMM. The 80B
        # checkpoint per-expert keys are stacked at load (bundle.materialize). Experts
        # are frozen (LoRA targets qkv/o_proj), so no adapter/weight-sync collision.
        import os as _os

        if self._sp_size > 1 and not _os.environ.get("UNIRL_HI3_GATHER_MOE"):
            import logging as _logging

            from unirl.train.backend.veomni.sp import hunyuan_image3 as _hi3sp

            if _hi3sp.is_hunyuan_image3_model(model):
                from unirl.train.backend.veomni.sp.hi3_moe import restructure_hi3_experts

                _n = restructure_hi3_experts(model)
                _logging.getLogger(__name__).info(
                    "HI3 grouped-MoE: restructured %d MoE blocks to stacked experts (pre-FSDP)", _n
                )

        # Shard + materialize (to_empty; init_weights is a bundle-stamped no-op).
        # Root-wrapped by VeOmni. ``wrap_root_leaves`` (HI3 / outer-wrapper
        # composites) additionally gives each root-leaf-with-params (wte/ln_f)
        # its own group so the outer wrapper can call them outside the decoder's
        # managed forward. The tie flag lives on the bundle WRAPPER config — the
        # wrapped decoder has no ``.config`` — and degrades to False (the safe
        # separate-wrap default for the embedding-DTensor bug).
        tie_word_embeddings = bool(
            getattr(getattr(self._bundle, "transformer", None), "config", None)
            and getattr(self._bundle.transformer.config, "tie_word_embeddings", False)
        )
        veomni_parallelize(
            model,
            block_class_names=tuple(block_class_names),
            param_dtype=fsdp_cfg.param_dtype,
            master_dtype=getattr(fsdp_cfg, "master_dtype", None),
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            activation_checkpointing=fsdp_cfg.activation_checkpointing,
            use_torch_compile=fsdp_cfg.use_torch_compile,
            tie_word_embeddings=tie_word_embeddings,
            wrap_root_leaves=wrap_root_leaves,
        )

        # Ulysses sequence parallelism (no-op at sp_size=1): route attention
        # through VeOmni's registered SP attn and wrap the decoder forward to
        # slice the sequence in / gather hidden out. Installed AFTER
        # veomni_parallelize (the GC -> FSDP -> SP order); gated at run time on
        # ulysses_enabled, so safe to call unconditionally.
        from unirl.train.backend.veomni.sp import apply_sequence_parallelism

        apply_sequence_parallelism(model, self._sp_size)

        # Real weights: load into the freshly-sharded module. Meta-init bundles
        # stash a safetensors dir; Pattern-A bundles materialize themselves; eager
        # bundles are rejected (eager_ok=False) — parallelize already to_empty'd,
        # so their weights are gone (FSDPBackend territory).
        load_trainable_weights(
            model,
            bundle,
            device=self._device,
            rank=self._rank,
            with_aux=with_aux,
            eager_ok=False,
        )

        # Post-materialize resets (LoRA adapter init, mirror copies).
        apply_deferred_ops(model)

        self._finalize_construction(
            model,
            shadow,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            lora_cfg=lora_cfg,
            ema_lora_cfg=ema_lora_cfg,
            ema_cfg=ema_cfg,
            fsdp_cfg=fsdp_cfg,
        )

    # ------------------------------------------------------------------
    # Engine hooks (VeOmni FSDP2)
    # ------------------------------------------------------------------

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        # VeOmni's clip takes the model (dispatches on EP / cpu-offload attrs).
        return clip_grad_norm(self.model, max_grad_norm)

    def _gather_optimizer_state(self) -> StateDict:
        # Plain per-rank state_dict (NOT DCP): VeOmni shards over a folded
        # dp_shard x ulysses mesh where the DCP optimizer gather is unverified.
        # Every rank computes its dict; only dist rank 0's is written.
        return self.optimizer.state_dict()

    def _load_optimizer_state(self, optimizer_state: StateDict) -> None:
        # Every rank loaded the full checkpoint, so the dict is real locally.
        self.optimizer.load_state_dict(optimizer_state)
        move_optimizer_state(self.optimizer, self._device)

    def _onload_model(self) -> None:
        veomni_onload(self.model, self._device)
        _memprobe(self.model, "post-onload (pre train-forward)")

    def _offload_model(self) -> None:
        veomni_offload(self.model)
        _memprobe(self.model, "post-offload")


# ----------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------


def _memprobe(model, tag: str) -> None:
    """One-line GPU-0 memory breakdown (diagnostic for the HI3 colocate OOM).

    Separates the trainable module's (sharded) param/buffer footprint from the
    total resident memory, so a fixed non-activation hog (frozen composite,
    optimizer, an unsharded copy) is distinguishable from forward activations.
    """
    import logging

    import torch.distributed as dist

    if not (torch.cuda.is_available() and (not dist.is_initialized() or dist.get_rank() == 0)):
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    resv = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    pmem = bmem = 0
    for p in model.parameters():
        t = p.to_local() if hasattr(p, "to_local") else p
        if t.is_cuda:
            pmem += t.numel() * t.element_size()
    for b in model.buffers():
        t = b.to_local() if hasattr(b, "to_local") else b
        if t.is_cuda:
            bmem += t.numel() * t.element_size()
    logging.getLogger(__name__).warning(
        "[MEMPROBE %s] allocated=%.1fGB reserved=%.1fGB peak=%.1fGB | trainable param-local=%.1fGB buffers=%.1fGB",
        tag, alloc, resv, peak, pmem / 1e9, bmem / 1e9,
    )
    torch.cuda.reset_peak_memory_stats()


def _validate_fsdp_cfg(fsdp_cfg: FSDPConfig) -> None:
    """Assert the v1-supported FSDPConfig subset (fail fast, actionably)."""
    if str(fsdp_cfg.fsdp_mode).strip().lower() != "full":
        raise ValueError(
            f"VeOmniBackend: fsdp_mode={fsdp_cfg.fsdp_mode!r} unsupported (v1 supports 'full'; "
            "HSDP/hybrid stays on FSDPBackend)."
        )
    if fsdp_cfg.cpu_offload:
        raise ValueError("VeOmniBackend: cpu_offload=true unsupported in v1 (use FSDPBackend).")
    if not fsdp_cfg.mixed_precision:
        raise ValueError("VeOmniBackend: mixed_precision=false unsupported in v1 (bf16-parity mode is fixed).")


__all__ = ["VeOmniBackend"]
