"""BaseFSDP2Backend — the shared training-state Remote for FSDP2 backends.

Holds everything identical across the torch-native and VeOmni FSDP2 backends:
the training step, the EMA eval-swap, the hardened checkpoint envelope, the
memory lifecycle, and the construction scaffolding (structural injection +
EMA/optimizer/scheduler build). The two concrete backends
(:class:`~unirl.train.backend.fsdp.FSDPBackend` and
:class:`~unirl.train.backend.veomni.VeOmniBackend`) subclass this and supply only
their engine-specific behavior:

* the constructor *lifecycle* (wrap strategy, distributed bring-up, sequence
  parallelism, eager-vs-meta weight load) stays in each leaf, written as a linear
  named sequence that calls the shared construction helpers here; and
* five small *hooks* — grad clip, optimizer-state gather/load, model on/offload —
  that the methods below dispatch through.

This module imports torch (and ema / lora / optim / deferred) at module level and
MUST NOT be imported from ``veomni/__init__`` — only from inside the two
``backend.py`` files. It imports neither ``veomni`` nor either leaf backend.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.sharded_state import (
    StateDict,
    _current_rank,
    gather_state_dict,
    load_model_state_dict,
    move_optimizer_state,
    trainable_params,
)
from unirl.train.configs import EmaFullConfig, EmaLoraConfig, FSDPConfig, LoraConfig
from unirl.train.ema import EMA, Shadow, inject_mirror, inject_nft, make_decay_fn
from unirl.train.lora import inject_lora
from unirl.train.optim import build_lr_scheduler, build_optimizer

logger = logging.getLogger(__name__)


class BaseFSDP2Backend(Remote):
    """Shared base for the single-track FSDP2 training backends.

    Subclasses own ``__init__`` (the engine-specific lifecycle) and the five
    hooks at the bottom of this class; everything else is shared. After a leaf
    ``__init__`` returns the backend is fully usable (model wrapped, weights
    loaded, optimizer/scheduler/EMA built).
    """

    # --- attribute contract (set by the leaf ctor + _finalize_construction) ---
    _bundle: object
    _rank: int
    _device: torch.device
    model: nn.Module
    ema: Optional[EMA]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    _optimizer_step_count: int
    _eval_ema_active: bool
    _lora_meta: Optional[Dict[str, object]]
    _rollout_adapter_name: str
    _defer_grad_sync: bool
    _grad_sync_enabled: bool

    # ------------------------------------------------------------------
    # Construction helpers (called in sequence from each leaf __init__)
    # ------------------------------------------------------------------

    def _check_lora_exclusivity(
        self,
        lora_cfg: Optional[LoraConfig],
        ema_lora_cfg: Optional[EmaLoraConfig],
    ) -> None:
        if lora_cfg is not None and ema_lora_cfg is not None:
            raise ValueError(
                f"{type(self).__name__}: lora_cfg and ema_lora_cfg are mutually "
                "exclusive (both inject LoRA adapters). Use ema_lora_cfg for "
                "NFT-style adapter EMA, or lora_cfg for plain LoRA."
            )

    def _inject_structural(
        self,
        model: nn.Module,
        lora_cfg: Optional[LoraConfig],
        ema_lora_cfg: Optional[EmaLoraConfig],
        ema_cfg: Optional[EmaFullConfig],
    ) -> Optional[Shadow]:
        """Structural injection on the (possibly meta) trainable module.

        LoRA / NFT-adapter-EMA / mirror-EMA injection — exactly the
        ``unirl.train.deferred`` contract (mutate now; the post-materialize
        resets are drained by ``apply_deferred_ops`` after the weight load).
        Returns the EMA :class:`Shadow` (or ``None``).
        """
        shadow: Optional[Shadow] = None
        if ema_lora_cfg is not None:
            shadow = inject_nft(
                model,
                rank=ema_lora_cfg.rank,
                alpha=ema_lora_cfg.alpha,
                target_modules=tuple(ema_lora_cfg.target_modules),
                default=ema_lora_cfg.default_adapter,
                shadow=ema_lora_cfg.shadow_adapter,
                dropout=ema_lora_cfg.dropout,
                bias=ema_lora_cfg.bias,
                task_type=ema_lora_cfg.task_type,
            )
        elif lora_cfg is not None:
            inject_lora(
                model,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                target_modules=tuple(lora_cfg.target_modules),
                dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
                task_type=lora_cfg.task_type,
            )
        if ema_cfg is not None:
            shadow = inject_mirror(model, prefix=ema_cfg.shadow_prefix)
        return shadow

    def _finalize_construction(
        self,
        model: nn.Module,
        shadow: Optional[Shadow],
        *,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        lora_cfg: Optional[LoraConfig],
        ema_lora_cfg: Optional[EmaLoraConfig],
        ema_cfg: Optional[EmaFullConfig],
        fsdp_cfg: FSDPConfig,
    ) -> None:
        """Build EMA / optimizer / scheduler and set the shared train state.

        Called at the end of each leaf constructor — after the model is wrapped,
        weight-loaded, and its deferred ops drained — so the backend is fully
        usable once the leaf ``__init__`` returns.
        """
        self.model = model

        self.ema = None
        if shadow is not None:
            active_cfg = ema_lora_cfg or ema_cfg
            self.ema = EMA(
                shadow=shadow,
                decay_fn=make_decay_fn(active_cfg),
                timing=active_cfg.timing,
            )

        self.optimizer = build_optimizer(
            optimizer_cfg,
            params=list(trainable_params(model)),
        )
        self.scheduler = build_lr_scheduler(
            scheduler_cfg,
            optimizer=self.optimizer,
        )

        self._optimizer_step_count = 0
        self._eval_ema_active = False
        # Checkpointed for export tooling: the LoRA fold needs scaling =
        # alpha / rank, and alpha is not derivable from the weights.
        active_lora = lora_cfg or ema_lora_cfg
        self._lora_meta = (
            {
                "rank": int(active_lora.rank),
                "alpha": int(active_lora.alpha),
                "target_modules": list(active_lora.target_modules),
            }
            if active_lora is not None
            else None
        )
        # Single source of truth for "which adapter the rollout samples under":
        # the EMA shadow ("old") for DiffusionNFT adapter-EMA, else the trainable
        # "default". The in-process eval-EMA swap and the weight sync to a
        # SEPARATE engine both derive from this, so they cannot disagree.
        self._rollout_adapter_name = (
            str(ema_lora_cfg.shadow_adapter) if ema_lora_cfg is not None else "default"
        )
        # No-sync gradient accumulation (see set_grad_sync). Only active under
        # ZeRO-2 (reshard_after_forward=False); a no-op under ZeRO-3, where the
        # per-micro reshard/re-gather interacts badly with deferred sync.
        self._defer_grad_sync = bool(fsdp_cfg.defer_grad_sync) and not bool(
            fsdp_cfg.reshard_after_forward
        )
        self._grad_sync_enabled = True

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def set_grad_sync(self, enable: bool) -> None:
        """Toggle the FSDP2 gradient reduce-scatter for no-sync accumulation.

        With ``defer_grad_sync`` on, the train loop disables sync on every
        micro-batch except the last, so every FSDP group accumulates gradients in
        its unsharded buffers and a single reduce-scatter runs per optimizer step
        instead of one per micro-batch (a multi-node win; ~no-op over NVLink).
        No-op when deferral is off (the common case) or the flag is already in
        the wanted state. ``set_is_last_backward`` does not recurse, so every
        FSDP module is toggled; ``set_requires_gradient_sync`` is idempotent
        across nesting.
        """
        if not self._defer_grad_sync or enable == self._grad_sync_enabled:
            return
        from torch.distributed.fsdp import FSDPModule

        for m in self.model.modules():
            if isinstance(m, FSDPModule):
                m.set_requires_gradient_sync(enable)
                m.set_is_last_backward(enable)
        self._grad_sync_enabled = enable

    @property
    def grad_sync_deferred(self) -> bool:
        """True when no-sync accumulation is active (``defer_grad_sync`` under ZeRO-2)."""
        return self._defer_grad_sync

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        """Clip (via the engine hook), optimizer step, scheduler step, EMA step.

        The algorithm sibling Remote populates grads on this backend's model
        (they share the bundle); caller invokes this only when ``has_backward``
        was True for the accumulated micro-batches.

        Skips the whole step on a non-finite (NaN/Inf) clipped grad norm:
        stepping would scale every parameter by the bad norm and poison the
        weights, crashing the next rollout's sampling. The clipped norm is an
        all-rank scalar so the skip is identical on every rank. This is the one
        optimizer-step chokepoint every v2 trainer routes through.
        """
        clipped = self._clip_grad_norm(float(max_grad_norm))
        grad_norm = float(clipped.item()) if isinstance(clipped, torch.Tensor) else float(clipped or 0.0)

        if not math.isfinite(grad_norm):
            logger.warning(
                "%s.optimizer_step: non-finite grad norm (%s) at step %d; skipping step.",
                type(self).__name__,
                grad_norm,
                self._optimizer_step_count,
            )
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norm

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.step(self._optimizer_step_count)
        self._optimizer_step_count += 1
        return grad_norm

    def on_rollout_end(self) -> None:
        if self.ema is not None:
            self.ema.on_rollout_end(self._optimizer_step_count)

    # ------------------------------------------------------------------
    # Eval-EMA swap
    # ------------------------------------------------------------------

    @property
    def rollout_adapter_name(self) -> str:
        """Adapter the rollout must sample under (single source of truth).

        The EMA shadow (``"old"``) for DiffusionNFT-style adapter EMA, else the
        trainable ``"default"``. The weight-sync handlers read this to decide
        which adapter to push to a SEPARATE engine, mirroring the in-process
        :meth:`apply_eval_ema` swap — so an off-policy algorithm rolls out under
        the same weights whether the engine is colocated or separate.
        """
        return self._rollout_adapter_name

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
        """Swap the EMA shadow ("old") adapter into live position for rollout.

        Driver-callable (each worker swaps its own model); the NFT trainer wraps
        ``rollout.generate`` with this + :meth:`restore_from_eval`. No-op when
        ``ema is None`` (GRPO) or already swapped in.
        """
        if self.ema is None or self._eval_ema_active:
            return
        self.ema.apply_shadow()
        self._eval_ema_active = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def restore_from_eval(self) -> None:
        if self.ema is None or not self._eval_ema_active:
            return
        self.ema.restore_shadow()
        self._eval_ema_active = False

    # ------------------------------------------------------------------
    # Checkpoint (one hardened envelope; optimizer mechanism is per-backend)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save(self, path: str, step: Optional[int] = None, mode: str = "full") -> None:
        """Gather state (collective on every rank); write ``path/checkpoint.pt`` on dist rank 0.

        ``step`` is the trainer's rollout step — :meth:`load` returns it so the
        loop resumes where it stopped. ``mode="adapter"`` keeps only the LoRA
        keys in the model state (MBs instead of GBs; the frozen base reloads from
        the pretrained snapshot on resume). The optimizer state is identical
        under both modes — it only ever covers the trainable (LoRA) params. The
        optimizer gather is per-backend (DCP for torch-native FSDP, plain
        ``state_dict()`` for VeOmni) via :meth:`_gather_optimizer_state`.
        """
        if mode not in ("full", "adapter"):
            raise ValueError(f"{type(self).__name__}.save: unknown mode {mode!r} (use 'full' or 'adapter')")
        if mode == "adapter" and not any("lora_" in name for name, _ in self.model.named_parameters()):
            raise RuntimeError(f"{type(self).__name__}.save: mode='adapter' but the model has no LoRA params")
        self._reject_meta_params("save")

        # Both gathers are collectives — every rank MUST enter them before any
        # rank bails; only dist rank 0 ends up with non-empty dicts.
        policy_state = gather_state_dict(self.model)
        if mode == "adapter":
            policy_state = {k: v for k, v in policy_state.items() if "lora_A" in k or "lora_B" in k}
        optimizer_state = self._gather_optimizer_state()

        state: Dict[str, object] = {
            "policy_state_dict": policy_state,
            "optimizer_state_dict": optimizer_state,
            "optimizer_step_count": self._optimizer_step_count,
            "step": step,
            "save_mode": mode,
            "lora_config": self._lora_meta,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        # The gathers above populate dist rank 0 only — that rank writes. (NOT
        # self._rank: that is a constructor kwarg, identical on every worker.)
        if _current_rank() != 0:
            return
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, "checkpoint.pt"))

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load(self, path: str) -> int:
        """Restore the state written by :meth:`save`; return the saved rollout step (0 if absent).

        Every rank runs this: each reads the checkpoint from shared storage to
        CPU, then the model tensors broadcast from dist rank 0 and re-shard, and
        the optimizer state is restored via the per-backend
        :meth:`_load_optimizer_state` hook. Adapter-mode checkpoints load
        non-strict — only the LoRA keys are present; the frozen base keeps the
        weights the bundle loaded.
        """
        self._reject_meta_params("load")
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        # Agree on visibility BEFORE the collectives: on multi-node, a rank whose
        # node does not mount the checkpoint path would raise alone and strand
        # the others in the broadcast until the NCCL timeout.
        exists = os.path.exists(checkpoint_path)
        if dist.is_available() and dist.is_initialized():
            verdicts: List[Optional[bool]] = [None] * dist.get_world_size()
            dist.all_gather_object(verdicts, exists)
            missing_on = [rank for rank, ok in enumerate(verdicts) if not ok]
        else:
            missing_on = [] if exists else [0]
        if missing_on:
            raise FileNotFoundError(
                f"{type(self).__name__}.load: checkpoint not visible on rank(s) {missing_on}: {checkpoint_path} "
                "(save_dir/load_dir must live on storage mounted on every node)"
            )
        # Every rank loads the file: the model broadcast tolerates {} on non-zero
        # ranks, but a plain-state_dict optimizer restore (VeOmni) needs the real
        # dict locally on every rank.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        strict = checkpoint.get("save_mode", "full") == "full"
        load_model_state_dict(self.model, checkpoint["policy_state_dict"], strict=strict)
        self._load_optimizer_state(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("optimizer_step_count") is not None:
            self._optimizer_step_count = int(checkpoint["optimizer_step_count"])
        return int(checkpoint.get("step") or 0)

    def _reject_meta_params(self, op: str) -> None:
        """Fail fast on never-materialized params (meta-init bundles, e.g. hi3 80B).

        Their frozen aux (vae / vit) stays on meta and a full-state-dict gather
        would die deep inside DCP ("Cannot copy out of meta tensor"). Same
        verdict on every rank, so raising here is collective-safe. (VeOmni v1
        forbids meta-resident trainables outright, so this never fires there.)
        """
        meta = [name for name, p in self.model.named_parameters() if p.is_meta]
        if meta:
            raise RuntimeError(
                f"{type(self).__name__}.{op}: {len(meta)} params are on meta (e.g. {meta[:3]}); "
                "full-state-dict checkpointing of meta-init bundles is not supported yet."
            )

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        """Move the train state (params + grads + optimizer) back to GPU.

        Driver-callable across all DP workers (each onloads its own FSDP shard).
        Inverse of :meth:`offload`; the colocate trainers call this before the
        train backward (gated by ``enable_fsdp_offload``)."""
        self._onload_model()
        move_optimizer_state(self.optimizer, self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        """Move the train state (params + grads + optimizer) to CPU.

        Frees GPU memory during the rollout phase so a colocate vLLM/SGLang
        engine fits. Driver-callable across all DP workers (each offloads its own
        FSDP shard). Gated by the trainer's ``enable_fsdp_offload``."""
        self._offload_model()
        move_optimizer_state(self.optimizer, "cpu")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def trainable_module(self) -> nn.Module:
        return self.model

    # ------------------------------------------------------------------
    # Smoke helpers
    # ------------------------------------------------------------------

    def compute_local_param_checksums(
        self,
        *,
        names: List[str],
        prefix: str = "",
    ) -> Dict[str, str]:
        from unirl.rollout.engine.vllm_omni.weight_sync.checksum import (
            fingerprint_tensor,
        )
        from unirl.utils.peft_merge import raw_state_dict

        target = set(names)
        out: Dict[str, str] = {}
        for raw_name, param in raw_state_dict(self.model):
            prefixed = prefix + raw_name
            if prefixed in target:
                out[prefixed] = fingerprint_tensor(param)
        return out

    def randomize_weights_for_smoke(self, seed: int = 0) -> None:
        from torch.distributed.tensor import DTensor

        gen = torch.Generator(device=self._device)
        gen.manual_seed(int(seed) + int(self._rank))
        with torch.no_grad():
            for p in trainable_params(self.model):
                local = p.data
                if isinstance(local, DTensor):
                    shard = local.to_local()
                    shard.copy_(
                        torch.randn(
                            shard.shape,
                            dtype=shard.dtype,
                            device=shard.device,
                            generator=gen,
                        )
                    )
                else:
                    local.copy_(
                        torch.randn(
                            local.shape,
                            dtype=local.dtype,
                            device=local.device,
                            generator=gen,
                        )
                    )
        logger.info(
            "Rank %s: randomize_weights_for_smoke complete (seed=%d)",
            self._rank,
            seed,
        )

    # ------------------------------------------------------------------
    # Engine-specific hooks (overridden by each leaf backend)
    # ------------------------------------------------------------------

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        """Clip gradients and return the (pre-clip) global grad norm."""
        raise NotImplementedError

    def _gather_optimizer_state(self) -> StateDict:
        """Rank-0 optimizer state for the checkpoint (collective for DCP backends)."""
        raise NotImplementedError

    def _load_optimizer_state(self, optimizer_state: StateDict) -> None:
        """Restore optimizer state from a checkpoint dict (real dict on every rank)."""
        raise NotImplementedError

    def _onload_model(self) -> None:
        """Move the wrapped model's params + grads back to the live device."""
        raise NotImplementedError

    def _offload_model(self) -> None:
        """Move the wrapped model's params + grads to CPU."""
        raise NotImplementedError


__all__ = ["BaseFSDP2Backend"]
