"""VeOmniBackend — single-track training-state Remote on VeOmni FSDP2.

Drop-in sibling of :class:`unirl.train.backend.fsdp.FSDPBackend` (byte-
identical constructor signature, same public surface) whose wrap, grad
clipping, and offload internals come from VeOmni's distributed layer via
the :mod:`._compat` selective-import shim.  Recipes select it purely by
``_target_``.

Lifecycle differences vs FSDPBackend (all internal to construction):

* The default process group is brought up *explicitly* (VeOmni builds its
  device meshes before any ``fully_shard`` call, so torch's lazy auto-init
  never gets the chance to fire), and ``init_parallel_state`` is invoked —
  one VeOmni-wrapped model per process.
* The trainable module must arrive on the **meta** device (the bundle's
  ``meta_init_transformer`` flag): VeOmni's parallelize materializes it via
  ``to_empty`` and calls its (no-op-stamped) ``init_weights``; the real
  weights are loaded *after* sharding — rank 0 reads the safetensors dir
  stashed by the bundle and broadcasts (``strict=False``: injected adapter
  params are legitimately absent and re-initialized by the deferred ops).
* LoRA/NFT/mirror injection runs on the meta module — exactly the contract
  ``unirl.train.deferred`` documents — and ``apply_deferred_ops`` drains
  the post-materialize resets *after* the weight load.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.types.bundle import Bundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.veomni.state import (
    StateDict,
    clip_grad_norm,
    gather_state_dict,
    load_model_state_dict,
    trainable_params,
    veomni_offload,
    veomni_onload,
)
from unirl.train.backend.veomni.wrap import veomni_parallelize
from unirl.train.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)
from unirl.train.deferred import apply_deferred_ops
from unirl.train.ema import EMA, Shadow, inject_mirror, inject_nft, make_decay_fn
from unirl.train.lora import inject_lora
from unirl.train.optim import build_lr_scheduler, build_optimizer

logger = logging.getLogger(__name__)


class VeOmniBackend(Remote):
    """Single-track VeOmni-FSDP2 training backend.

    One-shot construction: after ``__init__`` returns the backend is fully
    usable (model wrapped, weights loaded, optimizer/scheduler/EMA built).
    ``device`` / ``rank`` kwargs are accepted for signature parity with
    :class:`FSDPBackend` but resolved from the actor env + process group —
    backends are constructed before ``Remote.setup()`` delivers rank info.
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
        lora_cfg: Optional[LoraConfig] = None,
        ema_lora_cfg: Optional[EmaLoraConfig] = None,
        ema_cfg: Optional[EmaFullConfig] = None,
        with_aux: Tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        if lora_cfg is not None and ema_lora_cfg is not None:
            raise ValueError(
                "VeOmniBackend: lora_cfg and ema_lora_cfg are mutually exclusive "
                "(both inject LoRA adapters). Use ema_lora_cfg for NFT-style "
                "adapter EMA, or lora_cfg for plain LoRA."
            )
        _validate_fsdp_cfg(fsdp_cfg)

        from unirl.train.backend.veomni import _compat

        # 1-3. Distributed bring-up: device binding, default PG, VeOmni
        # parallel state (1D dp_shard mesh; re-init warns + no-ops, which
        # enforces one VeOmni-wrapped model per process).
        _, _, local_rank = _compat.rank_world_local()
        _compat.ensure_dist_initialized(local_rank)
        import torch.distributed as dist

        self._rank = dist.get_rank() if dist.is_initialized() else int(rank)
        world = dist.get_world_size() if dist.is_initialized() else 1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        api = _compat.load()
        api.init_parallel_state(
            dp_size=world,
            dp_mode="fsdp2",
            device_type=self._device.type,
        )

        self._bundle = bundle
        model = getattr(bundle, trainable_attr)

        # 4-5. Structural injection on the meta module (the documented
        # unirl.train.deferred contract: mutate on meta, stamp resets).
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

        # 6. Shard + materialize (to_empty; init_weights is a bundle-stamped
        # no-op). Root-wrapped by VeOmni — single-module trainables only.
        veomni_parallelize(
            model,
            block_class_names=tuple(block_class_names),
            param_dtype=fsdp_cfg.param_dtype,
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            activation_checkpointing=fsdp_cfg.activation_checkpointing,
            use_torch_compile=fsdp_cfg.use_torch_compile,
        )

        # 7. Real weights: rank 0 reads the bundle-stashed safetensors dir,
        # broadcast into the sharded module. strict=False — adapter params
        # are absent from the base checkpoint by design.
        weights_path = getattr(bundle, "_transformer_weights_path", None)
        if weights_path is not None:
            state_dict = _read_safetensors_dir(weights_path) if self._rank == 0 else {}
            if self._rank == 0:
                state_dict = _remap_lora_base_keys(state_dict, model)
            load_model_state_dict(model, state_dict, strict=False)
            logger.info("Rank %s: loaded transformer weights from %s", self._rank, weights_path)
        else:
            bundle_materialize = getattr(bundle, "materialize", None)
            if callable(bundle_materialize):
                bundle_materialize(device=self._device, with_aux=tuple(with_aux))
            else:
                raise ValueError(
                    "VeOmniBackend: trainable module has no weight source — the "
                    "bundle must either set meta-init (stashing "
                    "_transformer_weights_path) or provide materialize(). "
                    "Eagerly-loaded bundles are FSDPBackend territory: VeOmni's "
                    "parallelize would clobber their weights via to_empty()."
                )

        # 8. Post-materialize resets (LoRA adapter init, mirror copies).
        apply_deferred_ops(model)

        # 9-10. EMA, optimizer, scheduler — identical to FSDPBackend.
        self.ema: Optional[EMA] = None
        if shadow is not None:
            active_cfg = ema_lora_cfg or ema_cfg
            self.ema = EMA(
                shadow=shadow,
                decay_fn=make_decay_fn(active_cfg),
                timing=active_cfg.timing,
            )

        self.optimizer: torch.optim.Optimizer = build_optimizer(
            optimizer_cfg,
            params=list(trainable_params(model)),
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = build_lr_scheduler(
            scheduler_cfg,
            optimizer=self.optimizer,
        )

        self.model: nn.Module = model
        self._optimizer_step_count: int = 0
        self._eval_ema_active: bool = False

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        """Clip (VeOmni FSDP2 clip), optimizer step, scheduler step, EMA step.

        Same non-finite-norm safety valve as FSDPBackend: skip the whole
        step on a NaN/Inf clipped norm — the norm is an all-rank scalar so
        the skip is identical on every rank.
        """
        clipped = clip_grad_norm(self.model, float(max_grad_norm))
        grad_norm = float(clipped.item()) if isinstance(clipped, torch.Tensor) else float(clipped or 0.0)

        if not math.isfinite(grad_norm):
            logger.warning(
                "VeOmniBackend.optimizer_step: non-finite grad norm (%s) at step %d; skipping step.",
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

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
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
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Gather state on all ranks; write to ``path/checkpoint.pt`` on rank 0."""
        state: Dict[str, object] = {
            "policy_state_dict": gather_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if self._rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, "checkpoint.pt"))

    def load(self, path: str) -> None:
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VeOmniBackend.load: checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        load_model_state_dict(self.model, checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        """Move the train state (params + grads + optimizer) back to GPU."""
        veomni_onload(self.model, self._device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        """Move the train state to CPU (VeOmni reshards the root first)."""
        veomni_offload(self.model)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
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


# ----------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------


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


def _read_safetensors_dir(weights_path: str) -> StateDict:
    """Merge all ``*.safetensors`` shards in a (diffusers-layout) directory.

    Loading every shard makes the index json unnecessary and covers both
    single-file and sharded checkpoints."""
    import glob

    from safetensors.torch import load_file

    if not os.path.isdir(weights_path):
        raise FileNotFoundError(
            f"VeOmniBackend: transformer weights dir not found: {weights_path!r}. "
            "HF repo IDs are not supported here — point the recipe's checkpoint "
            "path at a local download."
        )
    shards = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"VeOmniBackend: no *.safetensors files under {weights_path!r}")
    state_dict: StateDict = {}
    for shard in shards:
        state_dict.update(load_file(shard, device="cpu"))
    return state_dict


def _remap_lora_base_keys(state_dict: StateDict, model: nn.Module) -> StateDict:
    """Translate base-checkpoint keys for LoRA-injected modules.

    ``peft.inject_adapter_in_model`` (via ``unirl.train.lora`` /
    ``unirl.train.ema``) rewires target Linears in place, so their original
    weight moves to ``<module>.base_layer.weight``.  The base checkpoint
    still uses the original key — insert the ``base_layer`` hop where (and
    only where) the model expects it."""
    model_keys = {n for n, _ in model.named_parameters()}
    model_keys.update(n for n, _ in model.named_buffers())
    remapped: StateDict = {}
    for key, value in state_dict.items():
        if key not in model_keys:
            stem, _, leaf = key.rpartition(".")
            candidate = f"{stem}.base_layer.{leaf}" if stem else key
            if candidate in model_keys:
                remapped[candidate] = value
                continue
        remapped[key] = value
    return remapped


__all__ = ["VeOmniBackend"]
