"""MegatronBackend — DP-only Megatron-Core training backend (M0).

A **Remote sibling**, copy-first: it does NOT subclass ``BaseFSDP2Backend`` (whose
checkpoint / optimizer-step / on-offload members all assume FSDP2 DTensors, which
do not describe mcore's ``parallel_state``-managed params). It re-presents the same
method surface ``TrainStack`` / the trainer / weight-sync require, mapping each to
its mcore implementation; the ~40-line genuinely-shared surface (EMA/eval swap,
adapter name) is copied, not inherited.

The heart is ``run_update`` (the train-loop control-inversion): mcore's
``get_forward_backward_func`` owns fwd+bwd+grad-reduce, so the backend owns the
whole update and ``TrainStack._run_update`` delegates to it.

STATUS: first-draft port pending GPU+mcore validation — this workspace has no
CUDA/mcore, so nothing here has been executed. Every ``# VERIFY:`` marks a
version-sensitive mcore/bridge API to confirm against the pinned combo.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.megatron import checkpoint as _ckpt
from unirl.train.backend.megatron import loss_bridge as _lb
from unirl.train.backend.megatron import state as _state
from unirl.train.backend.megatron._guards import assert_supported_save_mode, assert_supported_topology
from unirl.train.backend.megatron.model_provider import build_mcore_model
from unirl.train.configs import MegatronConfig


class MegatronBackend(Remote):
    """Single-stage Megatron-Core training backend (M0: DP only)."""

    # The delegation gate read by TrainStack._run_update. mcore owns the loop.
    owns_update_loop: bool = True

    def __init__(
        self,
        *,
        megatron_cfg: MegatronConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        device: Optional[torch.device] = None,
        rank: int = 0,
    ) -> None:
        super().__init__()
        assert_supported_topology(megatron_cfg, "M0")
        if not megatron_cfg.use_bridge:
            raise NotImplementedError("MegatronBackend M0 only supports the AutoBridge path (use_bridge=True).")
        if megatron_cfg.calculate_per_token_loss:
            # The run_update loss-scale reconciliation assumes mcore scales each
            # microbatch by 1/num_microbatches (calculate_per_token_loss=False).
            raise NotImplementedError("M0 requires calculate_per_token_loss=False (see loss-scale reconciliation).")

        self._cfg = megatron_cfg

        # 1. Distributed bring-up: default PG (harness/env) + mcore parallel_state.
        import torch.distributed as dist

        self._rank = dist.get_rank() if dist.is_initialized() else int(rank)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model_parallel(megatron_cfg)  # VERIFY: mpu.initialize_model_parallel(1,1,...)

        # 2. Megatron args namespace (optimizer + checkpoint read get_args()).
        self.megatron_args = self._build_megatron_args(megatron_cfg, optimizer_cfg, scheduler_cfg)

        # 3. Model via AutoBridge; seed weights from the HF checkpoint.
        self.model_chunks, self._bridge = build_mcore_model(megatron_cfg)
        self._bridge.load_hf_weights(self.model_chunks)  # VERIFY bridge API
        self.model = self.model_chunks[0]  # single chunk at M0 (pp=vpp=1)

        # 4. Distributed optimizer + scheduler (mcore-native).
        self.optimizer = self._build_optimizer()  # VERIFY get_megatron_optimizer
        self.scheduler = self._build_scheduler(scheduler_cfg)  # VERIFY OptimizerParamScheduler

        # 5. Attribute contract read directly by TrainStack / trainer / weight-sync.
        self.ema = None  # M0: no EMA/LoRA (guarded); EMA swap methods no-op below.
        self._optimizer_step_count = 0
        self._rollout_adapter_name = "default"
        self._defer_grad_sync = False
        # VERIFY: dp world + pad id (from mpu / the tokenizer the bridge loaded).
        from megatron.core import parallel_state as mpu

        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.pad_token_id = self._resolve_pad_token_id(megatron_cfg)

        _ckpt.apply_sharded_tensor_validation_bypass()

    # ------------------------------------------------------------------
    # Train loop — the backend owns the update (control-inversion)
    # ------------------------------------------------------------------

    def run_update(self, *, algorithm, resp_track, micros, training_progress, max_grad_norm):
        """One optimizer step over ``micros`` via mcore's monolithic scheduler.

        Mirrors slime ``train_one_step`` but returns UniRL's ``TrainStepResult``.
        Called in-process from ``TrainStack._run_update`` (already DP_SCATTER'd), so
        it is a plain local method — not a dispatched RPC.
        """
        from unirl.train.stack.base import TrainStepResult
        from unirl.utils.misc import aggregate_numeric_metrics

        if resp_track.advantages is None:
            raise ValueError("MegatronBackend.run_update: resp_track.advantages is None.")
        if not micros:
            raise ValueError("MegatronBackend.run_update: empty micros.")

        bs = int(resp_track.batch_size)
        update_total = sum(e - s for s, e in micros)
        num_mb = len(micros)
        temperature = float(getattr(algorithm, "sampling_temperature", 1.0))
        first_update = self._optimizer_step_count == 0

        # 1. zero grads — mcore needs BOTH (optimizer.zero_grad alone leaves main_grad dirty)
        self.zero_grad()

        # 2. data iterator + captured per-micro results (pp=1: same-rank closure sink)
        sink: List[Any] = []
        data_iterator = _lb._MicroDataIterator(resp_track, list(micros), update_total=update_total, bs=bs)

        def forward_step(data_iter, model):
            micro_track, loss_scale = data_iter.next_micro()
            batch = _lb.build_packed_micro_batch(micro_track, pad_token_id=self.pad_token_id, tp_pad=1)
            # VERIFY the mcore GPTModel forward kwargs (input_ids/position_ids/packed_seq_params).
            logits = model(
                input_ids=batch["tokens"],
                position_ids=None,  # FLAG-P: mcore derives per-seq RoPE from cu_seqlens (THD)
                attention_mask=None,
                packed_seq_params=batch["packed_seq_params"],
            )
            loss_closure = _lb.make_loss_closure(
                algorithm=algorithm, micro_track=micro_track, batch=batch,
                training_progress=training_progress, loss_scale=loss_scale, num_microbatches=num_mb,
                temperature=temperature, first_update=first_update, sink=sink,
            )
            return logits, loss_closure

        # 3. the monolithic schedule (owns fwd+bwd across all micros)
        from megatron.core.pipeline_parallel import get_forward_backward_func  # VERIFY import path

        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model_chunks,
            num_microbatches=num_mb,
            seq_length=None,  # pp=1 no-pipelining: per-mb packed length varies, no uniform shape needed
            micro_batch_size=1,
            forward_only=False,
        )

        # 4. step — grad_norm is already globally reduced across DP+MP by mcore.
        # mcore clips inside optimizer.step() using the optimizer config's clip_grad,
        # so wire the stack's max_grad_norm here for parity with the FSDP path (which
        # clips at max_grad_norm). VERIFY the config attribute path against the pin.
        self.optimizer.config.clip_grad = float(max_grad_norm)
        update_successful, grad_norm, _num_zeros = self.optimizer.step()
        if update_successful:
            self.scheduler.step(increment=bs * self.dp_world_size)  # FLAG-S: samples-keyed
        self._optimizer_step_count += 1
        self.zero_grad()  # release grad buffers

        metrics = aggregate_numeric_metrics([r.metrics for r in sink if r.metrics])
        return TrainStepResult(
            loss=sum(r.loss for r in sink),
            grad_norm=float(grad_norm) if update_successful else 0.0,
            lr=self._current_lr(),
            has_backward=any(r.has_backward for r in sink),
            micros=list(sink),
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Method surface required by TrainStack / trainer
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        for chunk in self.model_chunks:
            chunk.zero_grad_buffer()  # VERIFY mcore DDP API
        self.optimizer.zero_grad()

    def set_grad_sync(self, enable: bool) -> None:
        # No-op: mcore grad-sync is handled inside get_forward_backward_func /
        # finalize_model_grads. _defer_grad_sync=False makes TrainStack short-circuit.
        return None

    @property
    def grad_sync_deferred(self) -> bool:
        return self._defer_grad_sync

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        raise RuntimeError(
            "MegatronBackend.optimizer_step is superseded by run_update (owns_update_loop=True); "
            "it must never be called directly on the mcore path."
        )

    def trainable_module(self) -> Any:
        return self.model  # single chunk at M0

    def on_rollout_end(self) -> None:
        if self.ema is not None:
            self.ema.on_rollout_end(self._optimizer_step_count)

    @property
    def rollout_adapter_name(self) -> str:
        return self._rollout_adapter_name

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
        if self.ema is not None:
            self.ema.apply_shadow()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def restore_from_eval(self) -> None:
        if self.ema is not None:
            self.ema.restore_shadow()

    # ------------------------------------------------------------------
    # Weight resync — mcore -> HF via AutoBridge export (M0 walk body)
    # ------------------------------------------------------------------

    def iter_weight_sync_tensors(self, *, lora_merged: bool, adapter_name: Optional[str], dtype):
        """1 mcore-model in -> N HF-named tensors out. At tp=pp=ep=1 the bridge
        yields full params directly, lockstep-identical on every DP rank. This is
        the seam FullWeightSync._iter_full_tensors delegates to.
        """
        assert not lora_merged and adapter_name in (None, "default"), "M0: no mcore LoRA fold"
        # VERIFY: export_hf_weights streams (hf_name, weight, megatron_name).
        for hf_name, weight, _mcore_name in self._bridge.export_hf_weights(self.model):
            t = weight
            if dtype is not None and t.is_floating_point() and t.dtype != dtype:
                t = t.to(dtype)
            yield hf_name, t.contiguous()

    def compute_local_param_checksums(self, *, names: List[str], prefix: str = "") -> Mapping[str, str]:
        """Fingerprint HF-named tensors through the SAME walk the engine receives —
        else it hashes mcore-named local shards and never matches the engine.
        """
        from unirl.distributed.weight_sync.transfer.checksum import fingerprint_tensor

        wanted = set(names)
        out = {}
        for hf_name, tensor in self.iter_weight_sync_tensors(lora_merged=False, adapter_name="default", dtype=None):
            key = f"{prefix}{hf_name}"
            if not wanted or key in wanted:
                out[key] = fingerprint_tensor(tensor)
        return out

    # ------------------------------------------------------------------
    # Checkpoint (native mcore dist_checkpointing) + memory lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save(self, path: str, step: Optional[int] = None, mode: str = "full") -> None:
        assert_supported_save_mode(mode)
        _ckpt.save_native(
            path, step, mode, model_chunks=self.model_chunks, optimizer=self.optimizer,
            scheduler=self.scheduler, megatron_args=self.megatron_args, rank=self._rank,
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load(self, path: str) -> int:
        return _ckpt.load_native(
            path, model_chunks=self.model_chunks, optimizer=self.optimizer,
            scheduler=self.scheduler, megatron_args=self.megatron_args,
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        _state.onload_model_state(self.model_chunks, self.optimizer, self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        _state.offload_model_state(self.model_chunks, self.optimizer)

    # ------------------------------------------------------------------
    # Construction helpers (version-sensitive — VERIFY against the pin)
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _init_model_parallel(self, cfg: MegatronConfig) -> None:
        """mcore parallel_state on top of the default PG (port slime initialize.py,
        DP-only subset). VERIFY the initialize_model_parallel signature.
        """
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=cfg.tp_size,
                pipeline_model_parallel_size=cfg.pp_size,
                virtual_pipeline_model_parallel_size=cfg.vpp_size,
                context_parallel_size=cfg.cp_size,
                expert_model_parallel_size=cfg.ep_size,
                expert_tensor_parallel_size=cfg.etp_size,
            )

    def _build_megatron_args(self, cfg: MegatronConfig, opt: OptimizerConfig, sched: LrSchedulerConfig) -> Any:
        """Build the Megatron ``args`` namespace + register it via ``set_args``.

        VERIFY / TODO: this is the fiddliest port. mcore's optimizer + checkpoint
        read a large ``get_args()`` namespace (normally produced by Megatron's arg
        parser + validate_args). For M0 the AutoBridge derives the MODEL config, but
        the optimizer/checkpoint fields below must still be present. Seed from
        MegatronConfig + OptimizerConfig; mirror slime arguments.py
        (use_distributed_optimizer=True forced). Consider calling Megatron's own
        parse_args for a complete, validated namespace instead of hand-rolling.
        """
        from types import SimpleNamespace

        args = SimpleNamespace(
            # optimizer
            lr=opt.learning_rate,
            min_lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
            adam_beta1=opt.adam_beta1,
            adam_beta2=opt.adam_beta2,
            adam_eps=opt.adam_epsilon,
            use_distributed_optimizer=cfg.use_distributed_optimizer,
            bf16=cfg.bf16,
            fp16=not cfg.bf16 and False,
            # schedule
            lr_warmup_iters=sched.warmup_steps,
            lr_decay_iters=sched.total_steps,
            lr_decay_style=sched.type,
            # loss / model
            seq_length=cfg.seq_length,
            calculate_per_token_loss=cfg.calculate_per_token_loss,
            # checkpoint (set per-call in save/load)
            save=None,
            load=None,
        )
        # VERIFY: register with Megatron globals so get_args()/get_megatron_optimizer see it.
        from megatron.training.global_vars import set_args

        set_args(args)
        return args

    def _build_optimizer(self) -> Any:
        """mcore distributed optimizer (port slime setup_model_and_optimizer)."""
        # VERIFY: build a megatron OptimizerConfig from get_args() fields, then
        # get_megatron_optimizer(config=..., model_chunks=self.model_chunks).
        from megatron.core.optimizer import OptimizerConfig as McoreOptimizerConfig
        from megatron.core.optimizer import get_megatron_optimizer

        a = self.megatron_args
        mcfg = McoreOptimizerConfig(
            optimizer="adam",
            lr=a.lr,
            min_lr=a.min_lr,
            weight_decay=a.weight_decay,
            adam_beta1=a.adam_beta1,
            adam_beta2=a.adam_beta2,
            adam_eps=a.adam_eps,
            use_distributed_optimizer=a.use_distributed_optimizer,
            bf16=a.bf16,
        )
        return get_megatron_optimizer(config=mcfg, model_chunks=self.model_chunks)

    def _build_scheduler(self, sched: LrSchedulerConfig) -> Any:
        """mcore OptimizerParamScheduler (samples-keyed; port slime
        get_optimizer_param_scheduler). VERIFY the constructor args.
        """
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        a = self.megatron_args
        return OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0 if sched.warmup_steps > 0 else a.lr,
            max_lr=a.lr,
            min_lr=a.min_lr,
            lr_warmup_steps=sched.warmup_steps,
            lr_decay_steps=max(1, sched.total_steps),
            lr_decay_style=sched.type,
            start_wd=a.weight_decay,
            end_wd=a.weight_decay,
            wd_incr_steps=max(1, sched.total_steps),
            wd_incr_style="constant",
        )

    def _resolve_pad_token_id(self, cfg: MegatronConfig) -> int:
        """Pad id for the TE alignment filler. VERIFY: read from the tokenizer the
        bridge loaded (or the HF config); 0 is a safe filler since the fillers form
        an isolated sequence that gathers no logp.
        """
        return 0
