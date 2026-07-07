"""MegatronBackend — DP-only Megatron-Core training backend (M0, raw path, TE-free).

A Remote sibling (copy-first, NOT a BaseFSDP2Backend subclass). The mcore recipe
here — raw GPTModel (local spec), HF<->mcore converters, DDP + distributed
optimizer, padded forward — is validated end-to-end in scripts/megatron_probe/probe.py
(forward matches HF 1.000, weight round-trip err 0.0, train step grad_norm finite).

run_update owns the fwd/bwd/step loop via get_forward_backward_func (the
control-inversion TrainStack delegates to). iter_weight_sync_tensors exports mcore
params as HF-named tensors behind the FullWeightSync delegation seam.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.megatron import loss_bridge as _lb
from unirl.train.backend.megatron._guards import assert_supported_save_mode, assert_supported_topology
from unirl.train.backend.megatron.megatron_to_hf import convert_mcore_to_hf
from unirl.train.backend.megatron.model_provider import build_model_and_load
from unirl.train.configs import MegatronConfig


class _ConstantScheduler:
    """Minimal constant-LR scheduler (M0). mcore's OptimizerParamScheduler is the
    M1 upgrade; the M0 recipe uses a constant schedule, so a no-op step suffices."""

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, increment: int = 1) -> None:
        return None


class MegatronBackend(Remote):
    owns_update_loop: bool = True

    def __init__(
        self,
        *,
        megatron_cfg: MegatronConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        bundle: Any = None,
        device: Optional[torch.device] = None,
        rank: int = 0,
    ) -> None:
        super().__init__()
        assert_supported_topology(megatron_cfg, "M1")
        self._cfg = megatron_cfg
        self._bundle = bundle  # trainer injects it; mcore builds its own model from hf_checkpoint

        # Dist bring-up (backends are built before Remote.setup delivers rank info).
        from unirl.train.backend.veomni import _compat
        import torch.distributed as dist

        _, _, local_rank = _compat.rank_world_local()
        _compat.ensure_dist_initialized(local_rank)
        self._rank = dist.get_rank() if dist.is_initialized() else int(rank)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model_parallel()

        # Build mcore model + load TP-sharded HF weights.
        from megatron.core import parallel_state as mpu

        tp_rank = mpu.get_tensor_model_parallel_rank()
        self._gpt, self._tconf, self._hf = build_model_and_load(
            megatron_cfg.hf_checkpoint,
            tp_size=megatron_cfg.tp_size,
            pp_size=megatron_cfg.pp_size,
            tp_rank=tp_rank,
        )
        self.model = self._gpt  # single chunk (trainable_module / weight walk)

        # DDP wrap + distributed optimizer.
        self.model_chunks = [self._wrap_ddp(self._gpt)]
        self.optimizer = self._build_optimizer(optimizer_cfg)
        self.scheduler = _ConstantScheduler(float(optimizer_cfg.learning_rate))

        # Attribute contract read by TrainStack / trainer / weight-sync.
        self.ema = None
        self._optimizer_step_count = 0
        self._rollout_adapter_name = "default"
        self._defer_grad_sync = False
        from megatron.core import parallel_state as mpu

        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.pad_token_id = self._resolve_pad_token_id(megatron_cfg)

    # ------------------------------------------------------------------
    # Train loop — backend owns the update
    # ------------------------------------------------------------------

    def run_update(self, *, algorithm, resp_track, micros, training_progress, max_grad_norm):
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

        self.zero_grad()
        sink: List[Any] = []
        data_iterator = _lb._MicroDataIterator(resp_track, list(micros), update_total=update_total, bs=bs)

        def forward_step(data_iter, model):
            micro_track, loss_scale = next(data_iter)
            batch = _lb.build_padded_micro_batch(micro_track, pad_token_id=self.pad_token_id)
            logits = model(input_ids=batch["ids"], position_ids=batch["pos"], attention_mask=batch["mask"])
            closure = _lb.make_loss_closure(
                algorithm=algorithm, micro_track=micro_track, batch=batch,
                training_progress=training_progress, loss_scale=loss_scale, num_microbatches=num_mb,
                temperature=temperature, first_update=first_update, sink=sink,
            )
            return logits, closure

        from megatron.core.pipeline_parallel import get_forward_backward_func

        get_forward_backward_func()(
            forward_step_func=forward_step, data_iterator=data_iterator, model=self.model_chunks,
            num_microbatches=num_mb, seq_length=None, micro_batch_size=1, forward_only=False,
        )

        self.optimizer.config.clip_grad = float(max_grad_norm)
        update_successful, grad_norm, _nz = self.optimizer.step()
        self._optimizer_step_count += 1
        self.zero_grad()

        metrics = aggregate_numeric_metrics([r.metrics for r in sink if r.metrics])
        return TrainStepResult(
            loss=sum(r.loss for r in sink),
            grad_norm=float(grad_norm) if update_successful and grad_norm is not None else 0.0,
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
            chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

    def set_grad_sync(self, enable: bool) -> None:
        return None

    @property
    def grad_sync_deferred(self) -> bool:
        return self._defer_grad_sync

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        raise RuntimeError("MegatronBackend.optimizer_step is superseded by run_update (owns_update_loop=True).")

    def trainable_module(self) -> Any:
        return self.model

    def on_rollout_end(self) -> None:
        return None

    @property
    def rollout_adapter_name(self) -> str:
        return self._rollout_adapter_name

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
        return None

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def restore_from_eval(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Weight resync — mcore -> HF export (M0 walk body)
    # ------------------------------------------------------------------

    def iter_weight_sync_tensors(self, *, lora_merged: bool, adapter_name: Optional[str], dtype):
        assert not lora_merged and adapter_name in (None, "default"), "no mcore LoRA fold"
        # Under tp>1 each param is a shard; all-gather to the full tensor (lockstep
        # across the TP group) before converting to HF names. Inverse of the load
        # shard. At tp=1 this is a no-op. Colocate ships each rank's full model to
        # its own local engine, so no transport guard is needed.
        from megatron.core import parallel_state as mpu

        from unirl.train.backend.megatron.megatron_to_hf import all_gather_tp_param

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()
        for name, param in self._gpt.named_parameters():
            full = all_gather_tp_param(name, param.detach(), tp_group=tp_group, tp_size=tp_size)
            for hf_name, t in convert_mcore_to_hf(name, full, self._hf):
                if dtype is not None and t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                yield hf_name, t.contiguous()

    def compute_local_param_checksums(self, *, names: List[str], prefix: str = "") -> Mapping[str, str]:
        from unirl.distributed.weight_sync.transfer.checksum import fingerprint_tensor

        wanted = set(names)
        out = {}
        for hf_name, tensor in self.iter_weight_sync_tensors(lora_merged=False, adapter_name="default", dtype=None):
            key = f"{prefix}{hf_name}"
            if not wanted or key in wanted:
                out[key] = fingerprint_tensor(tensor)
        return out

    # ------------------------------------------------------------------
    # Checkpoint + memory lifecycle (M0: minimal — see notes)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save(self, path: str, step: Optional[int] = None, mode: str = "full") -> None:
        assert_supported_save_mode(mode)
        import os

        os.makedirs(path, exist_ok=True)
        if self._rank == 0:
            torch.save({"gpt": self._gpt.state_dict(), "step": step}, os.path.join(path, "megatron_ckpt.pt"))

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load(self, path: str) -> int:
        import os

        f = os.path.join(path, "megatron_ckpt.pt")
        if not os.path.exists(f):
            return 0
        state = torch.load(f, map_location="cpu", weights_only=False)
        self._gpt.load_state_dict(state["gpt"])
        return int(state.get("step") or 0)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        return None  # 0.6B M0 smoke fits alongside SGLang; real offload is M1.

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _init_model_parallel(self) -> None:
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self._cfg.tp_size,
                pipeline_model_parallel_size=self._cfg.pp_size,
            )
        model_parallel_cuda_manual_seed(1234)

    def _wrap_ddp(self, model):
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import DistributedDataParallelConfig

        ddp_cfg = DistributedDataParallelConfig(
            use_distributed_optimizer=self._cfg.use_distributed_optimizer, overlap_grad_reduce=False)
        return DDP(self._tconf, ddp_cfg, model)

    def _build_optimizer(self, opt: OptimizerConfig):
        from megatron.core.optimizer import OptimizerConfig as McoreOptCfg
        from megatron.core.optimizer import get_megatron_optimizer

        mcfg = McoreOptCfg(
            optimizer="adam",
            lr=opt.learning_rate,
            min_lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
            adam_beta1=opt.adam_beta1,
            adam_beta2=opt.adam_beta2,
            adam_eps=opt.adam_epsilon,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=self._cfg.use_distributed_optimizer,
            clip_grad=1.0,
        )
        return get_megatron_optimizer(mcfg, self.model_chunks)

    def _resolve_pad_token_id(self, cfg: MegatronConfig) -> int:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(cfg.hf_checkpoint)
            return int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        except Exception:
            return 0
