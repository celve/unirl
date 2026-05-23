"""Monkey-patch ``DiffusionLoRAManager._load_adapter`` to accept in-memory
LoRA tensors.

vLLM-Omni's stock ``DiffusionLoRAManager._load_adapter`` only loads LoRA
weights from a file path (calls ``LoRAModel.from_local_checkpoint``). For
RL we need to push freshly-trained adapter tensors directly without going
through disk. This module lifts the verl-omni hijack pattern verbatim:

- ``OmniTensorLoRARequest`` extends ``vllm_omni.lora.request.LoRARequest``
  with two extra fields (``peft_config`` dict + ``lora_tensors`` dict).
- ``VLLMOmniHijack.hijack()`` replaces ``DiffusionLoRAManager._load_adapter``
  with a version that branches on the request type: tensor requests go
  through ``LoRAModel.from_lora_tensors``, file-path requests still hit
  the original code path.

Origin: ``verl-omni/verl_omni/utils/vllm_omni/utils.py``. Lifted as-is.
Run ``VLLMOmniHijack.hijack()`` once per worker subprocess (typically
from a worker-extension's ``__new__``).
"""

from __future__ import annotations

import multiprocessing as mp

from msgspec import field

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel  # type: ignore[no-redef]

from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, logger
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest


class OmniTensorLoRARequest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


# ============================================================
# Subprocess propagation — make spawn children also run hijack
# ============================================================
#
# vllm-omni's ``multiproc_executor`` calls ``mp.set_start_method("spawn", force=True)``.
# Each spawn child is a fresh Python interpreter that does not inherit the
# parent's monkey-patches. Without this hook, ``patch_fp32_skip`` (and other
# patches whose targets are imported by the child) take effect in the driver
# but NOT in the worker subprocesses where vllm.lora.utils.from_layer is
# actually called during model loading — fp32 router gate then crashes punica.
#
# Mirrors the LIN-210 sglang pattern (``samplers/sglang/patches/_spawn_wrap.py``).


class _DiffrlPatchedTarget:
    """Pickleable top-level wrapper that installs patches in the child first.

    Must be a module-level class so spawn's pickler can serialise the wrapped
    target across the process boundary. Nested functions / closures cannot be
    pickled and would break spawn.
    """

    def __init__(self, target):
        self._target = target

    def __call__(self, *args, **kwargs):
        VLLMOmniHijack.hijack()
        return self._target(*args, **kwargs)


_WRAP_SENTINEL = "_diffrl_target_wrapped"


def wrap_mp_process_for_children() -> None:
    """Replace ``mp.Process.__init__`` so spawned targets install patches first."""
    if getattr(mp.Process, _WRAP_SENTINEL, False):
        return

    orig_init = mp.Process.__init__

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs=None,
        *,
        daemon=None,
    ):
        if target is not None and not isinstance(target, _DiffrlPatchedTarget):
            target = _DiffrlPatchedTarget(target)
        orig_init(
            self,
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs or {},
            daemon=daemon,
        )

    mp.Process.__init__ = __init__
    setattr(mp.Process, _WRAP_SENTINEL, True)


def patch_dit_lora_loader() -> None:
    """Patch ``DiffusionLoRAManager._load_adapter`` (DiT stage) to support in-memory tensors.

    vLLM-Omni's stock loader only accepts on-disk adapters. We branch on the
    request type: ``OmniTensorLoRARequest`` loads from in-memory tensors via
    ``LoRAModel.from_lora_tensors``; everything else falls through to the
    original on-disk loader via ``LoRAModel.from_local_checkpoint``.
    """

    def hijack__load_adapter(self, lora_request: OmniTensorLoRARequest) -> tuple[LoRAModel, PEFTHelper]:
        if not self._expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

        lora_tensors = None

        if isinstance(lora_request, OmniTensorLoRARequest):
            peft_config = lora_request.peft_config
            lora_tensors = lora_request.lora_tensors
            peft_helper = PEFTHelper.from_dict(peft_config)
        else:
            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            logger.debug("Resolved LoRA path: %s", lora_path)

            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                max_position_embeddings=None,  # no need in diffusion
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
            )

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.target_modules,
        )

        if isinstance(lora_request, OmniTensorLoRARequest):
            lora_model = LoRAModel.from_lora_tensors(
                tensors=lora_tensors,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",  # consistent w/ vllm's behavior
                dtype=self.dtype,
                model_vocab_size=None,
                weights_mapper=None,
            )
        else:
            lora_model = LoRAModel.from_local_checkpoint(
                lora_path,
                expected_lora_modules=self._expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",  # consistent w/ vllm's behavior
                dtype=self.dtype,
                model_vocab_size=None,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=None,
            )

        logger.info(
            "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
            lora_model.id,
            len(lora_model.loras),
            list(lora_model.loras.keys()),
        )

        for lora in lora_model.loras.values():
            lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

        return lora_model, peft_helper

    setattr(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)


def patch_ar_lora_loader() -> None:
    """Patch ``WorkerLoRAManager._load_adapter`` (AR stage) to support in-memory tensors.

    Best-effort: vllm's worker_manager is only importable in worker subprocesses
    that actually instantiate it. Returns just the ``LoRAModel`` (no peft_helper
    tuple). Mirrors the DiT shim for in-memory tensors and falls through to the
    original on-disk loader for plain ``LoRARequest``.
    """
    try:
        from vllm.lora.worker_manager import WorkerLoRAManager
    except ImportError:
        return

    _orig_ar_load_adapter = WorkerLoRAManager._load_adapter
    if getattr(_orig_ar_load_adapter, "_diffrl_hijacked", False):
        return

    def hijack_ar__load_adapter(self, lora_request, _orig=_orig_ar_load_adapter) -> LoRAModel:
        if not isinstance(lora_request, OmniTensorLoRARequest):
            return _orig(self, lora_request)

        peft_helper = PEFTHelper.from_dict(lora_request.peft_config or {})
        peft_helper.validate_legal(self.lora_config)

        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        lora = self._lora_model_cls.from_lora_tensors(
            tensors=lora_request.lora_tensors or {},
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",
            dtype=self.lora_config.lora_dtype,
            model_vocab_size=self.vocab_size,
            weights_mapper=hf_to_vllm_mapper,
        )
        return lora

    hijack_ar__load_adapter._diffrl_hijacked = True  # type: ignore[attr-defined]
    setattr(WorkerLoRAManager, "_load_adapter", hijack_ar__load_adapter)


def patch_fp32_skip() -> None:
    """Patch ``vllm.lora.utils.from_layer`` to skip non-fp16/bf16 layers.

    punica lora_shrink/expand kernels hard-assert inputs.dtype in [fp16, bf16].
    Skip LoRA wrap for fp32 layers (e.g. HI3 MoE router gate) and for
    non-fp16/bf16 dtypes (e.g. quantized) so the original layer.forward runs
    unmodified. If you intentionally want LoRA on such a layer, choose one:

      (a) cast the layer to bf16 in model code (lose precision)
      (b) wrap with a pure-pytorch LoRA variant (no punica),
          e.g. vllm_omni DiffusionBaseLinearLayerWithLoRA
      (c) filter target_modules so it does not match this layer

    Replaces pod-local patch: patches/vllm/0001-utils-skip-fp32-from-layer.patch
    """
    try:
        import torch as _torch
        import vllm.lora.utils as _lora_utils
    except (ImportError, AttributeError):
        return  # vllm not available in this process; skip

    _orig_from_layer = _lora_utils.from_layer
    if getattr(_orig_from_layer, "_diffrl_fp32_skip", False):
        return

    def _patched_from_layer(
        layer, max_loras, lora_config, packed_modules_list, model_config=None, _orig=_orig_from_layer
    ):
        _weight = getattr(layer, "weight", None)
        if _weight is not None and _weight.dtype not in (_torch.float16, _torch.bfloat16):
            _lora_utils.logger.warning_once(
                "Skipping LoRA wrap for layer=%s (weight.dtype=%s not in [fp16, bf16]). "
                "punica kernel does not support this dtype. See vllm/lora/utils.py:from_layer "
                "docstring for workarounds if you intended to LoRA this layer.",
                type(layer).__name__,
                _weight.dtype,
            )
            return layer
        return _orig(layer, max_loras, lora_config, packed_modules_list, model_config)

    _patched_from_layer._diffrl_fp32_skip = True  # type: ignore[attr-defined]
    _lora_utils.from_layer = _patched_from_layer


def patch_sigmas_passthrough() -> None:
    """Monkey-patch HunyuanImage3Pipeline to forward custom sigmas to DiT scheduler.

    Outer ``HunyuanImage3Pipeline.forward`` extracts sigmas from req and stashes
    on the instance; inner ``HunyuanImage3Text2ImagePipeline.__call__`` picks up
    via ``self.model`` (which references the outer instance) and injects as a
    kwarg so ``scheduler.set_timesteps`` gets the correct schedule.

    Without this, DiffusionRL's FlowMatchSchedulePolicy.sigmas is never
    forwarded to the DiT scheduler (rollout-train sigma mismatch
    max abs diff ~0.158 => GRPO log-prob replay incorrect).

    Replaces pod-local patch: patches/vllm_omni/0002-pipeline-sigmas-passthrough.patch
    """
    try:
        from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
            HunyuanImage3Pipeline,
            HunyuanImage3Text2ImagePipeline,
        )

        _orig_outer_forward = HunyuanImage3Pipeline.forward
        if not getattr(_orig_outer_forward, "_diffrl_sigmas_passthrough", False):

            def _patched_outer_forward(self, req, *args, _orig=_orig_outer_forward, **kwargs):
                sigmas = getattr(getattr(req, "sampling_params", None), "sigmas", None)
                self.diffusionrl_sigmas = sigmas
                try:
                    return _orig(self, req, *args, **kwargs)
                finally:
                    self.diffusionrl_sigmas = None

            _patched_outer_forward._diffrl_sigmas_passthrough = True  # type: ignore[attr-defined]
            HunyuanImage3Pipeline.forward = _patched_outer_forward

        _orig_inner_call = HunyuanImage3Text2ImagePipeline.__call__
        if not getattr(_orig_inner_call, "_diffrl_sigmas_passthrough", False):

            def _patched_inner_call(self, *args, _orig=_orig_inner_call, **kwargs):
                outer = getattr(self, "model", None)
                sigmas = getattr(outer, "diffusionrl_sigmas", None) if outer is not None else None
                if sigmas is not None and "sigmas" not in kwargs:
                    kwargs["sigmas"] = sigmas
                return _orig(self, *args, **kwargs)

            _patched_inner_call._diffrl_sigmas_passthrough = True  # type: ignore[attr-defined]
            HunyuanImage3Text2ImagePipeline.__call__ = _patched_inner_call
    except (ImportError, AttributeError):
        pass  # pipeline not available in this process; skip


class VLLMOmniHijack:
    """Monkey-patches vllm-omni internals to support in-memory LoRA tensors.

    Two managers need patching for HI3 t2i:

    - ``vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter``
      drives the DiT stage and returns ``(LoRAModel, PEFTHelper)``.
    - ``vllm.lora.worker_manager.WorkerLoRAManager._load_adapter`` drives the
      AR stage and returns just ``LoRAModel``.

    Both originally only accept on-disk adapters. We branch on the request
    type and load from in-memory tensors when ``OmniTensorLoRARequest`` is
    passed, otherwise fall through to the original loader.
    """

    @staticmethod
    def hijack() -> None:
        # MUST run first: install the mp.Process wrap so any subsequent
        # spawn-spawned subprocesses also run this hijack() at startup.
        # Without this, patches that target functions imported during the
        # child's model-loading phase (notably patch_fp32_skip → from_layer)
        # never take effect in the worker subprocesses.
        wrap_mp_process_for_children()

        patch_dit_lora_loader()
        patch_ar_lora_loader()
        patch_fp32_skip()
        patch_sigmas_passthrough()


__all__ = ["OmniTensorLoRARequest", "VLLMOmniHijack", "patch_sigmas_passthrough"]
