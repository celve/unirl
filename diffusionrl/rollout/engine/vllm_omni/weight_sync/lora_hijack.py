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
        def hijack__load_adapter(self, lora_request: OmniTensorLoRARequest) -> tuple[LoRAModel, PEFTHelper]:
            """Based on ``vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter``.

            Adds support for loading LoRA from in-memory tensors. vLLM-Omni
            does not natively support adding LoRA from tensors directly — it
            only supports adding LoRA via file paths. To synchronize LoRA
            tensors of the actor model, we need this workaround so vLLM-Omni
            can load memory-based LoRA tensors.
            """
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

        def do_hijack(target_cls, target_method_name, hooking_method) -> None:
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)
        # AR-side hijack is best-effort: vllm's worker_manager is only
        # importable in worker subprocesses that actually instantiate it.
        try:
            from vllm.lora.worker_manager import WorkerLoRAManager
        except ImportError:
            return
        _orig_ar_load_adapter = WorkerLoRAManager._load_adapter
        if getattr(_orig_ar_load_adapter, "_diffrl_hijacked", False):
            return

        def hijack_ar__load_adapter(self, lora_request, _orig=_orig_ar_load_adapter) -> LoRAModel:
            """AR-side counterpart: ``WorkerLoRAManager._load_adapter``.

            Returns just the ``LoRAModel`` (no peft_helper tuple). Mirrors the
            DiT shim above for in-memory tensors and falls through to the
            original on-disk loader for plain ``LoRARequest``.
            """
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
        do_hijack(WorkerLoRAManager, "_load_adapter", hijack_ar__load_adapter)

        # v44e/v45: Skip LoRA wrap for fp32 layers in vllm.lora.utils.from_layer.
        # punica lora_shrink/expand triton kernels hard-assert inputs.dtype in
        # [fp16, bf16]. fp32 layers (e.g. HI3 MoE router `self.gate`) crash
        # the moment add_lora() activates the wrapper. Fix: monkey-patch
        # from_layer to return the layer unchanged when weight.dtype is not
        # fp16/bf16. This eliminates the need for pod-local patches
        # (patches/vllm/0001 + patches/vllm_omni/0002).
        try:
            import torch as _torch
            import vllm.lora.utils as _lora_utils

            _orig_from_layer = _lora_utils.from_layer
            if not getattr(_orig_from_layer, "_diffrl_fp32_skip", False):

                def _patched_from_layer(
                    layer, max_loras, lora_config, packed_modules_list, model_config=None, _orig=_orig_from_layer
                ):
                    _weight = getattr(layer, "weight", None)
                    if _weight is not None and _weight.dtype not in (_torch.float16, _torch.bfloat16):
                        return layer
                    return _orig(layer, max_loras, lora_config, packed_modules_list, model_config)

                _patched_from_layer._diffrl_fp32_skip = True  # type: ignore[attr-defined]
                _lora_utils.from_layer = _patched_from_layer
        except (ImportError, AttributeError):
            pass  # vllm not available in this process; skip


__all__ = ["OmniTensorLoRARequest", "VLLMOmniHijack"]
