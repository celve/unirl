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

from multiprocessing.process import BaseProcess as _MpBaseProcess

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
    """Replace ``BaseProcess.__init__`` so spawned targets install patches first.

    Patching ``mp.Process.__init__`` alone misses spawn-context Process classes
    (vllm-omni's stage launcher uses ``get_mp_context().Process`` ==
    ``SpawnProcess``, a sibling class, not a subclass). All context-specific
    Process classes inherit from ``BaseProcess``, so patching the root catches
    every context in one shot.
    """
    if getattr(_MpBaseProcess, _WRAP_SENTINEL, False):
        return

    orig_init = _MpBaseProcess.__init__

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

    _MpBaseProcess.__init__ = __init__
    setattr(_MpBaseProcess, _WRAP_SENTINEL, True)


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


def patch_ar_merged_lora_fused_tensor() -> None:
    """Accept a single fused lora_b [q+k+v, rank] in MergedQKV set_lora.

    HI3 trains LoRA on a fused qkv_proj; vLLM expects a list [lora_b_q, lora_b_k,
    lora_b_v]. The checkpoint qkv_proj is GQA-interleaved, training loads it as-is,
    so lora_b rows are interleaved. vLLM base is block [q;k;v] after _split_qkv_weight
    — we mirror that reshape-split on lora_b. Falls back to plain split if the base
    layer lacks head_size/total_num_kv_heads.
    """
    try:
        import torch
        from vllm.lora.layers import column_parallel_linear as _cpl
    except (ImportError, AttributeError):
        return

    def _deinterleave_gqa(lora_b, output_sizes, base_layer):
        if len(output_sizes) != 3:
            return None
        head_size = getattr(base_layer, "head_size", None)
        num_kv_heads = getattr(base_layer, "total_num_kv_heads", None)
        if head_size is None or num_kv_heads is None:
            return None
        q_size, k_size, _v = output_sizes
        groups = q_size // k_size
        if groups * k_size != q_size or k_size != num_kv_heads * head_size:
            return None
        rank = lora_b.shape[1]
        try:
            lora_b_r = lora_b.reshape(num_kv_heads, groups + 2, head_size, rank)
        except RuntimeError:
            return None
        q_b, k_b, v_b = torch.split(lora_b_r, (groups, 1, 1), dim=1)
        return [q_b.reshape(-1, rank), k_b.reshape(-1, rank), v_b.reshape(-1, rank)]

    def _make(orig):
        def _set_lora(self, index, lora_a, lora_b, *args, _orig=orig, **kwargs):
            if isinstance(lora_b, torch.Tensor):
                output_sizes = list(getattr(self.base_layer, "output_sizes", []) or [])
                if output_sizes and int(lora_b.shape[0]) == sum(output_sizes):
                    slices = _deinterleave_gqa(lora_b, output_sizes, self.base_layer)
                    lora_b = slices if slices is not None else list(torch.split(lora_b, output_sizes, dim=0))
                    if isinstance(lora_a, torch.Tensor):
                        lora_a = [lora_a] * self.n_slices
            return _orig(self, index, lora_a, lora_b, *args, **kwargs)

        _set_lora._diffrl_fused_merged_tolerant = True  # type: ignore[attr-defined]
        return _set_lora

    # Patch every merged class that defines its own ``set_lora``; subclasses that
    # only inherit it are covered transitively by the base-class patch.
    for _name in (
        "MergedColumnParallelLinearWithLoRA",
        "MergedQKVParallelLinearWithLoRA",
    ):
        cls = getattr(_cpl, _name, None)
        if cls is None or "set_lora" not in cls.__dict__:
            continue
        orig = cls.__dict__["set_lora"]
        if getattr(orig, "_diffrl_fused_merged_tolerant", False):
            continue
        cls.set_lora = _make(orig)


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

    Replaces pod-local file patch on ``vllm/lora/utils.py``.
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

    # Rebind stale references in modules that did `from vllm.lora.utils import
    # from_layer` at top level before our patch ran.
    import importlib as _importlib

    for _modname in (
        "vllm.lora.lora_model",
        "vllm.lora.models",
        "vllm.lora.model_manager",
        "vllm.lora.worker_manager",
    ):
        try:
            _mod = _importlib.import_module(_modname)
        except ImportError:
            continue
        if getattr(_mod, "from_layer", None) is _orig_from_layer:
            _mod.from_layer = _patched_from_layer


def patch_lora_request_passthrough() -> None:
    """Forward ``lora_request`` through ``Omni.generate`` to ``engine.add_request``.

    Required for HI3-Instruct t2i RL (``think_recaption`` mode) so that the AR
    prelude stage in vllm-omni picks up the per-rollout LoRA adapter alongside
    the DiT stage. Without this, ``VLLMOmniRolloutEngine.generate`` cannot pass
    ``lora_request`` into the AR stage's request scheduler — the AR worker runs
    the base model while DiT runs the LoRA-adapted model (half-adapted
    trajectory => silent policy/rollout mismatch).

    Replaces pod-local file patch on ``vllm_omni/entrypoints/omni.py``.
    """
    try:
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
        from vllm_omni.entrypoints.omni import Omni
    except (ImportError, AttributeError):
        return  # vllm-omni not available in this process; skip

    # ── Omni.generate: stash lora_request on the engine instance ──────
    _orig_omni_generate = Omni.generate
    if not getattr(_orig_omni_generate, "_diffrl_lora_request_passthrough", False):

        def _patched_omni_generate(self, *args, lora_request=None, _orig=_orig_omni_generate, **kwargs):
            self.engine._diffrl_pending_lora_request = lora_request
            py_generator = kwargs.get("py_generator", False)
            try:
                result = _orig(self, *args, **kwargs)
            except Exception:
                self.engine._diffrl_pending_lora_request = None
                raise
            if py_generator:
                # ``_orig`` returned a generator — wrap so we clear the stash
                # only when the generator is exhausted / closed.
                def _wrapped(gen, engine):
                    try:
                        yield from gen
                    finally:
                        engine._diffrl_pending_lora_request = None

                return _wrapped(result, self.engine)
            self.engine._diffrl_pending_lora_request = None
            return result

        _patched_omni_generate._diffrl_lora_request_passthrough = True  # type: ignore[attr-defined]
        Omni.generate = _patched_omni_generate

    # ── AsyncOmniEngine.add_request: pickup from stash ────────────────
    _orig_add_request = AsyncOmniEngine.add_request
    if not getattr(_orig_add_request, "_diffrl_lora_request_passthrough", False):

        def _patched_add_request(self, *args, lora_request=None, _orig=_orig_add_request, **kwargs):
            if lora_request is None:
                lora_request = getattr(self, "_diffrl_pending_lora_request", None)
            return _orig(self, *args, lora_request=lora_request, **kwargs)

        _patched_add_request._diffrl_lora_request_passthrough = True  # type: ignore[attr-defined]
        AsyncOmniEngine.add_request = _patched_add_request


def patch_sigmas_passthrough() -> None:
    """Monkey-patch HunyuanImage3Pipeline to forward custom sigmas to DiT scheduler.

    Outer ``HunyuanImage3Pipeline.forward`` extracts sigmas from req and stashes
    on the instance; inner ``HunyuanImage3Text2ImagePipeline.__call__`` picks up
    via ``self.model`` (which references the outer instance) and injects as a
    kwarg so ``scheduler.set_timesteps`` gets the correct schedule.

    Without this, UniRL's FlowMatchSchedulePolicy.sigmas is never
    forwarded to the DiT scheduler (rollout-train sigma mismatch
    max abs diff ~0.158 => GRPO log-prob replay incorrect).

    Replaces pod-local file patch on ``vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py``.
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
                self.unirl_sigmas = sigmas
                try:
                    return _orig(self, req, *args, **kwargs)
                finally:
                    self.unirl_sigmas = None

            _patched_outer_forward._diffrl_sigmas_passthrough = True  # type: ignore[attr-defined]
            HunyuanImage3Pipeline.forward = _patched_outer_forward

        _orig_inner_call = HunyuanImage3Text2ImagePipeline.__call__
        if not getattr(_orig_inner_call, "_diffrl_sigmas_passthrough", False):

            def _patched_inner_call(self, *args, _orig=_orig_inner_call, **kwargs):
                outer = getattr(self, "model", None)
                sigmas = getattr(outer, "unirl_sigmas", None) if outer is not None else None
                if sigmas is not None and "sigmas" not in kwargs:
                    kwargs["sigmas"] = sigmas
                return _orig(self, *args, **kwargs)

            _patched_inner_call._diffrl_sigmas_passthrough = True  # type: ignore[attr-defined]
            HunyuanImage3Text2ImagePipeline.__call__ = _patched_inner_call
    except (ImportError, AttributeError):
        pass  # pipeline not available in this process; skip


def patch_per_request_ar_seed() -> None:
    """Stamp a fresh os.urandom seed onto every AR SamplingParams in add_request's
    sampling_params_list. Without this, a GRPO group's N parallel requests all
    re-seed from the same shared SamplingParams ref and collapse to byte-identical
    AR tokens despite temperature > 0.
    """
    try:
        import msgspec as _msgspec
        from vllm import SamplingParams as VLLMSamplingParams
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    except (ImportError, AttributeError):
        return

    _orig = AsyncOmniEngine.add_request
    if getattr(_orig, "_diffrl_per_request_ar_seed", False):
        return

    import os as _os

    def _patched(self, *args, sampling_params_list=None, _orig=_orig, **kwargs):
        if sampling_params_list is not None:
            # SamplingParams is a msgspec.Struct shared across the N add_request
            # calls; ``structs.replace`` produces a brand-new instance per request
            # so the worker queue does not see one object holding the last seed.
            sampling_params_list = [
                _msgspec.structs.replace(sp, seed=int.from_bytes(_os.urandom(4), "big"))
                if isinstance(sp, VLLMSamplingParams) and getattr(sp, "seed", None) is None
                else sp
                for sp in sampling_params_list
            ]
        return _orig(self, *args, sampling_params_list=sampling_params_list, **kwargs)

    _patched._diffrl_per_request_ar_seed = True  # type: ignore[attr-defined]
    AsyncOmniEngine.add_request = _patched


def patch_master_port_unstrip() -> None:
    """Keep ``master_port`` alive through ``AsyncOmniEngine._strip_single_engine_args``.

    At the v0.20.0 pin the ``stage_configs_path`` route strips parent
    ``EngineArgs`` fields (including ``master_port``) from the kwargs that
    become ``base_engine_args`` for the per-stage YAML merge
    (``async_omni_engine.py:1558``), and the post-resolution injection loop
    only re-adds ``enable_sleep_mode`` / ``lora_path`` / ``lora_scale``.
    Net effect: the engine-reserved per-replica master-port base NEVER
    reaches ``OmniDiffusionConfig``, so every stage settles from the shared
    ``(None or 30005) + random(0, 100)`` window with only the 37-stride
    bind-check scan for collision avoidance (``diffusion/data.py:578``).
    Eight colocated replicas race that window; fast-booting models (SD3.5)
    happened to win, slow-booting ones (Qwen-Image, ~35s weight load) lose
    the check-to-bind TOCTOU and die with ``DistNetworkError ... port:
    30005, code: -98`` (LIN-382 qwen probe, 2026-06-07).

    Re-attach the caller's ``master_port`` to the stripped dict so the
    existing ``load_stage_configs_from_yaml`` ``base_engine_args`` merge
    lands it per stage. Stage-YAML keys still win (none of ours define
    ``master_port``); the settle scan stays as the TOCTOU fallback.

    DELETE-WHEN: pin >= v0.21.0rc2 — #3803 honors the injected base
    verbatim (mind the env ``MASTER_PORT`` precedence landmine documented
    in ``docs/vllm-omni-v2-engine.md``).
    """
    try:
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

        _orig = AsyncOmniEngine._strip_single_engine_args
        if getattr(_orig, "_diffrl_master_port_unstrip", False):
            return

        def _patched_strip(kwargs, _orig=_orig):
            out = _orig(kwargs)
            if isinstance(kwargs, dict):
                master_port = kwargs.get("master_port")
                if master_port is not None:
                    out["master_port"] = master_port
            return out

        _patched_strip._diffrl_master_port_unstrip = True  # type: ignore[attr-defined]
        AsyncOmniEngine._strip_single_engine_args = staticmethod(_patched_strip)
    except (ImportError, AttributeError):
        pass  # vllm-omni not available in this process; skip


def patch_hi3_flow_alignment() -> None:
    """Port of bjf-frz/fix-hi3-flow (vllm-omni eed27812) to v0.20.0's older
    KV-cache API: store full 4-D first-step KV, then scatter live image KV by
    absolute position_ids on subsequent steps. Silent skip on non-v0.20.0.

    Threads position_ids through a thread-local so we only need to patch
    `_save_image_kv_caches`, `_update_image_kv_caches` and a tiny wrapper
    around `HunyuanImage3DecoderLayer.forward` (no need to reimplement
    `ImageKVCacheManager.__call__` for the sake of one line).

    Delete this function once vllm-omni upstream lands the fix in our pinned version.
    """
    try:
        from vllm_omni.diffusion.models.hunyuan_image3 import (
            hunyuan_image3_transformer as _trans,
        )
    except (ImportError, AttributeError):
        return

    _ImageKVCacheManager = _trans.ImageKVCacheManager
    _DecoderLayer = _trans.HunyuanImage3DecoderLayer

    if not hasattr(_ImageKVCacheManager, "_save_image_kv_caches"):
        return

    import threading as _threading

    # Thread-local position_ids stash. Single denoise call chain (DecoderLayer.forward
    # → self_attn → image_attn → _update_image_kv_caches) is synchronous in one
    # thread, so the wrapper sets _tls.position_ids on entry and the patched
    # _update reads it back down the stack.
    _tls = _threading.local()

    _orig_save = _ImageKVCacheManager._save_image_kv_caches
    if not getattr(_orig_save, "_diffrl_hi3_flow_aligned", False):

        def _patched_save_image_kv_caches(self, key, value, seq_len):
            assert key.shape[1] == seq_len, f"first-step q_len({key.shape[1]}) != seq_len({seq_len})"
            self.image_kv_cache_map = (key.contiguous(), value.contiguous())

        _patched_save_image_kv_caches._diffrl_hi3_flow_aligned = True  # type: ignore[attr-defined]
        _ImageKVCacheManager._save_image_kv_caches = _patched_save_image_kv_caches

    _orig_update = _ImageKVCacheManager._update_image_kv_caches
    if not getattr(_orig_update, "_diffrl_hi3_flow_aligned", False):

        def _patched_update_image_kv_caches(self, key, value, seq_len, position_ids=None):
            cached_key, cached_value = self.image_kv_cache_map
            bs, q_len = key.shape[0], key.shape[1]
            if position_ids is None:
                position_ids = getattr(_tls, "position_ids", None)
            assert cached_key.dim() == 4, (
                f"patch_hi3_flow_alignment expects a 4-D cache from the patched "
                f"_save_image_kv_caches; got dim={cached_key.dim()}."
            )
            assert position_ids is not None and position_ids.shape == (bs, q_len), (
                f"position_ids missing or wrong shape: {None if position_ids is None else tuple(position_ids.shape)} "
                f"!= ({bs}, {q_len})"
            )
            result_k = cached_key.clone()
            result_v = cached_value.clone()
            for b in range(bs):
                result_k[b].index_copy_(0, position_ids[b], key[b])
                result_v[b].index_copy_(0, position_ids[b], value[b])
            return result_k.contiguous(), result_v.contiguous()

        _patched_update_image_kv_caches._diffrl_hi3_flow_aligned = True  # type: ignore[attr-defined]
        _ImageKVCacheManager._update_image_kv_caches = _patched_update_image_kv_caches

    _orig_decoder = _DecoderLayer.forward
    if not getattr(_orig_decoder, "_diffrl_hi3_flow_aligned", False):

        def _patched_decoder_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            *args,
            _orig=_orig_decoder,
            **kwargs,
        ):
            _prev = getattr(_tls, "position_ids", None)
            _tls.position_ids = position_ids
            try:
                return _orig(self, hidden_states, attention_mask, position_ids, *args, **kwargs)
            finally:
                _tls.position_ids = _prev

        _patched_decoder_forward._diffrl_hi3_flow_aligned = True  # type: ignore[attr-defined]
        _DecoderLayer.forward = _patched_decoder_forward


def patch_sd3_shared_kernels() -> None:
    """Engine half of SD3 shared-kernel parity. Gated on
    ``UNIRL_VLLM_OMNI_PARITY=1`` (set by ``VLLMOmniBackend.boot`` when the
    engine config enables ``parity_mode``; spawn children inherit it).

    Runs pre-model-build in every worker child (hijack order), so class-attr
    patches land before ``Attention.__init__`` binds impl instances:

    - ``FlashAttentionImpl.forward_cuda`` → ``unirl.kernels.sd3.shared_attention``
      for the unmasked case (SD3 never passes a mask); masked callers fall
      through to the original implementation.
    - vllm ``RMSNorm.forward_cuda`` / ``forward_native`` (no-residual case) →
      ``shared_rms_norm`` — the DiT qk-norms. The fused-add residual path is
      not used by the SD3 DiT and passes through untouched.
    - Pins ``DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN`` so platform selection
      cannot silently fall back to SDPA (missing FA then fails loudly).
    - One-shot DiT forward shape/dtype log for geometry verification against
      the trainer replay logs.

    Trainer-side counterpart: ``unirl/models/sd3/parity.py`` (installed by
    ``SD3Bundle`` under ``shared_kernels: true``). Both sides import the SAME
    ``unirl.kernels.sd3`` functions — one kernel set, two hosts.
    """
    import os as _os

    if _os.environ.get("UNIRL_VLLM_OMNI_PARITY") != "1":
        return
    _os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")

    from vllm.model_executor.layers.layernorm import RMSNorm as _VllmRMSNorm
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    from unirl.kernels.sd3 import kernel_fingerprint, shared_attention, shared_rms_norm

    if not getattr(FlashAttentionImpl.forward_cuda, "_diffrl_sd3_parity", False):
        _orig_fa_forward = FlashAttentionImpl.forward_cuda

        def _parity_attn_forward(self, query, key, value, attn_metadata=None, *, _orig=_orig_fa_forward):
            mask = attn_metadata.attn_mask if attn_metadata is not None else None
            if mask is not None:
                return _orig(self, query, key, value, attn_metadata)
            return shared_attention(
                query, key, value, softmax_scale=self.softmax_scale, causal=self.causal
            )

        _parity_attn_forward._diffrl_sd3_parity = True  # type: ignore[attr-defined]
        FlashAttentionImpl.forward_cuda = _parity_attn_forward

    if not getattr(_VllmRMSNorm.forward_cuda, "_diffrl_sd3_parity", False):
        _orig_rms_cuda = _VllmRMSNorm.forward_cuda
        _orig_rms_native = _VllmRMSNorm.forward_native

        def _parity_rms_cuda(self, x, residual=None, *, _orig=_orig_rms_cuda):
            if residual is not None:
                return _orig(self, x, residual)
            return shared_rms_norm(x, self.weight, float(self.variance_epsilon))

        def _parity_rms_native(self, x, residual=None, *, _orig=_orig_rms_native):
            if residual is not None:
                return _orig(self, x, residual)
            return shared_rms_norm(x, self.weight, float(self.variance_epsilon))

        _parity_rms_cuda._diffrl_sd3_parity = True  # type: ignore[attr-defined]
        _parity_rms_native._diffrl_sd3_parity = True  # type: ignore[attr-defined]
        _VllmRMSNorm.forward_cuda = _parity_rms_cuda
        _VllmRMSNorm.forward_native = _parity_rms_native

    # One-shot geometry log — the actual forward batch is the parity contract
    # the trainer's replay must match (micro_batch_size == this batch dim).
    # With /tmp/unirl_parity_debug present, also SHA-log the first calls'
    # inputs/outputs for cross-process divergence hunting (see
    # unirl.kernels.sd3.parity_debug_sha; trainer counterpart in
    # SD3DiffusionStep.predict_noise).
    try:
        from vllm_omni.diffusion.models.sd3.sd3_transformer import SD3Transformer2DModel

        if not getattr(SD3Transformer2DModel.forward, "_diffrl_sd3_parity", False):
            _orig_dit_forward = SD3Transformer2DModel.forward
            _logged = {"count": 0}

            def _logged_forward(self, *args, _orig=_orig_dit_forward, **kwargs):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                ehs = kwargs.get("encoder_hidden_states")
                ts = kwargs.get("timestep")
                if _logged["count"] == 0:
                    logger.info(
                        "[sd3-parity] engine DiT forward: hidden=%s %s | encoder=%s | timestep=%s %s | %s",
                        tuple(hs.shape) if hs is not None else None,
                        hs.dtype if hs is not None else None,
                        tuple(ehs.shape) if ehs is not None else None,
                        tuple(ts.shape) if ts is not None else None,
                        ts.dtype if ts is not None else None,
                        kernel_fingerprint(),
                    )
                out = _orig(self, *args, **kwargs)
                import os as _os2

                if _logged["count"] < 14 and _os2.path.exists("/tmp/unirl_parity_debug"):
                    from unirl.kernels.sd3 import parity_debug_sha as _sha

                    pooled = kwargs.get("pooled_projections")
                    sample = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
                    logger.info(
                        "[sd3-parity-dbg] ENGINE call=%d t=%.6f x=%s enc=%s pool=%s out=%s",
                        _logged["count"],
                        float(ts.reshape(-1)[0]) if ts is not None else -1.0,
                        _sha(hs),
                        _sha(ehs),
                        _sha(pooled),
                        _sha(sample),
                    )
                _logged["count"] += 1
                return out

            _logged_forward._diffrl_sd3_parity = True  # type: ignore[attr-defined]
            SD3Transformer2DModel.forward = _logged_forward
    except Exception:  # noqa: BLE001 — the log is diagnostic, never block boot
        logger.warning("patch_sd3_shared_kernels: shape-log install failed", exc_info=True)

    logger.info("[sd3-parity] engine shared kernels installed: %s", kernel_fingerprint())


def patch_wan22_shared_kernels() -> None:
    """Engine half of Wan2.2 shared-kernel parity. Same gate and ordering as
    ``patch_sd3_shared_kernels`` (which already covers the attention path —
    Wan routes through the SAME ``FlashAttentionImpl`` class).

    Wan-specific targets:

    - ``vllm_omni.diffusion.layers.norm.RMSNorm`` (``forward_cuda`` /
      ``forward_native``) → ``shared_rms_norm``. Wan's qk-norms use this
      vllm-omni-local class (across-heads width, pre-head-split), NOT the
      ``vllm.model_executor`` RMSNorm the SD3 patch targets. Its fused CUDA
      kernel otherwise rounds differently from the trainer's eager norm.
      Blast radius: only the DiT qk-norms use this class in the wan2_2 model;
      the block norms are ``LayerNorm``/``AdaLayerNorm`` whose forward_cuda is
      already the pure-torch fp32 expression the trainer mirrors.
    - ``DistributedRMSNorm.__init__`` → raise. It only instantiates at TP>1,
      where RowParallel partial-sum order breaks bitwise parity anyway — fail
      at build instead of tripping the gate 40 layers deep.
    - One-shot Wan DiT forward shape/dtype log (+ SHA log of the first calls
      when ``/tmp/unirl_parity_debug`` exists) for geometry verification
      against the trainer replay logs.

    Trainer-side counterpart: ``unirl/models/wan22/parity.py`` (installed by
    the WAN22 bundle under ``shared_kernels: true``).
    """
    import os as _os

    if _os.environ.get("UNIRL_VLLM_OMNI_PARITY") != "1":
        return

    from vllm_omni.diffusion.layers.norm import RMSNorm as _OmniRMSNorm

    from unirl.kernels.sd3 import kernel_fingerprint, shared_rms_norm

    if not getattr(_OmniRMSNorm.forward_cuda, "_diffrl_wan22_parity", False):

        def _parity_wan_rms(self, x, *, _shared=shared_rms_norm):
            return _shared(x, self.weight, float(self.variance_epsilon))

        _parity_wan_rms._diffrl_wan22_parity = True  # type: ignore[attr-defined]
        _OmniRMSNorm.forward_cuda = _parity_wan_rms
        _OmniRMSNorm.forward_native = _parity_wan_rms

    try:
        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
            DistributedRMSNorm as _WanDistRMSNorm,
        )
        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
            WanTransformer3DModel as _WanDiT,
        )

        if not getattr(_WanDistRMSNorm.__init__, "_diffrl_wan22_parity", False):

            def _refuse_tp_rms(self, *args, **kwargs):
                raise RuntimeError(
                    "wan22 parity mode requires tensor_parallel_size=1 "
                    "(DistributedRMSNorm/RowParallel partial-sum order is not "
                    "bitwise-reproducible by the unsharded trainer)."
                )

            _refuse_tp_rms._diffrl_wan22_parity = True  # type: ignore[attr-defined]
            _WanDistRMSNorm.__init__ = _refuse_tp_rms

        if not getattr(_WanDiT.forward, "_diffrl_wan22_parity", False):
            _orig_wan_forward = _WanDiT.forward
            _wan_logged = {"count": 0}

            def _logged_wan_forward(self, *args, _orig=_orig_wan_forward, **kwargs):
                import os as _os3

                hs = kwargs.get("hidden_states", args[0] if args else None)
                ehs = kwargs.get("encoder_hidden_states")
                ts = kwargs.get("timestep")
                if _wan_logged["count"] == 0:
                    logger.info(
                        "[wan22-parity] engine DiT forward: hidden=%s %s | encoder=%s | timestep=%s %s | %s",
                        tuple(hs.shape) if hs is not None else None,
                        hs.dtype if hs is not None else None,
                        tuple(ehs.shape) if ehs is not None else None,
                        tuple(ts.shape) if ts is not None else None,
                        ts.dtype if ts is not None else None,
                        kernel_fingerprint(),
                    )
                # Per-block SHA capture at the divergent step (t≈989.58) —
                # gated on the same debug flag; hooks attach for ONE call.
                _hook_handles = []
                _t0 = float(ts.reshape(-1)[0]) if ts is not None else -1.0
                if (
                    _os3.path.exists("/tmp/unirl_parity_debug")
                    and 985.0 < _t0 < 995.0
                    and not _wan_logged.get("blk_fired")
                ):
                    _wan_logged["blk_fired"] = True
                    from unirl.kernels.sd3 import parity_debug_sha as _bsha

                    _os3.makedirs("/root/parity_dump", exist_ok=True)
                    torch.save(
                        {"t": ts.detach().cpu(), "enc": ehs.detach().cpu(), "x": hs.detach().cpu()},
                        "/root/parity_dump/engine_in.pt",
                    )

                    def _cond_dump(_m, _i, o):
                        torch.save(
                            tuple(v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in o),
                            "/root/parity_dump/engine_cond.pt",
                        )

                    _hook_handles.append(self.condition_embedder.register_forward_hook(_cond_dump))

                    def _mk(name):
                        def _h(_m, _i, o):
                            t = o[0] if isinstance(o, tuple) else o
                            print(f"[wan22-blk-dbg] ENGINE {name} {_bsha(t)}", flush=True)

                        return _h

                    _hook_handles.append(self.patch_embedding.register_forward_hook(_mk("patch_embedding")))
                    _hook_handles.append(self.condition_embedder.register_forward_hook(_mk("condition_embedder")))
                    for _bi, _blk in enumerate(self.blocks):
                        _hook_handles.append(_blk.register_forward_hook(_mk(f"block{_bi:02d}")))
                try:
                    out = _orig(self, *args, **kwargs)
                finally:
                    for _hh in _hook_handles:
                        _hh.remove()
                import os as _os2

                if _wan_logged["count"] < 26 and _os2.path.exists("/tmp/unirl_parity_debug"):
                    from unirl.kernels.sd3 import parity_debug_sha as _sha

                    sample = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
                    logger.info(
                        "[wan22-parity-dbg] ENGINE call=%d t=%.6f t_precise=%.17g t_dtype=%s x=%s enc=%s out=%s",
                        _wan_logged["count"],
                        float(ts.reshape(-1)[0]) if ts is not None else -1.0,
                        float(ts.reshape(-1)[0]) if ts is not None else -1.0,
                        ts.dtype if ts is not None else None,
                        _sha(hs),
                        _sha(ehs),
                        _sha(sample),
                    )
                _wan_logged["count"] += 1
                return out

            _logged_wan_forward._diffrl_wan22_parity = True  # type: ignore[attr-defined]
            _WanDiT.forward = _logged_wan_forward
    except Exception:  # noqa: BLE001 — diagnostics/guards, never block boot
        logger.warning("patch_wan22_shared_kernels: wan model guards install failed", exc_info=True)

    logger.info("[wan22-parity] engine shared kernels installed: %s", kernel_fingerprint())


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
        patch_ar_merged_lora_fused_tensor()
        patch_fp32_skip()
        patch_lora_request_passthrough()
        patch_per_request_ar_seed()
        patch_sigmas_passthrough()
        patch_hi3_flow_alignment()
        patch_master_port_unstrip()
        patch_sd3_shared_kernels()
        patch_wan22_shared_kernels()


__all__ = [
    "OmniTensorLoRARequest",
    "VLLMOmniHijack",
    "patch_hi3_flow_alignment",
    "patch_per_request_ar_seed",
    "patch_sd3_shared_kernels",
    "patch_sigmas_passthrough",
    "patch_wan22_shared_kernels",
]
