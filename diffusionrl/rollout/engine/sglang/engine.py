"""New-protocol SGLang rollout engine.

One class, one-shot construction. Implements
:class:`diffusionrl.rollout.engine.base.BaseRolloutEngine` and speaks the
``RolloutReq`` / ``RolloutResp`` types end-to-end.

Lifecycle:
    The ctor takes config + runtime deps (``device``, ``strategy``, ``rank``,
    ``model_config``) and returns a fully-usable engine: ``ServerArgs`` built,
    ``DiffGenerator.from_pretrained`` complete, scheduler reachable. There is
    no separate ``initialize(device)`` step.

Generation:
    ``generate(req)`` either reads the pre-shipped ``initial_latents`` off
    ``req.request_conditions['initial_latents']`` or computes Gaussian noise
    via :meth:`_compute_initial_noise` (gated on ``cfg.init_same_noise`` for
    per-group sharing). The translators in :mod:`request` / :mod:`response`
    handle the kwargs build and the result → ``RolloutResp`` packing.

Weight sync:
    Five direct forwards to SGLang's scheduler request types — each accepts an
    ignored ``stage_ids`` kwarg so the actor-side
    :class:`RolloutWeightSyncMixin` can pass it uniformly to vllm-omni and
    SGLang. ``update_weights_from_ipc`` raises ``NotImplementedError`` —
    SGLang has no bucketed-IPC receiver today.

What this engine intentionally does NOT do (vs the legacy SGLang adapter at
``diffusionrl/samplers/sglang/engine.py``):

- No ``initialize(device)`` step / ``_is_initialized`` flag — one-shot ctor.
- No ``update_weights(state_dict)`` / ``update_weights_from_path`` — the
  trainer-side ``UpdateWeightFromTensor`` handler packages and pushes via
  ``update_weights_from_tensor`` directly.
- No ``_infer_model_type`` substring match — ``cfg.model_family`` is an
  explicit enum.
- No instance-level ``_cached_runtime`` import hook — lazy module-level.
- No ``encode_prompt`` / ``decode_latents`` — not on the new ABC.
- No ``supports_distributed`` / ``requires_external_service`` properties.
- No ``get_last_weight_checksum`` / ``_verify_weight_checksum`` flag — use
  :meth:`loaded_param_checksums` on demand.
- No ``ForwardContext`` build inside the engine — trainer-side replay
  reconstructs typed conditions from ``resp.conditions``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from diffusionrl.config.require import require
from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sglang.request import _to_sglang_kwargs
from diffusionrl.rollout.engine.sglang.response import _to_rollout_resp
from diffusionrl.sde.noise import generate_latents
from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


def _import_sglang_runtime() -> Dict[str, Any]:
    """Lazy import of SGLang scheduler types. Imported once per process."""
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        GetWeightsChecksumReqInput,
        InitWeightsUpdateGroupReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import SetLoraFromTensorsReq
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    return {
        "DiffGenerator": DiffGenerator,
        "ServerArgs": ServerArgs,
        "GetWeightsChecksumReqInput": GetWeightsChecksumReqInput,
        "InitWeightsUpdateGroupReqInput": InitWeightsUpdateGroupReqInput,
        "DestroyWeightsUpdateGroupReqInput": DestroyWeightsUpdateGroupReqInput,
        "UpdateWeightsFromDistributedReqInput": UpdateWeightsFromDistributedReqInput,
        "UpdateWeightsFromTensorReqInput": UpdateWeightsFromTensorReqInput,
        "SetLoraFromTensorsReq": SetLoraFromTensorsReq,
        "sync_scheduler_client": sync_scheduler_client,
    }


class SGLangRolloutEngine(BaseRolloutEngine):
    """New-protocol rollout engine backed by ``sglang.multimodal_gen.DiffGenerator``."""

    _component_name = "sglang_new"

    def __init__(
        self,
        config: SGLangEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, SGLangEngineConfig),
            f"SGLangRolloutEngine requires SGLangEngineConfig; got {type(config).__name__}",
        )
        require(
            model_config is not None and bool(model_config.pretrained_model_ckpt_path),
            "SGLangRolloutEngine requires model_config.pretrained_model_ckpt_path",
        )
        if rank is not None:
            config = config.with_sglang_ports(int(rank))

        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sde_label = self._resolve_sde_label(strategy)
        self._target_modules: List[str] = list(self.cfg.target_modules or ("transformer",))
        self._runtime = _import_sglang_runtime()
        self._is_offloaded = False

        server_kwargs = self.cfg.build_server_kwargs(
            self._runtime["ServerArgs"],
            model_config=model_config,
        )

        # LoRA / target-module agreement check — pre-existing safety net.
        if model_config.use_lora and not server_kwargs.get("lora_target_modules"):
            logger.warning(
                "SGLang LoRA enabled without lora_target_modules; set bundle "
                "default_lora_target_modules() or --training.lora-target-modules."
            )

        logger.info(
            "Initializing new-protocol SGLang engine (rank=%s, local_mode=%s, "
            "target_modules=%s, model_family=%s, populate_conditions=%s, "
            "logprob_source=%s)",
            rank,
            self.cfg.local_mode,
            self._target_modules,
            self.cfg.model_family,
            self.cfg.populate_conditions,
            self.cfg.logprob_source,
        )

        disable_autocast = server_kwargs.get("disable_autocast")
        server_args = self._runtime["ServerArgs"].from_kwargs(**server_kwargs)
        if disable_autocast is not None:
            server_args.disable_autocast = disable_autocast

        self._server_args = server_args
        self._generator = self._runtime["DiffGenerator"].from_pretrained(
            server_args=server_args,
            local_mode=bool(self.cfg.local_mode),
        )

        # σ schedule policy — loaded once from the pretrained checkpoint
        # dir's JSONs (scheduler/transformer/vae configs). ``ensure_req_sigmas``
        # consumes it in ``generate`` to pin ``req.sigmas`` before the
        # request crosses the wire to the SGLang worker.
        #
        # ``shift`` source: ``model_config.shift`` is the single SOT.
        # New-path ``SD3PipelineConfig`` / ``WAN21PipelineConfig`` /
        # ``WAN22PipelineConfig`` / ``HunyuanImage3PipelineConfig`` all
        # carry it (SD3=3.0, Wan=5.0, etc.). Legacy ``ModelBundleConfig``
        # has no ``shift`` field — that path is no longer compatible
        # with the new engine; users must migrate to the new model
        # configs.
        if not hasattr(model_config, "shift"):
            raise RuntimeError(
                "SGLangRolloutEngine requires model_config.shift "
                "(legacy ModelBundleConfig has no shift field after the "
                "σ-consolidation refactor). Switch the experiment's "
                "``cfg.model`` to a new-path model config (e.g. "
                "``sd3_v2``, ``wan21_v2``, ``hunyuan_image3_v2``)."
            )
        # Same use_dynamic_shifting hook as vllm_omni engine. Generic —
        # any model config that declares it (Qwen-Image, future dynamic
        # models) gets the right policy without engine-side dispatch.
        require_dynamic = bool(getattr(model_config, "use_dynamic_shifting", False))
        dynamic_overrides = getattr(model_config, "dynamic_shift_overrides", None)
        self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
            model_config.pretrained_model_ckpt_path,
            shift=float(model_config.shift),
            require_dynamic=require_dynamic,
            dynamic_overrides=dynamic_overrides,
        )

    # ------------------------------------------------------------------
    # Strategy → SGLang SDE kernel label mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sde_label(strategy: Any) -> Optional[str]:
        """Resolve the SDE-strategy → SGLang kernel label once at ctor time.

        Mirrors legacy ``samplers/sglang/engine.py:_resolve_rollout_sde_type``.
        Returns ``None`` when strategy is missing — ODE-mode callers (eval,
        NFT) won't hit the SDE branch in the request translator anyway.
        """
        if strategy is None:
            return None
        canonical = type(strategy).canonical_name.strip().lower()
        if canonical == "flow":
            return "sde"
        if canonical == "cps":
            return "cps"
        raise ValueError(
            f"SGLang rollout currently supports only sde_type in {{'flow', 'cps'}} "
            f"(those have a verified SGLang-side kernel that matches DiffusionRL's math); "
            f"got canonical={canonical!r}. Either switch the SDE strategy on this engine, "
            f"or add an explicit mapping after verifying the SGLang-side kernel is "
            f"mathematically equivalent."
        )

    # ------------------------------------------------------------------
    # Scheduler request plumbing
    # ------------------------------------------------------------------

    def _send_scheduler_request(self, request: Any, *, operation: str) -> Any:
        response = self._runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(response, operation=operation)
        require(success, f"{operation} failed: {message}")
        return response

    @staticmethod
    def _extract_update_status(response: Any, *, operation: str) -> tuple[bool, str]:
        output = getattr(response, "output", None)
        require(isinstance(output, dict), f"Invalid SGLang response for {operation}: {response}")
        success = bool(output.get("success", False))
        message = str(output.get("message", "Unknown status"))
        return success, message

    def _call_memory_api(
        self,
        method_name: str,
        *,
        tags: Sequence[str],
        cpu_backup_tags: Optional[Sequence[str]] = None,
    ) -> Any:
        method = getattr(self._generator, method_name, None)
        require(callable(method), f"SGLang generator missing memory API: {method_name}.")
        kwargs: Dict[str, Any] = {"tags": list(tags)}
        if cpu_backup_tags is not None:
            kwargs["cpu_backup_tags"] = list(cpu_backup_tags)
        try:
            response = method(**kwargs)
        except TypeError:
            if cpu_backup_tags is None:
                raise
            response = method(tags=list(tags))
        if isinstance(response, dict):
            require(
                bool(response.get("success", True)),
                str(response.get("message", f"{method_name} failed")),
            )
        return response

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, req: RolloutReq) -> RolloutResp:
        require(
            int(req.batch_size) > 0,
            "SGLangRolloutEngine.generate requires non-empty req (batch_size > 0)",
        )

        # Main-repo SSOT for σ: pin once via the shared helper. Request
        # translator reads ``req.sigmas`` (no recompute) and forwards to
        # SGLang; response handler asserts SGLang echoed back what we sent.
        ensure_req_sigmas(req, self.schedule_policy)

        initial_noise = self._resolve_initial_noise(req)

        kwargs = _to_sglang_kwargs(
            req,
            cfg=self.cfg,
            sde_label=self._sde_label,
            initial_noise=initial_noise,
        )

        raw_results = self._generator.generate(sampling_params_kwargs=kwargs)
        require(
            raw_results is not None,
            "SGLang generator returned None — full-batch failure (see DiffGenerator.generate docstring)",
        )
        results = list(raw_results) if isinstance(raw_results, list) else [raw_results]

        diffusion = dict(req.stage_params.get("diffusion") or {})
        num_steps = int(diffusion["num_inference_steps"])
        sde_indices_raw = diffusion.get("sde_indices")
        sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None
        use_native_logprob = self.cfg.logprob_source == "native" and sde_indices is not None

        return _to_rollout_resp(
            req,
            results,
            cfg=self.cfg,
            num_steps=num_steps,
            sde_indices=sde_indices,
            use_native_logprob=use_native_logprob,
        )

    def _resolve_initial_noise(self, req: RolloutReq) -> Optional[torch.Tensor]:
        """Decide where ``initial_noise`` comes from for this generate call.

        Precedence:
        1. Pre-shipped ``req.request_conditions['initial_latents'].latents`` →
           use verbatim (caller owns the tensor).
        2. ``cfg.init_same_noise=True`` → engine-internal Gaussian noise keyed
           on ``req.group_ids`` + ``stage_params['diffusion']['seed']`` for
           per-group determinism.
        3. Otherwise → ``None`` (SGLang draws its own; matches legacy semantic
           when ``init_same_noise=False`` and no pre-shipped tensor).
        """
        # Path 1: pre-shipped
        cond = req.request_conditions.get("initial_latents")
        if cond is not None and getattr(cond, "latents", None) is not None:
            return cond.latents

        # Path 2: engine-computed (same-group sharing)
        if not bool(self.cfg.init_same_noise):
            return None

        diffusion = dict(req.stage_params.get("diffusion") or {})
        seed = diffusion.get("seed")
        require(
            seed is not None,
            "SGLangRolloutEngine: init_same_noise=True requires req.stage_params['diffusion']['seed']",
        )

        sp = self.cfg.sampling
        batch_size = int(req.batch_size)
        latent_shape = self._latent_shape(
            height=int(diffusion["height"]),
            width=int(diffusion["width"]),
            num_frames=int(sp.num_frames),
            batch_size=batch_size,
        )
        dtype = parse_torch_dtype(sp.autocast_precision, field_name="autocast_precision")
        return generate_latents(
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=self._device,
            dtype=dtype,
            init_same_noise=True,
            samples_per_prompt=int(diffusion.get("num_samples_per_prompt", 1)),
            noise_group_ids=[str(gid) for gid in req.group_ids],
            base_seed=int(seed),
        )

    def _latent_shape(
        self,
        *,
        height: int,
        width: int,
        num_frames: int,
        batch_size: int,
    ) -> tuple:
        """Resolve the per-sample latent shape via SGLang's pipeline_config."""
        from types import SimpleNamespace

        batch_stub = SimpleNamespace(height=height, width=width, num_frames=num_frames)
        full_shape = self._server_args.pipeline_config.prepare_latent_shape(
            batch_stub,
            batch_size,
            num_frames,
        )
        return tuple(full_shape[1:])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def sleep(self) -> None:
        self._call_memory_api(
            "release_memory_occupation",
            tags=["transformer", "vae", "text_encoder"],
            cpu_backup_tags=["vae", "text_encoder"],
        )
        self._is_offloaded = True
        logger.info("SGLang engine entered sleep state via release_memory_occupation().")

    def wake_up(self) -> None:
        if not self._is_offloaded:
            return
        self._call_memory_api(
            "resume_memory_occupation",
            tags=["transformer", "vae", "text_encoder"],
        )
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        if self._generator is None:
            return False
        try:
            return bool(self._runtime["sync_scheduler_client"].ping())
        except Exception as exc:
            logger.warning("SGLang health_check ping failed: %s", exc)
            return False

    def shutdown(self) -> None:
        if self._generator is not None:
            try:
                self._generator.shutdown()
            except Exception as exc:
                logger.warning("SGLang shutdown failed: %s", exc)
        self._generator = None

    # ------------------------------------------------------------------
    # Weight sync — direct forwards (stage_ids accepted and ignored;
    # SGLang is single-stage).
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        require(bool(serialized_named_tensors), "serialized_named_tensors must be non-empty")
        self._send_scheduler_request(
            self._runtime["UpdateWeightsFromTensorReqInput"](
                serialized_named_tensors=serialized_named_tensors,
                target_modules=list(target_modules or self._target_modules),
                load_format=load_format,
                flush_cache=flush_cache,
            ),
            operation="update_weights_from_tensor",
        )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        self._send_scheduler_request(
            self._runtime["InitWeightsUpdateGroupReqInput"](
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=int(rank_offset),
                world_size=int(world_size),
                group_name=str(group_name),
                backend=str(backend),
            ),
            operation="init_weights_update_group",
        )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        require(bool(names), "names must be non-empty for distributed update")
        self._send_scheduler_request(
            self._runtime["UpdateWeightsFromDistributedReqInput"](
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(shape) for shape in shapes],
                group_name=str(group_name),
                target_modules=list(target_modules or self._target_modules),
                flush_cache=flush_cache,
            ),
            operation="update_weights_from_distributed",
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        self._send_scheduler_request(
            self._runtime["DestroyWeightsUpdateGroupReqInput"](
                group_name=str(group_name),
            ),
            operation="destroy_weights_update_group",
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        # ``lora_tensors_for_vllm`` ships keys in vLLM's PEFT-envelope form
        # ``base_model.model.<prefix><module>.lora_A.weight``, where
        # ``<prefix>`` comes from ``weight_sync_param_name_prefix`` on the
        # pipeline config (e.g. ``transformer.`` for SD3, set so vllm-omni's
        # whole-pipeline loader matches its own module names).
        #
        # SGLang's ``LoRAPipeline._register_lora_state_dict`` only strips
        # ``diffusion_model.`` and ``.weight``; it does NOT strip
        # ``base_model.model.`` or any pipeline-level module prefix. SGLang's
        # ``lora_layers`` dict, however, is keyed by ``named_modules()`` of
        # ``self.modules["transformer"]`` — i.e. starting INSIDE the
        # transformer — so layer keys are bare ``transformer_blocks.<i>...``.
        #
        # Without normalization here the registered key ends up as
        # ``base_model.model.transformer.<module>.lora_A`` while SGLang's
        # layers are just ``<module>``. The mismatch silently degrades to
        # ``LoRA adapter ... applied to 0 layers`` and the rollout runs the
        # base model. Strip both the PEFT envelope and the pipeline-level
        # ``transformer.`` head here so the registered keys match SGLang's
        # module-name space exactly. This is a sglang_new-side normalization
        # only; trainer-side ``weight_sync_param_name_prefix`` stays as-is so
        # vllm-omni continues to work on the same recipe.
        stripped: Dict[str, torch.Tensor] = {}
        envelope = "base_model.model."
        pipeline_prefix = "transformer."
        for name, tensor in lora_tensors.items():
            key = name
            if key.startswith(envelope):
                key = key[len(envelope) :]
            if key.startswith(pipeline_prefix):
                key = key[len(pipeline_prefix) :]
            stripped[key] = tensor

        request = self._runtime["SetLoraFromTensorsReq"](
            lora_nickname=str(adapter_name),
            lora_tensors=stripped,
            target="all",
            strength=1.0,
        )
        response = self._runtime["sync_scheduler_client"].forward(request)
        error = getattr(response, "error", None)
        require(error is None, f"set_lora_from_tensors failed: {error}")
        # Count the distinct LoRA layer names we registered (each layer ships
        # two tensors: ``lora_A`` + ``lora_B``). With the trainer's PEFT-envelope
        # + pipeline prefix stripped above, every key here is the bare
        # SGLang-layer name + ``.lora_A`` / ``.lora_B`` suffix, so deduping on
        # that gives the layer count SGLang's ``_apply_lora_to_layers`` will
        # match. For SD3.5-medium this is ~191.
        layer_names = set()
        for key in stripped:
            base = key
            for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            layer_names.add(base)
        logger.info(
            "SGLang LoRA initialized from tensors (adapter=%s) — LoRA applied to %d layers",
            adapter_name,
            len(layer_names),
        )

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        stage_ids: Optional[List[int]] = None,
    ) -> None:
        """Bucketed-IPC weight sync is not implemented for SGLang.

        SGLang has no ``BucketedWeightReceiver`` today. Callers should use
        :meth:`update_weights_from_tensor` (SGLang-shape one-bag payload)
        or :meth:`update_weights_from_distributed` (NCCL broadcast) instead.
        """
        raise NotImplementedError(
            "SGLangRolloutEngine.update_weights_from_ipc: SGLang lacks a "
            "BucketedWeightReceiver. Use update_weights_from_tensor or "
            "update_weights_from_distributed instead."
        )

    # ------------------------------------------------------------------
    # Post-load value-correctness query (vllm-omni-shape return)
    # ------------------------------------------------------------------

    def loaded_param_checksums(
        self,
        *,
        names: List[str],
        stage_ids: Optional[List[int]] = None,
    ) -> Dict[int, List[Dict[str, str]]]:
        """Query SGLang for short SHA256 hashes of loaded parameter values.

        Returns ``{0: [{name: hex_short, ...}]}`` — a single-stage,
        single-rank-aggregated map to match the vllm-omni shape so that
        :mod:`diffusionrl.distributed.weight_sync.checksum` helpers work
        identically against either engine.

        Note: SGLang's ``GetWeightsChecksumReqInput`` returns one map per
        target module (TP-flat aggregated server-side), so the per-rank list
        on the result always has length 1.
        """
        response = self._runtime["sync_scheduler_client"].forward(
            self._runtime["GetWeightsChecksumReqInput"](
                module_names=list(names),
            )
        )
        output = getattr(response, "output", None)
        require(
            isinstance(output, dict) and bool(output),
            f"SGLang checksum query returned invalid payload: {output!r}",
        )
        return {0: [{str(k): str(v) for k, v in output.items()}]}


__all__ = ["SGLangRolloutEngine"]
