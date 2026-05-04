from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch

from diffusionrl.config.require import require
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.samplers.sglang.config import SGLangEngineConfig
from diffusionrl.samplers.sglang.request import SGLangRolloutRequest
from diffusionrl.samplers.sglang.response import SGLangRolloutResponse
from diffusionrl.samplers.utils.noise import generate_latents
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples
from diffusionrl.utils.dtypes import parse_torch_dtype

from ..engine import BaseRolloutEngine

logger = logging.getLogger(__name__)


class SGLangRolloutEngine(BaseRolloutEngine):
    """Inference engine backed by ``sglang.multimodal_gen`` DiffGenerator."""

    _component_name = "sglang"

    # ------------------------------------------------------------------
    # Capability + construction
    # ------------------------------------------------------------------
    @classmethod
    def declared_capabilities(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

    def __init__(
        self,
        *,
        config: SGLangEngineConfig,
        model_config: ModelBundleConfig,
        rank: Optional[int] = None,
    ):
        # Per-rank SGLang port rewrite. When caller passes ``rank`` (the cfg
        # path via ``build(engine_cfg, rank=...)`` from ``RolloutActor.__init__``),
        # offset the SGLang port range internally. Legacy argparse path
        # pre-rewrites and calls with ``rank=None``, so no double-offset.
        if rank is not None:
            config = config.with_sglang_ports(int(rank))
        require(
            bool(model_config.pretrained_model_ckpt_path),
            "SGLang engine requires model_config.pretrained_model_ckpt_path",
        )
        super().__init__(config)
        self._model_config = model_config
        self._device: Optional[torch.device] = None
        self._generator: Any = None
        self._server_args: Any = None
        self._local_mode: bool = True
        self._target_modules: List[str] = ["transformer"]
        self._verify_weight_checksum: bool = True
        self._last_weight_checksum: Dict[str, str] = {}
        self._cached_runtime: Optional[Dict[str, Any]] = None
        self.strategy: Any = None  # set by actor after construction

    # ------------------------------------------------------------------
    # Import / runtime helpers
    # ------------------------------------------------------------------
    def _import_sglang_runtime(self) -> Dict[str, Any]:
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )
        from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
            DestroyWeightsUpdateGroupReqInput,
            GetWeightsChecksumReqInput,
            InitWeightsUpdateGroupReqInput,
            UpdateWeightFromDiskReqInput,
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        return {
            "DiffGenerator": DiffGenerator,
            "ServerArgs": ServerArgs,
            "UpdateWeightFromDiskReqInput": UpdateWeightFromDiskReqInput,
            "GetWeightsChecksumReqInput": GetWeightsChecksumReqInput,
            "InitWeightsUpdateGroupReqInput": InitWeightsUpdateGroupReqInput,
            "DestroyWeightsUpdateGroupReqInput": DestroyWeightsUpdateGroupReqInput,
            "UpdateWeightsFromDistributedReqInput": UpdateWeightsFromDistributedReqInput,
            "UpdateWeightsFromTensorReqInput": UpdateWeightsFromTensorReqInput,
            "sync_scheduler_client": sync_scheduler_client,
        }

    @property
    def _runtime(self) -> Dict[str, Any]:
        if self._cached_runtime is None:
            self._cached_runtime = self._import_sglang_runtime()
        return self._cached_runtime

    def _send_scheduler_request(self, request: Any, *, operation: str) -> Any:
        """Forward a request to SGLang scheduler and raise on failure."""
        response = self._runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(response, operation=operation)
        require(success, f"{operation} failed: {message}")
        return response

    def _call_memory_api(
        self,
        method_name: str,
        *,
        tags: Sequence[str],
        cpu_backup_tags: Optional[Sequence[str]] = None,
    ) -> Any:
        require(self._generator is not None, "SGLang generator is not initialized.")

        method = getattr(self._generator, method_name, None)
        require(callable(method), f"SGLang generator missing required memory API: {method_name}.")

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
    # Model-type policy
    # ------------------------------------------------------------------
    def _infer_model_type(self) -> str:
        raw_override = str(self.config.engine_kwargs.get("model_type", ""))
        raw_path = str(self._model_config.pretrained_model_ckpt_path or "")
        value = (raw_override or raw_path).lower()
        for substring in ("hunyuan", "flux", "sd3", "mochi"):
            if substring in value:
                return substring
        raise ValueError(
            f"SGLangRolloutEngine could not infer model_type from "
            f"engine_kwargs.model_type={raw_override!r} or "
            f"pretrained_model_ckpt_path={raw_path!r}. "
            f"Set engine_kwargs.model_type explicitly to one of "
            f"{{'hunyuan', 'flux', 'sd3', 'mochi'}} (an exact match — the "
            f"pre-existing rule is a substring check, so e.g. "
            f"'stabilityai/stable-diffusion-3.5-medium' does not match 'sd3')."
        )

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------
    def initialize(self, device: torch.device) -> None:
        self._device = device

        self._local_mode = bool(self.config.local_mode)
        self._verify_weight_checksum = bool(self.config.verify_weight_checksum)
        target_modules = self.config.target_modules
        if target_modules:
            self._target_modules = [str(m) for m in target_modules]

        server_kwargs = self.config.build_server_kwargs(
            self._runtime["ServerArgs"],
            model_config=self._model_config,
        )

        # Training-side PEFT and rollout-side SGLang must agree on target modules,
        # else SGLang wraps every linear layer and silently disables LoRA on those
        # the training side doesn't ship weights for.
        if self._model_config.use_lora and not server_kwargs.get("lora_target_modules"):
            logger.warning(
                "SGLang LoRA enabled without lora_target_modules; set bundle "
                "default_lora_target_modules() or --training.lora-target-modules. Bundle: %s",
                getattr(self.config, "model_dotpath", None) or "<unknown>",
            )

        logger.info(
            "Initializing SGLang-diffusion engine (local_mode=%s, target_modules=%s)",
            self._local_mode,
            self._target_modules,
        )
        disable_autocast = server_kwargs.get("disable_autocast")
        server_args = self._runtime["ServerArgs"].from_kwargs(**server_kwargs)
        if disable_autocast is not None:
            server_args.disable_autocast = disable_autocast
        generator = self._runtime["DiffGenerator"].from_pretrained(
            server_args=server_args,
            local_mode=self._local_mode,
        )

        self._server_args = server_args
        self._generator = generator
        self._is_initialized = True

    # ------------------------------------------------------------------
    # Core inference API
    # ------------------------------------------------------------------
    def generate(self, request: RolloutRequest) -> RolloutSamples:
        sgl_request = SGLangRolloutRequest.from_rollout_request(request)
        sgl_request.initial_noise = self._compute_initial_noise(request)
        # ``rollout_sde_type`` is only consumed by SGLang in SDE mode (rollout=True).
        # Non-SDE mode (eval, NFT-train: sde_indices is None) omits the rollout
        # kwargs entirely; setting the type would be dead state. In SDE mode
        # ``_resolve_rollout_sde_type`` raises on unsupported strategies — the
        # check is intentionally gated on ``rollout_sde_indices is not None``
        # so ODE callers (NFT, eval) are never blocked by an SGLang kernel
        # they wouldn't have used anyway.
        if sgl_request.rollout_sde_indices is not None:
            sgl_request.rollout_sde_type = self._resolve_rollout_sde_type()

        raw_results = self._generator.generate(sampling_params_kwargs=sgl_request.to_kwargs())
        require(
            raw_results is not None,
            "SGLang generator returned None — full-batch failure (see DiffGenerator.generate docstring)",
        )
        results = list(raw_results) if isinstance(raw_results, list) else [raw_results]

        # Native log_probs are only meaningful when SGLang ran an SDE rollout
        # (ODE mode never emits ``trajectory_log_probs``). Strategy-level
        # validation already happens in ``_resolve_rollout_sde_type``; here
        # we only gate on whether SDE was actually requested.
        use_native_logprob = self.config.logprob_source == "native" and sgl_request.rollout_sde_indices is not None

        response = SGLangRolloutResponse.from_sglang_results(results)
        return response.to_rollout_samples(
            request,
            model_type=self._infer_model_type(),
            num_inference_steps=sgl_request.num_inference_steps,
            shift=float(request.sampling_params.sde_config.shift),
            sde_indices=sgl_request.rollout_sde_indices,
            guidance_scale=sgl_request.guidance_scale,
            height=sgl_request.height,
            width=sgl_request.width,
            use_native_logprob=use_native_logprob,
            return_decoded_for_reward=True,
        )

    def _compute_initial_noise(self, request: RolloutRequest) -> torch.Tensor:
        sp = request.sampling_params
        n_expanded = len(request.prompts.prompts)

        batch_stub = SimpleNamespace(height=sp.height, width=sp.width, num_frames=sp.num_frames)
        full_shape = self._server_args.pipeline_config.prepare_latent_shape(
            batch_stub,
            n_expanded,
            sp.num_frames,
        )

        dtype = parse_torch_dtype(sp.autocast_precision, field_name="autocast_precision")

        return generate_latents(
            batch_size=n_expanded,
            latent_shape=tuple(full_shape[1:]),
            device=self._device,
            dtype=dtype,
            init_same_noise=sp.init_same_noise,
            samples_per_prompt=sp.num_samples_per_prompt,
            noise_group_ids=request.prompts.noise_group_ids if sp.init_same_noise else None,
            base_seed=sp.seed if sp.init_same_noise else None,
        )

    def _resolve_rollout_sde_type(self) -> str:
        """Map the injected DiffusionRL strategy to its SGLang kernel label.

        Only flow / cps have a verified SGLang-side kernel that matches
        DiffusionRL's math. dance/dpm2 (and any future strategy) must add
        an explicit, math-equivalent mapping verified end-to-end before
        being accepted — otherwise SGLang would sample with the flow
        kernel while training-side replay recomputed log_prob with the
        user-configured kernel, producing silently wrong GRPO ratios.
        """
        if self.strategy is None:
            return "sde"
        canonical = type(self.strategy).canonical_name.strip().lower()
        if canonical == "flow":
            return "sde"
        if canonical == "cps":
            return "cps"
        raise ValueError(
            f"SGLang rollout currently supports only sde_type in {{'flow', 'cps'}} "
            f"(those have a verified SGLang-side kernel that matches DiffusionRL's math); "
            f"got canonical={canonical!r}. Either switch the SDE strategy on this engine, "
            f"or add an explicit mapping after verifying the SGLang-side kernel is "
            f"mathematically equivalent — do not let SGLang sample with the flow kernel "
            f"while training-side replay uses a different kernel."
        )

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_update_status(response: Any, *, operation: str) -> tuple[bool, str]:
        output = getattr(response, "output", None)
        require(isinstance(output, dict), f"Invalid SGLang response for {operation}: {response}")
        success = bool(output.get("success", False))
        message = str(output.get("message", "Unknown status"))
        return success, message

    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        require(self._is_initialized, "SGLang engine is not initialized")
        require(bool(state_dict), "state_dict must be non-empty")

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        named_tensors = [
            (str(name), tensor.detach().contiguous())
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
        ]
        require(bool(named_tensors), "state_dict contains no tensor entries")

        serialized = MultiprocessingSerializer.serialize(named_tensors, output_str=True)
        tp_size = max(1, int(getattr(self._server_args, "tp_size", 1) or 1))
        self._send_scheduler_request(
            self._runtime["UpdateWeightsFromTensorReqInput"](
                serialized_named_tensors=[serialized] * tp_size,
                target_modules=list(self._target_modules),
                load_format="direct",
                flush_cache=True,
            ),
            operation="update_weights",
        )

        logger.info(
            "SGLang weights updated from in-memory tensors (target_modules=%s, tensors=%d)",
            self._target_modules,
            len(named_tensors),
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Initialize LoRA adapter state on SGLang from in-memory tensors."""
        require(self._is_initialized, "SGLang engine is not initialized")
        from sglang.multimodal_gen.runtime.entrypoints.utils import SetLoraFromTensorsReq

        request = SetLoraFromTensorsReq(
            lora_nickname=str(adapter_name),
            lora_tensors=lora_tensors,
            target="all",
            strength=1.0,
        )
        response = self._runtime["sync_scheduler_client"].forward(request)
        error = getattr(response, "error", None)
        require(error is None, f"set_lora_from_tensors failed: {error}")
        logger.info("SGLang LoRA initialized from tensors (adapter=%s)", adapter_name)

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        require(self._is_initialized, "SGLang engine is not initialized")
        require(bool(checkpoint_path), "checkpoint_path must be non-empty")

        self._last_weight_checksum = {}

        self._send_scheduler_request(
            self._runtime["UpdateWeightFromDiskReqInput"](
                model_path=checkpoint_path,
                flush_cache=True,
                target_modules=list(self._target_modules),
            ),
            operation="update_weights_from_path",
        )

        if self._verify_weight_checksum:
            checksum_resp = self._runtime["sync_scheduler_client"].forward(
                self._runtime["GetWeightsChecksumReqInput"](
                    module_names=list(self._target_modules),
                )
            )
            checksum_output = getattr(checksum_resp, "output", None)
            require(
                isinstance(checksum_output, dict) and bool(checksum_output),
                f"SGLang checksum query returned invalid payload after weight update: {checksum_output!r}",
            )
            normalized = {str(k): str(v) for k, v in checksum_output.items()}
            bad_values = {k: v for k, v in normalized.items() if not v or v in {"not_found", "error"}}
            require(
                not bad_values,
                f"SGLang checksum query reported invalid modules after weight update: {bad_values}",
            )
            self._last_weight_checksum = normalized
            logger.info("SGLang weight checksum: %s", self._last_weight_checksum)

        logger.info(
            "SGLang weights updated from %s (target_modules=%s)",
            checkpoint_path,
            self._target_modules,
        )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str | bytes],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        require(self._is_initialized, "SGLang engine is not initialized")
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

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        self._send_scheduler_request(
            self._runtime["DestroyWeightsUpdateGroupReqInput"](
                group_name=str(group_name),
            ),
            operation="destroy_weights_update_group",
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

    def get_last_weight_checksum(self) -> Dict[str, str]:
        """Return checksum snapshot from the latest successful path sync."""
        return dict(self._last_weight_checksum)

    # ------------------------------------------------------------------
    # Memory + lifecycle
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
    def supports_distributed(self) -> bool:
        return True

    @property
    def requires_external_service(self) -> bool:
        return not self._local_mode

    def health_check(self) -> bool:
        if not self._is_initialized or self._generator is None:
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
        self._is_initialized = False
