"""SGLang-diffusion inference engine integration."""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from diffusionrl.samplers.log_prob import get_sigma_schedule
from diffusionrl.types import LogProbData, PromptEmbeddings, RolloutOutput, RolloutRequest

from ..engine import (
    BaseRolloutEngine,
    DistributedWeightSyncCapable,
    EngineCapabilities,
    EngineConfig,
    register_engine,
)

logger = logging.getLogger(__name__)

_SUPPORTED_PROMPT_ENCODER_MODEL_TYPES = {"hunyuan", "flux", "mochi", "sd3"}


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@register_engine("sglang")
class SGLangRolloutEngine(BaseRolloutEngine, DistributedWeightSyncCapable):
    """Inference engine backed by `sglang.multimodal_gen` DiffGenerator."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self._device: Optional[torch.device] = None
        self._generator: Any = None
        self._server_args: Any = None
        self._local_mode: bool = True
        self._target_modules: List[str] = ["transformer"]
        self._verify_weight_checksum: bool = True
        self._last_weight_checksum: Dict[str, str] = {}
        self._supports_prompt_encoding: bool = False
        self._prompt_encoder: Any = None
        self._encode_prompt_in_generate: bool = False
        self._supports_memory_api: bool = False
        self._require_memory_api: bool = False
        self._warned_missing_initial_noise: bool = False
        self._warned_missing_decoded: bool = False
        self._warned_ignored_external_embeddings: bool = False
        self._warned_logprob_shape: bool = False
        self._warned_unsupported_rollout_sde: bool = False
        self._warned_trimmed_logprob_prefix: bool = False
        self._warned_disabled_native_rollout: bool = False
        self._warned_missing_trajectory_with_optional_mode: bool = False
        self._warned_latent_encode_fallback: bool = False
        self._fallback_vae: Any = None
        self._fallback_vae_model_type: Optional[str] = None

    @classmethod
    def declared_capabilities(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": True,
            "requires_log_prob": False,
            "requires_embeddings": True,
        }

    # ---------------------------------------------------------------------
    # Import/runtime helpers
    # ---------------------------------------------------------------------
    def _candidate_sglang_python_paths(self) -> List[Path]:
        candidates: List[Path] = []

        env_path = os.getenv("SGLANG_PYTHON_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        # File path: .../diffusionRL/diffusionrl/samplers/sglang/engine.py
        # Workspace sibling path is usually ../sglang/python from diffusionRL repo root.
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root.parent / "sglang" / "python")

        # Also probe current working directory siblings.
        cwd = Path.cwd()
        candidates.append(cwd / "sglang" / "python")
        candidates.append(cwd.parent / "sglang" / "python")

        dedup: List[Path] = []
        seen: Set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(path)
        return dedup

    def _ensure_sglang_importable(self) -> None:
        last_error: Optional[BaseException] = None
        try:
            import sglang.multimodal_gen  # noqa: F401
            return
        except ModuleNotFoundError as exc:
            last_error = exc

        for candidate in self._candidate_sglang_python_paths():
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                import sglang.multimodal_gen  # noqa: F401
                logger.info("Added local sglang python path: %s", candidate_str)
                return
            except ModuleNotFoundError as exc:
                last_error = exc
                continue

        raise ModuleNotFoundError(
            "Cannot import sglang.multimodal_gen. "
            "Set SGLANG_PYTHON_PATH to your local sglang/python directory. "
            f"Last import error: {last_error!r}"
        )

    def _import_sglang_runtime(self) -> Dict[str, Any]:
        self._ensure_sglang_importable()

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

    def _has_memory_handler(self, method_name: str) -> bool:
        method = getattr(self._generator, method_name, None)
        return callable(method)

    def _validate_memory_api_contract(self) -> None:
        has_release = self._has_memory_handler("release_memory_occupation")
        has_resume = self._has_memory_handler("resume_memory_occupation")
        self._supports_memory_api = bool(has_release and has_resume)

        logger.info(
            "SGLang memory API contract: supports=%s require=%s release=%s resume=%s",
            self._supports_memory_api,
            self._require_memory_api,
            has_release,
            has_resume,
        )

        if self._require_memory_api and not self._supports_memory_api:
            raise RuntimeError(
                "SGLang offload is required but generator does not expose "
                "release_memory_occupation/resume_memory_occupation."
            )

    def _call_memory_api(self, method_name: str, *, tags: Sequence[str]) -> Any:
        if self._generator is None:
            raise RuntimeError("SGLang generator is not initialized.")

        method = getattr(self._generator, method_name, None)
        if not callable(method):
            raise RuntimeError(
                f"SGLang generator missing required memory API: {method_name}."
            )

        response = method(tags=list(tags))
        if isinstance(response, dict) and not bool(response.get("success", True)):
            raise RuntimeError(
                str(response.get("message", f"{method_name} failed"))
            )
        return response

    def _sglang_logprob_mode(self) -> str:
        mode = str((self.config.engine_kwargs or {}).get("sglang_logprob_mode", "replay")).strip().lower()
        if mode not in {"replay", "native"}:
            return "replay"
        return mode

    def _native_rollout_logprob_enabled(self) -> bool:
        return self._sglang_logprob_mode() == "native"

    def _build_server_kwargs(self, server_args_cls: Any) -> Dict[str, Any]:
        raw = dict(self.config.engine_kwargs or {})
        self._local_mode = _to_bool(raw.get("local_mode", True), default=True)
        self._verify_weight_checksum = _to_bool(
            raw.get("verify_weight_checksum", True),
            default=True,
        )
        self._require_memory_api = _to_bool(
            raw.get("require_memory_api", False),
            default=False,
        )
        self._encode_prompt_in_generate = _to_bool(
            raw.get(
                "encode_prompt_in_generate",
                raw.get("sglang_encode_prompt_in_generate", False),
            ),
            default=False,
        )

        target_modules = raw.get("target_modules")
        if isinstance(target_modules, (list, tuple)) and target_modules:
            self._target_modules = [str(m) for m in target_modules]

        reserved_keys = {
            "sampler_path",
            "base_gpu_id",
            "force_set_cuda_visible_devices",
            "local_mode",
            "target_modules",
            "verify_weight_checksum",
            "require_memory_api",
            "prompt_encoder_device",
            "prompt_encoder_dtype",
            "encode_prompt_in_generate",
            "sglang_encode_prompt_in_generate",
            "server_kwargs",
        }

        allowed_keys = {f.name for f in dataclasses.fields(server_args_cls)}
        server_kwargs: Dict[str, Any] = {}

        user_server_kwargs = raw.get("server_kwargs")
        if isinstance(user_server_kwargs, dict):
            for key, value in user_server_kwargs.items():
                if key in allowed_keys:
                    server_kwargs[key] = value

        for key, value in raw.items():
            if key in reserved_keys:
                continue
            normalized = key[7:] if key.startswith("sglang_") else key
            if normalized in allowed_keys and normalized not in server_kwargs:
                server_kwargs[normalized] = value

        model_path = self.config.pretrained_model_saved_path or self.config.model_path
        if not model_path:
            raise ValueError("SGLang engine requires pretrained_model_saved_path/model_path")

        server_kwargs.setdefault("model_path", model_path)
        server_kwargs.setdefault("num_gpus", int(raw.get("num_gpus", 1)))

        if "tp_size" not in server_kwargs and raw.get("tp_size") is not None:
            server_kwargs["tp_size"] = int(raw["tp_size"])
        if "sp_degree" not in server_kwargs:
            sp_degree = raw.get("sp_degree", raw.get("sp_size"))
            if sp_degree is not None:
                server_kwargs["sp_degree"] = int(sp_degree)

        return server_kwargs

    # ---------------------------------------------------------------------
    # Engine lifecycle
    # ---------------------------------------------------------------------
    def initialize(self, device: torch.device) -> None:
        self._device = device

        runtime = self._import_sglang_runtime()
        server_kwargs = self._build_server_kwargs(runtime["ServerArgs"])

        logger.info(
            "Initializing SGLang-diffusion engine (local_mode=%s, target_modules=%s)",
            self._local_mode,
            self._target_modules,
        )
        server_args = runtime["ServerArgs"].from_kwargs(**server_kwargs)
        generator = runtime["DiffGenerator"].from_pretrained(
            server_args=server_args,
            local_mode=self._local_mode,
        )

        self._server_args = server_args
        self._generator = generator
        self._validate_memory_api_contract()
        self._is_initialized = True

    def _infer_model_type(self) -> str:
        value = (
            str(self.config.engine_kwargs.get("model_type", ""))
            or str(self.config.model_path or "")
            or str(self.config.pretrained_model_saved_path or "")
        ).lower()
        if "hunyuan" in value:
            return "hunyuan"
        if "flux" in value:
            return "flux"
        if "sd3" in value:
            return "sd3"
        if "mochi" in value:
            return "mochi"
        return "unknown"

    def _ensure_prompt_encoder(self) -> None:
        if self._prompt_encoder is not None:
            return

        model_type = self._infer_model_type()
        engine_kwargs = dict(self.config.engine_kwargs or {})
        encoder_device = str(engine_kwargs.get("prompt_encoder_device", "cuda"))
        dtype_name = str(engine_kwargs.get("prompt_encoder_dtype", "auto")).lower()
        if dtype_name == "fp16" or dtype_name == "float16":
            encoder_dtype = torch.float16
        elif dtype_name == "bf16" or dtype_name == "bfloat16":
            encoder_dtype = torch.bfloat16
        elif dtype_name == "fp32" or dtype_name == "float32":
            encoder_dtype = torch.float32
        else:
            encoder_dtype = torch.bfloat16 if encoder_device.startswith("cuda") else torch.float32

        model_path = self.config.pretrained_model_saved_path or self.config.model_path

        if model_type == "hunyuan":
            from diffusionrl.models.hunyuan import HunyuanTextEncoderWrapper

            self._prompt_encoder = HunyuanTextEncoderWrapper(
                pretrained_path=model_path,
                device=encoder_device,
                dtype=encoder_dtype,
            )
        elif model_type == "flux":
            from diffusionrl.models.flux import FluxTextEncoderWrapper
            from transformers import (
                CLIPTextModel,
                CLIPTokenizer,
                T5EncoderModel,
                T5TokenizerFast,
            )

            clip_encoder = CLIPTextModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                torch_dtype=encoder_dtype,
            ).to(encoder_device)
            clip_encoder.eval()
            t5_encoder = T5EncoderModel.from_pretrained(
                model_path,
                subfolder="text_encoder_2",
                torch_dtype=encoder_dtype,
            ).to(encoder_device)
            t5_encoder.eval()
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
            )
            t5_tokenizer = T5TokenizerFast.from_pretrained(
                model_path,
                subfolder="tokenizer_2",
            )

            self._prompt_encoder = FluxTextEncoderWrapper(
                clip_encoder=clip_encoder,
                clip_tokenizer=clip_tokenizer,
                t5_encoder=t5_encoder,
                t5_tokenizer=t5_tokenizer,
                device=encoder_device,
                dtype=encoder_dtype,
            )
        elif model_type == "mochi":
            from diffusionrl.models.mochi import MochiTextEncoderWrapper
            from transformers import T5EncoderModel, T5TokenizerFast

            t5_encoder = T5EncoderModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                torch_dtype=encoder_dtype,
            ).to(encoder_device)
            t5_encoder.eval()
            tokenizer = T5TokenizerFast.from_pretrained(
                model_path,
                subfolder="tokenizer",
            )
            self._prompt_encoder = MochiTextEncoderWrapper(
                encoder=t5_encoder,
                tokenizer=tokenizer,
                device=encoder_device,
                dtype=encoder_dtype,
                max_length=int(engine_kwargs.get("prompt_encoder_max_length", 256)),
            )
        elif model_type == "sd3":
            from transformers import (
                CLIPTextModelWithProjection,
                CLIPTokenizer,
                T5EncoderModel,
                T5TokenizerFast,
            )

            class _SD3PromptEncoder:
                def __init__(
                    self,
                    *,
                    pretrained_path: str,
                    device: str,
                    dtype: torch.dtype,
                    max_sequence_length: int,
                ) -> None:
                    self.device = device
                    self.dtype = dtype
                    self.max_sequence_length = max_sequence_length

                    self.tokenizer_1 = CLIPTokenizer.from_pretrained(
                        pretrained_path, subfolder="tokenizer"
                    )
                    self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                        pretrained_path, subfolder="tokenizer_2"
                    )
                    self.tokenizer_3 = T5TokenizerFast.from_pretrained(
                        pretrained_path, subfolder="tokenizer_3"
                    )

                    self.text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
                        pretrained_path,
                        subfolder="text_encoder",
                        torch_dtype=dtype,
                    ).to(device)
                    self.text_encoder_1.eval()
                    self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                        pretrained_path,
                        subfolder="text_encoder_2",
                        torch_dtype=dtype,
                    ).to(device)
                    self.text_encoder_2.eval()
                    self.text_encoder_3 = T5EncoderModel.from_pretrained(
                        pretrained_path,
                        subfolder="text_encoder_3",
                        torch_dtype=dtype,
                    ).to(device)
                    self.text_encoder_3.eval()

                @torch.no_grad()
                def encode_prompt(self, prompts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
                    text_inputs_1 = self.tokenizer_1(
                        prompts,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs_2 = self.tokenizer_2(
                        prompts,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs_3 = self.tokenizer_3(
                        prompts,
                        padding="max_length",
                        max_length=self.max_sequence_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    clip_out_1 = self.text_encoder_1(
                        text_inputs_1.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    clip_out_2 = self.text_encoder_2(
                        text_inputs_2.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    t5_out = self.text_encoder_3(
                        text_inputs_3.input_ids.to(self.device),
                    )

                    pooled = torch.cat(
                        [clip_out_1.text_embeds, clip_out_2.text_embeds],
                        dim=-1,
                    ).to(dtype=self.dtype)
                    prompt_embeds = t5_out.last_hidden_state.to(dtype=self.dtype)
                    return prompt_embeds, pooled

            self._prompt_encoder = _SD3PromptEncoder(
                pretrained_path=model_path,
                device=encoder_device,
                dtype=encoder_dtype,
                max_sequence_length=int(engine_kwargs.get("prompt_encoder_max_length", 256)),
            )
        else:
            raise NotImplementedError(
                "SGLang prompt-only rollout input mode requires a built-in prompt encoder. "
                f"Unsupported model_type={model_type!r}. "
                f"Supported model types: {sorted(_SUPPORTED_PROMPT_ENCODER_MODEL_TYPES)}."
            )

        self._supports_prompt_encoding = True
        logger.info(
            "Initialized SGLang prompt encoder (model_type=%s, device=%s, dtype=%s)",
            model_type,
            encoder_device,
            encoder_dtype,
        )

    # ---------------------------------------------------------------------
    # Data conversion helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_tensor(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        try:
            import numpy as np
            from PIL import Image

            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            if isinstance(value, Image.Image):
                return torch.from_numpy(np.array(value))
            if isinstance(value, (list, tuple)) and value:
                if all(torch.is_tensor(v) for v in value):
                    return torch.stack([v.detach() for v in value], dim=0)
                if all(isinstance(v, np.ndarray) for v in value):
                    return torch.from_numpy(np.stack(value, axis=0))
                if all(isinstance(v, Image.Image) for v in value):
                    return torch.from_numpy(np.stack([np.array(v) for v in value], axis=0))
        except Exception:
            pass
        return None

    @staticmethod
    def _slice_result_tensor(value: torch.Tensor, idx: int) -> torch.Tensor:
        if value.dim() >= 5:
            # [B,C,T,H,W] or [B,T,C,H,W]
            if value.shape[0] > 1:
                return value[idx : idx + 1]
            return value
        if value.dim() >= 4:
            # [B,T,C,H,W] for trajectories is handled upstream.
            return value
        return value

    @staticmethod
    def _describe_result_fields(result: Any) -> str:
        keys = (
            "trajectory_latents",
            "trajectory",
            "trajectory_timesteps",
            "trajectory_log_probs",
            "latents",
            "sample_latents",
            "samples",
            "frames",
            "output_file_path",
        )
        parts: List[str] = []
        for key in keys:
            if not hasattr(result, key):
                continue
            value = getattr(result, key)
            if value is None:
                parts.append(f"{key}=None")
                continue
            if torch.is_tensor(value):
                parts.append(f"{key}=Tensor{tuple(value.shape)}")
                continue
            shape = getattr(value, "shape", None)
            if shape is not None:
                try:
                    parts.append(f"{key}={type(value).__name__}{tuple(shape)}")
                except Exception:
                    parts.append(f"{key}={type(value).__name__}")
            else:
                parts.append(f"{key}={type(value).__name__}")
        return ", ".join(parts) if parts else "no-known-fields"

    def _extract_trajectory_from_result(
        self,
        result: Any,
        *,
        required: bool = True,
    ) -> Optional[torch.Tensor]:
        traj = self._to_tensor(getattr(result, "trajectory_latents", None))
        if traj is None:
            traj = self._to_tensor(getattr(result, "trajectory", None))
        if traj is None:
            if required:
                raise RuntimeError(
                    "SGLang result missing trajectory_latents/trajectory. "
                    f"Available fields: {self._describe_result_fields(result)}"
                )
            if not self._warned_missing_trajectory_with_optional_mode:
                logger.warning(
                    "SGLang result is missing trajectory_latents in optional-trajectory mode. "
                    "Will try latent fallback from sample outputs."
                )
                self._warned_missing_trajectory_with_optional_mode = True
            return None

        model_type = self._infer_model_type()
        if model_type == "flux":
            # FLUX rollout trajectory is packed latent tokens:
            # - [T, S, D] for single sample
            # - [B, T, S, D] for batched samples
            if traj.dim() == 3:
                traj = traj.unsqueeze(0)
            elif traj.dim() == 4:
                if traj.shape[0] > 1:
                    idx = int(getattr(result, "prompt_index", 0))
                    idx = max(0, min(idx, traj.shape[0] - 1))
                    traj = traj[idx : idx + 1]
            else:
                raise RuntimeError(
                    f"Unexpected FLUX trajectory shape from SGLang: {tuple(traj.shape)}"
                )
            return traj.detach().cpu()

        if traj.dim() == 4:
            # [T,C,H,W]
            traj = traj.unsqueeze(0)
        elif traj.dim() in (5, 6):
            # [B,T,C,H,W] or [B,T,C,F,H,W]
            if traj.shape[0] > 1:
                idx = int(getattr(result, "prompt_index", 0))
                idx = max(0, min(idx, traj.shape[0] - 1))
                traj = traj[idx : idx + 1]
        else:
            raise RuntimeError(
                f"Unexpected trajectory shape from SGLang: {tuple(traj.shape)}"
            )

        return traj.detach().cpu()

    @staticmethod
    def _extract_sample_payload(result: Any) -> Any:
        sample = getattr(result, "samples", None)
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            sample = sample[0]
        return sample

    @staticmethod
    def _to_nchw_image_tensor(sample_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        tensor = sample_tensor.detach().cpu()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.dim() == 4 and tensor.shape[0] > 1 and tensor.shape[-1] in (1, 3, 4):
            # [N,H,W,C] -> pick first output for this prompt
            tensor = tensor[0]

        if tensor.dim() != 3:
            return None

        if tensor.shape[0] in (1, 3, 4):
            chw = tensor[:3]
        elif tensor.shape[-1] in (1, 3, 4):
            chw = tensor.permute(2, 0, 1)[:3]
        else:
            return None

        chw = chw.float()
        if chw.numel() == 0:
            return None
        max_val = float(chw.max().item())
        min_val = float(chw.min().item())
        if max_val > 1.5:
            chw = chw / 255.0
        if min_val >= 0.0:
            chw = chw * 2.0 - 1.0
        chw = chw.clamp(-1.0, 1.0)
        return chw.unsqueeze(0)

    def _extract_pixel_batch_for_latent_fallback(
        self,
        results: Sequence[Any],
    ) -> torch.Tensor:
        pixel_batches: List[torch.Tensor] = []
        for idx, result in enumerate(results):
            sample = self._extract_sample_payload(result)
            sample_tensor = self._to_tensor(sample)
            if sample_tensor is None:
                raise RuntimeError(
                    "SGLang trajectory fallback failed: cannot convert result.samples to tensor "
                    f"(idx={idx}). Fields: {self._describe_result_fields(result)}"
                )
            image = self._to_nchw_image_tensor(sample_tensor)
            if image is None:
                raise RuntimeError(
                    "SGLang trajectory fallback failed: result.samples is not an image tensor "
                    f"(idx={idx}, shape={tuple(sample_tensor.shape)})."
                )
            pixel_batches.append(image)
        return torch.cat(pixel_batches, dim=0)

    def _ensure_fallback_vae(self, *, model_type: str) -> None:
        if self._fallback_vae is not None and self._fallback_vae_model_type == model_type:
            return
        if model_type not in {"sd3", "flux"}:
            raise RuntimeError(
                f"SGLang latent fallback currently supports sd3/flux, got model_type={model_type}."
            )

        model_path = self.config.pretrained_model_saved_path or self.config.model_path
        if not model_path:
            raise RuntimeError("Missing model_path while initializing latent fallback VAE.")

        from diffusers import AutoencoderKL

        device = self._device or torch.device("cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=dtype,
        )
        vae = vae.to(device=device, dtype=dtype)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)

        self._fallback_vae = vae
        self._fallback_vae_model_type = model_type

    def _encode_pixels_to_latents(
        self,
        pixels: torch.Tensor,
        *,
        model_type: str,
    ) -> torch.Tensor:
        self._ensure_fallback_vae(model_type=model_type)
        if self._fallback_vae is None:
            raise RuntimeError("Fallback VAE was not initialized.")

        vae = self._fallback_vae
        device = next(vae.parameters()).device
        dtype = next(vae.parameters()).dtype
        pixel_values = pixels.to(device=device, dtype=dtype)

        with torch.no_grad():
            posterior = vae.encode(pixel_values).latent_dist
            latents = posterior.mode()
        scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
        latents = latents * scale
        return latents.detach().cpu()

    def _extract_final_latents_without_trajectory(
        self,
        *,
        results: Sequence[Any],
        model_type: str,
    ) -> torch.Tensor:
        # 1) Try direct latent-like fields first.
        direct_latents: List[torch.Tensor] = []
        for idx, result in enumerate(results):
            latent = None
            for key in ("latents", "sample_latents", "output_latents", "final_latents"):
                latent = self._to_tensor(getattr(result, key, None))
                if latent is not None:
                    break
            if latent is None:
                direct_latents = []
                break

            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            elif latent.dim() >= 4 and latent.shape[0] > 1:
                prompt_idx = int(getattr(result, "prompt_index", 0))
                prompt_idx = max(0, min(prompt_idx, latent.shape[0] - 1))
                latent = latent[prompt_idx : prompt_idx + 1]
            direct_latents.append(latent.detach().cpu())

            if idx == len(results) - 1 and direct_latents:
                return torch.cat(direct_latents, dim=0)

        # 2) Fallback: encode generated pixel samples back to latent space.
        pixels = self._extract_pixel_batch_for_latent_fallback(results)
        if not self._warned_latent_encode_fallback:
            logger.warning(
                "SGLang result has no trajectory latents; encoding generated images with VAE "
                "to recover clean latents for forward-process training."
            )
            self._warned_latent_encode_fallback = True
        return self._encode_pixels_to_latents(pixels, model_type=model_type)

    def _derive_timestep_alignment(
        self,
        *,
        trajectories_tensor: torch.Tensor,
        num_inference_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        traj_len = int(trajectories_tensor.shape[1])
        has_initial_noise = traj_len == int(num_inference_steps) + 1
        if has_initial_noise:
            timesteps = get_sigma_schedule(
                int(num_inference_steps),
                shift=float(self.config.shift),
            ).cpu()
            step_indices = torch.arange(timesteps.shape[0], dtype=torch.long)
        else:
            full_sigmas = get_sigma_schedule(traj_len, shift=float(self.config.shift)).cpu()
            timesteps = full_sigmas[1:]
            step_indices = torch.arange(1, full_sigmas.shape[0], dtype=torch.long)
            if not self._warned_missing_initial_noise:
                logger.warning(
                    "SGLang trajectory is missing initial x_T noise (len=T instead of T+1). "
                    "DiffusionRL applies step-index offset compatibility mode."
                )
                self._warned_missing_initial_noise = True
        if int(timesteps.shape[0]) != traj_len:
            raise RuntimeError(
                "SGLang timestep/trajectory length mismatch after conversion: "
                f"timesteps={timesteps.shape[0]}, trajectory_len={traj_len}"
            )
        return timesteps, step_indices, has_initial_noise

    @staticmethod
    def _tensor_to_pil(frame: torch.Tensor) -> Any:
        from PIL import Image

        if frame.dim() != 3:
            raise ValueError(f"Expected CHW frame tensor, got shape={tuple(frame.shape)}")

        frame = frame.detach().float().cpu()
        if frame.max().item() > 1.0:
            frame = frame / 255.0
        frame = frame.clamp(0.0, 1.0)
        if frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)
        img = frame.permute(1, 2, 0).mul(255).byte().numpy()
        return Image.fromarray(img)

    def _extract_decoded_media(self, result: Any) -> tuple[Optional[Any], Optional[torch.Tensor]]:
        sample = getattr(result, "samples", None)
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            sample = sample[0]

        sample_tensor = self._to_tensor(sample)
        if sample_tensor is None:
            return None, None

        sample_tensor = sample_tensor.detach().cpu()

        # Possible formats:
        # - [C,H,W] image
        # - [C,T,H,W] video
        # - [T,H,W,C] video
        # - [H,W,C] image
        if sample_tensor.dim() == 3:
            if sample_tensor.shape[0] in (1, 3, 4):
                return self._tensor_to_pil(sample_tensor[:3]), None
            if sample_tensor.shape[-1] in (1, 3, 4):
                chw = sample_tensor.permute(2, 0, 1)
                return self._tensor_to_pil(chw[:3]), None
            return None, None

        if sample_tensor.dim() == 4:
            if sample_tensor.shape[0] in (1, 3, 4):
                # [C,T,H,W]
                mid = sample_tensor[:, sample_tensor.shape[1] // 2]
                return self._tensor_to_pil(mid[:3]), sample_tensor
            if sample_tensor.shape[1] in (1, 3, 4):
                # [T,C,H,W]
                mid = sample_tensor[sample_tensor.shape[0] // 2]
                video = sample_tensor.permute(1, 0, 2, 3)
                return self._tensor_to_pil(mid[:3]), video
            if sample_tensor.shape[-1] in (1, 3, 4):
                # [T,H,W,C]
                mid = sample_tensor[sample_tensor.shape[0] // 2]
                chw = mid.permute(2, 0, 1)
                video = sample_tensor.permute(3, 0, 1, 2)
                return self._tensor_to_pil(chw[:3]), video
            return None, None

        return None, None

    @staticmethod
    def _extract_log_probs_from_result(result: Any) -> Optional[torch.Tensor]:
        value = getattr(result, "trajectory_log_probs", None)
        if value is None:
            return None
        if torch.is_tensor(value):
            lp = value
        else:
            try:
                lp = torch.as_tensor(value)
            except Exception:
                return None
        if lp.dim() == 1:
            lp = lp.unsqueeze(0)
        elif lp.dim() != 2:
            return None
        if lp.shape[0] > 1:
            idx = int(getattr(result, "prompt_index", 0))
            idx = max(0, min(idx, lp.shape[0] - 1))
            lp = lp[idx : idx + 1]
        return lp.detach().cpu()

    @staticmethod
    def _align_embedding_tensor(name: str, value: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be torch.Tensor when provided, got {type(value).__name__}")
        if name == "image_ids":
            # FLUX image_ids are often shared and not batched.
            return value.detach().cpu()
        if name == "text_ids" and value.dim() == 2:
            return value.unsqueeze(0).expand(batch_size, -1, -1).detach().cpu()
        if value.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch size mismatch: tensor batch={value.shape[0]}, expected={batch_size}"
            )
        return value.detach().cpu()

    @staticmethod
    def _build_flux_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        if prompt_embeds.dim() < 3:
            raise ValueError(
                f"FLUX prompt_embeds must be [B, seq, hidden], got shape={tuple(prompt_embeds.shape)}"
            )
        batch_size = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])
        return torch.zeros(
            batch_size,
            seq_len,
            3,
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )

    @staticmethod
    def _build_flux_image_ids(
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        latent_h = max(1, int(height) // 8)
        latent_w = max(1, int(width) // 8)
        packed_h = max(1, latent_h // 2)
        packed_w = max(1, latent_w // 2)

        image_ids = torch.zeros(
            packed_h,
            packed_w,
            3,
            device=device,
            dtype=dtype,
        )
        image_ids[..., 1] = torch.arange(packed_h, device=device, dtype=dtype)[:, None]
        image_ids[..., 2] = torch.arange(packed_w, device=device, dtype=dtype)[None, :]
        return image_ids.reshape(packed_h * packed_w, 3)

    # ---------------------------------------------------------------------
    # Core inference API
    # ---------------------------------------------------------------------
    def generate(self, request: RolloutRequest) -> RolloutOutput:
        if not self._is_initialized or self._generator is None:
            raise RuntimeError("SGLang engine is not initialized")

        # Extract fields from request
        prompts = request.prompts
        prompt_embeds = request.prompt_embeds
        pooled_prompt_embeds = request.pooled_prompt_embeds
        encoder_attention_mask = request.encoder_attention_mask
        num_inference_steps = request.num_inference_steps
        guidance_scale = request.guidance_scale
        height = request.height
        width = request.width
        num_frames = request.num_frames
        seed = request.seed
        sde_indices = request.sde_indices
        kwargs = dict(request.kwargs)

        # SGLang runtime samples noise internally.

        if prompts is None or len(prompts) == 0:
            raise ValueError("SGLang engine requires non-empty prompts")

        has_external_embeddings = bool(
            prompt_embeds is not None
            or pooled_prompt_embeds is not None
            or encoder_attention_mask is not None
            or kwargs.get("negative_prompt_embeds") is not None
            or kwargs.get("negative_pooled_prompt_embeds") is not None
            or kwargs.get("text_ids") is not None
            or kwargs.get("image_ids") is not None
        )
        if has_external_embeddings and not self._warned_ignored_external_embeddings:
            if self._encode_prompt_in_generate:
                logger.warning(
                    "SGLang engine ignores external prompt embedding tensors and recomputes prompt embeddings "
                    "from text prompts on every generate() call."
                )
            else:
                logger.warning(
                    "SGLang engine ignores external prompt embedding tensors in generate(); "
                    "expect rollout-level fallback embedding attachment."
                )
            self._warned_ignored_external_embeddings = True

        steps = int(num_inference_steps or self.config.num_inference_steps)
        scale = float(guidance_scale if guidance_scale is not None else self.config.guidance_scale)
        out_h = int(height or self.config.height)
        out_w = int(width or self.config.width)
        out_f = int(num_frames or self.config.num_frames)

        if self._encode_prompt_in_generate:
            encoded = self.encode_prompt(
                list(prompts),
                height=out_h,
                width=out_w,
                num_frames=out_f,
            )
            prompt_embeds = encoded.get("prompt_embeds")
            if prompt_embeds is None:
                raise RuntimeError("SGLang encode_prompt() returned no prompt_embeds")
            pooled_prompt_embeds = encoded.get("pooled_prompt_embeds")
            encoder_attention_mask = encoded.get("encoder_attention_mask")
            negative_prompt_embeds = encoded.get("negative_prompt_embeds")
            negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")
            text_ids = encoded.get("text_ids")
            image_ids = encoded.get("image_ids")
        else:
            prompt_embeds = None
            pooled_prompt_embeds = None
            encoder_attention_mask = None
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            text_ids = None
            image_ids = None
        model_type = self._infer_model_type()

        require_trajectory = bool(request.return_trajectories)
        require_log_probs = bool(request.return_log_probs)
        return_decoded_for_reward = bool(
            request.decode_for_reward or kwargs.pop("return_decoded_for_reward", False)
        )
        negative_prompt = kwargs.pop("negative_prompt", None)
        fps = kwargs.pop("fps", None)
        num_outputs_per_prompt = kwargs.pop("num_outputs_per_prompt", None)
        default_rollout_enabled = bool(
            require_log_probs and sde_indices is not None and self._native_rollout_logprob_enabled()
        )
        rollout_enabled = bool(kwargs.pop("enable_rollout_logprob", default_rollout_enabled))
        if rollout_enabled and not self._native_rollout_logprob_enabled():
            if not self._warned_disabled_native_rollout:
                logger.warning(
                    "enable_rollout_logprob requested but sglang_logprob_mode=%r. "
                    "Disabling native rollout logprob and using replay path.",
                    self._sglang_logprob_mode(),
                )
                self._warned_disabled_native_rollout = True
            rollout_enabled = False
        requested_rollout_sde = str(
            kwargs.pop("rollout_sde_type", getattr(self.config, "sde_type", "sde"))
        ).strip().lower()
        if requested_rollout_sde in {"sde", "flow", "flux_flow"}:
            rollout_sde_type = "sde"
        elif requested_rollout_sde in {"dance", "flux_dance"}:
            rollout_sde_type = "dance"
        elif requested_rollout_sde == "cps":
            rollout_sde_type = "cps"
        else:
            rollout_sde_type = "sde"
            if rollout_enabled and not self._warned_unsupported_rollout_sde:
                logger.warning(
                    "SGLang native rollout logprob currently supports only sde/cps, got sde_type=%r. "
                    "Disabling native rollout logprob and falling back to replay path.",
                    requested_rollout_sde,
                )
                self._warned_unsupported_rollout_sde = True
            rollout_enabled = False
        rollout_noise_level = float(
            kwargs.pop(
                "rollout_noise_level",
                kwargs.pop("eta", getattr(self.config, "eta", 1.0)),
            )
        )

        sampling_params_kwargs: Dict[str, Any] = {
            "num_inference_steps": steps,
            "guidance_scale": scale,
            "height": out_h,
            "width": out_w,
            "num_frames": out_f,
            "save_output": False,
            "return_file_paths_only": False,
            "return_trajectory_latents": bool(require_trajectory or require_log_probs),
            # Keep rollout path latent-only; reward-side image decoding is handled
            # by diffusionrl fallback in RolloutActor.generate().
            "return_trajectory_decoded": False,
        }
        if seed is not None:
            sampling_params_kwargs["seed"] = int(seed)
        if negative_prompt is not None:
            sampling_params_kwargs["negative_prompt"] = negative_prompt
        if fps is not None:
            sampling_params_kwargs["fps"] = int(fps)
        if num_outputs_per_prompt is not None:
            sampling_params_kwargs["num_outputs_per_prompt"] = int(num_outputs_per_prompt)
        if rollout_enabled:
            sampling_params_kwargs["rollout"] = True
            sampling_params_kwargs["rollout_sde_type"] = rollout_sde_type
            sampling_params_kwargs["rollout_noise_level"] = rollout_noise_level

        # The local DiffGenerator API currently validates `prompt` as a single
        # string before it expands batched inputs internally. Dispatch one
        # request per prompt here to keep diffusionrl compatible with both the
        # local checkout and installed sglang variants.
        results: List[Any] = []
        for prompt in prompts:
            request_kwargs = dict(sampling_params_kwargs)
            request_kwargs["prompt"] = str(prompt)
            if seed is not None:
                request_kwargs["seed"] = int(seed)

            raw_results = self._generator.generate(sampling_params_kwargs=request_kwargs)
            if raw_results is None:
                raise RuntimeError("SGLang generator returned no results for prompt batch")
            if isinstance(raw_results, list):
                results.extend(list(raw_results))
            else:
                results.append(raw_results)

        trajectory_items: List[Optional[torch.Tensor]] = [
            self._extract_trajectory_from_result(result, required=require_trajectory)
            for result in results
        ]
        missing_trajectory = sum(item is None for item in trajectory_items)
        use_trajectory = missing_trajectory == 0 and len(trajectory_items) > 0
        if missing_trajectory and not require_trajectory and missing_trajectory != len(trajectory_items):
            logger.warning(
                "SGLang returned mixed trajectory availability (%s/%s missing); "
                "falling back to trajectory-free adapter path for batch consistency.",
                missing_trajectory,
                len(trajectory_items),
            )
            use_trajectory = False

        trajectories_tensor: Optional[torch.Tensor] = None
        has_initial_noise: Optional[bool] = None
        if use_trajectory:
            trajectories_tensor = torch.cat(
                [item for item in trajectory_items if item is not None],
                dim=0,
            )
            timesteps, step_indices, has_initial_noise = self._derive_timestep_alignment(
                trajectories_tensor=trajectories_tensor,
                num_inference_steps=steps,
            )
            final_latents = trajectories_tensor[:, -1]
        else:
            final_latents = self._extract_final_latents_without_trajectory(
                results=results,
                model_type=model_type,
            )
            timesteps = get_sigma_schedule(steps, shift=float(self.config.shift)).cpu()
            step_indices = torch.arange(timesteps.shape[0], dtype=torch.long)

        per_result_log_probs: List[Optional[torch.Tensor]] = []
        if require_log_probs:
            per_result_log_probs = [self._extract_log_probs_from_result(result) for result in results]

        merged_log_probs: Optional[LogProbData] = None
        if require_log_probs and per_result_log_probs and all(lp is not None for lp in per_result_log_probs):
            log_prob_tensor = torch.cat([lp for lp in per_result_log_probs if lp is not None], dim=0)
            expected_steps = int(step_indices.shape[0]) - 1
            if (
                trajectories_tensor is not None
                and has_initial_noise is False
                and int(log_prob_tensor.shape[1]) == expected_steps + 1
            ):
                # Upstream rollout log_probs may include the first transition
                # x_T -> x_{T-1}, while DiffusionRL compatibility mode with
                # missing x_T can only train on transitions between stored
                # trajectory states. Drop the prefix to align with step_indices[:-1].
                log_prob_tensor = log_prob_tensor[:, 1:]
                if not self._warned_trimmed_logprob_prefix:
                    logger.warning(
                        "SGLang trajectory_log_probs includes an extra prefix step while trajectory is missing x_T; "
                        "dropping the first logprob column for alignment."
                    )
                    self._warned_trimmed_logprob_prefix = True
            if int(log_prob_tensor.shape[1]) == expected_steps:
                lp_dict = {
                    int(step_indices[i].item()): log_prob_tensor[:, i]
                    for i in range(expected_steps)
                }
                merged_log_probs = LogProbData.from_dict(lp_dict)
            elif not getattr(self, "_warned_logprob_shape", False):
                logger.warning(
                    "SGLang trajectory_log_probs shape mismatch: got %s, expected second dim=%s. "
                    "Ignoring rollout log_probs and keeping replay path enabled.",
                    tuple(log_prob_tensor.shape),
                    expected_steps,
                )
                self._warned_logprob_shape = True

        embeddings = None
        if prompt_embeds is not None:
            batch_size = int(final_latents.shape[0])
            embeddings = PromptEmbeddings(
                prompt_embeds=self._align_embedding_tensor("prompt_embeds", prompt_embeds, batch_size),
                pooled_prompt_embeds=self._align_embedding_tensor(
                    "pooled_prompt_embeds", pooled_prompt_embeds, batch_size
                ),
                encoder_attention_mask=self._align_embedding_tensor(
                    "encoder_attention_mask", encoder_attention_mask, batch_size
                ),
                negative_prompt_embeds=self._align_embedding_tensor(
                    "negative_prompt_embeds", negative_prompt_embeds, batch_size
                ),
                negative_pooled_prompt_embeds=self._align_embedding_tensor(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds, batch_size
                ),
                text_ids=self._align_embedding_tensor("text_ids", text_ids, batch_size),
                image_ids=self._align_embedding_tensor("image_ids", image_ids, batch_size),
            )

        decoded_images = None
        decoded_video_tensors: List[torch.Tensor] = []
        if return_decoded_for_reward:
            decoded_images = []
            for result in results:
                pil_img, video_tensor = self._extract_decoded_media(result)
                if pil_img is not None:
                    decoded_images.append(pil_img)
                if video_tensor is not None:
                    decoded_video_tensors.append(video_tensor)
            if not decoded_images:
                decoded_images = None
            if (
                decoded_images is None
                and not decoded_video_tensors
                and not self._warned_missing_decoded
            ):
                logger.warning(
                    "SGLang generate(return_decoded_for_reward=True) returned no decodable media. "
                    "Reward stage will fall back to latent tensors."
                )
                self._warned_missing_decoded = True

        if model_type == "flux":
            if trajectories_tensor is not None and trajectories_tensor.dim() == 4:
                trajectory_format = "packed_seq_c4"
            elif final_latents.dim() == 3:
                trajectory_format = "packed_seq_c4"
            else:
                trajectory_format = "dense_latent"
        else:
            if trajectories_tensor is not None:
                trajectory_format = (
                    "video_dense_latent" if trajectories_tensor.dim() == 6 else "dense_latent"
                )
            else:
                trajectory_format = (
                    "video_dense_latent" if final_latents.dim() == 5 else "dense_latent"
                )

        metadata: Dict[str, Any] = {
            "generator_type": "sglang",
            "engine_capabilities": self.get_capabilities_dict(),
            "sglang_logprob_mode": self._sglang_logprob_mode(),
            "encode_prompt_in_generate": bool(self._encode_prompt_in_generate),
            "trajectory_format": trajectory_format,
            "timestep_type": "sigma",
            "timestep_scale": 1.0,
            "sde_indices": sorted(int(i) for i in sde_indices) if sde_indices is not None else None,
            "has_initial_noise": has_initial_noise,
            "trajectory_available": bool(trajectories_tensor is not None),
        }
        if decoded_video_tensors:
            metadata["decoded_videos"] = torch.stack(decoded_video_tensors, dim=0)
        if self._last_weight_checksum:
            metadata["weight_checksum"] = dict(self._last_weight_checksum)

        return RolloutOutput(
            latents=final_latents,
            timesteps=timesteps,
            trajectories=trajectories_tensor if require_trajectory else None,
            log_probs=merged_log_probs,
            embeddings=embeddings,
            decoded_images=decoded_images,
            metadata=metadata,
            step_indices=step_indices,
        )

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if prompts is None or len(prompts) == 0:
            raise ValueError("encode_prompt requires non-empty prompts")

        self._ensure_prompt_encoder()
        encoded = self._prompt_encoder.encode_prompt(list(prompts))
        if not isinstance(encoded, (tuple, list)) or len(encoded) < 2:
            raise RuntimeError(
                f"Unexpected prompt encoder output type: {type(encoded).__name__}"
            )

        prompt_embeds = encoded[0]
        secondary = encoded[1]
        model_type = self._infer_model_type()

        output: Dict[str, torch.Tensor] = {
            "prompt_embeds": prompt_embeds,
        }
        if model_type == "mochi":
            output["encoder_attention_mask"] = secondary
        else:
            output["pooled_prompt_embeds"] = secondary
        if model_type == "flux":
            height = int(kwargs.get("height", self.config.height))
            width = int(kwargs.get("width", self.config.width))
            output["text_ids"] = self._build_flux_text_ids(prompt_embeds)
            output["image_ids"] = self._build_flux_image_ids(
                height=height,
                width=width,
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
        return output

    # ---------------------------------------------------------------------
    # Weight sync / memory management
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_update_status(response: Any, *, operation: str) -> tuple[bool, str]:
        output = getattr(response, "output", None)
        if not isinstance(output, dict):
            raise RuntimeError(f"Invalid SGLang response for {operation}: {response}")
        success = bool(output.get("success", False))
        message = str(output.get("message", "Unknown status"))
        return success, message

    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if not self._is_initialized:
            raise RuntimeError("SGLang engine is not initialized")
        if not state_dict:
            raise ValueError("state_dict must be non-empty")

        runtime = self._import_sglang_runtime()
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        named_tensors = [
            (str(name), tensor.detach().contiguous())
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
        ]
        if not named_tensors:
            raise ValueError("state_dict contains no tensor entries")

        serialized = MultiprocessingSerializer.serialize(named_tensors, output_str=True)
        tp_size = max(1, int(getattr(self._server_args, "tp_size", 1) or 1))
        request = runtime["UpdateWeightsFromTensorReqInput"](
            serialized_named_tensors=[serialized] * tp_size,
            target_modules=list(self._target_modules),
            load_format="direct",
            flush_cache=True,
        )
        response = runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(
            response,
            operation="update_weights_from_tensor",
        )
        if not success:
            raise RuntimeError(f"SGLang tensor weight update failed: {message}")

        logger.info(
            "SGLang weights updated from in-memory tensors (target_modules=%s, tensors=%d)",
            self._target_modules,
            len(named_tensors),
        )

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        if not self._is_initialized:
            raise RuntimeError("SGLang engine is not initialized")
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be non-empty")

        self._last_weight_checksum = {}

        runtime = self._import_sglang_runtime()
        request = runtime["UpdateWeightFromDiskReqInput"](
            model_path=checkpoint_path,
            flush_cache=True,
            target_modules=list(self._target_modules),
        )
        response = runtime["sync_scheduler_client"].forward(request)

        success, message = self._extract_update_status(
            response,
            operation="update_weights_from_disk",
        )
        if not success:
            raise RuntimeError(f"SGLang weight update failed: {message}")

        if self._verify_weight_checksum:
            checksum_req = runtime["GetWeightsChecksumReqInput"](
                module_names=list(self._target_modules),
            )
            checksum_resp = runtime["sync_scheduler_client"].forward(checksum_req)
            checksum_output = getattr(checksum_resp, "output", None)
            if not isinstance(checksum_output, dict) or not checksum_output:
                raise RuntimeError(
                    "SGLang checksum query returned invalid payload after weight update: "
                    f"{checksum_output!r}"
                )
            normalized = {str(k): str(v) for k, v in checksum_output.items()}
            bad_values = {
                k: v
                for k, v in normalized.items()
                if not v
                or v in {"not_found", "error"}
            }
            if bad_values:
                raise RuntimeError(
                    "SGLang checksum query reported invalid modules after weight update: "
                    f"{bad_values}"
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
        if not self._is_initialized:
            raise RuntimeError("SGLang engine is not initialized")
        if not serialized_named_tensors:
            raise ValueError("serialized_named_tensors must be non-empty")

        runtime = self._import_sglang_runtime()
        request = runtime["UpdateWeightsFromTensorReqInput"](
            serialized_named_tensors=serialized_named_tensors,
            target_modules=list(target_modules or self._target_modules),
            load_format=load_format,
            flush_cache=flush_cache,
        )
        response = runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(
            response,
            operation="update_weights_from_tensor",
        )
        if not success:
            raise RuntimeError(f"SGLang tensor weight update failed: {message}")

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
        runtime = self._import_sglang_runtime()
        request = runtime["InitWeightsUpdateGroupReqInput"](
            master_address=master_address,
            master_port=int(master_port),
            rank_offset=int(rank_offset),
            world_size=int(world_size),
            group_name=str(group_name),
            backend=str(backend),
        )
        response = runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(
            response,
            operation="init_weights_update_group",
        )
        if not success:
            raise RuntimeError(f"init_weights_update_group failed: {message}")

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        runtime = self._import_sglang_runtime()
        request = runtime["DestroyWeightsUpdateGroupReqInput"](
            group_name=str(group_name),
        )
        response = runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(
            response,
            operation="destroy_weights_update_group",
        )
        if not success:
            raise RuntimeError(f"destroy_weights_update_group failed: {message}")

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
        if not names:
            raise ValueError("names must be non-empty for distributed update")
        runtime = self._import_sglang_runtime()
        request = runtime["UpdateWeightsFromDistributedReqInput"](
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            target_modules=list(target_modules or self._target_modules),
            flush_cache=flush_cache,
        )
        response = runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(
            response,
            operation="update_weights_from_distributed",
        )
        if not success:
            raise RuntimeError(f"SGLang distributed weight update failed: {message}")

    def get_last_weight_checksum(self) -> Dict[str, str]:
        """Return checksum snapshot from the latest successful path sync."""
        return dict(self._last_weight_checksum)

    def sleep(self) -> None:
        self._call_memory_api("release_memory_occupation", tags=["weights"])
        self._is_offloaded = True
        logger.info("SGLang engine entered sleep state via release_memory_occupation().")

    def wake_up(self) -> None:
        if not self._is_offloaded:
            return

        self._call_memory_api("resume_memory_occupation", tags=["weights"])
        self._is_offloaded = False

    # ---------------------------------------------------------------------
    # Capability and lifecycle
    # ---------------------------------------------------------------------
    @property
    def supports_distributed(self) -> bool:
        return True

    @property
    def requires_external_service(self) -> bool:
        return not self._local_mode

    def get_capabilities(self) -> EngineCapabilities:
        supports_prompt_encoding = bool(
            self._supports_prompt_encoding
            or self._infer_model_type() in _SUPPORTED_PROMPT_ENCODER_MODEL_TYPES
        )
        supports_native_logprob = self._native_rollout_logprob_enabled()
        return EngineCapabilities(
            supports_logprob=supports_native_logprob,
            supports_trajectory=True,
            supports_prompt_embeddings=supports_prompt_encoding,
            supports_guidance_scale=True,
            weight_load_mode="state_dict",
        )

    def health_check(self) -> bool:
        if not self._is_initialized or self._generator is None:
            return False
        try:
            runtime = self._import_sglang_runtime()
            return bool(runtime["sync_scheduler_client"].ping())
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
