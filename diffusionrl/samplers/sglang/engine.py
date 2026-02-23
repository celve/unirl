"""SGLang-diffusion inference engine integration."""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import torch

from diffusionrl.samplers.log_prob import get_sigma_schedule
from diffusionrl.types import LogProbData, PromptEmbeddings, SamplerOutput

from ..engine import BaseInferenceEngine, EngineCapabilities, EngineConfig, register_engine

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
class SGLangInferenceEngine(BaseInferenceEngine):
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
        self._warned_sequential_requests: bool = False
        self._warned_missing_initial_noise: bool = False
        self._warned_missing_decoded: bool = False
        self._warned_ignored_external_embeddings: bool = False
        self._warned_logprob_shape: bool = False
        self._warned_noop_memory_api: bool = False
        self._warned_unsupported_rollout_sde: bool = False
        self._warned_trimmed_logprob_prefix: bool = False

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
            GetWeightsChecksumReqInput,
            UpdateWeightFromDiskReqInput,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        return {
            "DiffGenerator": DiffGenerator,
            "ServerArgs": ServerArgs,
            "UpdateWeightFromDiskReqInput": UpdateWeightFromDiskReqInput,
            "GetWeightsChecksumReqInput": GetWeightsChecksumReqInput,
            "sync_scheduler_client": sync_scheduler_client,
        }

    def _build_server_kwargs(self, server_args_cls: Any) -> Dict[str, Any]:
        raw = dict(self.config.engine_kwargs or {})
        self._local_mode = _to_bool(raw.get("local_mode", True), default=True)
        self._verify_weight_checksum = _to_bool(
            raw.get("verify_weight_checksum", True),
            default=True,
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
            "prompt_encoder_device",
            "prompt_encoder_dtype",
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
        encoder_device = str(engine_kwargs.get("prompt_encoder_device", "cpu"))
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

            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
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

    def _extract_trajectory_from_result(self, result: Any) -> torch.Tensor:
        traj = self._to_tensor(getattr(result, "trajectory_latents", None))
        if traj is None:
            raise RuntimeError("SGLang result missing trajectory_latents")

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
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> SamplerOutput:
        if not self._is_initialized or self._generator is None:
            raise RuntimeError("SGLang engine is not initialized")

        del latents  # SGLang runtime samples noise internally.

        if prompts is None or len(prompts) == 0:
            raise ValueError("SGLang engine requires non-empty prompts")
        if len(prompts) > 1 and not self._warned_sequential_requests:
            logger.warning(
                "SGLang DiffGenerator currently processes prompt batches sequentially. "
                "Large prompts_per_batch may reduce rollout throughput."
            )
            self._warned_sequential_requests = True

        if (
            prompt_embeds is not None
            or pooled_prompt_embeds is not None
            or encoder_attention_mask is not None
            or kwargs.get("negative_prompt_embeds") is not None
            or kwargs.get("negative_pooled_prompt_embeds") is not None
            or kwargs.get("text_ids") is not None
            or kwargs.get("image_ids") is not None
        ) and not self._warned_ignored_external_embeddings:
            logger.warning(
                "SGLang engine ignores external prompt embedding tensors and recomputes prompt embeddings "
                "from text prompts on every generate() call."
            )
            self._warned_ignored_external_embeddings = True

        steps = int(num_inference_steps or self.config.num_inference_steps)
        scale = float(guidance_scale if guidance_scale is not None else self.config.guidance_scale)
        out_h = int(height or self.config.height)
        out_w = int(width or self.config.width)
        out_f = int(num_frames or self.config.num_frames)

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
        model_type = self._infer_model_type()

        return_decoded_for_reward = bool(kwargs.pop("return_decoded_for_reward", False))
        negative_prompt = kwargs.pop("negative_prompt", None)
        fps = kwargs.pop("fps", None)
        num_outputs_per_prompt = kwargs.pop("num_outputs_per_prompt", None)
        rollout_enabled = bool(kwargs.pop("enable_rollout_logprob", sde_indices is not None))
        requested_rollout_sde = str(
            kwargs.pop("rollout_sde_type", getattr(self.config, "sde_type", "sde"))
        ).strip().lower()
        if requested_rollout_sde in {"sde", "flow", "flux_flow"}:
            rollout_sde_type = "sde"
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
            "prompt": list(prompts),
            "num_inference_steps": steps,
            "guidance_scale": scale,
            "height": out_h,
            "width": out_w,
            "num_frames": out_f,
            "save_output": False,
            "return_file_paths_only": False,
            "return_trajectory_latents": True,
            "return_trajectory_decoded": return_decoded_for_reward,
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

        raw_results = self._generator.generate(sampling_params_kwargs=sampling_params_kwargs)
        if raw_results is None:
            raise RuntimeError("SGLang generator returned no results")

        results: List[Any]
        if isinstance(raw_results, list):
            results = raw_results
        else:
            results = [raw_results]

        trajectories = [self._extract_trajectory_from_result(result) for result in results]
        if not trajectories:
            raise RuntimeError("SGLang result conversion produced no trajectories")
        trajectories_tensor = torch.cat(trajectories, dim=0)
        per_result_log_probs = [self._extract_log_probs_from_result(result) for result in results]

        traj_len = int(trajectories_tensor.shape[1])
        has_initial_noise = traj_len == steps + 1

        if has_initial_noise:
            timesteps = get_sigma_schedule(steps, shift=float(self.config.shift)).cpu()
            step_indices = torch.arange(timesteps.shape[0], dtype=torch.long)
        else:
            # Upstream currently returns [x_{T-1}, ..., x_0] without initial x_T.
            # Shift sigma labels by one step to keep timestep->trajectory alignment.
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
                f"SGLang timestep/trajectory length mismatch after conversion: "
                f"timesteps={timesteps.shape[0]}, trajectory_len={traj_len}"
            )

        merged_log_probs: Optional[LogProbData] = None
        if per_result_log_probs and all(lp is not None for lp in per_result_log_probs):
            log_prob_tensor = torch.cat([lp for lp in per_result_log_probs if lp is not None], dim=0)
            expected_steps = int(step_indices.shape[0]) - 1
            if (
                not has_initial_noise
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

        final_latents = trajectories_tensor[:, -1]

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

        metadata: Dict[str, Any] = {
            "generator_type": "sglang",
            "engine_capabilities": self.get_capabilities_dict(),
            "trajectory_format": (
                "packed_seq_c4"
                if model_type == "flux" and trajectories_tensor.dim() == 4
                else ("video_dense_latent" if trajectories_tensor.dim() == 6 else "dense_latent")
            ),
            "timestep_type": "sigma",
            "timestep_scale": 1.0,
            "sde_indices": sorted(int(i) for i in sde_indices) if sde_indices is not None else None,
            "has_initial_noise": has_initial_noise,
        }
        if decoded_video_tensors:
            metadata["decoded_videos"] = torch.stack(decoded_video_tensors, dim=0)
        if self._last_weight_checksum:
            metadata["weight_checksum"] = dict(self._last_weight_checksum)

        return SamplerOutput(
            latents=final_latents,
            timesteps=timesteps,
            trajectories=trajectories_tensor,
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
    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        del state_dict
        raise RuntimeError(
            "SGLang engine does not support direct state_dict push. "
            "Use checkpoint_path mode with update_weights_from_path()."
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

        output = getattr(response, "output", None)
        if not isinstance(output, dict):
            raise RuntimeError(f"Invalid SGLang weight update response: {response}")

        success = bool(output.get("success", False))
        message = str(output.get("message", ""))
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

    def get_last_weight_checksum(self) -> Dict[str, str]:
        """Return checksum snapshot from the latest successful path sync."""
        return dict(self._last_weight_checksum)

    def offload(self) -> None:
        if self._generator is not None and hasattr(self._generator, "release_memory_occupation"):
            try:
                response = self._generator.release_memory_occupation(tags=["weights"])
                if isinstance(response, dict) and not bool(response.get("success", True)):
                    raise RuntimeError(str(response.get("message", "unknown error")))
                self._is_offloaded = True
                logger.info("SGLang engine offloaded via release_memory_occupation().")
                return
            except Exception as exc:
                logger.warning(
                    "SGLang release_memory_occupation() failed; falling back to local no-op: %s",
                    exc,
                )

        # Fallback no-op when upstream memory handlers are unavailable.
        if not self._warned_noop_memory_api:
            logger.warning(
                "SGLang offload() is currently a local no-op: scheduler release/resume-memory handlers "
                "are not available in upstream sglang-diffusion."
            )
            self._warned_noop_memory_api = True
        self._is_offloaded = True

    def onload(self) -> None:
        if self._generator is not None and hasattr(self._generator, "resume_memory_occupation"):
            try:
                response = self._generator.resume_memory_occupation(tags=["weights"])
                if isinstance(response, dict) and not bool(response.get("success", True)):
                    raise RuntimeError(str(response.get("message", "unknown error")))
            except Exception as exc:
                logger.warning(
                    "SGLang resume_memory_occupation() failed; falling back to local mark-onload: %s",
                    exc,
                )
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
        default_sde_type = str(getattr(self.config, "sde_type", "sde")).strip().lower()
        supports_native_logprob = default_sde_type in {"sde", "cps", "flow", "flux_flow"}
        return EngineCapabilities(
            supports_logprob=supports_native_logprob,
            supports_trajectory=True,
            supports_prompt_embeddings=supports_prompt_encoding,
            supports_guidance_scale=True,
            supports_staged_onload=False,
            weight_sync_mode="checkpoint_path",
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
