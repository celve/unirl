"""SGLang-diffusion inference engine integration."""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from diffusionrl.samplers.noise_utils import MAX_TORCH_SEED, generate_latents
from diffusionrl.samplers.registry import register_rollout_engine
from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.forward_context import ForwardContext, get_forward_context_cls
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import LogProbData, RolloutSamples
from diffusionrl.types.trajectory_store import (
    Trajectory,
    compute_trajectory_positions,
)
from diffusionrl.utils.media import tensor_frame_to_pil

from ..engine import BaseRolloutEngine

logger = logging.getLogger(__name__)


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _deexpand_prompts(
    prompts: List[str],
    num_samples_per_prompt: int,
) -> Tuple[List[str], int]:
    """Collapse prompt-major repeated prompts back to unique prompts when possible."""
    k = int(num_samples_per_prompt)
    if k <= 1:
        return list(prompts), 1

    n = len(prompts)
    if n == 0 or n % k != 0:
        return list(prompts), 1

    num_unique = n // k
    unique_prompts: List[str] = []
    for i in range(num_unique):
        group_start = i * k
        base = prompts[group_start]
        for j in range(1, k):
            if prompts[group_start + j] != base:
                return list(prompts), 1
        unique_prompts.append(base)

    return unique_prompts, k


@dataclasses.dataclass
class _GenerateContext:
    """Derived parameters passed from request building to result parsing."""

    model_type: str
    steps: int
    guidance_scale: float
    height: int
    width: int
    sde_indices: Optional[Set[int]]
    require_trajectory: bool
    require_log_probs: bool
    return_decoded_for_reward: bool


@register_rollout_engine(component_name="sglang", component_cfg=EngineConfig)
class SGLangRolloutEngine(BaseRolloutEngine):
    """Inference engine backed by `sglang.multimodal_gen` DiffGenerator."""

    @classmethod
    def declared_capabilities(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self._device: Optional[torch.device] = None
        self._generator: Any = None
        self._server_args: Any = None
        self._local_mode: bool = True
        self._target_modules: List[str] = ["transformer"]
        self._verify_weight_checksum: bool = True
        self._last_weight_checksum: Dict[str, str] = {}
        self._supports_memory_api: bool = False
        self._require_memory_api: bool = False
        self._warned_missing_decoded: bool = False
        self._warned_logprob_shape: bool = False
        self._warned_unsupported_rollout_sde: bool = False
        self._warned_missing_trajectory_with_optional_mode: bool = False
        self._warned_latent_encode_fallback: bool = False
        self._informed_sglang_initial_noise_policy: bool = False
        self._fallback_vae: Any = None
        self._fallback_vae_model_type: Optional[str] = None
        self._cached_runtime: Optional[Dict[str, Any]] = None

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
            importlib.import_module("sglang.multimodal_gen")
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
                importlib.import_module("sglang.multimodal_gen")
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

    @property
    def _runtime(self) -> Dict[str, Any]:
        if self._cached_runtime is None:
            self._cached_runtime = self._import_sglang_runtime()
        return self._cached_runtime

    def _send_scheduler_request(self, request: Any, *, operation: str) -> Any:
        """Forward a request to SGLang scheduler and raise on failure."""
        response = self._runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(response, operation=operation)
        if not success:
            raise RuntimeError(f"{operation} failed: {message}")
        return response

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

    def _call_memory_api(
        self,
        method_name: str,
        *,
        tags: Sequence[str],
        cpu_backup_tags: Optional[Sequence[str]] = None,
    ) -> Any:
        if self._generator is None:
            raise RuntimeError("SGLang generator is not initialized.")

        method = getattr(self._generator, method_name, None)
        if not callable(method):
            raise RuntimeError(f"SGLang generator missing required memory API: {method_name}.")

        kwargs: Dict[str, Any] = {"tags": list(tags)}
        if cpu_backup_tags is not None:
            kwargs["cpu_backup_tags"] = list(cpu_backup_tags)
        try:
            response = method(**kwargs)
        except TypeError:
            if cpu_backup_tags is None:
                raise
            response = method(tags=list(tags))
        if isinstance(response, dict) and not bool(response.get("success", True)):
            raise RuntimeError(str(response.get("message", f"{method_name} failed")))
        return response

    def _logprob_source(self) -> str:
        return self.config.logprob_source

    def _native_rollout_logprob_enabled(self) -> bool:
        return self._logprob_source() == "native"

    def _build_server_kwargs(self, server_args_cls: Any) -> Dict[str, Any]:
        """Build ServerArgs kwargs from typed EngineConfig fields + escape hatch."""
        self._local_mode = bool(self.config.local_mode)
        self._verify_weight_checksum = bool(self.config.verify_weight_checksum)
        self._require_memory_api = bool(self.config.require_memory_api)

        target_modules = self.config.target_modules
        if target_modules:
            self._target_modules = [str(m) for m in target_modules]

        model_path = self.config.pretrained_model_ckpt_path or self.config.model_dotpath
        if not model_path:
            raise ValueError("SGLang engine requires pretrained_model_ckpt_path or model_dotpath")

        return self.config.build_server_kwargs(server_args_cls)

    # ---------------------------------------------------------------------
    # Engine lifecycle
    # ---------------------------------------------------------------------
    def initialize(self, device: torch.device) -> None:
        self._device = device

        server_kwargs = self._build_server_kwargs(self._runtime["ServerArgs"])

        # Cross-process LoRA contract check.  Training-side PEFT and rollout-side
        # SGLang MUST agree on the target-module set; otherwise SGLang defaults
        # to wrapping every linear layer and emits a wall of ``LoRA adapter
        # None does not contain the weights for layer '...'`` warnings.
        if self.config.use_lora and not server_kwargs.get("lora_target_modules"):
            logger.warning(
                "SGLang LoRA enabled but lora_target_modules is not provided. "
                "SGLang will wrap EVERY linear layer in the transformer while "
                "the training side only ships LoRA weights for a subset, "
                "leading to 'LoRA adapter None does not contain the weights "
                "for layer ...' warnings and silently disabled LoRA on those "
                "layers. Override %s.default_lora_target_modules() on the "
                "model bundle or pass --training.lora-target-modules.",
                type(self).__name__,
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
        self._validate_memory_api_contract()
        self._is_initialized = True

    def _infer_model_type(self) -> str:
        value = (
            str(self.config.engine_kwargs.get("model_type", ""))
            or str(self.config.model_dotpath or "")
            or str(self.config.pretrained_model_ckpt_path or "")
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
                raise RuntimeError(f"Unexpected FLUX trajectory shape from SGLang: {tuple(traj.shape)}")
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
            raise RuntimeError(f"Unexpected trajectory shape from SGLang: {tuple(traj.shape)}")

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
            raise RuntimeError(f"SGLang latent fallback currently supports sd3/flux, got model_type={model_type}.")

        model_path = self.config.pretrained_model_ckpt_path or self.config.model_dotpath
        if not model_path:
            raise RuntimeError(
                "Missing pretrained_model_ckpt_path or model_dotpath while initializing latent fallback VAE."
            )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_len = int(trajectories_tensor.shape[1])
        if traj_len != int(num_inference_steps) + 1:
            raise ValueError(
                f"SGLang trajectory missing initial x_T noise: traj_len={traj_len}, "
                f"expected num_inference_steps + 1 = {num_inference_steps + 1}. "
                f"Modern SGLang prepends initial latents at "
                f"sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1036-1037, "
                f"so traj_len should always equal T+1. Upgrade SGLang or fix the sampler "
                f"to emit a T+1 trajectory."
            )
        timesteps = get_sigma_schedule(
            int(num_inference_steps),
            shift=float(self.config.shift),
        ).cpu()
        step_indices = torch.arange(timesteps.shape[0], dtype=torch.long)
        if int(timesteps.shape[0]) != traj_len:
            raise RuntimeError(
                "SGLang timestep/trajectory length mismatch after conversion: "
                f"timesteps={timesteps.shape[0]}, trajectory_len={traj_len}"
            )
        return timesteps, step_indices

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
                return tensor_frame_to_pil(sample_tensor[:3]), None
            if sample_tensor.shape[-1] in (1, 3, 4):
                chw = sample_tensor.permute(2, 0, 1)
                return tensor_frame_to_pil(chw[:3]), None
            return None, None

        if sample_tensor.dim() == 4:
            if sample_tensor.shape[0] in (1, 3, 4):
                # [C,T,H,W]
                mid = sample_tensor[:, sample_tensor.shape[1] // 2]
                return tensor_frame_to_pil(mid[:3]), sample_tensor
            if sample_tensor.shape[1] in (1, 3, 4):
                # [T,C,H,W]
                mid = sample_tensor[sample_tensor.shape[0] // 2]
                video = sample_tensor.permute(1, 0, 2, 3)
                return tensor_frame_to_pil(mid[:3]), video
            if sample_tensor.shape[-1] in (1, 3, 4):
                # [T,H,W,C]
                mid = sample_tensor[sample_tensor.shape[0] // 2]
                chw = mid.permute(2, 0, 1)
                video = sample_tensor.permute(3, 0, 1, 2)
                return tensor_frame_to_pil(chw[:3]), video
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
            raise ValueError(f"{name} batch size mismatch: tensor batch={value.shape[0]}, expected={batch_size}")
        return value.detach().cpu()

    @staticmethod
    def _build_flux_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        if prompt_embeds.dim() < 3:
            raise ValueError(f"FLUX prompt_embeds must be [B, seq, hidden], got shape={tuple(prompt_embeds.shape)}")
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
    # Embedding extraction from sglang results
    # ---------------------------------------------------------------------
    @staticmethod
    def _unwrap_embed_field(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            tensors = [item for item in value if torch.is_tensor(item)]
            if not tensors:
                return None
            if len(tensors) == 1:
                value = tensors[0]
            elif tensors[0].dim() >= 3:
                value = torch.cat(tensors, dim=-2)
            else:
                value = torch.cat(tensors, dim=-1)
        if not torch.is_tensor(value):
            return None
        return value

    def _build_forward_context_from_results(
        self,
        *,
        results: Sequence[Any],
        model_type: str,
        batch_size: int,
        height: int,
        width: int,
        guidance_scale: float,
    ) -> Optional["ForwardContext"]:
        """Build a ForwardContext directly from SGLang results via the registry.

        Instead of hard-coding model-specific branches (e.g. ``if model_type == "flux"``),
        this method collects raw embedding tensors from results, then uses
        ``get_forward_context_cls(model_type)`` to instantiate the right subclass.
        The ForwardContext ClassVar field classification handles model-specific
        shared vs stackable semantics automatically.
        """
        from dataclasses import fields as dataclass_fields

        prompt_embeds_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        negative_prompt_list: List[torch.Tensor] = []
        negative_pooled_list: List[torch.Tensor] = []

        for result in results:
            prompt_embeds = self._unwrap_embed_field(getattr(result, "prompt_embeds", None))
            if prompt_embeds is None:
                return None
            prompt_embeds_list.append(prompt_embeds.detach().cpu())

            pooled_prompt_embeds = self._unwrap_embed_field(getattr(result, "pooled_prompt_embeds", None))
            if pooled_prompt_embeds is not None:
                pooled_list.append(pooled_prompt_embeds.detach().cpu())

            encoder_attention_mask = self._unwrap_embed_field(getattr(result, "encoder_attention_mask", None))
            if encoder_attention_mask is not None:
                mask_list.append(encoder_attention_mask.detach().cpu())

            negative_prompt_embeds = self._unwrap_embed_field(getattr(result, "negative_prompt_embeds", None))
            if negative_prompt_embeds is not None:
                negative_prompt_list.append(negative_prompt_embeds.detach().cpu())

            negative_pooled_prompt_embeds = self._unwrap_embed_field(
                getattr(result, "negative_pooled_prompt_embeds", None)
            )
            if negative_pooled_prompt_embeds is None:
                negative_pooled_prompt_embeds = self._unwrap_embed_field(
                    getattr(result, "neg_pooled_prompt_embeds", None)
                )
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_list.append(negative_pooled_prompt_embeds.detach().cpu())

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0) if prompt_embeds_list else None
        if prompt_embeds is None:
            return None

        raw_tensors: Dict[str, Any] = {
            "prompt_embeds": self._align_embedding_tensor("prompt_embeds", prompt_embeds, batch_size),
            "pooled_prompt_embeds": self._align_embedding_tensor(
                "pooled_prompt_embeds",
                torch.cat(pooled_list, dim=0) if pooled_list else None,
                batch_size,
            ),
            "encoder_attention_mask": self._align_embedding_tensor(
                "encoder_attention_mask",
                torch.cat(mask_list, dim=0) if mask_list else None,
                batch_size,
            ),
            "negative_prompt_embeds": self._align_embedding_tensor(
                "negative_prompt_embeds",
                (torch.cat(negative_prompt_list, dim=0) if negative_prompt_list else None),
                batch_size,
            ),
            "negative_pooled_prompt_embeds": self._align_embedding_tensor(
                "negative_pooled_prompt_embeds",
                (torch.cat(negative_pooled_list, dim=0) if negative_pooled_list else None),
                batch_size,
            ),
            "guidance_scale": guidance_scale,
        }

        target_model_type = model_type or "default"
        try:
            ctx_cls = get_forward_context_cls(target_model_type)
        except KeyError:
            ctx_cls = get_forward_context_cls("default")

        valid_fields = {f.name for f in dataclass_fields(ctx_cls)}

        if "text_ids" in valid_fields and prompt_embeds is not None:
            raw_tensors["text_ids"] = self._build_flux_text_ids(prompt_embeds)
        if "image_ids" in valid_fields and prompt_embeds is not None:
            raw_tensors["image_ids"] = self._build_flux_image_ids(
                height=height,
                width=width,
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )

        filtered = {k: v for k, v in raw_tensors.items() if k in valid_fields and v is not None}
        return ctx_cls(**filtered)

    # ---------------------------------------------------------------------
    # Core inference API
    # ---------------------------------------------------------------------
    def generate(self, request: RolloutRequest) -> RolloutSamples:
        if not self._is_initialized or self._generator is None:
            raise RuntimeError("SGLang engine is not initialized")

        request_kwargs, ctx = self._build_generate_kwargs(request)
        raw_results = self._generator.generate(sampling_params_kwargs=request_kwargs)

        if raw_results is None:
            raise RuntimeError("SGLang generator returned no results for prompt batch")
        results = list(raw_results) if isinstance(raw_results, list) else [raw_results]

        return self._parse_generate_results(results, ctx, request)

    def _build_generate_kwargs(self, request: RolloutRequest) -> Tuple[Dict[str, Any], _GenerateContext]:
        """Build SGLang generator kwargs and derive context for result parsing."""
        sp = request.sampling_params
        prompts = request.prompts.prompts
        steps = int(sp.num_inference_steps)
        scale = float(sp.guidance_scale)
        out_h = int(sp.height)
        out_w = int(sp.width)
        out_f = int(sp.num_frames)
        seed = int(sp.seed) if sp.seed is not None else None
        sde_indices = None if sp.sde_indices is None else {int(v) for v in sp.sde_indices}
        kwargs = dict(sp.sampler_kwargs or {})

        if not prompts:
            raise ValueError("SGLang engine requires non-empty prompts")

        model_type = self._infer_model_type()

        require_trajectory = True
        require_log_probs = True
        return_decoded_for_reward = True
        negative_prompt = kwargs.pop("negative_prompt", None)
        fps = kwargs.pop("fps", None)
        num_outputs_per_prompt = kwargs.pop("num_outputs_per_prompt", None)
        init_same_noise = bool(sp.init_same_noise)
        samples_per_prompt = int(
            kwargs.pop("num_samples_per_prompt", sp.num_samples_per_prompt)
        )
        # SGLang MUST run SDE sampling whenever the algorithm wants SDE steps
        # (i.e. ``sde_indices`` is populated) — otherwise SGLang falls back to
        # a deterministic ODE step (``scheduler.step(...)`` with eta=0), which
        # makes every sampled ``x_{t-1}`` equal its mean ``μ_t``. Replay then
        # computes ``log p(x_{t-1} | x_t)`` on that (near-)zero residual, so
        # both old_log_prob and new_log_prob are dominated by ``-log std`` and
        # GRPO's policy gradient collapses after the first optimizer step —
        # the exact reward-crash symptom observed under
        # ``--sampling.logprob-source replay``.
        #
        # Decouple "SDE vs ODE kernel" (``rollout_enabled``) from "who computes
        # log_prob" (``logprob_source``):
        #   • rollout_enabled = True  → SGLang runs SDE step (required for GRPO)
        #   • logprob_source=native   → also return trajectory_log_probs; use them
        #   • logprob_source=replay   → preserve the SDE trajectory but return
        #                                ``log_probs=None`` so ReplayLogProbPatch
        #                                recomputes log_prob on the training side
        # Keeping replay on the SDE path is still much cheaper than silently
        # going ODE and losing a whole training run.
        default_rollout_enabled = bool(
            require_log_probs and sde_indices is not None
        )
        rollout_enabled = bool(kwargs.pop("enable_rollout_logprob", default_rollout_enabled))
        requested_rollout_sde = (
            str(
                normalize_sde_type(
                    kwargs.pop("rollout_sde_type", getattr(self.config, "sde_type", "flow"))
                )
            )
            .strip()
            .lower()
        )
        # Internal config only uses canonical flow/cps/dance/dpm2 names.
        # The native SGLang backend still expects "sde" as the flow-kernel label,
        # so translate only at this external boundary.
        if requested_rollout_sde == "flow":
            rollout_sde_type = "sde"
        elif requested_rollout_sde == "cps":
            rollout_sde_type = "cps"
        else:
            # Unknown SDE label — we still want SGLang to run its SDE kernel
            # (rather than falling back to a deterministic ODE step) but we
            # can't promise the returned log_probs match DiffusionRL's math,
            # so suppress the native log_prob transport.  Replay path still
            # computes log_prob on the training side from the SDE trajectory,
            # so GRPO keeps working.
            rollout_sde_type = "sde"
            if (
                rollout_enabled
                and self._native_rollout_logprob_enabled()
                and not self._warned_unsupported_rollout_sde
            ):
                logger.warning(
                    "SGLang native rollout logprob currently supports only flow/cps, got sde_type=%r. "
                    "Forcing rollout_sde_type='sde' and suppressing native log_prob transport; "
                    "training will recompute log_prob via replay.",
                    requested_rollout_sde,
                )
                self._warned_unsupported_rollout_sde = True
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
            # Keep rollout path latent-only; decoded reward media must come from
            # the sampler output itself or explicit decode_latents().
            "return_trajectory_decoded": False,
            "return_prompt_embeds": True,
            "return_negative_prompt_embeds": bool(negative_prompt is not None),
        }
        if negative_prompt is not None:
            sampling_params_kwargs["negative_prompt"] = negative_prompt
        if fps is not None:
            sampling_params_kwargs["fps"] = int(fps)
        if num_outputs_per_prompt is not None:
            sampling_params_kwargs["num_outputs_per_prompt"] = int(num_outputs_per_prompt)
        sampling_params_kwargs["sigmas"] = get_sigma_schedule(
            steps,
            shift=float(self.config.shift),
        )[:-1].tolist()
        if rollout_enabled:
            sampling_params_kwargs["rollout"] = True
            sampling_params_kwargs["rollout_sde_type"] = rollout_sde_type
            sampling_params_kwargs["rollout_noise_level"] = rollout_noise_level
            if sde_indices is not None:
                sampling_params_kwargs["rollout_sde_indices"] = sorted(int(i) for i in sde_indices)

        unique_prompts, validated_k = _deexpand_prompts(list(prompts), samples_per_prompt)
        if num_outputs_per_prompt is not None:
            validated_k = 1
            unique_prompts = list(prompts)
            sampling_params_kwargs["num_outputs_per_prompt"] = int(num_outputs_per_prompt)

        if validated_k > 1:
            logger.info(
                "SGLang batched generation: %d expanded prompts -> %d unique prompts x K=%d (init_same_noise=%s)",
                len(prompts),
                len(unique_prompts),
                validated_k,
                init_same_noise,
            )

        request_kwargs = dict(sampling_params_kwargs)
        request_kwargs["prompt"] = unique_prompts if len(unique_prompts) > 1 else unique_prompts[0]
        if validated_k > 1:
            request_kwargs["num_outputs_per_prompt"] = validated_k
            # Precomputed x_T in DiffusionRL; SGLang must not alias one RNG
            # stream for all K when init_same was True, or we lose per-sample
            # independent SDE step noise. Initial sharing is from ``initial_noise``.
            request_kwargs["init_same_noise"] = False

        # Precompute x_T in DiffusionRL (same blake2 group seeds as FSDP) so SGLang
        # does not need ``noise_group_ids`` for the denoising loop — forwarding
        # those IDs also drives deterministic per-step SDE generators in
        # SGLang, which we avoid (see ``diffusionrl/sde/runtime.py``).
        if self._server_args is None:
            raise RuntimeError(
                "SGLangRolloutEngine._build_generate_kwargs: initialize() must run first "
                "(server_args is required to prepare_latent_shape / initial_noise)."
            )
        n_expanded = len(prompts)
        assert n_expanded > 0, "non-empty prompts checked above"
        batch_stub = SimpleNamespace(
            height=int(out_h), width=int(out_w), num_frames=int(out_f)
        )
        full_shape = self._server_args.pipeline_config.prepare_latent_shape(
            batch_stub, int(n_expanded), int(out_f)
        )
        if int(full_shape[0]) != n_expanded:
            raise ValueError(
                "prepare_latent_shape first dim != expanded prompt count: "
                f"full_shape[0]={full_shape[0]}, n_expanded={n_expanded}."
            )
        per_sample_shape = tuple(full_shape[1:])
        device = self._device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Resolve initial-noise dtype from sampling config; SGLang's
        # latent_preparation will `.to(dtype=model_dtype)` anyway, but
        # generating at the right precision avoids a redundant cast.
        _prec = getattr(sp, "autocast_precision", "bf16")
        _prec_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        latent_dtype = _prec_map.get(str(_prec), torch.bfloat16) if device.type == "cuda" else torch.float32
        raw_prompts = request.prompts
        raw_noise_group_ids = getattr(raw_prompts, "noise_group_ids", None)
        if init_same_noise:
            if not isinstance(raw_noise_group_ids, list) or len(
                raw_noise_group_ids
            ) != n_expanded:
                raise ValueError(
                    "SGLang: init_same_noise=True requires request.prompts."
                    f"noise_group_ids length {len(raw_noise_group_ids) if isinstance(raw_noise_group_ids, list) else 0} "
                    f"to match expanded batch {n_expanded}."
                )
            ng_for_latents = [str(x) for x in raw_noise_group_ids]
            base_for_latents = int(seed) if seed is not None else None
            if base_for_latents is None:
                raise ValueError("init_same_noise=True requires sampling_params.seed")
        else:
            ng_for_latents = None
            base_for_latents = None
        initial = generate_latents(
            batch_size=n_expanded,
            latent_shape=per_sample_shape,
            device=device,
            dtype=latent_dtype,
            init_same_noise=init_same_noise,
            samples_per_prompt=samples_per_prompt,
            noise_group_ids=ng_for_latents,
            base_seed=base_for_latents,
        )
        request_kwargs["initial_noise"] = initial
        if not self._informed_sglang_initial_noise_policy:
            logger.info(
                "SGLang: initial latents are precomputed in DiffusionRL; "
                "noise_group_ids are not sent to SGLang so per-step SDE noise is not group-seeded."
            )
            self._informed_sglang_initial_noise_policy = True
        # High-entropy batch seed: per-step SDE noise is not tied to the training
        # config seed (sp.seed is used for init noise in DiffusionRL only after
        # rollout mix in ``plan_requests``).
        request_kwargs["seed"] = int.from_bytes(os.urandom(8), "big") % (MAX_TORCH_SEED + 1)

        ctx = _GenerateContext(
            model_type=model_type,
            steps=steps,
            guidance_scale=scale,
            height=out_h,
            width=out_w,
            sde_indices=sde_indices,
            require_trajectory=require_trajectory,
            require_log_probs=require_log_probs,
            return_decoded_for_reward=return_decoded_for_reward,
        )
        return request_kwargs, ctx

    def _parse_generate_results(
        self,
        results: List[Any],
        ctx: _GenerateContext,
        request: RolloutRequest,
    ) -> RolloutSamples:
        """Parse raw SGLang results into RolloutSamples."""
        trajectory_items: List[Optional[torch.Tensor]] = [
            self._extract_trajectory_from_result(result, required=ctx.require_trajectory) for result in results
        ]
        missing_trajectory = sum(item is None for item in trajectory_items)
        use_trajectory = missing_trajectory == 0 and len(trajectory_items) > 0
        if missing_trajectory and not ctx.require_trajectory and missing_trajectory != len(trajectory_items):
            logger.warning(
                "SGLang returned mixed trajectory availability (%s/%s missing); "
                "falling back to trajectory-free adapter path for batch consistency.",
                missing_trajectory,
                len(trajectory_items),
            )
            use_trajectory = False

        trajectories_tensor: Optional[torch.Tensor] = None
        trajectory_store: Optional[Trajectory] = None
        if use_trajectory:
            trajectories_tensor = torch.cat(
                [item for item in trajectory_items if item is not None],
                dim=0,
            )
            timesteps, step_indices = self._derive_timestep_alignment(
                trajectories_tensor=trajectories_tensor,
                num_inference_steps=ctx.steps,
            )
            final_latents = trajectories_tensor[:, -1].clone()

            # Client-side selective trim: reduce GPU memory on training side
            traj_len = int(trajectories_tensor.shape[1])

            # Attempt selective trim when only a subset of positions is needed.
            # With the T+1 invariant enforced by _derive_timestep_alignment,
            # column i == trajectory position i (0-indexed).
            trimmed_cols = None
            if ctx.sde_indices is not None and len(ctx.sde_indices) < ctx.steps:
                needed_original = set(compute_trajectory_positions(ctx.sde_indices, ctx.steps))
                keep_cols = sorted(p for p in needed_original if 0 <= p < traj_len)
                if keep_cols and len(keep_cols) < traj_len:
                    trimmed_cols = keep_cols

            if trimmed_cols is not None:
                trimmed = trajectories_tensor[:, trimmed_cols]
                trajectory_store = Trajectory.from_selective(trimmed, trimmed_cols, traj_len)
            else:
                trajectory_store = Trajectory.from_full(trajectories_tensor)
            del trajectories_tensor  # Free raw tensor; trajectory_store owns the data now
        else:
            final_latents = self._extract_final_latents_without_trajectory(
                results=results,
                model_type=ctx.model_type,
            )
            timesteps = get_sigma_schedule(ctx.steps, shift=float(self.config.shift)).cpu()
            step_indices = torch.arange(timesteps.shape[0], dtype=torch.long)

        per_result_log_probs: List[Optional[torch.Tensor]] = []
        # Only consume SGLang-returned log_probs when logprob_source='native'.
        # Under ``logprob_source='replay'`` we still ask SGLang to run SDE
        # sampling (so the trajectory has real Gaussian residuals), but the
        # authoritative log_prob must be recomputed on the training side via
        # ``ReplayLogProbPatch`` — that path is gated on
        # ``batch.log_probs is None``, so we intentionally return None here.
        if ctx.require_log_probs and self._native_rollout_logprob_enabled():
            per_result_log_probs = [self._extract_log_probs_from_result(result) for result in results]

        merged_log_probs: Optional[LogProbData] = None
        if ctx.require_log_probs and per_result_log_probs and all(lp is not None for lp in per_result_log_probs):
            log_prob_tensor = torch.cat([lp for lp in per_result_log_probs if lp is not None], dim=0)
            expected_steps = int(step_indices.shape[0]) - 1
            if int(log_prob_tensor.shape[1]) == expected_steps:
                lp_dict = {int(step_indices[i].item()): log_prob_tensor[:, i] for i in range(expected_steps)}
                merged_log_probs = LogProbData.from_dict(lp_dict)
                if not getattr(self, "_logged_logprob_attach", False):
                    logger.info(
                        "SGLang native logprobs attached: tensor_shape=%s, step_keys=%s, require_log_probs=%s",
                        tuple(log_prob_tensor.shape),
                        sorted(lp_dict.keys()),
                        ctx.require_log_probs,
                    )
                    self._logged_logprob_attach = True
            elif not getattr(self, "_warned_logprob_shape", False):
                logger.warning(
                    "SGLang trajectory_log_probs shape mismatch: got %s, expected second dim=%s. "
                    "Ignoring rollout log_probs and keeping replay path enabled.",
                    tuple(log_prob_tensor.shape),
                    expected_steps,
                )
                self._warned_logprob_shape = True

        rollout_noise_preds_tensor: Optional[torch.Tensor] = None
        per_result_noise_preds = [getattr(result, "trajectory_noise_preds", None) for result in results]
        if any(item is not None for item in per_result_noise_preds):
            valid_noise_preds = [item for item in per_result_noise_preds if item is not None]
            if valid_noise_preds:
                rollout_noise_preds_tensor = torch.cat(valid_noise_preds, dim=0)

        forward_context = self._build_forward_context_from_results(
            results=results,
            model_type=ctx.model_type,
            batch_size=int(final_latents.shape[0]),
            height=ctx.height,
            width=ctx.width,
            guidance_scale=ctx.guidance_scale,
        )

        decoded_images = None
        decoded_video_tensors: List[torch.Tensor] = []
        if ctx.return_decoded_for_reward:
            decoded_images = []
            for result in results:
                pil_img, video_tensor = self._extract_decoded_media(result)
                if pil_img is not None:
                    decoded_images.append(pil_img)
                if video_tensor is not None:
                    decoded_video_tensors.append(video_tensor)
            if not decoded_images:
                decoded_images = None
            if decoded_images is None and not decoded_video_tensors and not self._warned_missing_decoded:
                logger.warning(
                    "SGLang generate(return_decoded_for_reward=True) returned no decodable media. "
                    "Reward stage will fail without decoded media."
                )
                self._warned_missing_decoded = True

        if ctx.model_type == "flux":
            if trajectory_store is not None and trajectory_store.data.dim() == 4:
                trajectory_format = "packed_seq_c4"
            elif final_latents.dim() == 3:
                trajectory_format = "packed_seq_c4"
            else:
                trajectory_format = "dense_latent"
        else:
            if trajectory_store is not None:
                trajectory_format = "video_dense_latent" if trajectory_store.data.dim() == 6 else "dense_latent"
            else:
                trajectory_format = "video_dense_latent" if final_latents.dim() == 5 else "dense_latent"

        metadata: Dict[str, Any] = {
            "generator_type": "sglang",
            "logprob_source": self._logprob_source(),
            "prompt_embeds_source": "sglang",
            "trajectory_format": trajectory_format,
            "timestep_type": "sigma",
            "timestep_scale": 1.0,
            "sde_indices": sorted(int(i) for i in ctx.sde_indices) if ctx.sde_indices is not None else None,
            "trajectory_available": bool(trajectory_store is not None),
        }
        if rollout_noise_preds_tensor is not None:
            metadata["rollout_noise_preds"] = rollout_noise_preds_tensor
        if decoded_video_tensors:
            metadata["decoded_videos"] = torch.stack(decoded_video_tensors, dim=0)
        if self._last_weight_checksum:
            metadata["weight_checksum"] = dict(self._last_weight_checksum)

        decoded_videos = metadata.pop("decoded_videos", None)

        return RolloutSamples(
            latents=final_latents,
            timesteps=timesteps,
            sampling_params=request.sampling_params,
            prompts=request.prompts,
            trajectories=trajectory_store,
            log_probs=merged_log_probs,
            forward_context=forward_context,
            step_indices=step_indices,
            rewards=None,
            component_rewards=None,
            decoded_images=decoded_images,
            decoded_videos=decoded_videos if isinstance(decoded_videos, torch.Tensor) else None,
        )

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if prompts is None or len(prompts) == 0:
            raise ValueError("encode_prompt requires non-empty prompts")
        if not self._is_initialized or self._generator is None:
            raise RuntimeError("SGLang engine is not initialized")

        encode_fn = getattr(self._generator, "encode_prompt", None)
        if not callable(encode_fn):
            raise NotImplementedError(
                "SGLang DiffGenerator does not support encode_prompt(). "
                "Upgrade sglang to a version with encode_prompt support."
            )

        output = encode_fn(list(prompts))
        if isinstance(output, dict) and "error" in output:
            raise RuntimeError(f"SGLang encode_prompt failed: {output['error']}")
        if not isinstance(output, dict):
            raise RuntimeError(f"Unexpected SGLang encode_prompt output type: {type(output).__name__}")

        model_type = self._infer_model_type()
        prompt_embeds = output.get("prompt_embeds")
        if model_type == "flux":
            if prompt_embeds is None:
                raise RuntimeError("SGLang encode_prompt() returned no prompt_embeds for FLUX")
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
        if not self._is_initialized:
            raise RuntimeError("SGLang engine is not initialized")
        try:
            from sglang.multimodal_gen.runtime.entrypoints.utils import SetLoraFromTensorsReq
        except Exception as exc:
            raise RuntimeError("Installed sglang runtime does not expose SetLoraFromTensorsReq.") from exc

        request = SetLoraFromTensorsReq(
            lora_nickname=str(adapter_name),
            lora_tensors=lora_tensors,
            target="all",
            strength=1.0,
        )
        response = self._runtime["sync_scheduler_client"].forward(request)
        error = getattr(response, "error", None)
        if error is not None:
            raise RuntimeError(f"set_lora_from_tensors failed: {error}")
        logger.info("SGLang LoRA initialized from tensors (adapter=%s)", adapter_name)

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        if not self._is_initialized:
            raise RuntimeError("SGLang engine is not initialized")
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be non-empty")

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
            if not isinstance(checksum_output, dict) or not checksum_output:
                raise RuntimeError(
                    f"SGLang checksum query returned invalid payload after weight update: {checksum_output!r}"
                )
            normalized = {str(k): str(v) for k, v in checksum_output.items()}
            bad_values = {k: v for k, v in normalized.items() if not v or v in {"not_found", "error"}}
            if bad_values:
                raise RuntimeError(f"SGLang checksum query reported invalid modules after weight update: {bad_values}")
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
        if not names:
            raise ValueError("names must be non-empty for distributed update")
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

    # ---------------------------------------------------------------------
    # Capability and lifecycle
    # ---------------------------------------------------------------------
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
