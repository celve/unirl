"""Base class for all FSDP-native samplers.

Provides shared state (model, text_encoder, vae, model_bundle) and concrete
orchestration methods: sampler creation, input encoding, sample generation
with optional adapter switching, latent decoding, and module discovery.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from diffusionrl.samplers.base import BaseSampler
from diffusionrl.sde.kernels import StepStrategy
from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.types.media import MediaRef
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples
from diffusionrl.utils import load_function
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context manager (module-level helper)
# ---------------------------------------------------------------------------


@contextmanager
def _temporary_sampler_overrides(
    sampler: Any,
    overrides: Optional[Dict[str, Any]],
):
    """Temporarily override mutable sampler attributes for one request."""
    if sampler is None or not isinstance(overrides, dict) or not overrides:
        yield
        return

    original_values: Dict[str, Any] = {}
    try:
        for raw_key, raw_value in overrides.items():
            key = str(raw_key).strip()
            if key not in {"eta", "sde_type"}:
                continue
            if not hasattr(sampler, key):
                continue
            original_values[key] = getattr(sampler, key)
            value = raw_value
            if key == "sde_type":
                value = normalize_sde_type(str(raw_value))
            else:
                value = float(raw_value)
            setattr(sampler, key, value)
        yield
    finally:
        for key, value in original_values.items():
            setattr(sampler, key, value)


# ---------------------------------------------------------------------------
# FSDPBaseSampler
# ---------------------------------------------------------------------------


class FSDPBaseSampler(BaseSampler):
    """Base class for all FSDP-native samplers.

    Extends ``BaseSampler`` with:
    - Shared instance state (model, text_encoder, vae, model_bundle,
      precision dtypes).
    - Orchestration methods for the full sampling lifecycle: input
      encoding, adapter-aware sampling, latent decoding, and module
      discovery.
    - A ``from_config`` factory classmethod for dotpath-based
      instantiation.

    Concrete subclasses (FluxSampler, SD3Sampler, FSDPHunyuanVideoSampler)
    only need to implement ``sample()``.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 1.0,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        model_bundle: Optional[Any] = None,
        autocast_precision: Any = "bf16",
        trajectory_precision: Any = "fp16",
        logprob_precision: Any = "fp32",
        **kwargs: Any,
    ):
        super().__init__(eta=eta, strategy=strategy, shift=shift)
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.model_bundle = model_bundle
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_runtime_device(
        self,
        prompt_embeds: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
    ) -> torch.device:
        """Resolve sampling compute device robustly under FSDP CPU offload.

        FSDP with cpu_offload can keep parameters on CPU outside forward, so
        ``next(self.model.parameters()).device`` is not a reliable runtime
        signal.  Prefer actual runtime tensors and fall back to current CUDA
        device.
        """
        if latents is not None and torch.is_tensor(latents) and latents.is_cuda:
            return latents.device
        if prompt_embeds is not None and torch.is_tensor(prompt_embeds) and prompt_embeds.is_cuda:
            return prompt_embeds.device
        if torch.cuda.is_available():
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                return torch.device("cuda")
        if latents is not None and torch.is_tensor(latents):
            return latents.device
        if prompt_embeds is not None and torch.is_tensor(prompt_embeds):
            return prompt_embeds.device
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Input encoding
    # ------------------------------------------------------------------

    def encode_inputs(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Encode sampler conditioning inputs via *model_bundle*.

        Raises:
            RuntimeError: If model_bundle is None or lacks the encoding method.
        """
        if self.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")
        return self.model_bundle.encode_inputs(prompts, **kwargs)

    def prepare_multimodal_encode_kwargs(
        self,
        request: RolloutRequest,
        *,
        height: int,
        width: int,
        num_frames: int,
    ) -> Dict[str, Any]:
        """Build model-bundle encode kwargs from typed per-sample media refs.

        First version supports condition images. Video/audio refs remain typed
        on the request for rewards and future samplers, but are not loaded here
        until a model-side contract exists.
        """
        image_refs = self._select_media_refs(request.prompts, modality="image", role="condition")
        if image_refs is None:
            return {}

        if self.model_bundle is None or not getattr(self.model_bundle, "accepts_image_input", False):
            model_name = type(self.model_bundle).__name__ if self.model_bundle is not None else "<missing>"
            raise ValueError(f"{model_name} received condition image media refs but does not accept image input.")

        return {
            "image": self._load_image_refs(image_refs, height=height, width=width),
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }

    def _select_media_refs(
        self,
        prompts: Any,
        *,
        modality: str,
        role: str,
    ) -> Optional[List[MediaRef]]:
        """Select exactly one media ref per sample for a modality/role pair."""
        if not isinstance(prompts, Prompts):
            return None

        selected: List[Optional[MediaRef]] = []
        has_any = False
        modality = str(modality).strip().lower()
        role = str(role).strip().lower()
        for sample_idx, refs in enumerate(prompts.media_refs):
            matches = [
                ref for ref in refs if ref.modality.strip().lower() == modality and ref.role.strip().lower() == role
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Expected at most one {role} {modality} media ref per sample; "
                    f"got {len(matches)} at sample index {sample_idx}."
                )
            if matches:
                has_any = True
                selected.append(matches[0])
            else:
                selected.append(None)

        if not has_any:
            return None
        missing = [idx for idx, ref in enumerate(selected) if ref is None]
        if missing:
            raise ValueError(f"Batch has mixed {role} {modality} media refs. Missing refs at sample indices: {missing}")
        return [ref for ref in selected if ref is not None]

    def _load_image_refs(self, refs: List[MediaRef], *, height: int, width: int) -> torch.Tensor:
        """Load image refs to a [B, 3, H, W] tensor in [-1, 1]."""
        try:
            from PIL import Image as PILImage
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "Image media refs require torchvision and Pillow. Install with: pip install torchvision pillow"
            ) from exc

        transform = self._get_image_transform(height, width, transforms)
        tensors: List[torch.Tensor] = []
        for ref in refs:
            with PILImage.open(ref.uri) as pil_img:
                tensors.append(transform(pil_img.convert("RGB")))
        device = self._resolve_runtime_device(prompt_embeds=None, latents=None)
        return torch.stack(tensors, dim=0).to(device=device)

    def _get_image_transform(self, height: int, width: int, transforms: Any) -> Any:
        """Return a cached image transform pipeline for the given resolution."""
        cache = getattr(self, "_image_transform_cache", None)
        if cache is None:
            self._image_transform_cache: Dict[Tuple[int, int], Any] = {}
            cache = self._image_transform_cache
        key = (int(height), int(width))
        if key not in cache:
            cache[key] = transforms.Compose(
                [
                    transforms.Resize(key, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        return cache[key]

    # ------------------------------------------------------------------
    # Latent decoding
    # ------------------------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors to pixel space ``[0, 1]`` via VAE.

        Always uses float32 for VAE decoding (bfloat16 is unsupported by most
        VAE implementations).

        Raises:
            RuntimeError: If *vae* is None.
        """
        if self.vae is None:
            raise RuntimeError("VAE not available for decoding")
        with torch.no_grad():
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "scaling_factor"):
                scaling_factor = self.vae.config.scaling_factor
            else:
                scaling_factor = 0.18215
            shift_factor = getattr(getattr(self.vae, "config", None), "shift_factor", None)
            latents_float = latents.to(dtype=torch.float32)
            vae_input = latents_float / scaling_factor
            if shift_factor is not None:
                vae_input = vae_input + float(shift_factor)
            decoded = self.vae.to(torch.float32).decode(vae_input).sample
            return (decoded + 1) / 2  # [-1, 1] -> [0, 1]

    # ------------------------------------------------------------------
    # Module discovery (for offload / eval-context)
    # ------------------------------------------------------------------

    def iter_offloadable_modules(
        self,
        *,
        include_transformer: bool = True,
    ) -> List[Tuple[str, nn.Module]]:
        """Discover ``nn.Module`` attributes on *self* suitable for GPU offloading.

        Scans ``self.__dict__`` for attributes whose name matches well-known
        component names (transformer, text_encoder*, vae, image_encoder).

        Args:
            include_transformer: If False, skip attributes containing
                ``"transformer"`` in their name.

        Returns:
            List of ``(attr_name, module)`` pairs.
        """
        known_names = {
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
            "image_encoder",
        }
        results: List[Tuple[str, nn.Module]] = []
        for name, value in self.__dict__.items():
            if not isinstance(value, nn.Module):
                continue
            base_name = name.lstrip("_").lower()
            if not include_transformer and "transformer" in base_name:
                continue
            if base_name in known_names or any(token in base_name for token in ("encoder", "vae", "transformer")):
                results.append((name, value))
        return results

    # ------------------------------------------------------------------
    # Core sampling call
    # ------------------------------------------------------------------

    def run_sample(
        self,
        *,
        sampling_adapter: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> RolloutSamples:
        """Call ``self.sample()`` with optional LoRA adapter switching.

        Args:
            sampling_adapter: If set, temporarily switch to this adapter name
                before sampling (e.g. ``"old"`` for NFT).
            **sample_kwargs: Forwarded verbatim to ``self.sample()``.

        Returns:
            ``RolloutSamples`` from the sampler.
        """
        sampler_overrides = sample_kwargs.pop("sampler_overrides", None)
        with _temporary_sampler_overrides(self, sampler_overrides):
            if sampling_adapter and self.model is not None:
                with switch_adapter(self.model, sampling_adapter):
                    return self.sample(**sample_kwargs)
            return self.sample(**sample_kwargs)

    # ------------------------------------------------------------------
    # Full generation flow
    # ------------------------------------------------------------------

    def generate(
        self,
        request: RolloutRequest,
        *,
        host_label: str,
        sampling_adapter: Optional[str] = None,
    ) -> RolloutSamples:
        """Run the full prompt-only FSDP sampling flow.

        Reads resolved parameters from ``request.sampling_params``,
        encodes conditioning inputs, then calls :meth:`run_sample`.
        """
        from diffusionrl.types.prompts import Prompts as _PromptsType

        sp = request.sampling_params
        raw_prompts = request.prompts
        prompts = list(raw_prompts.prompts) if isinstance(raw_prompts, _PromptsType) else list(raw_prompts or [])
        if not prompts:
            raise ValueError(
                f"{host_label} requires non-empty text prompts. Prompt-embedding-only input is not supported."
            )

        kwargs = dict(sp.sampler_kwargs or {})
        base_seed = int(sp.seed)
        num_inference_steps = int(sp.num_inference_steps)
        guidance_scale = float(sp.guidance_scale)
        height = int(sp.height)
        width = int(sp.width)
        num_frames = int(sp.num_frames)
        sde_indices_raw = sp.sde_indices
        sde_indices = None if sde_indices_raw is None else {int(v) for v in sde_indices_raw}
        noise_group_ids = list(raw_prompts.noise_group_ids) if isinstance(raw_prompts, _PromptsType) else None

        encode_kwargs = dict(kwargs)
        encode_kwargs.update(
            self.prepare_multimodal_encode_kwargs(
                request,
                height=height,
                width=width,
                num_frames=num_frames,
            )
        )
        encoded = self.encode_inputs(prompts, **encode_kwargs)
        if encoded.get("prompt_embeds") is None:
            raise RuntimeError(f"{host_label} input encoder returned no prompt_embeds.")

        output = self.run_sample(
            sampling_adapter=sampling_adapter,
            prompts=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            latents=None,
            base_seed=base_seed,
            sde_indices=sde_indices,
            init_same_noise=bool(sp.init_same_noise),
            samples_per_prompt=max(1, int(sp.num_samples_per_prompt)),
            noise_group_ids=noise_group_ids,
            **encoded,
            **kwargs,
        )
        # Attach request-level metadata the sampler doesn't have access to.
        from dataclasses import replace as _replace

        return _replace(output, sampling_params=sp, prompts=raw_prompts)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        sampler_dotpath: str,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[Any] = None,
        eta: float = 1.0,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        model_bundle: Optional[Any] = None,
        **sampler_kwargs: Any,
    ) -> "FSDPBaseSampler":
        """Instantiate a sampler from a dotpath, merging model_bundle extra kwargs.

        Args:
            sampler_dotpath: Fully-qualified dotpath to the sampler class.
            model: Transformer / denoiser module.
            text_encoder: Text encoder (may be None for embedding-only mode).
            vae: VAE decoder (may be None if decoding is not needed).
            eta: SDE noise scale.
            strategy: SDE step strategy instance built from
                ``cfg.sampling.sde_strategy`` (Flow/CPS/Dance/DPM2). When
                ``None``, the concrete sampler subclass picks a default.
            shift: Time-shift parameter.
            model_bundle: Optional ModelBundle that may provide extra kwargs.
            **sampler_kwargs: Forwarded to sampler constructor.

        Returns:
            A ``FSDPBaseSampler`` subclass instance ready for ``sampler.sample()``.
        """
        sampler_cls = load_function(sampler_dotpath)
        if model_bundle is not None and hasattr(model_bundle, "get_sampler_extra_kwargs"):
            extra_kwargs = model_bundle.get_sampler_extra_kwargs() or {}
            for key, value in extra_kwargs.items():
                sampler_kwargs.setdefault(key, value)
        return sampler_cls(
            model=model,
            text_encoder=text_encoder,
            vae=vae,
            eta=eta,
            strategy=strategy,
            shift=shift,
            model_bundle=model_bundle,
            **sampler_kwargs,
        )


__all__ = ["FSDPBaseSampler"]
