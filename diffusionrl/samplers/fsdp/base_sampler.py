"""Base class for all FSDP-native samplers.

Provides shared state (model, text_encoder, vae, model_bundle) and concrete
orchestration methods that were previously free functions in sampler_runner.py:
sampler creation, prompt encoding, sample generation with optional adapter
switching, latent decoding, module discovery, and output post-processing.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from diffusionrl.samplers.base import BaseSampler
from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples
from diffusionrl.utils import load_function
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.dtypes import parse_torch_dtype
from diffusionrl.utils.media import tensor_to_pil

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
    - Orchestration methods for the full sampling lifecycle: prompt
      encoding, adapter-aware sampling, latent decoding, module
      discovery, and output post-processing.
    - A ``from_config`` factory classmethod for dotpath-based
      instantiation.

    Concrete subclasses (FluxSampler, SD3Sampler, FSDPHunyuanSampler)
    only need to implement ``sample()``.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 1.0,
        sde_type: str = "flow",
        shift: float = 3.0,
        model_bundle: Optional[Any] = None,
        autocast_precision: Any = "bf16",
        trajectory_precision: Any = "fp16",
        logprob_precision: Any = "fp32",
        **kwargs: Any,
    ):
        super().__init__(eta=eta, sde_type=sde_type, shift=shift)
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
    # Prompt encoding
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts to embeddings via *model_bundle*.

        Raises:
            RuntimeError: If model_bundle is None or lacks the encoding method.
        """
        if self.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")
        if not hasattr(self.model_bundle, "encode_prompt_for_inference"):
            raise RuntimeError("Model bundle does not support inference prompt encoding")
        return self.model_bundle.encode_prompt_for_inference(prompts, **kwargs)

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
            latents_float = latents.to(dtype=torch.float32)
            decoded = self.vae.to(torch.float32).decode(latents_float / scaling_factor).sample
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
        decode_for_reward: bool = True,
    ) -> RolloutSamples:
        """Run the full prompt-only FSDP sampling flow.

        Reads resolved parameters from ``request.sampling_params``,
        encodes prompts, then calls :meth:`run_sample`.
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

        encoded = self.encode_prompt(prompts)
        if encoded.get("prompt_embeds") is None:
            raise RuntimeError(f"{host_label} prompt encoder returned no prompt_embeds.")

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
    # Output post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_metadata_defaults(
        output: RolloutSamples,
        metadata_defaults: Optional[Dict[str, Any]],
    ) -> RolloutSamples:
        """Fill missing keys in ``aux['metadata']`` from *metadata_defaults*."""
        if not metadata_defaults:
            return output
        raw_metadata = output.aux.get("metadata")
        metadata = dict(raw_metadata or {})
        changed = False
        for key, value in metadata_defaults.items():
            if key not in metadata:
                metadata[key] = value
                changed = True
        if changed:
            output.aux["metadata"] = metadata
        return output

    def _decode_for_reward_if_needed(
        self,
        output: RolloutSamples,
        *,
        decode_for_reward: bool,
        host_label: str,
    ) -> RolloutSamples:
        """VAE-decode final latents when the request asks for reward-ready pixels."""
        if not decode_for_reward:
            return output
        if output.decoded_videos is not None or output.decoded_images is not None:
            return output
        try:
            decoded = self.decode_latents(output.latents)
            decoded_images = tensor_to_pil(decoded)
        except Exception as exc:
            raise RuntimeError(
                f"decode_for_reward requested but {host_label} produced no "
                f"decoded media and latent decoding failed: {exc}"
            ) from exc
        from dataclasses import replace as _replace

        return _replace(output, decoded_images=decoded_images)

    def finalize_output(
        self,
        *,
        output: RolloutSamples,
        host_label: str,
        decode_for_reward: bool = True,
        metadata_defaults: Optional[Dict[str, Any]] = None,
        local_reward_attach_fn: Optional[Callable[[RolloutSamples], RolloutSamples]] = None,
        transport_optimize_fn: Optional[Callable[[RolloutSamples], RolloutSamples]] = None,
        move_output_to_cpu: bool = True,
    ) -> RolloutSamples:
        """Apply shared sampling-output post-processing after raw generation.

        Pipeline: metadata defaults -> decode-for-reward -> local reward ->
        transport optimisation -> move to CPU.
        """
        output = self._attach_metadata_defaults(output, metadata_defaults)
        output = self._decode_for_reward_if_needed(
            output,
            decode_for_reward=decode_for_reward,
            host_label=host_label,
        )
        if decode_for_reward and local_reward_attach_fn is not None:
            output = local_reward_attach_fn(output)
        if transport_optimize_fn is not None:
            output = transport_optimize_fn(output)
        if move_output_to_cpu:
            output = output.to_device("cpu")
        return output

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
        sde_type: str = "flow",
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
            sde_type: Transition rule ("flow", "cps", "dance", "dpm2").
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
            sde_type=sde_type,
            shift=shift,
            model_bundle=model_bundle,
            **sampler_kwargs,
        )


__all__ = ["FSDPBaseSampler"]
