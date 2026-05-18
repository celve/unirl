"""HunyuanVideo-1.5 model bundle (T2V + I2V).

Mirrors the diffusers ``HunyuanVideo15Pipeline`` and
``HunyuanVideo15ImageToVideoPipeline`` while plugging into DiffusionRL's
existing FSDP sampling stack. The legacy 1.0 ``HunyuanModelBundle`` keeps its
own ``model_type='hunyuan'`` registration; this bundle introduces a separate
``model_type='hunyuan_veido1p5'`` so both versions can coexist.

Upstream contract this bundle re-implements:

* Text streams: Qwen2.5-VL MLLM (chat-template + ``crop_start``) + ByT5
  glyph encoder (regex-extracted ``Text "..."`` snippets).
* Vision conditioning: SigLIP ``last_hidden_state`` (T2V uses zeros).
* Latent packing: ``cat([latents, cond_latents, mask], dim=1)`` before the
  transformer forward (T2V → zeros for cond/mask; I2V → first-frame image
  latent + binary first-frame mask).

Reference:
    diffusers/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py
    diffusers/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusionrl.config.registration import register_config

from .base import ModelBundle
from .config import ModelBundleConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants copied from the upstream pipeline
# ---------------------------------------------------------------------------

# fmt: off
_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "1. The main content and theme of the video. "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "4. background environment, light, style and atmosphere. "
    "5. camera angles, movements, and transitions used in the video."
)
# fmt: on

_GLYPH_PATTERN = re.compile(r"\"(.*?)\"|\u201c(.*?)\u201d")


def _extract_glyph_texts(prompt: str) -> Optional[str]:
    """Extract quoted glyph snippets and reformat to ``Text "...". `` form.

    Mirrors ``pipeline_hunyuan_video1_5.extract_glyph_texts``: returns ``None``
    when no quoted glyph text is present so callers can substitute a zero
    embedding tensor.
    """
    matches = _GLYPH_PATTERN.findall(prompt)
    result = [m[0] or m[1] for m in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    if not result:
        return None
    return ". ".join([f'Text "{text}"' for text in result]) + ". "


def _format_chat_template(prompts: List[str], system_message: str) -> List[List[Dict[str, str]]]:
    """Build the (system, user) chat conversation list expected by Qwen2.5-VL."""
    return [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": p if p else " "},
        ]
        for p in prompts
    ]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@register_config(
    group="model",
    name="hunyuan_veido1p5",
    target="diffusionrl.models.hunyuan_veido1p5.HunyuanVeido1p5ModelBundle",
)
@dataclass
class HunyuanVeido1p5ModelBundleConfig(ModelBundleConfig):
    """Bundle config for HunyuanVideo-1.5.

    All optional checkpoint paths default to ``pretrained_model_ckpt_path``;
    override individually when the encoders / VAE / SigLIP live in different
    folders.
    """

    text_encoder_2_ckpt_path: Optional[str] = None
    image_encoder_ckpt_path: Optional[str] = None
    mllm_max_length: int = 1000
    byt5_max_length: int = 256
    mllm_skip_layers: int = 2
    mllm_crop_start: int = 108
    vision_num_semantic_tokens: int = 729
    vision_states_dim: int = 1152
    # Pure-T2V recipes can disable the SigLIP vision encoder to free ~1.6 GB
    # of permanently-resident GPU memory: ``_build_image_conditioning`` only
    # touches the encoder on the I2V branch (``image is not None``); the T2V
    # branch returns a zero ``image_embeds`` placeholder of shape
    # ``[B, vision_num_semantic_tokens, vision_states_dim]`` regardless. Set
    # to ``False`` in the T2V experiment recipes; leave ``True`` (default) for
    # I2V or any flow that may pass a conditioning image at runtime.
    load_vision_encoder: bool = True


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


class HunyuanVeido1p5ModelBundle(ModelBundle):
    """HunyuanVideo-1.5 bundle supporting T2V and I2V conditioning."""

    TIMESTEP_SCALE = 1000.0
    DEFAULT_SPATIAL_DOWNSAMPLE = 16
    DEFAULT_TEMPORAL_DOWNSAMPLE = 4

    def __init__(self, config: HunyuanVeido1p5ModelBundleConfig):
        super().__init__(config)

        self.vae_ckpt_path = config.vae_ckpt_path or config.pretrained_model_ckpt_path
        self.text_encoder_ckpt_path = config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path
        self.text_encoder_2_ckpt_path = config.text_encoder_2_ckpt_path or config.pretrained_model_ckpt_path
        self.image_encoder_ckpt_path = config.image_encoder_ckpt_path or config.pretrained_model_ckpt_path

        self.mllm_max_length = int(config.mllm_max_length)
        self.byt5_max_length = int(config.byt5_max_length)
        self.mllm_skip_layers = int(config.mllm_skip_layers)
        self.mllm_crop_start = int(config.mllm_crop_start)
        self.vision_num_semantic_tokens = int(config.vision_num_semantic_tokens)
        self.vision_states_dim = int(config.vision_states_dim)
        self.load_vision_encoder_flag = bool(config.load_vision_encoder)

        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.training_only = config.training_only
        self.skip_device_move = config.skip_device_move

        self._mllm_tokenizer = None
        self._mllm_encoder: Optional[nn.Module] = None
        self._byt5_tokenizer = None
        self._byt5_encoder: Optional[nn.Module] = None
        self._image_processor = None
        self._scheduler = None

        self.load()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return "hunyuan_veido1p5"

    @property
    def media_type(self) -> str:
        return "tiv2iv"

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        return "diffusionrl.samplers.fsdp.hunyuan_veido1p5_sampler.FSDPHunyuanVeido1p5Sampler"

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        return "fsdp"

    @classmethod
    def supports_sglang_prompt_mode(cls) -> bool:
        # Qwen2.5-VL + ByT5 + SigLIP are not part of the SGLang prompt-only path
        # in this repo; rollout always runs through FSDP.
        return False

    @classmethod
    def default_lora_target_modules(cls) -> Optional[List[str]]:
        """Canonical LoRA targets for HunyuanVideo15TransformerBlock.

        The transformer's MMDiT block uses a ``diffusers`` ``Attention`` with
        added-KV joint-attention + a feed-forward module. Targets are the
        attention Q/K/V/out projections (both streams) and the FFN inner / outer
        linears, mirroring the SD3 / WAN convention used elsewhere in this repo.
        """
        return [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.training_only:
            logger.info("Loading HunyuanVideo-1.5 transformer only (training_only mode)...")
        else:
            logger.info("Loading HunyuanVideo-1.5 model bundle...")

        self._load_transformer()

        if self.use_lora:
            self._add_lora_adapters()

        if not self.training_only:
            self._load_vae()
            self._load_text_encoders()
            self._load_vision_encoder()
            self._load_scheduler()

    def _load_transformer(self) -> None:
        try:
            from diffusers import HunyuanVideo15Transformer3DModel
        except ImportError:
            try:
                from diffusers.models.transformers.transformer_hunyuan_video15 import (
                    HunyuanVideo15Transformer3DModel,
                )
            except ImportError as exc:
                logger.warning(
                    "Could not import HunyuanVideo15Transformer3DModel from diffusers (%s). "
                    "Install a diffusers build that ships the HunyuanVideo-1.5 transformer.",
                    exc,
                )
                self._transformer = None
                return

        try:
            transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                self.pretrained_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
        except Exception as exc:
            logger.warning("Could not load HunyuanVideo-1.5 transformer: %s", exc)
            self._transformer = None
            return

        # Fail fast on meanflow until we wire ``timestep_r`` through the bundle.
        if bool(getattr(getattr(transformer, "config", None), "use_meanflow", False)):
            raise NotImplementedError(
                "HunyuanVeido1p5ModelBundle does not yet support transformers configured with "
                "use_meanflow=True (timestep_r is not threaded through the training forward)."
            )

        self._transformer = transformer
        if self.skip_device_move:
            first_param_device = next(self._transformer.parameters()).device
            if str(first_param_device) != "cpu":
                self._transformer.to("cpu")
        else:
            # Diffusers from_pretrained leaves a few buffers in fp32; FSDP
            # _init_mp_dtypes asserts a uniform dtype across the wrapped
            # module, so realign explicitly.
            self._transformer.to(self.device, dtype=self.dtype)
        logger.info("Loaded HunyuanVideo-1.5 transformer from %s", self.pretrained_path)

    def _add_lora_adapters(self) -> None:
        if self._transformer is None:
            logger.warning("Cannot add LoRA: HunyuanVideo-1.5 transformer not loaded")
            return
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install with: pip install peft")
            return

        target_modules = self.lora_target_modules
        if target_modules is None:
            target_modules = type(self).default_lora_target_modules()
            if target_modules is None:
                raise RuntimeError(
                    "HunyuanVideo-1.5 LoRA requested but lora_target_modules is unresolved. "
                    "Pass --training.lora-target-modules or ensure "
                    f"{type(self).__name__}.default_lora_target_modules() returns a list."
                )

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        self._transformer = get_peft_model(self._transformer, lora_config)
        # Second adapter for NFT dual-adapter mechanism, mirroring SD3 / FLUX / WAN.
        self._transformer.add_adapter("old", lora_config)
        self._transformer.set_adapter("default")

        for param in self._transformer.parameters():
            if param.requires_grad and param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)

        logger.info(
            "LoRA adapters added to HunyuanVideo-1.5 (rank=%s, alpha=%s)",
            self.lora_rank,
            self.lora_alpha,
        )

    def _load_vae(self) -> None:
        try:
            from diffusers import AutoencoderKLHunyuanVideo15
        except ImportError:
            try:
                from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
                    AutoencoderKLHunyuanVideo15,
                )
            except ImportError as exc:
                logger.warning(
                    "Could not import AutoencoderKLHunyuanVideo15 from diffusers (%s).",
                    exc,
                )
                self._vae = None
                return
        try:
            self._vae = AutoencoderKLHunyuanVideo15.from_pretrained(
                self.vae_ckpt_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()
            self._vae.requires_grad_(False)
            logger.info("Loaded HunyuanVideo-1.5 VAE from %s", self.vae_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load HunyuanVideo-1.5 VAE: %s", exc)
            self._vae = None

    def _load_text_encoders(self) -> None:
        try:
            from transformers import (
                ByT5Tokenizer,
                Qwen2_5_VLTextModel,
                Qwen2Tokenizer,
                T5EncoderModel,
            )
        except ImportError as exc:
            logger.warning(
                "Could not import HunyuanVideo-1.5 text encoder dependencies (%s). "
                "Install a recent transformers release.",
                exc,
            )
            return

        try:
            self._mllm_tokenizer = Qwen2Tokenizer.from_pretrained(
                self.text_encoder_ckpt_path,
                subfolder="tokenizer",
            )
            self._mllm_encoder = Qwen2_5_VLTextModel.from_pretrained(
                self.text_encoder_ckpt_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
            )
            self._mllm_encoder.to(self.device)
            self._mllm_encoder.eval()
            self._mllm_encoder.requires_grad_(False)
            logger.info("Loaded Qwen2.5-VL MLLM encoder from %s", self.text_encoder_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load Qwen2.5-VL MLLM encoder: %s", exc)
            self._mllm_tokenizer = None
            self._mllm_encoder = None

        try:
            self._byt5_tokenizer = ByT5Tokenizer.from_pretrained(
                self.text_encoder_2_ckpt_path,
                subfolder="tokenizer_2",
            )
            self._byt5_encoder = T5EncoderModel.from_pretrained(
                self.text_encoder_2_ckpt_path,
                subfolder="text_encoder_2",
                torch_dtype=self.text_encoder_dtype,
            )
            self._byt5_encoder.to(self.device)
            self._byt5_encoder.eval()
            self._byt5_encoder.requires_grad_(False)
            logger.info("Loaded ByT5 glyph encoder from %s", self.text_encoder_2_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load ByT5 glyph encoder: %s", exc)
            self._byt5_tokenizer = None
            self._byt5_encoder = None

        if self._mllm_encoder is not None or self._byt5_encoder is not None:
            self._text_encoder = HunyuanVeido1p5TextEncoderWrapper(self)

    def _load_vision_encoder(self) -> None:
        # Pure-T2V recipes set ``load_vision_encoder=False`` to free ~1.6 GB of
        # permanently-resident GPU memory. Gating here (rather than only in
        # ``load()``) is required because ``ModelBundle.load_aux_components``
        # — invoked by the FSDP engine after ``training_only`` actor init —
        # re-calls this loader whenever ``self._vision_encoder is None``,
        # which would silently re-attach SigLIP and undo the opt-out.
        if not getattr(self, "load_vision_encoder_flag", True):
            self._vision_encoder = None
            self._image_processor = None
            logger.info(
                "Skipping SigLIP vision encoder load (load_vision_encoder=False); "
                "image_embeds will use the T2V zero-placeholder path."
            )
            return

        try:
            from transformers import SiglipImageProcessor, SiglipVisionModel
        except ImportError as exc:
            logger.warning("Could not import SigLIP vision components (%s).", exc)
            self._vision_encoder = None
            self._image_processor = None
            return

        try:
            self._image_processor = SiglipImageProcessor.from_pretrained(
                self.image_encoder_ckpt_path,
                subfolder="feature_extractor",
            )
            self._vision_encoder = SiglipVisionModel.from_pretrained(
                self.image_encoder_ckpt_path,
                subfolder="image_encoder",
                torch_dtype=self.dtype,
            )
            self._vision_encoder.to(self.device)
            self._vision_encoder.eval()
            self._vision_encoder.requires_grad_(False)
            logger.info("Loaded SigLIP image encoder from %s", self.image_encoder_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load SigLIP image encoder: %s", exc)
            self._vision_encoder = None
            self._image_processor = None

    def _load_scheduler(self) -> None:
        try:
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        except ImportError:
            return
        try:
            self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.pretrained_path,
                subfolder="scheduler",
            )
        except Exception as exc:
            logger.warning("Could not load HunyuanVideo-1.5 scheduler: %s", exc)
            self._scheduler = None

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts to ``(mllm_embeds, mllm_mask, byt5_embeds, byt5_mask)``.

        ``negative_prompt`` is intentionally **not** stacked into the returned
        tensors: HunyuanVideo-1.5 keeps positive and negative streams separate
        until CFG batching inside ``forward_denoiser``. Negative encoding is
        driven via :meth:`encode_inputs` instead.
        """
        del negative_prompt
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if self._text_encoder is None:
            self._raise_aux_component_not_loaded("text encoder")
        return self._text_encoder.encode_prompt(prompts)

    def _encode_prompt_to_input_dict(
        self,
        prompts: Union[str, List[str]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Map the four-tuple from :meth:`encode_prompt` into named fields.

        Override is required because the upstream tuple has 4 elements and the
        second one is an attention mask, not a pooled embedding.
        """
        del kwargs
        prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
        embeds, mask, embeds_2, mask_2 = self.encode_prompt(prompts_list)
        return {
            "prompt_embeds": embeds,
            "prompt_embeds_mask": mask,
            "prompt_embeds_2": embeds_2,
            "prompt_embeds_mask_2": mask_2,
        }

    # ------------------------------------------------------------------
    # Multimodal input encoding
    # ------------------------------------------------------------------

    def encode_inputs(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if video is not None:
            raise NotImplementedError("HunyuanVeido1p5ModelBundle does not support video conditioning yet")

        prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
        batch_size = len(prompts_list)

        height = kwargs.get("height")
        width = kwargs.get("width")
        num_frames = kwargs.get("num_frames")

        output = self._encode_prompt_to_input_dict(prompts_list)

        # CFG is part of HunyuanVideo-1.5's inference contract. Always emit
        # negative streams (default to empty strings) so the sampler can hand
        # them to ``forward_denoiser`` without a special-case branch.
        if negative_prompt is None:
            negative_prompts: List[str] = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt] * batch_size
        else:
            negative_prompts = list(negative_prompt)
            if len(negative_prompts) != batch_size:
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompts)} does not match prompt batch size {batch_size}"
                )

        negative_output = self._encode_prompt_to_input_dict(negative_prompts)
        output["negative_prompt_embeds"] = negative_output["prompt_embeds"]
        output["negative_prompt_embeds_mask"] = negative_output["prompt_embeds_mask"]
        output["negative_prompt_embeds_2"] = negative_output["prompt_embeds_2"]
        output["negative_prompt_embeds_mask_2"] = negative_output["prompt_embeds_mask_2"]

        cond_dict = self._build_image_conditioning(
            image=image,
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        output.update(cond_dict)
        return output

    def _build_image_conditioning(
        self,
        *,
        image: Optional[Any],
        batch_size: int,
        height: Optional[int],
        width: Optional[int],
        num_frames: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        """Produce ``image_embeds`` / ``cond_latents`` / ``cond_mask`` tensors.

        T2V uses zero placeholders matching the upstream shapes:
          * ``image_embeds``: ``[B, vision_num_semantic_tokens, vision_states_dim]``
          * ``cond_latents``: ``[B, C_lat, F_lat, H_lat, W_lat]``
          * ``cond_mask``:    ``[B, 1,    F_lat, H_lat, W_lat]``

        I2V encodes the image with the SigLIP vision encoder and bakes the
        first-frame VAE latent into ``cond_latents`` (mask first-frame=1, rest
        zero) — the exact layout consumed by the transformer's channel-dim
        concat.
        """
        target_dtype = self.dtype
        device = self.device

        if image is None:
            # T2V path: ``image_embeds`` is a latent-grid-independent zero
            # placeholder, always safe to allocate here.
            image_embeds = torch.zeros(
                batch_size,
                self.vision_num_semantic_tokens,
                self.vision_states_dim,
                device=device,
                dtype=target_dtype,
            )
            # ``cond_latents`` / ``cond_mask`` are channel-dim ``cat``-ed with
            # the noise tensor inside ``_forward_single`` and so must match
            # the sampler's actual latent grid. Eagerly size them only when
            # the caller provided all three shape kwargs (the test contract);
            # otherwise leave them ``None`` and let ``forward_denoiser``'s
            # lazy fallback (~line 714) size them from the real ``latents``
            # tensor at denoise time — the only authoritative grid source.
            #
            # Without this guard, when ``base_sampler.generate`` calls
            # ``encode_inputs`` with no shape kwargs (the production path),
            # ``_latent_shape`` would fall back to its own defaults
            # (121 frames → ``F_lat = 31``) and collide with the sampler's
            # actual latents (e.g. ``F_lat = 14`` for ``num_frames=53``) at
            # the channel-dim cat.
            if height is not None and width is not None and num_frames is not None:
                latent_t, latent_h, latent_w = self._latent_shape(
                    height=height,
                    width=width,
                    num_frames=num_frames,
                )
                latent_channels = self._latent_channels()
                cond_latents: Optional[torch.Tensor] = torch.zeros(
                    (batch_size, latent_channels, latent_t, latent_h, latent_w),
                    device=device,
                    dtype=target_dtype,
                )
                cond_mask: Optional[torch.Tensor] = torch.zeros(
                    (batch_size, 1, latent_t, latent_h, latent_w),
                    device=device,
                    dtype=target_dtype,
                )
            else:
                cond_latents = None
                cond_mask = None
            return {
                "image_embeds": image_embeds,
                "cond_latents": cond_latents,
                "cond_mask": cond_mask,
            }

        latent_t, latent_h, latent_w = self._latent_shape(
            height=height,
            width=width,
            num_frames=num_frames,
        )
        latent_channels = self._latent_channels()
        cond_shape = (batch_size, latent_channels, latent_t, latent_h, latent_w)
        mask_shape = (batch_size, 1, latent_t, latent_h, latent_w)

        if self._vision_encoder is None or self._image_processor is None:
            self._raise_aux_component_not_loaded("vision encoder")
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")

        image_tensor = self._prepare_conditioning_image_tensor(image)
        if image_tensor.shape[0] == 1 and batch_size > 1:
            image_tensor = image_tensor.repeat(batch_size, 1, 1, 1)
        elif image_tensor.shape[0] != batch_size:
            raise ValueError(
                f"Conditioning image batch size {image_tensor.shape[0]} does not match prompt batch size {batch_size}"
            )

        # SigLIP path — feed pixels in [0, 1] via the upstream processor.
        clip_images = ((image_tensor + 1.0) / 2.0).clamp(0.0, 1.0).detach().cpu()
        siglip_inputs = self._image_processor(
            images=clip_images,
            return_tensors="pt",
            do_resize=True,
            do_convert_rgb=True,
        )
        vision_dtype = next(self._vision_encoder.parameters()).dtype
        pixel_values = siglip_inputs.pixel_values.to(device=device, dtype=vision_dtype)
        with torch.no_grad():
            image_embeds = self._vision_encoder(pixel_values=pixel_values).last_hidden_state
        image_embeds = image_embeds.to(dtype=target_dtype)

        # Resize to the actual latent grid before VAE encoding (matches
        # ``HunyuanVideo15ImageProcessor.resize`` + ``_get_image_latents``).
        spatial_scale = self._spatial_compression_ratio()
        target_h = latent_h * spatial_scale
        target_w = latent_w * spatial_scale
        resized = F.interpolate(image_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False)

        vae_dtype = next(self._vae.parameters()).dtype
        with torch.no_grad():
            video_input = resized.unsqueeze(2).to(device=device, dtype=vae_dtype)
            encoded = self._vae.encode(video_input)
            latent_dist = getattr(encoded, "latent_dist", encoded)
            image_latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
        scaling_factor = float(getattr(self._vae.config, "scaling_factor", 1.0))
        image_latents = image_latents * scaling_factor
        image_latents = image_latents.to(device=device, dtype=target_dtype)

        # Place the encoded image at frame 0; remaining frames are zero, with
        # mask first-frame=1 (matches ``prepare_cond_latents_and_mask`` of the
        # I2V pipeline).
        cond_latents = torch.zeros(cond_shape, device=device, dtype=target_dtype)
        # ``image_latents`` shape: [B, C, 1, H_lat, W_lat]
        cond_latents[:, :, :1, :, :] = image_latents[:, :, :1, :, :]
        cond_mask = torch.zeros(mask_shape, device=device, dtype=target_dtype)
        cond_mask[:, :, 0, :, :] = 1.0

        return {
            "image_embeds": image_embeds,
            "cond_latents": cond_latents,
            "cond_mask": cond_mask,
        }

    def _prepare_conditioning_image_tensor(self, image: Any) -> torch.Tensor:
        """Normalise PIL / numpy / torch image inputs to ``[B, 3, H, W]`` in [-1, 1]."""
        if torch.is_tensor(image):
            image_tensor = image.detach()
        else:
            import numpy as np

            image_tensor = torch.from_numpy(np.array(image))

        if image_tensor.ndim == 3 and image_tensor.shape[0] not in (1, 3) and image_tensor.shape[-1] in (1, 3):
            image_tensor = image_tensor.permute(2, 0, 1)
        elif image_tensor.ndim == 4 and image_tensor.shape[1] not in (1, 3) and image_tensor.shape[-1] in (1, 3):
            image_tensor = image_tensor.permute(0, 3, 1, 2)

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 4:
            raise ValueError(f"Unsupported conditioning image shape: {tuple(image_tensor.shape)}")

        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)
        if image_tensor.max() > 1.5:
            image_tensor = image_tensor / 255.0
        if image_tensor.min() >= 0.0:
            image_tensor = image_tensor * 2.0 - 1.0
        return image_tensor

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward_denoiser(
        self,
        *,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        prompt_embeds = ctx.prompt_embeds
        if prompt_embeds is None:
            raise ValueError("HunyuanVeido1p5ModelBundle.forward_denoiser requires ctx.prompt_embeds.")
        prompt_embeds_mask = getattr(ctx, "prompt_embeds_mask", None)
        prompt_embeds_2 = getattr(ctx, "prompt_embeds_2", None)
        prompt_embeds_mask_2 = getattr(ctx, "prompt_embeds_mask_2", None)
        if prompt_embeds_mask is None or prompt_embeds_2 is None or prompt_embeds_mask_2 is None:
            raise ValueError(
                "HunyuanVeido1p5ModelBundle.forward_denoiser requires the full text-stream tuple "
                "(prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2)."
            )

        negative_prompt_embeds = getattr(ctx, "negative_prompt_embeds", None)
        negative_prompt_embeds_mask = getattr(ctx, "negative_prompt_embeds_mask", None)
        negative_prompt_embeds_2 = getattr(ctx, "negative_prompt_embeds_2", None)
        negative_prompt_embeds_mask_2 = getattr(ctx, "negative_prompt_embeds_mask_2", None)

        image_embeds = getattr(ctx, "image_embeds", None)
        cond_latents = getattr(ctx, "cond_latents", None)
        cond_mask = getattr(ctx, "cond_mask", None)
        attention_kwargs = getattr(ctx, "attention_kwargs", None)
        guidance_scale = float(getattr(ctx, "guidance_scale", 6.0))

        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype

        if cond_latents is None or cond_mask is None:
            cond_latents = torch.zeros_like(latents).to(dtype)
            cond_mask = torch.zeros(
                batch_size,
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                device=device,
                dtype=dtype,
            )
        else:
            cond_latents = cond_latents.to(device=device, dtype=dtype)
            cond_mask = cond_mask.to(device=device, dtype=dtype)

        if image_embeds is None:
            image_embeds = torch.zeros(
                batch_size,
                self.vision_num_semantic_tokens,
                self.vision_states_dim,
                device=device,
                dtype=dtype,
            )
        else:
            image_embeds = image_embeds.to(device=device, dtype=dtype)

        timestep = self._prepare_training_timestep(sigma, batch_size, device, dtype) * self.TIMESTEP_SCALE

        with self._build_training_autocast_ctx(device):
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_pred = self._forward_with_cfg(
                    latents=latents,
                    cond_latents=cond_latents,
                    cond_mask=cond_mask,
                    image_embeds=image_embeds,
                    timestep=timestep,
                    dtype=dtype,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    prompt_embeds_2=prompt_embeds_2,
                    prompt_embeds_mask_2=prompt_embeds_mask_2,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                    negative_prompt_embeds_2=negative_prompt_embeds_2,
                    negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
                    attention_kwargs=attention_kwargs,
                )
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            return self._forward_single(
                latents=latents,
                cond_latents=cond_latents,
                cond_mask=cond_mask,
                image_embeds=image_embeds,
                timestep=timestep,
                dtype=dtype,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                prompt_embeds_2=prompt_embeds_2,
                prompt_embeds_mask_2=prompt_embeds_mask_2,
                attention_kwargs=attention_kwargs,
            )

    def _forward_single(
        self,
        *,
        latents: torch.Tensor,
        cond_latents: torch.Tensor,
        cond_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        timestep: torch.Tensor,
        dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_embeds_mask_2: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        latent_model_input = torch.cat([latents.to(dtype), cond_latents, cond_mask], dim=1)
        kwargs: Dict[str, Any] = {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_embeds_mask,
            "encoder_hidden_states_2": prompt_embeds_2,
            "encoder_attention_mask_2": prompt_embeds_mask_2,
            "image_embeds": image_embeds,
            "return_dict": False,
        }
        if attention_kwargs is not None:
            kwargs["attention_kwargs"] = attention_kwargs
        return self.transformer(**kwargs)[0]

    def _forward_with_cfg(
        self,
        *,
        latents: torch.Tensor,
        cond_latents: torch.Tensor,
        cond_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        timestep: torch.Tensor,
        dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_embeds_mask_2: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        negative_prompt_embeds_2: Optional[torch.Tensor],
        negative_prompt_embeds_mask_2: Optional[torch.Tensor],
        attention_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if negative_prompt_embeds_mask is None:
            negative_prompt_embeds_mask = torch.ones_like(prompt_embeds_mask)
        if negative_prompt_embeds_2 is None:
            negative_prompt_embeds_2 = torch.zeros_like(prompt_embeds_2)
        if negative_prompt_embeds_mask_2 is None:
            negative_prompt_embeds_mask_2 = torch.zeros_like(prompt_embeds_mask_2)

        latent_model_input = torch.cat([latents.to(dtype), cond_latents, cond_mask], dim=1)
        latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
        timestep_batched = torch.cat([timestep, timestep], dim=0)
        encoder_hidden_states = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        encoder_attention_mask = torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0)
        encoder_hidden_states_2 = torch.cat([prompt_embeds_2, negative_prompt_embeds_2], dim=0)
        encoder_attention_mask_2 = torch.cat([prompt_embeds_mask_2, negative_prompt_embeds_mask_2], dim=0)
        image_embeds_batched = torch.cat([image_embeds, image_embeds], dim=0)

        kwargs: Dict[str, Any] = {
            "hidden_states": latent_model_input,
            "timestep": timestep_batched,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_hidden_states_2": encoder_hidden_states_2,
            "encoder_attention_mask_2": encoder_attention_mask_2,
            "image_embeds": image_embeds_batched,
            "return_dict": False,
        }
        if attention_kwargs is not None:
            kwargs["attention_kwargs"] = attention_kwargs
        return self.transformer(**kwargs)[0]

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")
        vae_dtype = next(self._vae.parameters()).dtype
        scale = float(getattr(self._vae.config, "scaling_factor", 1.0))
        with torch.no_grad():
            decoded = self._vae.decode(latents.to(dtype=vae_dtype) / scale, return_dict=False)[0]
        return decoded

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")
        vae_dtype = next(self._vae.parameters()).dtype
        with torch.no_grad():
            encoded = self._vae.encode(images.to(device=self.device, dtype=vae_dtype))
            latent_dist = getattr(encoded, "latent_dist", encoded)
            latents = latent_dist.sample() if hasattr(latent_dist, "sample") else latent_dist
            scale = float(getattr(self._vae.config, "scaling_factor", 1.0))
            return latents * scale

    # ------------------------------------------------------------------
    # Sampler hooks
    # ------------------------------------------------------------------

    def get_sampler_extra_kwargs(self) -> Dict[str, Any]:
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is None:
            return {}
        return {"scheduler": scheduler}

    def _spatial_compression_ratio(self) -> int:
        vae_cfg = getattr(self._vae, "config", None) if self._vae is not None else None
        if self._vae is not None:
            attr = getattr(self._vae, "spatial_compression_ratio", None)
            if attr is not None:
                return int(attr)
        return int(getattr(vae_cfg, "spatial_compression_ratio", self.DEFAULT_SPATIAL_DOWNSAMPLE))

    def _temporal_compression_ratio(self) -> int:
        if self._vae is not None:
            attr = getattr(self._vae, "temporal_compression_ratio", None)
            if attr is not None:
                return int(attr)
        vae_cfg = getattr(self._vae, "config", None) if self._vae is not None else None
        return int(getattr(vae_cfg, "temporal_compression_ratio", self.DEFAULT_TEMPORAL_DOWNSAMPLE))

    def _latent_channels(self) -> int:
        if self._vae is not None:
            channels = getattr(self._vae.config, "latent_channels", None)
            if channels is not None:
                return int(channels)
        config = getattr(self.transformer, "config", None) if self._transformer is not None else None
        return int(getattr(config, "out_channels", 32))

    def _latent_shape(
        self,
        *,
        height: Optional[int],
        width: Optional[int],
        num_frames: Optional[int],
    ) -> Tuple[int, int, int]:
        spatial = self._spatial_compression_ratio()
        temporal = self._temporal_compression_ratio()
        height_v = int(height) if height is not None else 480
        width_v = int(width) if width is not None else 848
        num_frames_v = int(num_frames) if num_frames is not None else 121
        latent_t = (num_frames_v - 1) // temporal + 1
        latent_h = max(1, height_v // spatial)
        latent_w = max(1, width_v // spatial)
        return latent_t, latent_h, latent_w

    # ------------------------------------------------------------------
    # FSDP module discovery
    # ------------------------------------------------------------------

    def get_no_split_modules(self) -> Tuple[Type[nn.Module], ...]:
        try:
            from diffusers.models.transformers.transformer_hunyuan_video15 import (
                HunyuanVideo15PatchEmbed,
                HunyuanVideo15TokenRefiner,
                HunyuanVideo15TransformerBlock,
            )
        except ImportError:
            return ()
        return (
            HunyuanVideo15TransformerBlock,
            HunyuanVideo15PatchEmbed,
            HunyuanVideo15TokenRefiner,
        )

    def iter_offloadable_modules(self, include_transformer: bool = True):
        # Surface the additional Hunyuan-1.5-specific encoders so the rollout
        # actor can offload them alongside the transformer/VAE when memory
        # pressure forces a sleep cycle.
        seen = set()
        for name, value in self.__dict__.items():
            if not isinstance(value, nn.Module):
                continue
            base_name = name.lstrip("_").lower()
            if not include_transformer and "transformer" in base_name:
                continue
            seen.add(name)
            yield name, value
        # Defer to the base implementation for any names we haven't emitted.
        for name, value in super().iter_offloadable_modules(include_transformer=include_transformer):
            if name in seen:
                continue
            yield name, value


# ---------------------------------------------------------------------------
# Text encoder wrapper (Qwen2.5-VL + ByT5)
# ---------------------------------------------------------------------------


class HunyuanVeido1p5TextEncoderWrapper:
    """Encapsulates the dual-stream prompt encoder used by HunyuanVideo-1.5.

    Returns a 4-tuple ``(mllm_embeds, mllm_mask, byt5_embeds, byt5_mask)``.
    Bundle ``encode_prompt`` simply forwards to this wrapper.
    """

    def __init__(self, bundle: HunyuanVeido1p5ModelBundle) -> None:
        self.bundle = bundle

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @property
    def dtype(self) -> torch.dtype:
        return self.bundle.text_encoder_dtype

    def encode_prompt(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        if bundle._mllm_encoder is None or bundle._mllm_tokenizer is None:
            raise RuntimeError(
                "HunyuanVideo-1.5 MLLM encoder is unavailable. "
                "Ensure the Qwen2.5-VL components are loaded before calling encode_prompt."
            )
        if bundle._byt5_encoder is None or bundle._byt5_tokenizer is None:
            raise RuntimeError(
                "HunyuanVideo-1.5 ByT5 glyph encoder is unavailable. "
                "Ensure text_encoder_2 / tokenizer_2 components are loaded before calling encode_prompt."
            )

        mllm_embeds, mllm_mask = self._encode_mllm(prompts)
        byt5_embeds, byt5_mask = self._encode_byt5(prompts)
        return mllm_embeds, mllm_mask, byt5_embeds, byt5_mask

    def _encode_mllm(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        tokenizer = bundle._mllm_tokenizer
        text_encoder = bundle._mllm_encoder
        device = bundle.device
        crop_start = int(bundle.mllm_crop_start)

        chat = _format_chat_template(prompts, _DEFAULT_SYSTEM_MESSAGE)
        text_inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=int(bundle.mllm_max_length) + crop_start,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device=device)
        attention_mask = text_inputs.attention_mask.to(device=device)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        prompt_embeds = outputs.hidden_states[-(int(bundle.mllm_skip_layers) + 1)]

        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        return prompt_embeds, attention_mask

    def _encode_byt5(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        tokenizer = bundle._byt5_tokenizer
        text_encoder = bundle._byt5_encoder
        device = bundle.device
        max_length = int(bundle.byt5_max_length)
        d_model = int(getattr(text_encoder.config, "d_model", 1472))

        embeds_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        for raw in prompts:
            glyph = _extract_glyph_texts(raw or "")
            if glyph is None:
                emb = torch.zeros(1, max_length, d_model, device=device, dtype=text_encoder.dtype)
                mask = torch.zeros(1, max_length, device=device, dtype=torch.int64)
            else:
                tokens = tokenizer(
                    glyph,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                input_ids = tokens.input_ids.to(device=device)
                attn = tokens.attention_mask.to(device=device)
                with torch.no_grad():
                    out = text_encoder(input_ids=input_ids, attention_mask=attn.float())[0]
                emb = out.to(device=device)
                mask = attn.to(device=device)
            embeds_list.append(emb)
            masks_list.append(mask)

        prompt_embeds_2 = torch.cat(embeds_list, dim=0).to(dtype=self.dtype)
        prompt_embeds_mask_2 = torch.cat(masks_list, dim=0)
        return prompt_embeds_2, prompt_embeds_mask_2

    def to(self, device: Union[str, torch.device]) -> "HunyuanVeido1p5TextEncoderWrapper":
        if self.bundle._mllm_encoder is not None:
            self.bundle._mllm_encoder.to(device)
        if self.bundle._byt5_encoder is not None:
            self.bundle._byt5_encoder.to(device)
        return self


__all__ = [
    "HunyuanVeido1p5ModelBundle",
    "HunyuanVeido1p5ModelBundleConfig",
    "HunyuanVeido1p5TextEncoderWrapper",
]
