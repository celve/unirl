"""WAN 2.1 multi-task model bundle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusionrl.config.registration import register_config

from .base import ModelBundle
from .config import ModelBundleConfig

logger = logging.getLogger(__name__)


@register_config(
    group="model",
    name="wan21",
    target="diffusionrl.models.wan21.WAN21ModelBundle",
)
@dataclass
class WAN21ModelBundleConfig(ModelBundleConfig):
    image_encoder_ckpt_path: Optional[str] = None
    max_sequence_length: int = 512


class WAN21ModelBundle(ModelBundle):
    """WAN 2.1 bundle supporting text-to-video and image-conditioned video tasks."""

    TIMESTEP_SCALE = 1000.0

    def __init__(self, config: WAN21ModelBundleConfig):
        super().__init__(config)
        self.vae_ckpt_path = config.vae_ckpt_path or config.pretrained_model_ckpt_path
        self.text_encoder_ckpt_path = config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path
        self.image_encoder_ckpt_path = config.image_encoder_ckpt_path or config.pretrained_model_ckpt_path
        self.max_sequence_length = config.max_sequence_length
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.training_only = config.training_only
        self.skip_device_move = config.skip_device_move

        self._umt5_encoder: Optional[nn.Module] = None
        self._tokenizer = None
        self._clip_vision_processor = None
        self._is_i2v_model: bool = False
        self._uses_clip_vision: bool = False

        self.load()

    @property
    def model_type(self) -> str:
        return "wan21"

    @property
    def media_type(self) -> str:
        return "tiv2iv"

    @property
    def is_i2v_capable(self) -> bool:
        """Whether the loaded transformer supports image-to-video conditioning."""
        return self._is_i2v_model

    @property
    def uses_clip_vision(self) -> bool:
        """Whether this I2V model uses CLIP vision encoder for image cross-attention."""
        return self._uses_clip_vision

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        return "diffusionrl.samplers.fsdp.wan_sampler.FSDPWanSampler"

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        return "fsdp"

    @classmethod
    def supports_sglang_prompt_mode(cls) -> bool:
        return False

    @classmethod
    def default_lora_target_modules(cls) -> Optional[List[str]]:
        """WAN transformer LoRA targets from diffusers WanTransformer3DModel.

        Includes self-attention, cross-attention, feed-forward, output
        projection, and the optional added image key/value projections used by
        I2V checkpoints.
        """
        return [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "attn2.add_k_proj",
            "attn2.add_v_proj",
            "ffn.net.0.proj",
            "ffn.net.2",
            "proj_out",
        ]

    def load(self) -> None:
        if self.training_only:
            logger.info("Loading WAN transformer only (training_only mode)...")
        else:
            logger.info("Loading WAN model bundle...")

        self._load_transformer()
        if self.use_lora:
            self._add_lora_adapters()
        if not self.training_only:
            self._load_vae()
            self._load_text_encoder()
            self._load_vision_encoder()

    def _load_transformer(self) -> None:
        try:
            try:
                from diffusers import WanTransformer3DModel
            except ImportError:
                from diffusers import AutoModel

                WanTransformer3DModel = AutoModel

            self._transformer = WanTransformer3DModel.from_pretrained(
                self.pretrained_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            if self.skip_device_move:
                first_param_device = next(self._transformer.parameters()).device
                if str(first_param_device) != "cpu":
                    self._transformer.to("cpu")
            else:
                # Dtype unification matters here even though from_pretrained got
                # torch_dtype=self.dtype: diffusers leaves some parameters /
                # buffers (timestep embeddings, RoPE freqs, ...) in fp32, and
                # FSDP's _init_mp_dtypes asserts a uniform original-param dtype
                # across the wrapped module.
                self._transformer.to(self.device, dtype=self.dtype)

            # Detect I2V capability and cache CLIP vision flag
            config = getattr(self._transformer, "config", None)
            added_kv_proj_dim = getattr(config, "added_kv_proj_dim", None)
            in_channels = getattr(config, "in_channels", None)
            out_channels = getattr(config, "out_channels", None)
            self._is_i2v_model = (added_kv_proj_dim is not None and added_kv_proj_dim > 0) or (
                in_channels is not None and out_channels is not None and in_channels > out_channels
            )
            image_dim = getattr(config, "image_dim", None)
            self._uses_clip_vision = image_dim is not None and image_dim > 0
            variant_str = "I2V" if self._is_i2v_model else "T2V"
            logger.info("Loaded WAN transformer from %s (variant: %s)", self.pretrained_path, variant_str)
        except Exception as exc:
            logger.warning("Could not load WAN transformer: %s", exc)
            self._transformer = None

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to the WAN transformer for efficient fine-tuning."""
        if self._transformer is None:
            logger.warning("Cannot add LoRA: WAN transformer not loaded")
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
                    "WAN LoRA requested but lora_target_modules is unresolved. "
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
        self._transformer.add_adapter("old", lora_config)
        self._transformer.set_adapter("default")

        for param in self._transformer.parameters():
            if param.requires_grad and param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)

        logger.info("LoRA adapters added to WAN (rank=%s, alpha=%s)", self.lora_rank, self.lora_alpha)

    def _load_vae(self) -> None:
        try:
            try:
                from diffusers import AutoencoderKLWan
            except ImportError:
                from diffusers import AutoModel

                AutoencoderKLWan = AutoModel

            self._vae = AutoencoderKLWan.from_pretrained(
                self.vae_ckpt_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()
            self._vae.requires_grad_(False)
            self._vae_norm_cache: Dict[Tuple[torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
            logger.info("Loaded WAN VAE from %s", self.vae_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load WAN VAE: %s", exc)
            self._vae = None

    def _vae_norm_params(self, device: torch.device, dtype: torch.dtype) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return cached (mean, std_reciprocal) tensors for VAE latent normalization.

        Returns None if the VAE does not use per-channel normalization.
        """
        latents_mean = getattr(self._vae.config, "latents_mean", None)
        latents_std = getattr(self._vae.config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            return None

        cache_key = (device, dtype)
        cached = self._vae_norm_cache.get(cache_key)
        if cached is not None:
            return cached

        z_dim = int(getattr(self._vae.config, "z_dim", 16))
        mean = torch.tensor(latents_mean, device=device, dtype=dtype).view(1, z_dim, 1, 1, 1)
        std_reciprocal = (1.0 / torch.tensor(latents_std, device=device, dtype=dtype)).view(1, z_dim, 1, 1, 1)
        self._vae_norm_cache[cache_key] = (mean, std_reciprocal)
        return mean, std_reciprocal

    def _load_text_encoder(self) -> None:
        try:
            from transformers import AutoTokenizer

            try:
                from transformers import UMT5EncoderModel as WANTextEncoderModel
            except ImportError:
                from transformers import T5EncoderModel as WANTextEncoderModel

            self._umt5_encoder = WANTextEncoderModel.from_pretrained(
                self.text_encoder_ckpt_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
            )
            self._umt5_encoder.to(self.device)
            self._umt5_encoder.eval()
            self._umt5_encoder.requires_grad_(False)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.text_encoder_ckpt_path,
                subfolder="tokenizer",
            )
            self._text_encoder = WANTextEncoderWrapper(
                encoder=self._umt5_encoder,
                tokenizer=self._tokenizer,
                device=self.device,
                dtype=self.text_encoder_dtype,
                max_length=self.max_sequence_length,
            )
            logger.info("Loaded WAN text encoder from %s", self.text_encoder_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load WAN text encoder: %s", exc)
            self._umt5_encoder = None
            self._tokenizer = None
            self._text_encoder = None

    def _load_vision_encoder(self) -> None:
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self._clip_vision_processor = CLIPImageProcessor.from_pretrained(
                self.image_encoder_ckpt_path,
                subfolder="image_processor",
            )
            self._vision_encoder = CLIPVisionModel.from_pretrained(
                self.image_encoder_ckpt_path,
                subfolder="image_encoder",
                torch_dtype=self.dtype,
            )
            self._vision_encoder.to(self.device)
            self._vision_encoder.eval()
            self._vision_encoder.requires_grad_(False)
            logger.info("Loaded WAN vision encoder from %s", self.image_encoder_ckpt_path)
        except Exception as exc:
            logger.warning("Could not load WAN vision encoder: %s", exc)
            self._clip_vision_processor = None
            self._vision_encoder = None

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._text_encoder is None:
            self._raise_aux_component_not_loaded("text encoder")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        negatives = None
        if isinstance(negative_prompt, str):
            negatives = [negative_prompt] * len(prompts)
        elif negative_prompt is not None:
            negatives = list(negative_prompt)
            if len(negatives) != len(prompts):
                raise ValueError(
                    f"negative_prompt batch size {len(negatives)} does not match prompt batch size {len(prompts)}"
                )
        return self._text_encoder.encode_prompt(prompts, negatives)

    def encode_inputs(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if video is not None:
            raise NotImplementedError("WAN21ModelBundle does not support video conditioning yet")
        prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)

        prompt_embeds, _ = self.encode_prompt(prompts_list)
        output: Dict[str, torch.Tensor] = {
            "prompt_embeds": prompt_embeds,
        }

        if negative_prompt is None:
            negative_prompts = [""] * len(prompts_list)
        elif isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt] * len(prompts_list)
        else:
            negative_prompts = list(negative_prompt)
            if len(negative_prompts) != len(prompts_list):
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompts)} does not match prompt batch size {len(prompts_list)}"
                )
        negative_prompt_embeds, _ = self.encode_prompt(negative_prompts)
        output["negative_prompt_embeds"] = negative_prompt_embeds

        if image is not None:
            output.update(
                self.encode_image_inputs(
                    image,
                    batch_size=len(prompts_list),
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    num_frames=kwargs.get("num_frames"),
                    expand_timesteps=kwargs.get("expand_timesteps"),
                )
            )
        return output

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")
        vae_dtype = next(self._vae.parameters()).dtype
        with torch.no_grad():
            latents = latents.to(dtype=vae_dtype)
            norm = self._vae_norm_params(latents.device, vae_dtype)
            if norm is not None:
                mean, std_reciprocal = norm
                latents = latents / std_reciprocal + mean
            else:
                scaling_factor = getattr(self._vae.config, "scaling_factor", 1.0)
                shift_factor = getattr(self._vae.config, "shift_factor", 0.0)
                latents = latents / scaling_factor
                if shift_factor and not (isinstance(shift_factor, (int, float)) and shift_factor == 0.0):
                    latents = latents + shift_factor
            return self._vae.decode(latents).sample

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")
        vae_dtype = next(self._vae.parameters()).dtype
        with torch.no_grad():
            encoded = self._vae.encode(images.to(device=self.device, dtype=vae_dtype))
            latent_dist = getattr(encoded, "latent_dist", encoded)
            latents = latent_dist.sample()
            norm = self._vae_norm_params(latents.device, vae_dtype)
            if norm is not None:
                mean, std_reciprocal = norm
                latents = (latents - mean) * std_reciprocal
            else:
                scale = getattr(self._vae.config, "scaling_factor", 1.0)
                latents = latents * scale
            return latents

    def encode_image_inputs(
        self,
        image: Any,
        *,
        batch_size: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        expand_timesteps: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")

        image_tensor = self._prepare_conditioning_image_tensor(image)
        if image_tensor.shape[0] == 1 and batch_size > 1:
            image_tensor = image_tensor.repeat(batch_size, 1, 1, 1)
        elif image_tensor.shape[0] != batch_size:
            raise ValueError(
                f"Conditioning image batch size {image_tensor.shape[0]} does not match prompt batch size {batch_size}"
            )

        output: Dict[str, torch.Tensor] = {}

        # WAN 2.1 I2V: encode image via CLIP vision for cross-attention
        if self.uses_clip_vision:
            if self._vision_encoder is None or self._clip_vision_processor is None:
                missing = []
                if self._vision_encoder is None:
                    missing.append("_vision_encoder")
                if self._clip_vision_processor is None:
                    missing.append("_clip_vision_processor")
                self._raise_aux_component_not_loaded("vision encoder/image processor", missing=missing)

            clip_images = ((image_tensor + 1.0) / 2.0).clamp(0.0, 1.0).detach().cpu()
            image_inputs = self._clip_vision_processor(images=clip_images, return_tensors="pt")
            vision_dtype = next(self._vision_encoder.parameters()).dtype
            pixel_values = image_inputs.pixel_values.to(self.device, dtype=vision_dtype)

            with torch.no_grad():
                image_embeds = self._vision_encoder(pixel_values=pixel_values, output_hidden_states=True).hidden_states[
                    -2
                ]
            output["encoder_hidden_states_image"] = image_embeds.to(dtype=self.dtype)

        # Both paths: VAE latent conditioning (first frame + mask)
        image_conditioning_latents, first_frame_mask = self.prepare_image_conditioning_latents(
            image_tensor,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=self.dtype,
            expand_timesteps=expand_timesteps,
        )
        output["image_conditioning_latents"] = image_conditioning_latents
        if first_frame_mask is not None:
            output["first_frame_mask"] = first_frame_mask
        return output

    def prepare_image_conditioning_latents(
        self,
        image: torch.Tensor,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        expand_timesteps: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")
        if image.ndim != 4:
            raise ValueError(f"Expected conditioning image tensor [B, C, H, W], got {tuple(image.shape)}")

        batch_size = image.shape[0]
        if expand_timesteps is None:
            expand_timesteps = bool(getattr(getattr(self._transformer, "config", None), "expand_timesteps", False))

        spatial_scale = int(getattr(self._vae.config, "scale_factor_spatial", 8))
        temporal_scale = int(getattr(self._vae.config, "scale_factor_temporal", 4))
        num_frames = int(num_frames or (temporal_scale * 20 + 1))

        if not expand_timesteps and (num_frames - 1) % temporal_scale != 0:
            raise ValueError(
                f"I2V image conditioning requires (num_frames - 1) % temporal_scale == 0. "
                f"Got num_frames={num_frames}, temporal_scale={temporal_scale}. "
                f"Valid num_frames: {temporal_scale + 1}, {2 * temporal_scale + 1}, "
                f"{3 * temporal_scale + 1}, ... (i.e., 4k+1 for temporal_scale=4)."
            )

        target_height = int(height or image.shape[-2])
        target_width = int(width or image.shape[-1])
        latent_height = max(1, target_height // spatial_scale)
        latent_width = max(1, target_width // spatial_scale)
        target_height = latent_height * spatial_scale
        target_width = latent_width * spatial_scale

        resized = F.interpolate(image, size=(target_height, target_width), mode="bicubic", align_corners=False)
        if expand_timesteps:
            video_condition = resized.unsqueeze(2)
        else:
            video_condition = torch.cat(
                [
                    resized.unsqueeze(2),
                    resized.new_zeros(
                        batch_size,
                        resized.shape[1],
                        max(0, num_frames - 1),
                        target_height,
                        target_width,
                    ),
                ],
                dim=2,
            )

        vae_dtype = next(self._vae.parameters()).dtype
        with torch.no_grad():
            encoded = self._vae.encode(video_condition.to(device=self.device, dtype=vae_dtype))
            latent_dist = getattr(encoded, "latent_dist", encoded)
            latent_condition = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()

        out_dtype = dtype or self.dtype
        latent_condition = latent_condition.to(device=self.device, dtype=out_dtype)
        norm = self._vae_norm_params(self.device, out_dtype)
        if norm is not None:
            mean, std_reciprocal = norm
            latent_condition = (latent_condition - mean) * std_reciprocal
        else:
            scale = getattr(self._vae.config, "scaling_factor", None)
            if scale is not None:
                latent_condition = latent_condition * float(scale)

        if expand_timesteps:
            first_frame_mask = torch.ones(
                batch_size,
                1,
                latent_condition.shape[2],
                latent_height,
                latent_width,
                device=self.device,
                dtype=out_dtype,
            )
            first_frame_mask[:, :, 0, :, :] = 0
            return latent_condition, first_frame_mask

        mask = torch.ones(batch_size, 1, num_frames, latent_height, latent_width, device=self.device, dtype=out_dtype)
        mask[:, :, 1:, :, :] = 0
        first_frame_mask = torch.repeat_interleave(mask[:, :, 0:1, :, :], repeats=temporal_scale, dim=2)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:, :, :]], dim=2)
        mask = mask.view(batch_size, -1, temporal_scale, latent_height, latent_width)
        mask = mask.transpose(1, 2).contiguous()
        return torch.cat([mask, latent_condition], dim=1), None

    def forward_denoiser(self, *, latents: torch.Tensor, sigma: torch.Tensor, ctx) -> torch.Tensor:
        prompt_embeds = ctx.prompt_embeds
        if prompt_embeds is None:
            raise ValueError("WAN21ModelBundle.forward_denoiser requires ctx.prompt_embeds.")

        guidance_scale = float(getattr(ctx, "guidance_scale", 5.0))
        negative_prompt_embeds = getattr(ctx, "negative_prompt_embeds", None)
        encoder_hidden_states_image = getattr(ctx, "encoder_hidden_states_image", None)
        negative_encoder_hidden_states_image = getattr(ctx, "negative_encoder_hidden_states_image", None)
        image_conditioning_latents = getattr(ctx, "image_conditioning_latents", None)
        first_frame_mask = getattr(ctx, "first_frame_mask", None)
        attention_kwargs = getattr(ctx, "attention_kwargs", None)

        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        timestep = self._prepare_training_timestep(sigma, batch_size, device, dtype) * self.TIMESTEP_SCALE

        with self._build_training_autocast_ctx(device):
            if guidance_scale > 1.0:
                uncond_embeds = (
                    negative_prompt_embeds if negative_prompt_embeds is not None else torch.zeros_like(prompt_embeds)
                )
                latents_batched = torch.cat([latents, latents], dim=0)
                embeds_batched = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                timestep_batched = torch.cat([timestep, timestep], dim=0)
                hidden_states_batched = latents_batched.to(dtype)

                if first_frame_mask is not None and image_conditioning_latents is not None:
                    mask_batched = torch.cat([first_frame_mask, first_frame_mask], dim=0).to(device=device, dtype=dtype)
                    condition_batched = torch.cat(
                        [image_conditioning_latents, image_conditioning_latents],
                        dim=0,
                    ).to(device=device, dtype=dtype)
                    hidden_states_batched = (
                        1 - mask_batched
                    ) * condition_batched + mask_batched * hidden_states_batched
                    spatial_patch = self._spatial_patch_size()
                    timestep_template = (
                        first_frame_mask[0, 0, :, ::spatial_patch, ::spatial_patch].to(device=device, dtype=dtype)
                        * timestep[0]
                    ).flatten()
                    timestep_batched = timestep_template.unsqueeze(0).expand(latents_batched.shape[0], -1)
                elif image_conditioning_latents is not None:
                    hidden_states_batched = torch.cat(
                        [
                            hidden_states_batched,
                            torch.cat([image_conditioning_latents, image_conditioning_latents], dim=0).to(
                                device=device,
                                dtype=dtype,
                            ),
                        ],
                        dim=1,
                    )

                model_kwargs: Dict[str, Any] = {
                    "hidden_states": hidden_states_batched,
                    "encoder_hidden_states": embeds_batched,
                    "timestep": timestep_batched,
                    "return_dict": False,
                }
                if encoder_hidden_states_image is not None:
                    uncond_image_embeds = (
                        negative_encoder_hidden_states_image
                        if negative_encoder_hidden_states_image is not None
                        else encoder_hidden_states_image
                    )
                    model_kwargs["encoder_hidden_states_image"] = torch.cat(
                        [uncond_image_embeds, encoder_hidden_states_image],
                        dim=0,
                    )
                if attention_kwargs is not None:
                    model_kwargs["attention_kwargs"] = attention_kwargs

                noise_pred = self.transformer(**model_kwargs)[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            return self.transformer(**self._prepare_forward_kwargs(latents, timestep, prompt_embeds, ctx))[0]

    def _prepare_forward_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        ctx: Any,
    ) -> Dict[str, Any]:
        dtype = prompt_embeds.dtype
        device = latents.device
        hidden_states = latents.to(dtype)
        image_conditioning_latents = getattr(ctx, "image_conditioning_latents", None)
        first_frame_mask = getattr(ctx, "first_frame_mask", None)
        if first_frame_mask is not None and image_conditioning_latents is not None:
            mask = first_frame_mask.to(device=device, dtype=dtype)
            condition = image_conditioning_latents.to(device=device, dtype=dtype)
            hidden_states = (1 - mask) * condition + mask * hidden_states
            spatial_patch = self._spatial_patch_size()
            timestep_template = (
                first_frame_mask[0, 0, :, ::spatial_patch, ::spatial_patch].to(device=device, dtype=dtype) * timestep[0]
            ).flatten()
            timestep = timestep_template.unsqueeze(0).expand(latents.shape[0], -1)
        elif image_conditioning_latents is not None:
            hidden_states = torch.cat([hidden_states, image_conditioning_latents.to(device=device, dtype=dtype)], dim=1)

        model_kwargs: Dict[str, Any] = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep,
            "return_dict": False,
        }
        encoder_hidden_states_image = getattr(ctx, "encoder_hidden_states_image", None)
        if encoder_hidden_states_image is not None:
            model_kwargs["encoder_hidden_states_image"] = encoder_hidden_states_image
        attention_kwargs = getattr(ctx, "attention_kwargs", None)
        if attention_kwargs is not None:
            model_kwargs["attention_kwargs"] = attention_kwargs
        return model_kwargs

    def _prepare_conditioning_image_tensor(self, image: Any) -> torch.Tensor:
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

    def _spatial_patch_size(self) -> int:
        patch_size = getattr(getattr(self.transformer, "config", None), "patch_size", (1, 2, 2))
        if isinstance(patch_size, (list, tuple)):
            return int(patch_size[-1])
        return int(patch_size)

    def get_no_split_modules(self) -> Tuple[Type[nn.Module], ...]:
        try:
            from diffusers.models.transformers import transformer_wan

            block_types = []
            for name in ("WanTransformerBlock", "WanAttnProcessorBlock"):
                block = getattr(transformer_wan, name, None)
                if isinstance(block, type) and issubclass(block, nn.Module):
                    block_types.append(block)
            return tuple(block_types)
        except Exception:
            return tuple()


class WANTextEncoderWrapper:
    """Wrapper for WAN's UMT5/T5 text encoder."""

    def __init__(
        self,
        encoder: Optional[nn.Module],
        tokenizer,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is None or self.tokenizer is None:
            batch_size = len(prompt)
            return (
                torch.zeros(batch_size, self.max_length, 4096, dtype=self.dtype, device=self.device),
                torch.ones(batch_size, self.max_length, dtype=torch.long, device=self.device),
            )

        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            prompt_embeds = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(
                dtype=self.dtype
            )
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(dtype=prompt_embeds.dtype)

        if negative_prompt is not None:
            negative_embeds, negative_mask = self.encode_prompt(negative_prompt, None)
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            attention_mask = torch.cat([negative_mask, attention_mask], dim=0)

        return prompt_embeds, attention_mask

    def to(self, device: Union[str, torch.device]) -> "WANTextEncoderWrapper":
        self.device = device
        if self.encoder is not None:
            self.encoder.to(device)
        return self
